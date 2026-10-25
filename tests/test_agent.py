"""Agent tests: the real graph, tools and retriever, with a scripted model instead of an LLM.

ScriptedModel returns pre-written replies (tool requests or a final answer) in
order and records every conversation it was sent, so each test can check both
what the agent did and what the model saw.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from app import main
from app.agent import AgentRunner, build_agent
from app.agent.graph import is_refusal
from app.agent.models import make_chat_model
from app.agent.prompts import BUDGET_NOTICE, NOT_EXECUTED
from app.agent.tools import READ_SECTION, SEARCH_DOCS, tool_specs
from app.config import Settings
from app.pipeline import RAGPipeline
from app.retrieval import Retriever
from tests.conftest import FakeReranker

NOT_FOUND = "I could not find this in the Shopware documentation."


class ScriptedModel:
    def __init__(self, replies: list[AIMessage]):
        self.replies = list(replies)
        self.seen: list[list] = []
        self.tools: list[dict] | None = None

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        if not self.replies:
            raise AssertionError("model called more often than scripted")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def call(name: str, id: str, **args) -> dict:
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


def ai(text: str = "", *calls: dict) -> AIMessage:
    return AIMessage(content=text, tool_calls=list(calls),
                     usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
                     response_metadata={"model_name": "scripted-model"})


def tool_results(model: ScriptedModel, turn: int) -> list[ToolMessage]:
    return [m for m in model.seen[turn] if isinstance(m, ToolMessage)]


@pytest.fixture
def make_runner(settings, indexed):
    qdrant, dense, sparse = indexed
    retriever = Retriever(settings, qdrant, dense, sparse, FakeReranker())

    def make(replies, **overrides):
        for k, v in overrides.items():
            setattr(settings, k, v)
        model = ScriptedModel(replies)
        return AgentRunner(settings, retriever, model), model

    return make


# ------------------------------------------------------------------ loop


def test_search_read_answer_with_verified_citations(make_runner):
    runner, model = make_runner([
        ai("", call(SEARCH_DOCS, "c1", query="register event subscriber kernel.event_subscriber")),
        ai("", call(READ_SECTION, "c2", source=1)),
        ai("Tag the service with kernel.event_subscriber [1]. Made up [9]."),
    ])
    res = runner.run("How do I register an event subscriber?")

    assert res.answer == "Tag the service with kernel.event_subscriber [1]. Made up."
    assert res.answerable is True and res.stopped_by == "answer"
    assert [c.n for c in res.citations] == [1]
    assert res.citations[0].doc_path == "guides/events.md"
    assert [s.tool for s in res.steps] == [SEARCH_DOCS, READ_SECTION]
    assert res.steps[1].sources == [1] and res.steps[1].error is None
    assert res.tool_calls == 2 and res.model_calls == 3
    assert res.usage.input_tokens == 300 and res.usage.output_tokens == 60
    assert res.llm_model == "scripted-model"

    # What the model saw: system prompt with the budget, numbered results, then the full section.
    assert isinstance(model.seen[0][0], SystemMessage) and "at most 8 tool calls" in model.seen[0][0].content
    search_result = tool_results(model, 1)[0].content
    assert search_result.startswith('Results for "register event subscriber') and "[1] " in search_result
    full_text = tool_results(model, 2)[-1].content
    assert "$services->set(MySubscriber::class)" in full_text


def test_tool_specs_use_snake_case_names(make_runner):
    _, model = make_runner([ai("x")])
    names = [t["function"]["name"] for t in model.tools]
    assert names == [SEARCH_DOCS, READ_SECTION] == [t["function"]["name"] for t in tool_specs()]
    assert model.tools[0]["function"]["parameters"]["required"] == ["query"]


def test_source_numbers_stay_stable_across_searches(make_runner):
    runner, _ = make_runner([
        ai("", call(SEARCH_DOCS, "c1", query="theme compile scss")),
        ai("", call(SEARCH_DOCS, "c2", query="compile theme storefront")),
        ai("Run bin/console theme:compile [1]."),
    ])
    res = runner.run("How do I compile the theme?")
    first, second = res.steps[0].sources, res.steps[1].sources
    # Both searches return the same few chunks; a chunk keeps its number the second time.
    assert sorted(first) == list(range(1, len(first) + 1))
    assert set(second) <= set(first) | set(range(len(first) + 1, len(first) + len(second) + 1))
    assert max(first + second) <= 3  # the test index has 3 sections


# ---------------------------------------------------------------- budget


def test_budget_caps_tool_calls_and_answers_every_request(make_runner):
    runner, model = make_runner([
        ai("Let me search.", call(SEARCH_DOCS, "a", query="theme compile"),
           call(SEARCH_DOCS, "b", query="EqualsFilter Criteria"),
           call(SEARCH_DOCS, "c", query="event subscriber")),
        ai("One more.", call(READ_SECTION, "d", source=1)),  # asks again after the budget is gone
    ], agent_max_tool_calls=2)
    res = runner.run("Three things at once?")

    assert res.tool_calls == 2 and res.stopped_by == "budget"
    assert res.answerable is False and res.answer == NOT_FOUND  # narration is not an answer
    assert [s.error for s in res.steps] == [None, None, "skipped: tool-call budget used up"]
    replies = tool_results(model, 1)
    assert [m.tool_call_id for m in replies] == ["a", "b", "c"]  # every request got a reply
    assert replies[2].content.startswith(NOT_EXECUTED) and replies[2].content.endswith(BUDGET_NOTICE)


def test_request_can_lower_the_budget(make_runner):
    runner, model = make_runner([ai("", call(SEARCH_DOCS, "a", query="theme")), ai("Done [1].")])
    res = runner.run("theme?", max_tool_calls=1)
    assert res.tool_calls == 1 and res.answerable is True
    assert "at most 1 tool calls" in model.seen[0][0].content
    assert tool_results(model, 1)[0].content.endswith(BUDGET_NOTICE)


# ------------------------------------------------------------ bad calls


def test_bad_tool_calls_return_errors_to_the_model(make_runner):
    runner, model = make_runner([
        ai("", call(READ_SECTION, "a", source=99), call(SEARCH_DOCS, "b"), call("delete_docs", "c")),
        ai(NOT_FOUND),
    ])
    res = runner.run("Something odd?")

    errors = [s.error for s in res.steps]
    assert errors[0].startswith("Unknown source 99. Run search_docs first.")
    assert errors[1].startswith("Invalid arguments for search_docs: query: Field required")
    assert errors[2].startswith("Unknown tool 'delete_docs'")
    assert all(m.status == "error" for m in tool_results(model, 1))
    assert res.tool_calls == 3  # failed calls count too, so a confused model can't loop forever
    assert res.answerable is False and res.citations == []


def test_read_section_lists_known_sources(make_runner):
    runner, _ = make_runner([
        ai("", call(SEARCH_DOCS, "a", query="theme compile")),
        ai("", call(READ_SECTION, "b", source=7)),
        ai("Run theme:compile [1]."),
    ])
    res = runner.run("theme?")
    assert res.steps[1].error.startswith("Unknown source 7. Known sources: [1,")


# ---------------------------------------------------------------- answer


@pytest.mark.parametrize("text", [NOT_FOUND, f'"{NOT_FOUND}"', "i could not find this in the shopware documentation"])
def test_not_found_variants_are_refusals(text):
    assert is_refusal(text, NOT_FOUND)


def test_partial_answer_is_not_a_refusal():
    text = "Run theme:compile [1]. " + NOT_FOUND + " for the SCSS part."
    assert not is_refusal(text, NOT_FOUND)


def test_refusal_with_explanation_keeps_the_explanation(make_runner):
    # Seen in the multi-hop eval: Claude opens with the sentence, then says what it searched.
    runner, _ = make_runner([
        ai("", call(SEARCH_DOCS, "a", query="HubSpot connector")),
        ai(f"{NOT_FOUND}\n\nI searched for a HubSpot connector; no page mentions it. Custom fields are "
           "defined declaratively instead [1]. Invented [7]."),
    ])
    res = runner.run("How do I use the HubSpot connector?")
    assert res.answerable is False
    assert res.answer.startswith(NOT_FOUND) and "I searched for a HubSpot connector" in res.answer
    assert "[7]" not in res.answer and [c.n for c in res.citations] == [1]


def test_trace_can_be_left_out(make_runner):
    runner, _ = make_runner([ai("", call(SEARCH_DOCS, "a", query="theme")), ai("Compile it [1].")])
    res = runner.run("theme?", include_trace=False)
    assert res.steps is None and res.tool_calls == 1


def test_model_failure_becomes_agent_error(make_runner):
    from app.agent import AgentError

    runner, _ = make_runner([ConnectionError("ollama is not running")])
    with pytest.raises(AgentError, match="ConnectionError: ollama is not running"):
        runner.run("theme?")


# ------------------------------------------------------------ providers


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_agent_follows_the_generation_provider_by_default():
    s = _settings(llm_provider="ollama", llm_model="qwen2.5:3b")
    assert (s.resolved_agent_provider, s.resolved_agent_model) == ("ollama", "qwen2.5:3b")
    s = _settings(llm_provider="ollama", agent_provider="anthropic", agent_api_key="k")
    assert (s.resolved_agent_provider, s.resolved_agent_model) == ("anthropic", "claude-opus-5-5")
    assert s.resolved_agent_api_key == "k"
    assert _settings(agent_provider="", agent_model="").agent_provider is None  # empty .env values


def test_make_chat_model_per_provider():
    from langchain_anthropic import ChatAnthropic
    from langchain_ollama import ChatOllama
    from langchain_openai import ChatOpenAI

    claude = make_chat_model(_settings(agent_provider="anthropic", agent_model="claude-haiku-5-5",
                                       agent_api_key="k", anthropic_effort="medium"))
    assert isinstance(claude, ChatAnthropic) and claude.effort == "medium" and claude.max_tokens == 2000
    local = make_chat_model(_settings(llm_provider="ollama", ollama_num_ctx=16384))
    assert isinstance(local, ChatOllama) and local.num_ctx == 16384 and local.model == "qwen2.5:3b"
    gpt = make_chat_model(_settings(agent_provider="openai", agent_api_key="k"))
    assert isinstance(gpt, ChatOpenAI) and gpt.model_name == "gpt-4.1-mini"
    assert make_chat_model(_settings(llm_provider="extractive")) is None


# ------------------------------------------------------------------- api


@pytest.fixture
def api(settings, indexed, monkeypatch):
    qdrant, dense, sparse = indexed
    pipeline = RAGPipeline(settings, qdrant, dense, sparse, FakeReranker(), None)
    monkeypatch.setattr(main.RAGPipeline, "from_settings", classmethod(lambda cls, s: pipeline))
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    def make(model: ScriptedModel | None):
        monkeypatch.setattr(main, "build_agent",
                            lambda s, p: AgentRunner(s, p.retriever, model) if model else None)
        return TestClient(main.app)

    return make


def test_agent_endpoint_returns_answer_and_trace(api):
    model = ScriptedModel([ai("", call(SEARCH_DOCS, "a", query="EqualsFilter EntityRepository")),
                           ai("Use an EqualsFilter on the Criteria [1].")])
    with api(model) as client:
        r = client.post("/agent", json={"question": "How do I filter products by name?"})
        health = client.get("/health").json()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answerable"] is True and body["citations"][0]["doc_path"] == "guides/dal.md"
    assert body["steps"][0]["tool"] == "search_docs" and body["steps"][0]["sources"]
    assert set(body["timings_ms"]) == {"model", "tools", "total"}
    assert health["agent"]["enabled"] is True


def test_agent_endpoint_disabled_without_llm(api):
    with api(None) as client:
        r = client.post("/agent", json={"question": "anything at all"})
        health = client.get("/health").json()
    assert r.status_code == 503 and "AGENT_PROVIDER" in r.json()["detail"]
    assert health["agent"] == {"enabled": False, "provider": "extractive", "model": None}


def test_agent_endpoint_maps_model_errors_to_502(api):
    with api(ScriptedModel([TimeoutError("read timed out")])) as client:
        r = client.post("/agent", json={"question": "anything at all"})
    assert r.status_code == 502 and "TimeoutError" in r.json()["detail"]


def test_agent_endpoint_validates_input(api):
    with api(ScriptedModel([])) as client:
        assert client.post("/agent", json={"question": "hi"}).status_code == 422
        assert client.post("/agent", json={"question": "valid question", "max_tool_calls": 99}).status_code == 422


def test_build_agent_returns_none_for_extractive(settings):
    assert build_agent(settings, pipeline=None) is None


def test_sources_show_exactly_what_the_model_saw(make_runner):
    runner, model = make_runner([
        ai("", call(SEARCH_DOCS, "a", query="register event subscriber kernel.event_subscriber")),
        ai("", call(READ_SECTION, "b", source=1)),
        ai("Tag it [1]."),
    ])
    res = runner.run("How do I register a subscriber?", include_sources=True)
    by_n = {s.n: s for s in res.sources}
    assert sorted(by_n) == list(range(1, len(by_n) + 1))
    assert by_n[1].read is True and "$services->set(MySubscriber::class)" in by_n[1].text
    unread = [s for s in res.sources if not s.read]
    assert unread and all(s.text in tool_results(model, 1)[0].content for s in unread)  # the preview it saw


def test_sources_are_off_by_default(make_runner):
    runner, _ = make_runner([ai("", call(SEARCH_DOCS, "a", query="theme")), ai("Compile it [1].")])
    assert runner.run("theme?").sources is None

"""Request-shape tests for the LLM providers - no network, the SDK clients are faked."""

from types import SimpleNamespace

import httpx
import pytest

from app import llm as llm_mod
from app.generation import ANSWER_SCHEMA
from app.llm import AnthropicLLM, LLMError, OllamaLLM


class _FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _anthropic_response(text: str, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        model="claude-opus-5-5",
    )


@pytest.fixture
def fake_anthropic(monkeypatch):
    import anthropic

    holder = {}

    class FakeClient:
        def __init__(self, **kwargs):
            holder["init"] = kwargs
            self.messages = holder.setdefault("messages", _FakeMessages(None))
            self.beta = SimpleNamespace(messages=holder.setdefault("beta", _FakeMessages(None)))

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    return holder


def test_anthropic_uses_structured_output_effort_and_fallbacks(fake_anthropic):
    client = AnthropicLLM("claude-opus-5-5", api_key=None, effort="low", timeout_s=30)
    fake_anthropic["beta"].response = _anthropic_response('{"answer": "x [1]", "citations": [1], "answerable": true}')
    res = client.generate_json("sys", "user", ANSWER_SCHEMA, "answer", 500)
    call = fake_anthropic["beta"].calls[0]
    assert call["output_config"] == {"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}, "effort": "low"}
    assert call["betas"] == ["server-side-fallback-2026-07-01"] and call["fallbacks"] == "default"
    assert "temperature" not in call and "tool_choice" not in call  # rejected by current Claude models
    assert call["system"] == "sys" and call["messages"] == [{"role": "user", "content": "user"}]
    assert res.data["citations"] == [1] and res.input_tokens == 100


def test_anthropic_workspace_header_only_when_configured(fake_anthropic):
    AnthropicLLM("claude-opus-5-5", api_key="k", effort="low", timeout_s=30, workspace_id="wrkspc_123")
    assert fake_anthropic["init"]["default_headers"] == {"anthropic-workspace-id": "wrkspc_123"}
    AnthropicLLM("claude-opus-5-5", api_key="k", effort="low", timeout_s=30)
    assert fake_anthropic["init"]["default_headers"] is None


def test_anthropic_model_without_fallback_uses_plain_endpoint(fake_anthropic):
    client = AnthropicLLM("claude-haiku-5-5", api_key="k", effort="low", timeout_s=30)
    fake_anthropic["messages"].response = _anthropic_response('{"answer": "a", "citations": [], "answerable": true}')
    client.generate_json("s", "u", ANSWER_SCHEMA, "answer", 100)
    call = fake_anthropic["messages"].calls[0]
    assert "fallbacks" not in call and call["output_config"]["effort"] == "low"


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_anthropic_stop_reasons_raise(fake_anthropic, stop_reason):
    client = AnthropicLLM("claude-opus-5-5", api_key=None, effort="low", timeout_s=30)
    fake_anthropic["beta"].response = _anthropic_response("{}", stop_reason=stop_reason)
    with pytest.raises(LLMError):
        client.generate_json("s", "u", ANSWER_SCHEMA, "answer", 100)


def test_ollama_retries_once_on_repetition_abort(monkeypatch):
    bodies: list[dict] = []
    responses = [
        httpx.Response(500, json={"error": "prediction aborted, token repeat limit reached"}),
        httpx.Response(200, json={"message": {"content": '{"answer": "ok", "citations": [], "answerable": true}'},
                                  "prompt_eval_count": 10, "eval_count": 5, "model": "qwen2.5:3b"}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion"]})
        bodies.append(__import__("json").loads(request.content))
        return responses.pop(0)

    client = OllamaLLM("qwen2.5:3b", "http://ollama:11434", 0.0, 42, 30)
    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    res = client.generate_json("s", "u", ANSWER_SCHEMA, "answer", 100)
    assert res.data["answer"] == "ok"
    assert bodies[0]["options"]["num_ctx"] == 8192 and bodies[0]["format"] == ANSWER_SCHEMA
    assert "think" not in bodies[0]  # model has no thinking capability
    assert bodies[1]["options"]["repeat_penalty"] > bodies[0]["options"]["repeat_penalty"]


def test_parse_json_tolerates_wrapped_output():
    assert llm_mod._parse_json('Sure! ```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(LLMError):
        llm_mod._parse_json("no json here")

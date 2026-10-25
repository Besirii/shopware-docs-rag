"""The agent as a LangGraph state machine.

    START -> agent --(asked for tools, budget left)--> tools -> agent -> ...
               |
               +--(wrote an answer, or asked for tools with no budget left)--> END

* `agent` sends the conversation to the chat model, which has the tools bound.
  The model either asks for tool calls or writes the final answer.
* `tools` runs the requested calls against the retriever, numbers the sources,
  and enforces the tool-call budget.

The loop is written out as two nodes instead of using LangGraph's prebuilt
ReAct agent, so the budget, the source numbering and the trace are explicit and
testable - and so the next step (separate planner / writer / reviewer nodes) is
a matter of adding nodes, not rewriting.

Everything the run produces lives in `AgentState`: the messages, the numbered
sources, one trace entry per tool call, and the counters. Plain dicts and lists
only, so a finished run can be stored as JSON for auditing.
"""

from __future__ import annotations

import json
import logging
import operator
import re
import time
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agent.prompts import BUDGET_NOTICE, NOT_EXECUTED, SYSTEM_PROMPT
from app.agent.tools import DocsTools, Source, preview, tool_specs
from app.config import Settings
from app.generation import snippet, validate_citations
from app.retrieval import Retriever
from app.schemas import AgentResponse, AgentSource, AgentStep, AgentTimings, Citation, Usage

log = logging.getLogger("rag.agent")


class AgentError(RuntimeError):
    """The model call failed (API error, timeout, bad key, Ollama not running ...)."""


def _merge(old: dict, new: dict) -> dict:
    return {**old, **new}


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # appended to by every node
    sources: Annotated[dict[str, Source], _merge]  # chunk_id -> numbered source
    steps: Annotated[list[dict], operator.add]  # one entry per tool call, in order
    model_ms: Annotated[float, operator.add]
    tool_calls: int  # tool calls used so far (including failed ones)
    max_tool_calls: int


def build_graph(model, tools: DocsTools, not_found: str):
    """`model` is a chat model with the tools already bound (anything with `.invoke(messages)`)."""

    def agent(state: AgentState) -> dict:
        system = SystemMessage(SYSTEM_PROMPT.format(max_tool_calls=state["max_tool_calls"], not_found=not_found))
        t0 = time.perf_counter()
        try:
            reply = model.invoke([system, *state["messages"]])
        except Exception as e:  # provider SDKs raise many different types
            raise AgentError(f"{type(e).__name__}: {e}") from e
        return {"messages": [reply], "model_ms": (time.perf_counter() - t0) * 1000}

    def run_tools(state: AgentState) -> dict:
        request: AIMessage = state["messages"][-1]  # type: ignore[assignment]
        registry = dict(state["sources"])
        used, budget = state["tool_calls"], state["max_tool_calls"]
        replies: list[ToolMessage] = []
        steps: list[dict] = []
        added: dict[str, Source] = {}
        for call in request.tool_calls:
            if used >= budget:
                # Every tool call needs an answer, or the next model request is invalid.
                replies.append(ToolMessage(NOT_EXECUTED, tool_call_id=call["id"], name=call["name"], status="error"))
                steps.append({"tool": call["name"], "args": call["args"], "sources": [], "ms": 0.0,
                              "error": "skipped: tool-call budget used up"})
                continue
            outcome = tools.run(call["name"], call["args"], registry)
            registry.update(outcome.sources)
            added.update(outcome.sources)
            used += 1
            replies.append(ToolMessage(outcome.content, tool_call_id=call["id"], name=call["name"],
                                       status="error" if outcome.error else "success"))
            steps.append(outcome.step)
        if used >= budget and replies:
            replies[-1].content += BUDGET_NOTICE
        return {"messages": replies, "sources": added, "steps": steps, "tool_calls": used}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls and state["tool_calls"] < state["max_tool_calls"]:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["tools", END])
    graph.add_edge("tools", "agent")
    return graph.compile()


class AgentRunner:
    """Builds the graph once at startup; `run` answers one question."""

    def __init__(self, settings: Settings, retriever: Retriever, chat_model: BaseChatModel) -> None:
        self.s = settings
        self.model_name = settings.resolved_agent_model
        self.tools = DocsTools(retriever, top_k=settings.agent_search_top_k)
        self.graph = build_graph(chat_model.bind_tools(tool_specs()), self.tools, settings.not_found_message)

    def run(self, question: str, max_tool_calls: int | None = None, include_trace: bool = True,
            include_sources: bool = False) -> AgentResponse:
        budget = max_tool_calls or self.s.agent_max_tool_calls
        t0 = time.perf_counter()
        initial: AgentState = {"messages": [HumanMessage(question.strip())], "sources": {}, "steps": [],
                               "model_ms": 0.0, "tool_calls": 0, "max_tool_calls": budget}
        try:
            # Each tool round is two graph steps (agent + tools); the budget ends the loop long before this.
            state = self.graph.invoke(initial, config={"recursion_limit": 2 * budget + 6})
        except GraphRecursionError as e:
            raise AgentError("agent did not finish within the step limit") from e
        response = finalize(state, self.s.not_found_message, self.model_name, include_trace, include_sources)
        response.timings_ms.total = round((time.perf_counter() - t0) * 1000, 1)
        log.info(json.dumps({
            "event": "agent", "tool_calls": response.tool_calls, "model_calls": response.model_calls,
            "stopped_by": response.stopped_by, "answerable": response.answerable,
            "n_citations": len(response.citations), "usage": response.usage.model_dump(),
            "timings_ms": response.timings_ms.model_dump(),
        }))
        return response


# ------------------------------------------------------------------ result


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def is_refusal(answer: str, not_found: str) -> bool:
    """The answer opens with the not-found sentence (often followed by what was searched),
    or is (roughly) just that sentence. Mentioning it mid-answer for a missing part is not a refusal."""
    a, nf = _norm(answer), _norm(not_found)
    return a.startswith(nf) or (nf in a and len(a) <= len(nf) + 20)


def citation(src: Source) -> Citation:
    return Citation(n=src["n"], title=src["title"], url=src["url"],
                    heading=src["headings"][-1] if src["headings"] else None,
                    snippet=snippet(src["text"]), doc_path=src["doc_path"])


def source_view(src: Source) -> AgentSource:
    """A source as the model saw it: full text only if it was read, else the search preview."""
    return AgentSource(n=src["n"], chunk_id=src["chunk_id"], doc_path=src["doc_path"], title=src["title"],
                       url=src["url"], heading=src["headings"][-1] if src["headings"] else None, read=src["read"],
                       text=src["text"].strip() if src["read"] else preview(src["text"]))


def finalize(state: AgentState, not_found: str, model_name: str, include_trace: bool = True,
             include_sources: bool = False) -> AgentResponse:
    """Turn the final state into the API response: clean citations, totals, trace."""
    last = state["messages"][-1]
    stopped_by = "budget" if isinstance(last, AIMessage) and last.tool_calls else "answer"
    # Text next to a tool request is narration ("Let me search for ..."), not an answer.
    text = last.text.strip() if isinstance(last, AIMessage) and stopped_by == "answer" else ""
    by_n = {src["n"]: src for src in state["sources"].values()}

    if not text:
        answer, answerable, numbers = not_found, False, []
    else:
        # Numbers are handed out as 1..N, so "valid" means 1 <= n <= N - same check as /ask.
        # A refusal keeps its explanation (what was searched, related pages) and the sources it cites.
        answer, numbers = validate_citations(text, [], len(by_n))
        answerable = not is_refusal(text, not_found)

    ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
    usage = Usage(
        input_tokens=sum((m.usage_metadata or {}).get("input_tokens", 0) for m in ai_messages),
        output_tokens=sum((m.usage_metadata or {}).get("output_tokens", 0) for m in ai_messages),
    )
    meta = ai_messages[-1].response_metadata if ai_messages else {}
    steps = [AgentStep(step=i, **s) for i, s in enumerate(state["steps"], start=1)]
    return AgentResponse(
        answer=answer,
        answerable=answerable,
        citations=[citation(by_n[n]) for n in numbers],
        tool_calls=state["tool_calls"],
        model_calls=len(ai_messages),
        stopped_by=stopped_by,
        timings_ms=AgentTimings(model=round(state["model_ms"], 1), tools=round(sum(s.ms for s in steps), 1)),
        usage=usage,
        llm_model=meta.get("model_name") or meta.get("model") or model_name,
        steps=steps if include_trace else None,
        sources=[source_view(by_n[n]) for n in sorted(by_n)] if include_sources else None,
    )

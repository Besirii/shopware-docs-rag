"""Chat model for the agent, built from the same settings as app/llm.py.

The /ask pipeline talks to each provider's SDK directly because it needs one
structured JSON answer. The agent needs multi-turn tool calling instead, and
LangChain's chat models give that with one interface for all three providers
(`bind_tools` + `invoke`), so the graph does not care which one it runs on.
"""

from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel

from app.config import Settings

# Claude models that take `effort` (cost/latency knob) instead of a temperature.
_CLAUDE_5 = re.compile(r"claude-(opus|sonnet|haiku|fable|mythos)-5")


def make_chat_model(settings: Settings) -> BaseChatModel | None:
    """The configured chat model, or None when the provider is 'extractive' (no LLM -> no agent)."""
    provider = settings.resolved_agent_provider
    model = settings.resolved_agent_model
    key = settings.resolved_agent_api_key

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs: dict = {"model": model, "max_tokens": settings.agent_max_tokens,
                        "timeout": settings.llm_timeout_s, "max_retries": 3}
        if key:  # otherwise the SDK reads ANTHROPIC_API_KEY
            kwargs["api_key"] = key
        if settings.anthropic_workspace_id:
            kwargs["default_headers"] = {"anthropic-workspace-id": settings.anthropic_workspace_id}
        if _CLAUDE_5.match(model):
            kwargs["effort"] = settings.anthropic_effort
        return ChatAnthropic(**kwargs)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=key, base_url=settings.llm_base_url,
                          temperature=settings.llm_temperature, seed=settings.llm_seed,
                          max_tokens=settings.agent_max_tokens, timeout=settings.llm_timeout_s, max_retries=3)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # num_ctx matters even more here than in /ask: the conversation grows with every tool result.
        return ChatOllama(model=model, base_url=settings.ollama_base_url, temperature=settings.llm_temperature,
                          seed=settings.llm_seed, num_ctx=settings.ollama_num_ctx,
                          num_predict=settings.agent_max_tokens, repeat_penalty=settings.ollama_repeat_penalty)

    return None

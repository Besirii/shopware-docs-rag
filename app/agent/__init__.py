"""Agentic mode (POST /agent): the model plans its own searches with tool calls.

/ask runs one fixed search and hands the top sections to the LLM. That is fast
and works for single-topic questions, but a question like "add a custom field
to products and show it in the storefront" touches several parts of the docs
(DAL, administration, Twig) that one search rarely finds together. The agent
searches as many times as it needs (within a budget), reads the sections it
will use, and answers with the same verifiable citations as /ask.

    graph.py    the LangGraph state machine + AgentRunner
    tools.py    search_docs / read_section over the existing Retriever
    models.py   LangChain chat model for the configured provider
    prompts.py  the system prompt
"""

from __future__ import annotations

import logging

from app.agent.graph import AgentError, AgentRunner
from app.agent.models import make_chat_model
from app.config import Settings

__all__ = ["AgentError", "AgentRunner", "build_agent"]

log = logging.getLogger("rag.agent")


def build_agent(settings: Settings, pipeline) -> AgentRunner | None:
    """The agent for the configured provider, reusing the pipeline's retriever. None = disabled."""
    try:
        model = make_chat_model(settings)
    except Exception as e:  # e.g. ChatOpenAI without any API key - keep /ask working
        log.warning("agent disabled: %s", e)
        return None
    if model is None:
        return None
    return AgentRunner(settings, pipeline.retriever, model)

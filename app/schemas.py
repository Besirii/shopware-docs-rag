"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]
Chunking = Literal["heading", "fixed"]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=20, description="chunks passed to the LLM")
    mode: Mode = Field(default="hybrid", description="hybrid (default, measured best) / hybrid_rerank / dense / bm25")
    chunking: Chunking = Field(default="heading", description="which index to query (fixed = baseline)")
    sparse_weight: float | None = Field(default=None, ge=0, le=5,
                                        description="BM25 weight in RRF (hybrid modes); null = server default")
    generate: bool = Field(default=True, description="false = retrieval only, no LLM call (cheap eval runs)")
    include_contexts: bool = Field(default=False, description="return the full ranked candidate list")

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "How do I register an event subscriber in a plugin?", "top_k": 6,
                          "mode": "hybrid"}]
        }
    }


class Citation(BaseModel):
    n: int
    title: str
    url: str
    heading: str | None
    snippet: str
    doc_path: str


class RetrievedContext(BaseModel):
    rank: int
    chunk_id: str
    doc_path: str
    title: str
    url: str
    heading: str | None
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None
    token_count: int | None = None
    text: str | None = None


class Timings(BaseModel):
    retrieve: float = 0.0
    rerank: float = 0.0
    generate: float = 0.0
    total: float = 0.0


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class AskResponse(BaseModel):
    answer: str
    answerable: bool
    citations: list[Citation]
    timings_ms: Timings
    usage: Usage
    mode: Mode
    chunking: Chunking
    llm_model: str | None = None
    refused_by: Literal["threshold", "llm"] | None = None
    top_rerank_score: float | None = None
    contexts: list[RetrievedContext] | None = None


# ------------------------------------------------------------------- agent


class AgentRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    max_tool_calls: int | None = Field(default=None, ge=1, le=15,
                                       description="tool-call budget for this question; null = server default")
    include_trace: bool = Field(default=True, description="return every tool call the agent made")
    include_sources: bool = Field(default=False, description="return every source the agent saw, with the text "
                                                             "it saw (eval + auditing)")

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "How do I add a custom field to products and show it in the storefront?"}]
        }
    }


class AgentStep(BaseModel):
    """One tool call, in order. This is the audit trail of how the answer was found."""

    step: int
    tool: str
    args: dict
    sources: list[int] = Field(default_factory=list, description="source numbers the tool returned or read")
    ms: float = 0.0
    error: str | None = None


class AgentSource(BaseModel):
    """A numbered source the agent saw. `text` is exactly what the model saw: the full
    section if it called read_section, otherwise the search preview."""

    n: int
    chunk_id: str
    doc_path: str
    title: str
    url: str
    heading: str | None
    read: bool
    text: str


class AgentTimings(BaseModel):
    model: float = 0.0
    tools: float = 0.0
    total: float = 0.0


class AgentResponse(BaseModel):
    answer: str
    answerable: bool
    citations: list[Citation]
    tool_calls: int
    model_calls: int
    stopped_by: Literal["answer", "budget"]
    timings_ms: AgentTimings
    usage: Usage
    llm_model: str | None = None
    steps: list[AgentStep] | None = None
    sources: list[AgentSource] | None = None

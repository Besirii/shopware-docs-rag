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

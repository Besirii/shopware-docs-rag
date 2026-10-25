"""Central configuration.

Every tunable lives here and is read from environment variables (or `.env`) via
pydantic-settings. The point is that the *same* Docker image can run against
Anthropic, OpenAI, or a local Ollama model, with any embedding/reranker model,
without touching code - and that an eval result can record exactly which
configuration produced it (see `Settings.fingerprint`).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LLMProvider = Literal["anthropic", "openai", "ollama", "extractive"]
EmbeddingProvider = Literal["fastembed", "openai"]
FusionBackend = Literal["python", "qdrant"]

# Sensible per-provider defaults so that switching LLM_PROVIDER alone is enough.
DEFAULT_LLM_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-4.1-mini",
    "ollama": "qwen2.5:3b",
    "extractive": "extractive",
}


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "shopware-docs-rag" / "models"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ------------------------------------------------------------------ source
    docs_repo_url: str = "https://github.com/shopware/docs.git"
    docs_ref: str = "main"
    # Pin an exact commit for reproducible evals. Empty = latest commit of DOCS_REF.
    docs_commit: str | None = None
    docs_dir: Path = Path("data/raw/shopware-docs")
    docs_include: Annotated[list[str], NoDecode] = ["guides", "concepts", "resources/references"]
    docs_exclude: Annotated[list[str], NoDecode] = [
        "**/changelog*",
        "**/CHANGELOG*",
        "**/release-notes*",
    ]
    docs_base_url: str = "https://developer.shopware.com/docs"
    # Index pages with fewer prose words than this (and mostly links) are pure navigation.
    nav_page_max_prose_words: int = 30

    # ---------------------------------------------------------------- chunking
    tokenizer_encoding: str = "cl100k_base"
    chunk_max_tokens: int = 600
    chunk_min_tokens: int = 120
    chunk_overlap_tokens: int = 80
    baseline_chunk_tokens: int = 512
    baseline_overlap_tokens: int = 64

    # -------------------------------------------------------------- embeddings
    embedding_provider: EmbeddingProvider = "fastembed"
    embedding_model: str = "ibm-granite/granite-embedding-small-english-r2"
    embedding_batch_size: int = 16
    # Dense input is cut at this many *model* tokens (explicitly, and counted by
    # the indexer). The full chunk is still stored, shown to the LLM and BM25-indexed.
    embedding_max_tokens: int = 1024
    openai_api_key: SecretStr | None = None
    sparse_model: str = "Qdrant/bm25"
    model_cache_dir: Path = Field(default_factory=_default_cache_dir)
    # ONNX Runtime threads for local models (0 = let ORT decide).
    onnx_threads: int = 0

    # ------------------------------------------------------------------ qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    collection_name: str = "shopware_docs"

    # --------------------------------------------------------------- retrieval
    retrieval_candidates: int = 30
    rrf_k: int = 60
    # Weight of the BM25 list in RRF (dense = 1.0). 1.0 = classic RRF; 0.5 measured better here.
    hybrid_sparse_weight: float = 0.5
    fusion_backend: FusionBackend = "python"
    default_top_k: int = 6
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # What the cross-encoder reads: "embed_text" (title + breadcrumb + chunk) or "text" (chunk only).
    rerank_input: Literal["embed_text", "text"] = "embed_text"
    # Top rerank score below this => "not found". None disables the gate.
    # Calibrate with `python -m eval.calibrate`; scores are model-specific.
    rerank_threshold: float | None = None

    # --------------------------------------------------------------- generation
    llm_provider: LLMProvider = "anthropic"
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    # Only for OpenAI-compatible endpoints (vLLM, LM Studio, Azure, ...).
    llm_base_url: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 8192
    ollama_repeat_penalty: float = 1.1
    llm_max_tokens: int = 2000
    # Applied only where the provider accepts it (OpenAI, Ollama). Current Claude
    # models reject sampling parameters, so it is never sent to Anthropic.
    llm_temperature: float = 0.0
    llm_seed: int = 42
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    # Only for API keys that are not scoped to a workspace (sent as anthropic-workspace-id).
    anthropic_workspace_id: str | None = None
    llm_timeout_s: float = 180.0
    not_found_message: str = "I could not find this in the Shopware documentation."

    # -------------------------------------------------------------------- eval
    judge_provider: LLMProvider | None = None
    judge_model: str | None = None

    # ------------------------------------------------------------------- agent
    # POST /agent: the model plans its own searches with tool calls. Empty = same
    # provider/model as generation. Tool calling needs a capable model - a small
    # Claude (claude-haiku-5-5) is the cheap, reliable choice; 3B local models
    # often skip the tools or call them with broken arguments.
    agent_provider: LLMProvider | None = None
    agent_model: str | None = None
    # Only needed when the agent uses a different provider than generation
    # (otherwise LLM_API_KEY is reused; empty = the SDK's own env variable).
    agent_api_key: SecretStr | None = None
    # Hard cap on tool calls per question (search_docs + read_section together).
    agent_max_tool_calls: int = 8
    # Sections returned by one search_docs call.
    agent_search_top_k: int = 5
    agent_max_tokens: int = 2000

    # --------------------------------------------------------------------- api
    admin_token: SecretStr | None = None
    # When set, POST /ask and POST /agent require it in the X-API-Key header. Set it on any
    # public deployment: both endpoints spend LLM tokens on your account.
    api_key: SecretStr | None = None
    log_level: str = "INFO"

    @field_validator("docs_include", "docs_exclude", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("docs_commit", "llm_model", "llm_base_url", "judge_model", "anthropic_workspace_id",
                     "agent_provider", "agent_model", "agent_api_key", "api_key", "admin_token", mode="before")
    @classmethod
    def _empty_to_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("rerank_threshold", mode="before")
    @classmethod
    def _threshold_none(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "off"}:
            return None
        return value

    # ----------------------------------------------------------------- helpers
    @property
    def resolved_llm_model(self) -> str:
        return self.llm_model or DEFAULT_LLM_MODELS[self.llm_provider]

    @property
    def resolved_agent_provider(self) -> LLMProvider:
        return self.agent_provider or self.llm_provider

    @property
    def resolved_agent_model(self) -> str:
        if self.agent_model:
            return self.agent_model
        if self.resolved_agent_provider == self.llm_provider:
            return self.resolved_llm_model
        return DEFAULT_LLM_MODELS[self.resolved_agent_provider]

    @property
    def resolved_agent_api_key(self) -> str | None:
        if self.agent_api_key:
            return self.agent_api_key.get_secret_value()
        if self.resolved_agent_provider == self.llm_provider and self.llm_api_key:
            return self.llm_api_key.get_secret_value()
        if self.resolved_agent_provider == "openai" and self.openai_api_key:
            return self.openai_api_key.get_secret_value()
        return None

    @property
    def resolved_judge_provider(self) -> LLMProvider:
        return self.judge_provider or self.llm_provider

    @property
    def resolved_judge_model(self) -> str:
        return self.judge_model or DEFAULT_LLM_MODELS[self.resolved_judge_provider]

    def collection_for(self, chunking: str) -> str:
        """Each chunking strategy gets its own collection.

        Keeping them apart matters for BM25: Qdrant computes IDF per collection,
        so mixing two chunkings of the same corpus would skew term statistics.
        """
        return self.collection_name if chunking == "heading" else f"{self.collection_name}_{chunking}"

    def fingerprint(self) -> dict[str, object]:
        """Non-secret settings that determine results - stored with every eval run."""
        return {
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "sparse_model": self.sparse_model,
            "rerank_model": self.rerank_model,
            "rerank_input": self.rerank_input,
            "rerank_threshold": self.rerank_threshold,
            "retrieval_candidates": self.retrieval_candidates,
            "rrf_k": self.rrf_k,
            "hybrid_sparse_weight": self.hybrid_sparse_weight,
            "fusion_backend": self.fusion_backend,
            "chunk_max_tokens": self.chunk_max_tokens,
            "chunk_min_tokens": self.chunk_min_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
            "baseline_chunk_tokens": self.baseline_chunk_tokens,
            "baseline_overlap_tokens": self.baseline_overlap_tokens,
            "llm_provider": self.llm_provider,
            "llm_model": self.resolved_llm_model,
            "collection_name": self.collection_name,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()

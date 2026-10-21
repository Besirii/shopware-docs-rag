"""Embedding, sparse (BM25) and reranking models behind small interfaces.

The same classes are used at index time and at query time - a mismatch between
the two (different model, different prefix) silently destroys retrieval quality,
so there is exactly one place that decides how text becomes a vector.

All local models run on ONNX Runtime via fastembed: no PyTorch, a small Docker
image, and acceptable CPU latency. OpenAI embeddings are an optional drop-in.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from functools import cached_property
from typing import ClassVar, Protocol

from qdrant_client import models

from app.config import Settings

log = logging.getLogger(__name__)

# Asymmetric models expect different prefixes for queries and documents.
# Getting this wrong costs several points of recall without any error.
_PREFIXES: dict[str, tuple[str, str]] = {
    "nomic-ai/nomic-embed-text-v1.5": ("search_query: ", "search_document: "),
    "nomic-ai/nomic-embed-text-v1.5-Q": ("search_query: ", "search_document: "),
    "nomic-ai/nomic-embed-text-v1": ("search_query: ", "search_document: "),
    "BAAI/bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-large-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "snowflake/snowflake-arctic-embed-s": ("Represent this sentence for searching relevant passages: ", ""),
    "snowflake/snowflake-arctic-embed-m": ("Represent this sentence for searching relevant passages: ", ""),
    "snowflake/snowflake-arctic-embed-m-long": ("Represent this sentence for searching relevant passages: ", ""),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
}


class DenseEmbedder(Protocol):
    model_name: str

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedDense:
    """Local ONNX embedder with explicit truncation and length-aware batching.

    Attention memory grows with batch_size x seq_len^2, and a batch is padded to
    its longest member. One 5,000-token code block in a batch of 16 would
    allocate gigabytes. So inputs are (1) truncated to `max_tokens` model tokens
    - explicitly, and reported by the indexer instead of happening silently -
    and (2) sorted by length and batched under a token budget.
    """

    def __init__(self, model_name: str, cache_dir: str, threads: int | None = None, batch_size: int = 16,
                 max_tokens: int = 1024, batch_token_budget: int = 8192) -> None:
        from fastembed import TextEmbedding

        t0 = time.perf_counter()
        self.model_name = model_name
        self.batch_size = batch_size
        self.batch_token_budget = batch_token_budget
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=threads or None)
        self._tokenizer = self._model.model.tokenizer  # type: ignore[attr-defined]
        model_max = (self._tokenizer.truncation or {}).get("max_length", max_tokens)
        self.max_tokens = min(model_max, max_tokens)
        self._tokenizer.enable_truncation(max_length=self.max_tokens)
        self._q_prefix, self._d_prefix = _PREFIXES.get(model_name, ("", ""))
        log.info("Loaded dense model %s (max %d tokens) in %.1fs", model_name, self.max_tokens, time.perf_counter() - t0)

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        """Untruncated model-token lengths (for truncation statistics)."""
        self._tokenizer.no_truncation()
        try:
            return [sum(e.attention_mask) for e in self._tokenizer.encode_batch([self._d_prefix + t for t in texts])]
        finally:
            self._tokenizer.enable_truncation(max_length=self.max_tokens)

    @cached_property
    def dim(self) -> int:
        from fastembed import TextEmbedding

        for desc in TextEmbedding.list_supported_models():
            if desc["model"] == self.model_name:
                return int(desc["dim"])
        return len(self.embed_query("dimension probe"))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        texts = [self._d_prefix + t for t in texts]
        lengths = [sum(e.attention_mask) for e in self._tokenizer.encode_batch(texts)]  # non-padding tokens
        order = sorted(range(len(texts)), key=lambda i: lengths[i])
        out: list[list[float] | None] = [None] * len(texts)
        batch: list[int] = []
        for idx in order:
            # Batch is sorted ascending, so the newest item is the longest (= padded length).
            too_big = (len(batch) + 1) * lengths[idx] > self.batch_token_budget
            if batch and (too_big or len(batch) >= self.batch_size):
                self._embed_batch(texts, batch, out)
                batch = []
            batch.append(idx)
        if batch:
            self._embed_batch(texts, batch, out)
        return out  # type: ignore[return-value]

    def _embed_batch(self, texts: Sequence[str], idxs: list[int], out: list) -> None:
        vecs = self._model.embed([texts[i] for i in idxs], batch_size=len(idxs))
        for i, v in zip(idxs, vecs, strict=True):
            out[i] = v.tolist()

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([self._q_prefix + text]))).tolist()


class OpenAIDense:
    _DIMS: ClassVar[dict[str, int]] = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536}

    def __init__(self, model_name: str, api_key: str | None, batch_size: int = 128) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self.batch_size = batch_size
        self._client = OpenAI(api_key=api_key)

    @cached_property
    def dim(self) -> int:
        return self._DIMS.get(self.model_name) or len(self.embed_query("dimension probe"))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = self._client.embeddings.create(model=self.model_name, input=list(texts[i:i + self.batch_size]))
            out.extend(d.embedding for d in resp.data)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._client.embeddings.create(model=self.model_name, input=[text]).data[0].embedding


class SparseBM25:
    """BM25 as a sparse vector: term frequencies on the document side, IDF in Qdrant.

    Qdrant applies IDF at query time (collection created with `Modifier.IDF`),
    so the collection statistics are always current. `avg_len` must be the
    average document length of *this* corpus, otherwise BM25's length
    normalisation (parameter b) penalises the wrong documents.
    """

    def __init__(self, model_name: str, cache_dir: str, avg_len: float = 256.0) -> None:
        from fastembed import SparseTextEmbedding

        self.model_name = model_name
        self.avg_len = avg_len
        self._model = SparseTextEmbedding(model_name=model_name, cache_dir=cache_dir, avg_len=avg_len)

    @staticmethod
    def _to_qdrant(emb) -> models.SparseVector:
        return models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())

    def embed_documents(self, texts: Sequence[str]) -> list[models.SparseVector]:
        return [self._to_qdrant(e) for e in self._model.embed(list(texts), batch_size=64)]

    def embed_query(self, text: str) -> models.SparseVector:
        return self._to_qdrant(next(iter(self._model.query_embed(text))))


class Reranker:
    """Cross-encoder: reads (question, passage) together and outputs a relevance logit.

    Much more accurate than comparing two independently computed vectors, but
    one forward pass per candidate - which is why it only reranks the top ~30.
    """

    def __init__(self, model_name: str, cache_dir: str, threads: int | None = None) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        t0 = time.perf_counter()
        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir, threads=threads or None)
        log.info("Loaded reranker %s in %.1fs", model_name, time.perf_counter() - t0)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self._model.rerank(query, list(passages), batch_size=8)]


def make_dense(settings: Settings) -> DenseEmbedder:
    if settings.embedding_provider == "openai":
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        return OpenAIDense(settings.embedding_model, key)
    return FastEmbedDense(settings.embedding_model, str(settings.model_cache_dir),
                          settings.onnx_threads, settings.embedding_batch_size,
                          max_tokens=settings.embedding_max_tokens)


def make_sparse(settings: Settings, avg_len: float = 256.0) -> SparseBM25:
    return SparseBM25(settings.sparse_model, str(settings.model_cache_dir), avg_len=avg_len)


def make_reranker(settings: Settings) -> Reranker:
    return Reranker(settings.rerank_model, str(settings.model_cache_dir), settings.onnx_threads)

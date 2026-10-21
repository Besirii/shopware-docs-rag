"""Retrieval: dense, BM25, hybrid (RRF) and cross-encoder reranking.

Why hybrid matters for Shopware: questions often contain exact identifiers -
`EntityRepository`, `theme:compile`, `ProductPageLoadedEvent` - that BM25 matches
precisely while a dense model may blur them into "something about products".
Loosely phrased questions ("how do I change how the product page looks?") are
the opposite case. Reciprocal Rank Fusion combines the two ranked lists using
ranks only, so it needs no score calibration between BM25 and cosine similarity:

    RRF(d) = sum over lists L of  1 / (k + rank_L(d))      with k = 60

The cross-encoder then re-scores the fused top-N jointly with the question.
Its top score doubles as a confidence signal for the "not found" gate.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

from qdrant_client import QdrantClient, models

from app.config import Settings
from app.embeddings import DenseEmbedder, Reranker, SparseBM25
from app.store import DENSE, SPARSE

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    point_id: str
    payload: dict
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None

    @property
    def doc_path(self) -> str:
        return self.payload["doc_path"]


@dataclass
class RetrievalResult:
    candidates: list[Candidate]
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    top_rerank_score: float | None = None
    details: dict = field(default_factory=dict)


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60, weights: list[float] | None = None) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. Ranks are 1-based. Ties broken by id for determinism."""
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = defaultdict(float)
    for w, ids in zip(weights, ranked_lists, strict=True):
        for rank, pid in enumerate(ids, start=1):
            scores[pid] += w / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class Retriever:
    def __init__(self, settings: Settings, client: QdrantClient, dense: DenseEmbedder,
                 sparse: SparseBM25, reranker: Reranker | None = None) -> None:
        self.s = settings
        self.client = client
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        # Eval runs ask the same question in several modes - embed it once.
        self._embed_dense = lru_cache(maxsize=1024)(lambda q: tuple(self.dense.embed_query(q)))
        self._embed_sparse = lru_cache(maxsize=1024)(self.sparse.embed_query)

    # ------------------------------------------------------------- searches
    def _dense_request(self, question: str, limit: int) -> models.QueryRequest:
        return models.QueryRequest(query=list(self._embed_dense(question)), using=DENSE, limit=limit, with_payload=True)

    def _sparse_request(self, question: str, limit: int) -> models.QueryRequest:
        return models.QueryRequest(query=self._embed_sparse(question), using=SPARSE, limit=limit, with_payload=True)

    @staticmethod
    def _to_candidates(points, rank_attr: str | None) -> list[Candidate]:
        out = []
        for rank, p in enumerate(points, start=1):
            c = Candidate(point_id=str(p.id), payload=p.payload or {}, score=float(p.score))
            if rank_attr:
                setattr(c, rank_attr, rank)
            out.append(c)
        return out

    def search(self, question: str, mode: str, chunking: str = "heading",
               sparse_weight: float | None = None) -> list[Candidate]:
        collection = self.s.collection_for(chunking)
        n = self.s.retrieval_candidates
        if mode == "dense":
            res = self.client.query_points(collection, query=list(self._embed_dense(question)), using=DENSE,
                                           limit=n, with_payload=True)
            return self._to_candidates(res.points, "dense_rank")
        if mode == "bm25":
            res = self.client.query_points(collection, query=self._embed_sparse(question), using=SPARSE,
                                           limit=n, with_payload=True)
            return self._to_candidates(res.points, "sparse_rank")
        if mode in ("hybrid", "hybrid_rerank"):
            w = self.s.hybrid_sparse_weight if sparse_weight is None else sparse_weight
            if self.s.fusion_backend == "qdrant":
                return self._hybrid_native(collection, question, n, w)
            return self._hybrid_python(collection, question, n, w)
        raise ValueError(f"unknown mode {mode!r}")

    def _hybrid_python(self, collection: str, question: str, n: int, sparse_weight: float) -> list[Candidate]:
        # Both searches in one round trip, fused here so the formula is explicit and testable.
        dense_res, sparse_res = self.client.query_batch_points(
            collection, requests=[self._dense_request(question, n), self._sparse_request(question, n)]
        )
        by_id: dict[str, Candidate] = {}
        for rank, p in enumerate(dense_res.points, start=1):
            by_id[str(p.id)] = Candidate(str(p.id), p.payload or {}, 0.0, dense_rank=rank)
        for rank, p in enumerate(sparse_res.points, start=1):
            c = by_id.setdefault(str(p.id), Candidate(str(p.id), p.payload or {}, 0.0))
            c.sparse_rank = rank
        fused = rrf_fuse([[str(p.id) for p in dense_res.points], [str(p.id) for p in sparse_res.points]],
                         k=self.s.rrf_k, weights=[1.0, sparse_weight])
        out = []
        for pid, score in fused[:n]:
            c = by_id[pid]
            c.score = score
            out.append(c)
        return out

    def _hybrid_native(self, collection: str, question: str, n: int, sparse_weight: float) -> list[Candidate]:
        # Same thing, fused server-side by Qdrant's Query API (prefetch + RRF).
        res = self.client.query_points(
            collection,
            prefetch=[
                models.Prefetch(query=list(self._embed_dense(question)), using=DENSE, limit=n),
                models.Prefetch(query=self._embed_sparse(question), using=SPARSE, limit=n),
            ],
            query=models.RrfQuery(rrf=models.Rrf(k=self.s.rrf_k, weights=[1.0, sparse_weight])),
            limit=n,
            with_payload=True,
        )
        return self._to_candidates(res.points, None)

    # --------------------------------------------------------------- rerank
    def _score(self, question: str, candidates: list[Candidate]) -> None:
        key = self.s.rerank_input
        texts = [c.payload.get(key) or c.payload["text"] for c in candidates]
        for c, s in zip(candidates, self.reranker.score(question, texts), strict=True):  # type: ignore[union-attr]
            c.rerank_score = s

    def rerank(self, question: str, candidates: list[Candidate]) -> list[Candidate]:
        if not self.reranker or not candidates:
            return candidates
        self._score(question, candidates)
        return sorted(candidates, key=lambda c: (-(c.rerank_score or 0.0), c.point_id))

    def confidence(self, question: str, context: list[Candidate]) -> float | None:
        """Cross-encoder used as a relevance *classifier* rather than a ranker.

        Scores only the chunks that will be sent to the LLM and returns the best
        score - "is anything in this context actually about the question?". Used
        by the not-found gate in modes that do not rerank.
        """
        if not self.reranker or not context:
            return None
        self._score(question, context)
        return max(c.rerank_score for c in context if c.rerank_score is not None)

    # ----------------------------------------------------------------- main
    def retrieve(self, question: str, mode: str, chunking: str = "heading", top_k: int | None = None,
                 sparse_weight: float | None = None) -> RetrievalResult:
        t0 = time.perf_counter()
        candidates = self.search(question, mode, chunking, sparse_weight)
        t1 = time.perf_counter()
        result = RetrievalResult(candidates=candidates, retrieve_ms=(t1 - t0) * 1000)
        if mode == "hybrid_rerank":
            if self.reranker is None:
                raise RuntimeError("mode hybrid_rerank requires a reranker")
            result.candidates = self.rerank(question, candidates)
            result.top_rerank_score = result.candidates[0].rerank_score if result.candidates else None
        elif top_k is not None:
            result.top_rerank_score = self.confidence(question, candidates[:top_k])
        result.rerank_ms = (time.perf_counter() - t1) * 1000
        return result

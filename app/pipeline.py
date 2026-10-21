"""The RAG pipeline used by both the API and the in-process eval runner."""

from __future__ import annotations

import json
import logging
import time

from qdrant_client import QdrantClient

from app.config import Settings
from app.embeddings import (
    DenseEmbedder,
    Reranker,
    SparseBM25,
    make_dense,
    make_reranker,
    make_sparse,
)
from app.generation import Generation, generate, snippet
from app.llm import LLMClient, make_llm
from app.retrieval import Candidate, Retriever
from app.schemas import AskRequest, AskResponse, Citation, RetrievedContext, Timings, Usage
from app.store import collection_metadata, get_client

log = logging.getLogger("rag")


class RAGPipeline:
    def __init__(self, settings: Settings, client: QdrantClient, dense: DenseEmbedder, sparse: SparseBM25,
                 reranker: Reranker | None, llm: LLMClient | None) -> None:
        self.s = settings
        self.client = client
        self.llm = llm
        self.retriever = Retriever(settings, client, dense, sparse, reranker)

    @classmethod
    def from_settings(cls, settings: Settings, with_llm: bool = True) -> RAGPipeline:
        return cls(
            settings,
            get_client(settings),
            make_dense(settings),
            make_sparse(settings),
            make_reranker(settings),
            make_llm(settings) if with_llm else None,
        )

    # ------------------------------------------------------------------ ask
    def ask(self, req: AskRequest) -> AskResponse:
        t0 = time.perf_counter()
        retrieval = self.retriever.retrieve(req.question, req.mode, req.chunking, top_k=req.top_k,
                                            sparse_weight=req.sparse_weight)
        context = retrieval.candidates[: req.top_k]

        threshold = self.s.rerank_threshold
        refused_by = None
        gen = Generation(answer="", answerable=True, citation_numbers=[])
        below = (
            threshold is not None
            and retrieval.top_rerank_score is not None
            and retrieval.top_rerank_score < threshold
        )
        if below:
            # Confidence gate: nothing relevant enough was found - say so instead of inventing.
            gen = Generation(answer=self.s.not_found_message, answerable=False, citation_numbers=[])
            refused_by = "threshold"
        elif req.generate:
            gen = generate(self.llm, req.question, context, self.s.llm_max_tokens, self.s.not_found_message)
            if not gen.answerable:
                refused_by = "llm"

        citations = [self._citation(n, context[n - 1]) for n in gen.citation_numbers if 0 < n <= len(context)]
        timings = Timings(
            retrieve=round(retrieval.retrieve_ms, 1),
            rerank=round(retrieval.rerank_ms, 1),
            generate=round(gen.latency_ms, 1),
            total=round((time.perf_counter() - t0) * 1000, 1),
        )
        response = AskResponse(
            answer=gen.answer,
            answerable=gen.answerable,
            citations=citations,
            timings_ms=timings,
            usage=Usage(input_tokens=gen.input_tokens, output_tokens=gen.output_tokens),
            mode=req.mode,
            chunking=req.chunking,
            llm_model=gen.model,
            refused_by=refused_by,
            top_rerank_score=retrieval.top_rerank_score,
            contexts=[self._context(i, c) for i, c in enumerate(retrieval.candidates, 1)] if req.include_contexts else None,
        )
        log.info(json.dumps({
            "event": "ask", "mode": req.mode, "chunking": req.chunking, "top_k": req.top_k,
            "timings_ms": timings.model_dump(), "usage": response.usage.model_dump(),
            "answerable": gen.answerable, "refused_by": refused_by, "n_citations": len(citations),
            "top_rerank_score": retrieval.top_rerank_score,
        }))
        return response

    @staticmethod
    def _citation(n: int, c: Candidate) -> Citation:
        p = c.payload
        headings = p.get("headings") or []
        return Citation(n=n, title=p.get("title", ""), url=p.get("url", ""),
                        heading=headings[-1] if headings else None, snippet=snippet(p.get("text", "")),
                        doc_path=p.get("doc_path", ""))

    @staticmethod
    def _context(rank: int, c: Candidate) -> RetrievedContext:
        p = c.payload
        headings = p.get("headings") or []
        return RetrievedContext(
            rank=rank, chunk_id=p.get("chunk_id", c.point_id), doc_path=p.get("doc_path", ""),
            title=p.get("title", ""), url=p.get("url", ""), heading=headings[-1] if headings else None,
            score=round(c.score, 6), dense_rank=c.dense_rank, sparse_rank=c.sparse_rank,
            rerank_score=None if c.rerank_score is None else round(c.rerank_score, 4),
            token_count=p.get("token_count"), text=p.get("text"),
        )

    # --------------------------------------------------------------- health
    def health(self) -> dict:
        out: dict = {"qdrant": {"ok": False}, "collections": {}, "llm": {"provider": self.s.llm_provider}}
        try:
            names = {c.name for c in self.client.get_collections().collections}
            out["qdrant"] = {"ok": True, "url": self.s.qdrant_url}
            for chunking in ("heading", "fixed"):
                name = self.s.collection_for(chunking)
                if name not in names:
                    out["collections"][chunking] = {"name": name, "exists": False}
                    continue
                meta = collection_metadata(self.client, name)
                out["collections"][chunking] = {
                    "name": name,
                    "exists": True,
                    "points": self.client.count(name, exact=True).count,
                    "commit_sha": meta.get("commit_sha"),
                    "embedding_model": meta.get("embedding_model"),
                    "model_matches": meta.get("embedding_model") == self.retriever.dense.model_name,
                }
        except Exception as e:
            out["qdrant"] = {"ok": False, "error": str(e)[:200]}
        if self.llm is None:
            out["llm"].update({"ok": True, "model": "extractive (no LLM)"})
        else:
            out["llm"].update({"model": self.llm.model, **self.llm.health()})
        main = out["collections"].get("heading", {})
        out["status"] = "ok" if out["qdrant"]["ok"] and main.get("exists") and main.get("model_matches") and out["llm"].get("ok") else "degraded"
        return out

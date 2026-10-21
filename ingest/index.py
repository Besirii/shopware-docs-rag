"""Index pipeline: docs -> chunks -> dense + BM25 vectors -> Qdrant.

Idempotent by construction:
  * a chunk's point id is a hash of its content, so re-running upserts the
    same ids instead of creating duplicates;
  * only chunks whose id is not yet in the collection are embedded (the
    embedding "cache" is the collection itself);
  * points whose id no longer exists in the corpus (edited or deleted docs)
    are removed.

Usage:
    python -m ingest.index --chunking heading          # production index
    python -m ingest.index --chunking fixed            # baseline index (experiment #0)
    python -m ingest.index --chunking all --recreate   # rebuild everything
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

from qdrant_client import QdrantClient, models

from app.config import Settings, get_settings
from app.embeddings import FastEmbedDense, SparseBM25, make_dense, make_sparse
from app.store import (
    DENSE,
    SPARSE,
    all_point_ids,
    collection_metadata,
    ensure_collection,
    get_client,
)
from ingest.chunking import Chunk, chunk_documents, make_chunker
from ingest.corpus import load_documents
from ingest.fetch_docs import fetch_docs

log = logging.getLogger(__name__)

STRATEGIES = ("heading", "fixed")
UPSERT_BATCH = 64


def bm25_doc_length(sparse: SparseBM25, text: str) -> int:
    """Document length exactly as fastembed's BM25 counts it (stemmed, no stopwords)."""
    model = sparse._model.model  # type: ignore[attr-defined]
    from fastembed.sparse.utils.tokenizer import SimpleTokenizer

    return len(model._stem(SimpleTokenizer.tokenize(text)))


def _percentiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    s = sorted(values)
    pick = lambda q: s[min(len(s) - 1, int(q * (len(s) - 1)))]  # noqa: E731
    return {"p10": pick(0.1), "p50": pick(0.5), "p90": pick(0.9), "max": s[-1], "mean": round(statistics.mean(s))}


def index_strategy(settings: Settings, client: QdrantClient, strategy: str, docs, commit_sha: str,
                   recreate: bool = False) -> dict:
    t_start = time.perf_counter()
    collection = settings.collection_for(strategy)
    chunks: list[Chunk] = chunk_documents(docs, make_chunker(strategy, settings))
    log.info("[%s] %d documents -> %d chunks", strategy, len(docs), len(chunks))

    dense = make_dense(settings)
    sparse_probe = make_sparse(settings)
    avg_len = statistics.mean(bm25_doc_length(sparse_probe, c.embed_text) for c in chunks)
    sparse = make_sparse(settings, avg_len=avg_len)

    metadata = {
        "embedding_provider": settings.embedding_provider,
        "embedding_model": dense.model_name,
        "embedding_dim": dense.dim,
        "embedding_max_tokens": getattr(dense, "max_tokens", None),
        "sparse_model": settings.sparse_model,
        "bm25_avg_len": round(avg_len, 2),
        "chunking": strategy,
        "chunk_params": {
            "max_tokens": settings.chunk_max_tokens, "min_tokens": settings.chunk_min_tokens,
            "overlap_tokens": settings.chunk_overlap_tokens,
        } if strategy == "heading" else {
            "chunk_tokens": settings.baseline_chunk_tokens, "overlap_tokens": settings.baseline_overlap_tokens,
        },
        "commit_sha": commit_sha,
        "docs_repo_url": settings.docs_repo_url,
    }
    created = ensure_collection(client, collection, dense.dim, metadata, recreate=recreate)

    existing = set() if created else all_point_ids(client, collection)
    current = {c.point_id: c for c in chunks}
    to_add = [c for pid, c in current.items() if pid not in existing]
    stale = sorted(existing - current.keys())
    log.info("[%s] %d new/changed chunks to embed, %d unchanged, %d stale to delete",
             strategy, len(to_add), len(current) - len(to_add), len(stale))

    # If BM25 statistics drifted (corpus changed a lot), refresh sparse vectors of unchanged points too.
    if not created:
        old_avg = float(collection_metadata(client, collection).get("bm25_avg_len") or avg_len)
        if abs(old_avg - avg_len) / old_avg > 0.05:
            unchanged = [c for pid, c in current.items() if pid in existing]
            log.info("[%s] BM25 avg_len drift %.1f -> %.1f: refreshing %d sparse vectors",
                     strategy, old_avg, avg_len, len(unchanged))
            for i in range(0, len(unchanged), UPSERT_BATCH):
                batch = unchanged[i:i + UPSERT_BATCH]
                vecs = sparse.embed_documents([c.embed_text for c in batch])
                client.update_vectors(collection, points=[
                    models.PointVectors(id=c.point_id, vector={SPARSE: v}) for c, v in zip(batch, vecs, strict=True)
                ])

    # Truncation report: how much of each chunk the dense model actually sees.
    truncated = 0
    if isinstance(dense, FastEmbedDense) and to_add:
        lengths = dense.token_lengths([c.embed_text for c in to_add])
        truncated = sum(n > dense.max_tokens for n in lengths)

    t_embed = time.perf_counter()
    for i in range(0, len(to_add), UPSERT_BATCH):
        batch = to_add[i:i + UPSERT_BATCH]
        texts = [c.embed_text for c in batch]
        dense_vecs = dense.embed_documents(texts)
        sparse_vecs = sparse.embed_documents(texts)
        client.upsert(collection, wait=True, points=[
            models.PointStruct(
                id=c.point_id,
                vector={DENSE: dv, SPARSE: sv},
                payload={**c.payload(), "embedding_model": dense.model_name},
            )
            for c, dv, sv in zip(batch, dense_vecs, sparse_vecs, strict=True)
        ])
        done = i + len(batch)
        rate = done / max(time.perf_counter() - t_embed, 1e-6)
        log.info("[%s] embedded %d/%d (%.1f chunks/s)", strategy, done, len(to_add), rate)

    if stale:
        client.delete(collection, points_selector=models.PointIdsList(points=stale), wait=True)
    # Unchanged chunks may come from a newer commit - keep their provenance current.
    client.set_payload(collection, payload={"commit_sha": commit_sha},
                       points=models.FilterSelector(filter=models.Filter()), wait=True)
    client.update_collection(collection, metadata=metadata)

    count = client.count(collection, exact=True).count
    stats = {
        "collection": collection,
        "strategy": strategy,
        "commit_sha": commit_sha,
        "documents": len(docs),
        "chunks": len(chunks),
        "points_in_collection": count,
        "embedded_now": len(to_add),
        "deleted_stale": len(stale),
        "dense_truncated_chunks": truncated,
        "chunk_tokens": _percentiles([c.token_count for c in chunks]),
        "bm25_avg_len": round(avg_len, 2),
        "embedding_model": dense.model_name,
        "seconds": round(time.perf_counter() - t_start, 1),
    }
    assert count == len(current), f"collection has {count} points, corpus has {len(current)} chunks"
    out = Path("data/index_stats")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{collection}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("[%s] done: %s", strategy, json.dumps(stats))
    return stats


def run(strategies: list[str], recreate: bool = False, fetch: bool = False) -> list[dict]:
    settings = get_settings()
    if fetch or not (settings.docs_dir / ".git").exists():
        fetch_docs(settings)
    docs, _ = load_documents(settings)
    commit_sha = docs[0].commit_sha if docs else ""
    client = get_client(settings)
    return [index_strategy(settings, client, s, docs, commit_sha, recreate=recreate) for s in strategies]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunking", choices=[*STRATEGIES, "all"], default="all")
    parser.add_argument("--recreate", action="store_true", help="drop and rebuild the collection(s)")
    parser.add_argument("--fetch", action="store_true", help="(re)fetch the docs before indexing")
    args = parser.parse_args()
    logging.basicConfig(level=get_settings().log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    strategies = list(STRATEGIES) if args.chunking == "all" else [args.chunking]
    for stats in run(strategies, recreate=args.recreate, fetch=args.fetch):
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

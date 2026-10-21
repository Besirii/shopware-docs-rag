"""Qdrant collection schema and helpers.

One collection per chunking strategy, each holding for every chunk:
  * a named dense vector  `dense` (cosine)          -> semantic search
  * a named sparse vector `bm25`  (IDF modifier)    -> keyword search
  * the chunk metadata as payload, with keyword indexes for filtering

Collection-level metadata records how the vectors were produced (embedding
model, BM25 avg_len, docs commit). A query-time model that differs from the
index-time model is a silent, catastrophic bug - so we check it at startup.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient, models

from app.config import Settings

log = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "bm25"


def get_client(settings: Settings) -> QdrantClient:
    key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    if settings.qdrant_url == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(url=settings.qdrant_url, api_key=key, timeout=60)


def collection_metadata(client: QdrantClient, name: str) -> dict:
    info = client.get_collection(name)
    return dict(info.config.metadata or {})


def ensure_collection(client: QdrantClient, name: str, dim: int, metadata: dict, recreate: bool = False) -> bool:
    """Create the collection if missing. Returns True if it was (re)created.

    If the collection exists but was built with a different embedding model or
    dimension, refuse to mix vectors from two models - require --recreate.
    """
    exists = client.collection_exists(name)
    if exists and not recreate:
        current = collection_metadata(client, name)
        size = client.get_collection(name).config.params.vectors[DENSE].size  # type: ignore[index]
        if size != dim or current.get("embedding_model") != metadata.get("embedding_model"):
            raise RuntimeError(
                f"Collection {name!r} was built with {current.get('embedding_model')} (dim {size}); "
                f"current model is {metadata.get('embedding_model')} (dim {dim}). Re-run with --recreate."
            )
        return False
    if exists:
        log.warning("Recreating collection %s", name)
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        metadata=metadata,
    )
    for field in ("doc_path", "strategy", "commit_sha"):
        client.create_payload_index(name, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD)
    log.info("Created collection %s (dense dim %d)", name, dim)
    return True


def all_point_ids(client: QdrantClient, name: str) -> set[str]:
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(name, limit=1000, offset=offset, with_payload=False, with_vectors=False)
        ids.update(str(p.id) for p in points)
        if offset is None:
            return ids

"""Shared fixtures: fake models + in-memory Qdrant, so tests need no network or GPU."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from ingest.corpus import Document

_WORD = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _bucket(token: str, n: int) -> int:
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % n


class FakeDense:
    """Bag-of-words hashed into a small vector - similar texts get similar vectors."""

    model_name = "fake-dense"
    dim = 64
    max_tokens = 10_000

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim)
        for t in _tokens(text):
            v[_bucket(t, self.dim)] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def token_lengths(self, texts):
        return [len(_tokens(t)) for t in texts]


class FakeSparse:
    model_name = "fake-bm25"
    avg_len = 100.0

    def _vec(self, text: str) -> models.SparseVector:
        counts: dict[int, float] = {}
        for t in _tokens(text):
            i = _bucket(t, 100_000)
            counts[i] = counts.get(i, 0.0) + 1.0
        return models.SparseVector(indices=list(counts), values=list(counts.values()))

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return models.SparseVector(indices=sorted({_bucket(t, 100_000) for t in _tokens(text)}),
                                   values=[1.0] * len({_bucket(t, 100_000) for t in _tokens(text)}))


class FakeReranker:
    """Scores by word overlap with the query - enough to test ordering logic."""

    model_name = "fake-reranker"

    def score(self, query, passages):
        q = set(_tokens(query))
        return [float(len(q & set(_tokens(p)))) for p in passages]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        qdrant_url=":memory:",
        collection_name="test_docs",
        llm_provider="extractive",
        docs_dir=tmp_path / "docs",
        rerank_threshold=None,
    )


def make_doc(path: str, text: str, title: str = "Title", parents: list[str] | None = None) -> Document:
    return Document(path=path, url=f"https://example.test/{path[:-3]}.html", title=title,
                    parents=parents or ["Guides"], text=text, commit_sha="abc123")


@pytest.fixture
def docs() -> list[Document]:
    return [
        make_doc("guides/events.md", "# Listening to events\n\n## Register the subscriber\n\n"
                 "Tag the subscriber service with kernel.event_subscriber in services.php.\n\n"
                 "```php\n$services->set(MySubscriber::class)->tag('kernel.event_subscriber');\n```\n",
                 title="Listening to events"),
        make_doc("guides/theme.md", "# Themes\n\n## Compile\n\nRun bin/console theme:compile after changing SCSS "
                 "variables so the storefront picks them up.\n", title="Themes"),
        make_doc("guides/dal.md", "# Reading data\n\n## Filters\n\nUse an EqualsFilter on the Criteria and call "
                 "search on the EntityRepository to read products by name.\n", title="Reading data"),
    ]


@pytest.fixture
def indexed(settings, docs):
    """In-memory Qdrant with the heading index built from the fake docs."""
    from app.store import DENSE, SPARSE, ensure_collection
    from ingest.chunking import chunk_documents, make_chunker

    client = QdrantClient(location=":memory:")
    dense, sparse = FakeDense(), FakeSparse()
    for strategy in ("heading", "fixed"):
        chunks = chunk_documents(docs, make_chunker(strategy, settings))
        name = settings.collection_for(strategy)
        ensure_collection(client, name, dense.dim, {"embedding_model": dense.model_name, "commit_sha": "abc123"})
        client.upsert(name, points=[
            models.PointStruct(id=c.point_id, vector={DENSE: dv, SPARSE: sv}, payload=c.payload())
            for c, dv, sv in zip(chunks, dense.embed_documents([c.embed_text for c in chunks]),
                                 sparse.embed_documents([c.embed_text for c in chunks]), strict=True)
        ])
    return client, dense, sparse

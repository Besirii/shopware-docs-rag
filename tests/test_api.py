"""API tests: the real FastAPI app with an in-memory pipeline (fake models, extractive answers)."""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.llm import LLMResult
from app.pipeline import RAGPipeline
from tests.conftest import FakeReranker


class StubLLM:
    provider = "stub"
    model = "stub-model"

    def __init__(self, data):
        self.data = data

    def generate_json(self, system, user, schema, schema_name, max_tokens):
        assert "<documentation>" in user and schema["required"] == ["answer", "citations", "answerable"]
        return LLMResult(data=self.data, input_tokens=120, output_tokens=30, model=self.model, latency_ms=5.0)

    def health(self):
        return {"ok": True}


@pytest.fixture
def client_factory(settings, indexed, monkeypatch):
    qdrant, dense, sparse = indexed

    def make(llm=None, threshold=None):
        settings.rerank_threshold = threshold
        pipeline = RAGPipeline(settings, qdrant, dense, sparse, FakeReranker(), llm)
        monkeypatch.setattr(main.RAGPipeline, "from_settings", classmethod(lambda cls, s: pipeline))
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        return TestClient(main.app)

    return make


def test_ask_extractive_returns_cited_answer(client_factory):
    with client_factory() as client:
        r = client.post("/ask", json={"question": "How do I register a subscriber with kernel.event_subscriber?",
                                      "include_contexts": True})
    assert r.status_code == 200
    body = r.json()
    assert body["answerable"] is True
    assert body["citations"][0]["n"] == 1
    assert body["citations"][0]["doc_path"] == "guides/events.md"
    assert body["citations"][0]["url"].startswith("https://example.test/guides/events.html")
    assert body["contexts"][0]["rank"] == 1
    assert set(body["timings_ms"]) >= {"retrieve", "rerank", "generate", "total"}


def test_ask_with_llm_strips_invalid_citations(client_factory):
    llm = StubLLM({"answer": "Tag it [1]. Invented [42].", "citations": [1, 42], "answerable": True})
    with client_factory(llm=llm) as client:
        body = client.post("/ask", json={"question": "subscriber tag?", "top_k": 2}).json()
    assert body["answer"] == "Tag it [1]. Invented."
    assert [c["n"] for c in body["citations"]] == [1]
    assert body["usage"] == {"input_tokens": 120, "output_tokens": 30}
    assert body["llm_model"] == "stub-model"


def test_llm_says_not_answerable(client_factory):
    llm = StubLLM({"answer": "The documentation does not cover this.", "citations": [], "answerable": False})
    with client_factory(llm=llm) as client:
        body = client.post("/ask", json={"question": "GraphQL setup?"}).json()
    assert body["answerable"] is False and body["refused_by"] == "llm" and body["citations"] == []


def test_threshold_refuses_without_calling_llm(client_factory):
    class ExplodingLLM(StubLLM):
        def generate_json(self, *a, **k):
            raise AssertionError("LLM must not be called below the threshold")

    with client_factory(llm=ExplodingLLM({}), threshold=1000.0) as client:
        body = client.post("/ask", json={"question": "anything at all", "mode": "hybrid_rerank"}).json()
    assert body["answerable"] is False and body["refused_by"] == "threshold"
    assert body["answer"] == "I could not find this in the Shopware documentation."


def test_retrieval_only_mode(client_factory):
    with client_factory(llm=StubLLM({})) as client:
        body = client.post("/ask", json={"question": "theme compile", "generate": False, "mode": "dense",
                                         "chunking": "fixed", "include_contexts": True}).json()
    assert body["answer"] == "" and body["chunking"] == "fixed" and body["contexts"]


def test_validation_errors(client_factory):
    with client_factory() as client:
        assert client.post("/ask", json={"question": "hi"}).status_code == 422  # too short
        assert client.post("/ask", json={"question": "valid question", "mode": "magic"}).status_code == 422
        assert client.post("/ask", json={"question": "valid question", "top_k": 99}).status_code == 422


def test_health_reports_collections(client_factory):
    with client_factory() as client:
        r = client.get("/health")
    body = r.json()
    assert r.status_code == 200, body
    assert body["collections"]["heading"]["points"] > 0 and body["collections"]["heading"]["model_matches"]


def test_ingest_requires_admin_token(client_factory):
    with client_factory() as client:
        assert client.post("/ingest", json={}).status_code == 503  # disabled without ADMIN_TOKEN


def test_confidence_gate_also_works_without_reranking(client_factory):
    with client_factory(threshold=1000.0) as client:
        body = client.post("/ask", json={"question": "anything at all", "mode": "hybrid", "sparse_weight": 0.5}).json()
    assert body["refused_by"] == "threshold" and body["top_rerank_score"] is not None


def test_marker_only_answer_becomes_not_found(client_factory):
    llm = StubLLM({"answer": "[1]", "citations": [1], "answerable": False})
    with client_factory(llm=llm) as client:
        body = client.post("/ask", json={"question": "GraphQL setup?"}).json()
    assert body["answer"] == "I could not find this in the Shopware documentation."
    assert body["citations"] == [] and body["answerable"] is False


def test_ingest_with_admin_token(client_factory, settings, monkeypatch):
    from pydantic import SecretStr

    import ingest.index

    calls = []
    monkeypatch.setattr(ingest.index, "run", lambda strategies, recreate, fetch: calls.append(strategies) or [{"ok": 1}])
    settings.admin_token = SecretStr("s3cret")
    with client_factory() as client:
        assert client.post("/ingest", json={}, headers={"X-Admin-Token": "wrong"}).status_code == 401
        r = client.post("/ingest", json={"chunking": "heading", "fetch": False}, headers={"X-Admin-Token": "s3cret"})
        assert r.status_code == 202 and r.json()["status"] == "running"
        status = client.get("/ingest/status").json()  # TestClient runs background tasks before returning
    assert calls == [["heading"]] and status["status"] == "done"


def test_demo_page_is_served(client_factory):
    with client_factory() as client:
        r = client.get("/")
    assert r.status_code == 200 and "Shopware 6 Docs RAG" in r.text and "/ask" in r.text

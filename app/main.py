"""FastAPI app: POST /ask, GET /health, POST /ingest (admin)."""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.config import get_settings
from app.llm import LLMError
from app.pipeline import RAGPipeline
from app.schemas import AskRequest, AskResponse


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        base = {"ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
                "level": record.levelname, "logger": record.name}
        try:
            payload = json.loads(msg)
            if isinstance(payload, dict):
                return json.dumps({**base, **payload})
        except ValueError:
            pass
        return json.dumps({**base, "message": msg})


def _configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level)
    t0 = time.perf_counter()
    # Load every model once at startup so the first request is not slow.
    app.state.pipeline = RAGPipeline.from_settings(settings)
    app.state.ingest = {"status": "idle"}
    app.state.ingest_lock = threading.Lock()
    logging.getLogger("rag").info(json.dumps({"event": "startup", "seconds": round(time.perf_counter() - t0, 1),
                                              **settings.fingerprint()}))
    yield


app = FastAPI(
    title="Shopware 6 Docs RAG",
    description="Answers Shopware 6 developer questions from the official documentation, with citations.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
def ask(req: AskRequest) -> AskResponse:
    # Sync endpoint on purpose: model inference is CPU-bound, FastAPI runs it in a threadpool.
    try:
        return app.state.pipeline.ask(req)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}") from e


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Minimal demo UI (single static page, no build step)."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health() -> JSONResponse:
    report = app.state.pipeline.health()
    return JSONResponse(report, status_code=200 if report["status"] == "ok" else 503)


class IngestRequest(BaseModel):
    chunking: str = "all"
    recreate: bool = False
    fetch: bool = True


def _run_ingest(state, req: IngestRequest) -> None:
    from ingest.index import STRATEGIES, run

    try:
        strategies = list(STRATEGIES) if req.chunking == "all" else [req.chunking]
        stats = run(strategies, recreate=req.recreate, fetch=req.fetch)
        state.ingest = {"status": "done", "finished_at": datetime.now(UTC).isoformat(), "stats": stats}
    except Exception as e:  # report, don't crash the server
        logging.getLogger("rag").exception("ingest failed")
        state.ingest = {"status": "failed", "error": str(e)}
    finally:
        state.ingest_lock.release()


@app.post("/ingest", status_code=202)
def ingest(req: IngestRequest, background: BackgroundTasks, x_admin_token: str | None = Header(default=None)) -> dict:
    token = get_settings().admin_token
    if token is None:
        raise HTTPException(status_code=503, detail="ingest endpoint disabled (set ADMIN_TOKEN)")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, token.get_secret_value()):
        raise HTTPException(status_code=401, detail="invalid admin token")
    if req.chunking not in ("all", "heading", "fixed"):
        raise HTTPException(status_code=422, detail="chunking must be all|heading|fixed")
    if not app.state.ingest_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="an ingest job is already running")
    app.state.ingest = {"status": "running", "started_at": datetime.now(UTC).isoformat(), "request": req.model_dump()}
    background.add_task(_run_ingest, app.state, req)
    return app.state.ingest


@app.get("/ingest/status")
def ingest_status() -> dict:
    return app.state.ingest

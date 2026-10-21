# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Shopware 6 Docs RAG - API image
#   * dependencies installed with uv from the lockfile (reproducible)
#   * no model weights inside the image: embedding / BM25 / reranker models are
#     downloaded on first start into the /models volume and reused afterwards
#   * runs as an unprivileged user
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# git is needed by `python -m ingest` (shallow sparse clone of shopware/docs).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first: this layer is cached until pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY ingest ./ingest
COPY eval ./eval

# Match your host UID on Linux so bind-mounted ./data and ./eval/results stay writable.
ARG UID=1000
RUN useradd --create-home --uid ${UID} rag \
    && mkdir -p /models /app/data /app/eval/results \
    && chown -R rag:rag /models /app

# GIT_CONFIG_*: ./data is bind-mounted from the host and owned by another uid;
# without safe.directory git refuses to touch the docs checkout ("dubious ownership").
ENV PATH="/opt/venv/bin:$PATH" \
    MODEL_CACHE_DIR=/models \
    HF_HOME=/models/hf \
    TIKTOKEN_CACHE_DIR=/models/tiktoken \
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0=safe.directory \
    GIT_CONFIG_VALUE_0=*

USER rag
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/docs', timeout=4)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Shopware 6 Docs RAG - common tasks. Windows without make: use ./tasks.ps1 <target>.
.DEFAULT_GOAL := help
COMPOSE       ?= docker compose
OLLAMA_MODEL  ?= qwen2.5:3b
API           ?= http://localhost:8000
Q             ?= How do I register an event subscriber in a plugin?

.PHONY: help up up-ollama pull-model down logs ingest ingest-local eval eval-retrieval eval-judge report calibrate test lint ask health

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

up: ## build and start qdrant + api
	$(COMPOSE) up -d --build

up-ollama: ## same as up, plus a local LLM (Ollama) - then run `make pull-model`
	$(COMPOSE) --profile ollama up -d --build

pull-model: ## download the local LLM into the ollama container
	$(COMPOSE) exec ollama ollama pull $(OLLAMA_MODEL)

down: ## stop everything (volumes are kept)
	$(COMPOSE) --profile ollama down

logs: ## follow api logs
	$(COMPOSE) logs -f api

ingest: ## fetch the docs + build both indexes (inside the api container)
	$(COMPOSE) run --rm api python -m ingest

ingest-local: ## same, with the local virtualenv (uv)
	uv run python -m ingest

eval: ## full eval (answers) of experiments 0-3 through the running API
	$(COMPOSE) exec api python -m eval.run_eval --experiments main --api-url http://localhost:8000

eval-retrieval: ## retrieval-only eval (no LLM, free, deterministic) of every experiment
	$(COMPOSE) exec api python -m eval.run_eval --experiments all --no-generate --api-url http://localhost:8000

eval-judge: ## eval with LLM-judge metrics (uses JUDGE_PROVIDER / JUDGE_MODEL)
	$(COMPOSE) exec api python -m eval.run_eval --experiments main --judge --api-url http://localhost:8000

report: ## regenerate the results tables in docs/DESIGN.md
	uv run python -m eval.report --update-docs

calibrate: ## suggest RERANK_THRESHOLD from the latest 3_rerank result
	uv run python -m eval.calibrate

test: ## unit + API tests (no network, no models)
	uv run pytest

lint:
	uv run ruff check .

health:
	curl -s $(API)/health

ask: ## make ask Q="your question"
	curl -s -X POST $(API)/ask -H 'Content-Type: application/json' \
		-d '{"question": "$(Q)", "top_k": 6, "mode": "hybrid"}'

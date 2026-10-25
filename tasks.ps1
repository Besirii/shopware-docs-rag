# Windows equivalent of the Makefile:  .\tasks.ps1 <target> ["question"] [-Model qwen2.5:3b]
param(
    [Parameter(Position = 0)][string]$Target = "help",
    [Parameter(Position = 1)][string]$Question = "How do I register an event subscriber in a plugin?",
    [string]$Model = "qwen2.5:3b",
    [string]$Api = "http://localhost:8000"
)
$ErrorActionPreference = "Stop"

switch ($Target) {
    "up"             { docker compose up -d --build }
    "up-ollama"      { docker compose --profile ollama up -d --build }
    "pull-model"     { docker compose exec ollama ollama pull $Model }
    "down"           { docker compose --profile ollama down }
    "logs"           { docker compose logs -f api }
    "ingest"         { docker compose run --rm api python -m ingest }
    "ingest-local"   { uv run python -m ingest }
    "eval"           { docker compose exec api python -m eval.run_eval --experiments main --api-url http://localhost:8000 }
    "eval-retrieval" { docker compose exec api python -m eval.run_eval --experiments all --no-generate --api-url http://localhost:8000 }
    "eval-judge"     { docker compose exec api python -m eval.run_eval --experiments main --judge --api-url http://localhost:8000 }
    "eval-agent"     { docker compose exec api python -m eval.run_eval --experiments compare --dataset eval/questions_multihop.jsonl --api-url http://localhost:8000 }
    "eval-agent-single" { docker compose exec api python -m eval.run_eval --experiments compare --api-url http://localhost:8000 }
    "report"         { uv run python -m eval.report --update-docs }
    "calibrate"      { uv run python -m eval.calibrate }
    "test"           { uv run pytest }
    "lint"           { uv run ruff check . }
    "health"         { Invoke-RestMethod "$Api/health" | ConvertTo-Json -Depth 6 }
    "ask" {
        $body = @{ question = $Question; top_k = 6; mode = "hybrid" } | ConvertTo-Json
        $r = Invoke-RestMethod -Method Post -Uri "$Api/ask" -ContentType "application/json; charset=utf-8" -Body $body
        $r.answer
        ""
        $r.citations | ForEach-Object { "[{0}] {1} - {2}" -f $_.n, $_.title, $_.url }
    }
    "agent" {
        $body = @{ question = $Question } | ConvertTo-Json
        $r = Invoke-RestMethod -Method Post -Uri "$Api/agent" -ContentType "application/json; charset=utf-8" -Body $body
        $r.steps | ForEach-Object { "{0}. {1} {2} -> {3} {4}" -f $_.step, $_.tool, ($_.args | ConvertTo-Json -Compress), ($_.sources -join ","), $_.error }
        ""
        $r.answer
        ""
        $r.citations | ForEach-Object { "[{0}] {1} - {2}" -f $_.n, $_.title, $_.url }
        ""
        "{0} tool calls, {1} model calls, {2:N1}s ({3})" -f $r.tool_calls, $r.model_calls, ($r.timings_ms.total / 1000), $r.llm_model
    }
    default {
        "Targets: up, up-ollama, pull-model, down, logs, ingest, ingest-local, eval, eval-retrieval,"
        "         eval-judge, eval-agent, eval-agent-single, report, calibrate, test, lint, health, ask '...', agent '...'"
    }
}

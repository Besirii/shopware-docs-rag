"""Ask the agent from the terminal and watch every tool call.

    uv run python -m app.agent "How do I add a custom field to products and show it in the storefront?"
    uv run python -m app.agent "..." --max-tool-calls 4 --json

Needs Qdrant with the index (`make up` + `make ingest`) and AGENT_PROVIDER / LLM_PROVIDER set.
"""

from __future__ import annotations

import argparse
import json
import sys

from app.agent import build_agent
from app.config import get_settings
from app.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.agent", description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--max-tool-calls", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="print the full API response as JSON")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]  # Windows consoles + "…"

    settings = get_settings()
    runner = build_agent(settings, RAGPipeline.from_settings(settings, with_llm=False))
    if runner is None:
        sys.exit("Agent disabled: set AGENT_PROVIDER (or LLM_PROVIDER) to anthropic, openai or ollama.")
    res = runner.run(args.question, args.max_tool_calls)

    if args.json:
        print(res.model_dump_json(indent=2))
        return
    for st in res.steps or []:
        status = f"  ERROR: {st.error}" if st.error else f"  -> sources {st.sources}"
        print(f"{st.step}. {st.tool}({json.dumps(st.args, ensure_ascii=False)}){status}")
    print(f"\n{res.answer}\n")
    for c in res.citations:
        print(f"[{c.n}] {c.title} - {c.url}")
    print(f"\n{res.tool_calls} tool calls, {res.model_calls} model calls, "
          f"{res.usage.input_tokens} in / {res.usage.output_tokens} out tokens, "
          f"{res.timings_ms.total / 1000:.1f}s ({res.llm_model}, stopped by {res.stopped_by})")


if __name__ == "__main__":
    main()

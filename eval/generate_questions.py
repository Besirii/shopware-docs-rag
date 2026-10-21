"""Draft extra eval questions with an LLM from random chunks - for HUMAN REVIEW.

Writes eval/candidates.jsonl (never questions.jsonl directly). Review every line:
fix the question so it sounds like a real developer, check the ground truth
against the page, then move the good ones into questions.jsonl.

Synthetic questions tend to copy the wording of the chunk they came from, which
flatters keyword search. The prompt asks for a paraphrase, but the review is
what keeps the eval honest.

Usage:  python -m eval.generate_questions --n 20 [--seed 7]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from app.config import get_settings
from app.llm import make_llm
from app.store import get_client

OUT = Path(__file__).parent / "candidates.jsonl"

PROMPT = """\
Below is an excerpt from the Shopware 6 developer documentation (page: {title}).

Write ONE question a Shopware developer could realistically ask that this excerpt answers.
Rules: phrase it in your own words (do not copy distinctive phrases or headings from the excerpt),
make it specific, and do not mention "the excerpt" or "the documentation".
Then write a short ground-truth answer (1-3 sentences) using only the excerpt.
If the excerpt is too thin to support a useful question, set "usable" to false.

EXCERPT:
{text}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "question": {"type": "string"},
        "ground_truth": {"type": "string"},
    },
    "required": ["usable", "question", "ground_truth"],
    "additionalProperties": False,
}


def category_for(path: str) -> str:
    if "/framework/data-handling/" in path or "/dal-reference/" in path:
        return "dal"
    if "/storefront/" in path or "/themes/" in path:
        return "storefront"
    if "store-api" in path or "integrations-api" in path or path.startswith("concepts/api"):
        return "api"
    if path.startswith("guides/hosting") or "commands-reference" in path:
        return "cli-config"
    if path.startswith("guides/plugins"):
        return "plugin-dev"
    return "concepts"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    s = get_settings()
    llm = make_llm(s)
    if llm is None:
        sys.exit("set LLM_PROVIDER to a real model to generate questions")
    client = get_client(s)
    points, _ = client.scroll(s.collection_for("heading"), limit=10_000, with_payload=True, with_vectors=False)
    rng = random.Random(args.seed)
    pool = [p for p in points if 150 <= (p.payload or {}).get("token_count", 0) <= 700
            and "/adr/" not in p.payload["doc_path"]]
    written = 0
    with OUT.open("a", encoding="utf-8") as f:
        for p in rng.sample(pool, min(len(pool), args.n * 2)):
            if written >= args.n:
                break
            payload = p.payload
            res = llm.generate_json("You write evaluation data for a RAG system.",
                                    PROMPT.format(title=payload["title"], text=payload["text"]), SCHEMA,
                                    "eval_question", 600)
            if not res.data.get("usable"):
                continue
            row = {
                "id": f"g{written + 1:02d}_{args.seed}",
                "category": category_for(payload["doc_path"]),
                "answerable": True,
                "question": res.data["question"].strip(),
                "ground_truth": res.data["ground_truth"].strip(),
                "relevant_docs": [payload["doc_path"]],
                "source": "generated",
                "needs_review": True,
                "source_chunk": payload["chunk_id"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            print(f"[{row['category']}] {row['question']}")
    print(f"\n{written} candidates appended to {OUT} - review them before using!")


if __name__ == "__main__":
    main()

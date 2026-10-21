"""Eval dataset: one JSON object per line in eval/questions.jsonl."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent / "questions.jsonl"
REQUIRED = ("id", "question", "ground_truth", "relevant_docs", "category", "answerable")


@dataclass
class EvalQuestion:
    id: str
    question: str
    ground_truth: str
    relevant_docs: list[str]
    category: str
    answerable: bool = True
    source: str = "curated"
    # Key facts (identifiers, commands) the context must contain to answer -> evidence@k.
    evidence: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def load_questions(path: Path = DEFAULT_PATH) -> list[EvalQuestion]:
    out: list[EvalQuestion] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        row = json.loads(line)
        missing = [k for k in REQUIRED if k not in row]
        if missing:
            raise ValueError(f"{path}:{lineno} missing fields {missing}")
        if row["id"] in seen:
            raise ValueError(f"{path}:{lineno} duplicate id {row['id']}")
        if row["answerable"] and not row["relevant_docs"]:
            raise ValueError(f"{path}:{lineno} answerable question without relevant_docs")
        seen.add(row["id"])
        known = {k: row.pop(k) for k in list(row) if k in EvalQuestion.__dataclass_fields__}
        out.append(EvalQuestion(**known, extra=row))
    return out

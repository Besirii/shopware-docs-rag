"""Retrieval and refusal metrics - deterministic, free, run them on every change.

All retrieval metrics are computed at *document* level from the ranked list of
retrieved chunks: a chunk counts as relevant if it comes from one of the
question's `relevant_docs`. The rank of a document is the rank of its first
chunk, because that is the position at which the LLM first sees it.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence


def first_relevant_rank(ranked_doc_paths: Sequence[str], relevant: Iterable[str]) -> int | None:
    """1-based rank of the first chunk from a relevant document, or None."""
    rel = set(relevant)
    for rank, path in enumerate(ranked_doc_paths, start=1):
        if path in rel:
            return rank
    return None


def hit_at_k(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: int | None, k: int = 10) -> float:
    """MRR@k contribution: 1/rank if found within k, else 0."""
    return 1.0 / rank if rank is not None and rank <= k else 0.0


def evidence_recall(context_texts: Sequence[str], evidence: Sequence[str]) -> float | None:
    """Share of the key facts that literally appear in the chunks given to the LLM.

    Doc-level hits say "the right page was found"; this says "the passage that
    actually contains the answer was in the prompt" - a chunk-level signal that
    is still deterministic and free.
    """
    if not evidence:
        return None
    blob = "\n".join(context_texts).lower()
    return sum(e.lower() in blob for e in evidence) / len(evidence)


def doc_recall_at_k(ranked_doc_paths: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Share of the relevant documents that appear in the top-k chunks."""
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(rel & set(ranked_doc_paths[:k])) / len(rel)


def average_precision(relevance: Sequence[bool]) -> float:
    """Average precision over a ranked list of binary judgements.

    This is RAGAS' "context precision": it rewards rankings that put the useful
    chunks first. 1.0 = every relevant chunk is above every irrelevant one.
    """
    hits = 0
    total = 0.0
    for i, is_rel in enumerate(relevance, start=1):
        if is_rel:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


def refusal_outcome(answerable_truth: bool, refused: bool) -> str:
    """Confusion-matrix cell for the "I don't know" behaviour."""
    if not answerable_truth:
        return "correct_refusal" if refused else "hallucinated_answer"
    return "false_refusal" if refused else "answered"


def mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.mean(vals), 4) if vals else None


def percentile(values: Iterable[float], q: float) -> float | None:
    vals = sorted(values)
    if not vals:
        return None
    return round(vals[min(len(vals) - 1, int(q * (len(vals) - 1)))], 1)


def bootstrap_ci(values: Sequence[float], n_resamples: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float] | None:
    """Percentile bootstrap confidence interval of the mean.

    With ~40 questions one question moving from rank 1 to rank 2 changes MRR by
    ~0.012, so differences between configurations must be read against this
    interval - not as absolute truth.
    """
    import random

    vals = list(values)
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(vals, k=len(vals))) for _ in range(n_resamples))
    lo = means[int(alpha / 2 * n_resamples)]
    hi = means[int((1 - alpha / 2) * n_resamples) - 1]
    return round(lo, 3), round(hi, 3)


def paired_changes(before: dict[str, int | None], after: dict[str, int | None]) -> dict[str, int]:
    """How many questions got a better / worse / same first-relevant rank (None = not found)."""
    inf = 10**9
    better = worse = same = 0
    for qid, b in before.items():
        a = after.get(qid)
        b_, a_ = (b or inf), (a or inf)
        if a_ < b_:
            better += 1
        elif a_ > b_:
            worse += 1
        else:
            same += 1
    return {"better": better, "worse": worse, "same": same}

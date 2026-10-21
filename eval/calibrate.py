"""Calibrate RERANK_THRESHOLD - the "I could not find this" gate.

Reads a hybrid_rerank result file (which records the top reranker score of every
question, answerable or not) and sweeps candidate thresholds. A question is
refused when its top score is below the threshold. We pick the threshold with
the best accuracy; ties go to the lower threshold (prefer answering).

With only a handful of unanswerable questions the in-sample number is
optimistic, so we also report leave-one-out accuracy: for each question the
threshold is chosen on all *other* questions, then applied to the held-out one.

Usage:  python -m eval.calibrate [--result eval/results/<file>.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from eval.run_eval import RESULTS_DIR


def _accuracy(rows: list[tuple[float, bool]], threshold: float) -> tuple[float, int, int]:
    correct = false_refusals = hallucinations = 0
    for score, answerable in rows:
        refused = score < threshold
        if refused != answerable:
            correct += 1
        elif refused and answerable:
            false_refusals += 1
        else:
            hallucinations += 1
    return correct / len(rows), false_refusals, hallucinations


def best_threshold(rows: list[tuple[float, bool]]) -> float:
    scores = sorted({s for s, _ in rows})
    # Candidates: midpoints between observed scores, plus "never refuse".
    candidates = [scores[0] - 1.0] + [(a + b) / 2 for a, b in itertools.pairwise(scores)]
    return max(candidates, key=lambda t: (_accuracy(rows, t)[0], -t))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", type=Path, help="result file (default: newest *_3_rerank.json)")
    args = parser.parse_args()
    path = args.result or max(RESULTS_DIR.glob("*_3_rerank.json"), default=None)
    if path is None:
        sys.exit("no 3_rerank result found - run `python -m eval.run_eval --experiments 3_rerank --no-generate`")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [(q["top_rerank_score"], q["answerable_truth"]) for q in data["questions"]
            if q.get("top_rerank_score") is not None]
    if not any(not a for _, a in rows):
        sys.exit("dataset has no unanswerable questions - cannot calibrate")

    print(f"Result: {path}  (reranker: {data['settings'].get('rerank_model')})")
    print("\nTop reranker score per question (sorted):")
    for score, answerable in sorted(rows):
        print(f"  {score:8.3f}  {'answerable' if answerable else 'UNANSWERABLE'}")

    t = best_threshold(rows)
    acc, fr, hal = _accuracy(rows, t)
    print(f"\nBest threshold (in-sample): {t:.3f}  accuracy={acc:.1%}  false_refusals={fr}  hallucinations={hal}")

    loo_correct = 0
    for i, (score, answerable) in enumerate(rows):
        t_i = best_threshold(rows[:i] + rows[i + 1:])
        loo_correct += int((score < t_i) != answerable)
    print(f"Leave-one-out accuracy: {loo_correct / len(rows):.1%}  (honest estimate for unseen questions)")

    # With generated answers we can also simulate "gate OR the LLM says not answerable".
    gen = [(q["top_rerank_score"], q["answerable_truth"], q.get("refused_by") == "llm")
           for q in data["questions"] if q.get("top_rerank_score") is not None and q.get("refused") is not None]
    if gen and data["config"].get("generate"):
        def combined(th: float) -> tuple[float, int, int]:
            fr = sum(1 for s, a, llm in gen if a and (llm or s < th))
            hal = sum(1 for s, a, llm in gen if not a and not (llm or s < th))
            return 1 - (fr + hal) / len(gen), fr, hal

        llm_only = combined(float("-inf"))
        print(f"\nLLM refusals only (no gate):  accuracy={llm_only[0]:.1%}  false_refusals={llm_only[1]}  "
              f"hallucinations={llm_only[2]}")
        cands = sorted({s for s, _, _ in gen})
        options = [float("-inf")] + [(a + b) / 2 for a, b in itertools.pairwise(cands)]
        best = max(options, key=lambda th: (combined(th)[0], -th))
        acc, fr, hal = combined(best)
        if best == float("-inf"):
            print("No threshold improves on the LLM's own refusals - leave RERANK_THRESHOLD empty.")
            return
        print(f"Gate + LLM, best threshold {best:.3f}:  accuracy={acc:.1%}  false_refusals={fr}  hallucinations={hal}")
        t = best
    print(f"\nSet in .env:  RERANK_THRESHOLD={t:.3f}")


if __name__ == "__main__":
    main()

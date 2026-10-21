"""Run the eval set against the RAG system and store the results.

Each *experiment* is one row of the README's before/after table:

    0_baseline  fixed 512-token chunks + dense only
    1_heading   + heading-aware chunking
    2_hybrid    + hybrid search (BM25 + dense, RRF)
    3_rerank    + cross-encoder reranker (+ "not found" threshold if configured)

Usage:
    python -m eval.run_eval --experiments all                  # retrieval + answers (+ judge if --judge)
    python -m eval.run_eval --experiments all --no-generate    # retrieval metrics only: free & fast
    python -m eval.run_eval --experiments 3_rerank --judge     # add LLM-judge metrics
    python -m eval.run_eval --api-url http://localhost:8000    # go through the HTTP API (default: in-process)

Results: eval/results/<date>_<experiment>.json (one file per experiment run).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx

from app.config import get_settings
from eval import metrics as m
from eval.dataset import DEFAULT_PATH, EvalQuestion, load_questions

log = logging.getLogger("eval")

RESULTS_DIR = Path(__file__).parent / "results"

EXPERIMENTS: dict[str, dict] = {
    "0_baseline": {"chunking": "fixed", "mode": "dense", "label": "Baseline: fixed 512-token chunks + dense only"},
    "1_heading": {"chunking": "heading", "mode": "dense", "label": "+ heading-aware chunking"},
    "2_hybrid": {"chunking": "heading", "mode": "hybrid", "sparse_weight": 1.0,
                 "label": "+ hybrid BM25 + dense (RRF, k=60)"},
    "2b_hybrid_weighted": {"chunking": "heading", "mode": "hybrid", "sparse_weight": 0.5,
                           "label": "+ weighted RRF (BM25 x 0.5)"},
    "3_rerank": {"chunking": "heading", "mode": "hybrid_rerank", "sparse_weight": 0.5,
                 "label": "+ cross-encoder reranker"},
    # Ablations - not in the main table, useful for understanding.
    "x_bm25": {"chunking": "heading", "mode": "bm25", "label": "BM25 only (heading chunks)"},
    "x_fixed_hybrid": {"chunking": "fixed", "mode": "hybrid", "sparse_weight": 1.0, "label": "fixed chunks + hybrid"},
    "x_fixed_rerank": {"chunking": "fixed", "mode": "hybrid_rerank", "sparse_weight": 1.0,
                       "label": "fixed chunks + hybrid + reranker"},
}
MAIN = ["0_baseline", "1_heading", "2_hybrid", "2b_hybrid_weighted", "3_rerank"]
KS = (1, 3, 5, 10)


# ------------------------------------------------------------------ clients


class ApiClient:
    def __init__(self, url: str) -> None:
        self.http = httpx.Client(base_url=url.rstrip("/"), timeout=900)

    def ask(self, body: dict) -> dict:
        r = self.http.post("/ask", json=body)
        r.raise_for_status()
        return r.json()

    def info(self) -> dict:
        r = self.http.get("/health")
        return r.json()


class InProcessClient:
    def __init__(self, with_llm: bool) -> None:
        from app.pipeline import RAGPipeline

        self.pipeline = RAGPipeline.from_settings(get_settings(), with_llm=with_llm)

    def ask(self, body: dict) -> dict:
        from app.schemas import AskRequest

        return self.pipeline.ask(AskRequest(**body)).model_dump(exclude_none=True)

    def info(self) -> dict:
        return self.pipeline.health()


# ------------------------------------------------------------------ running


def evaluate_question(client, q: EvalQuestion, exp: dict, top_k: int, generate: bool) -> dict:
    """Phase 1: ask the system and compute the deterministic metrics."""
    body = {"question": q.question, "top_k": top_k, "mode": exp["mode"], "chunking": exp["chunking"],
            "sparse_weight": exp.get("sparse_weight"), "generate": generate, "include_contexts": True}
    resp = None
    for attempt in range(3):
        t0 = time.perf_counter()
        try:
            resp = client.ask(body)
            break
        except Exception as e:  # keep going; a question that still fails is visible in the results
            log.error("%s failed (attempt %d/3): %s", q.id, attempt + 1, e)
            if attempt == 2:
                return {"id": q.id, "category": q.category, "error": str(e)}
            time.sleep(5 * (attempt + 1))
    wall_ms = (time.perf_counter() - t0) * 1000

    contexts = resp.get("contexts") or []
    ranked_docs = [c["doc_path"] for c in contexts]
    rank = m.first_relevant_rank(ranked_docs, q.relevant_docs) if q.answerable else None
    refused = resp.get("answerable") is False
    citations = resp.get("citations") or []

    row: dict = {
        "id": q.id,
        "category": q.category,
        "question": q.question,
        "answerable_truth": q.answerable,
        "relevant_docs": q.relevant_docs,
        "first_relevant_rank": rank,
        **{f"hit@{k}": m.hit_at_k(rank, k) for k in KS},
        "rr@10": m.reciprocal_rank(rank, 10),
        "doc_recall@5": m.doc_recall_at_k(ranked_docs, q.relevant_docs, 5) if q.answerable else None,
        "evidence@k": m.evidence_recall([c.get("text") or "" for c in contexts[:top_k]], q.evidence)
        if q.answerable else None,
        "top_rerank_score": resp.get("top_rerank_score"),
        "refused": refused if (generate or resp.get("refused_by") == "threshold") else None,
        "refused_by": resp.get("refused_by"),
        "answer": resp.get("answer"),
        "citations": [{"n": c["n"], "doc_path": c["doc_path"], "url": c["url"]} for c in citations],
        "citation_hit": any(c["doc_path"] in q.relevant_docs for c in citations) if citations and q.answerable else None,
        "retrieved": [
            {k: c.get(k) for k in ("rank", "doc_path", "heading", "score", "dense_rank", "sparse_rank", "rerank_score")}
            for c in contexts[:10]
        ],
        "timings_ms": {**resp.get("timings_ms", {}), "wall": round(wall_ms, 1)},
        "usage": resp.get("usage", {}),
        "context_tokens": sum(c.get("token_count") or 0 for c in contexts[:top_k]),
        # Kept only until the judge phase has run (not written to the results file).
        "_contexts": [c.get("text") or "" for c in contexts[:top_k]],
    }
    if row["refused"] is not None:
        row["refusal_outcome"] = m.refusal_outcome(q.answerable, refused)
    return row


def judge_row(row: dict, q: EvalQuestion, judge) -> dict:
    """Phase 2 (after all answers exist): LLM-judge metrics.

    Running it as a separate pass means a local setup with one small GPU loads
    the generator once and the judge once, instead of swapping per question.
    """
    if "error" in row:
        return row
    ctx = row["_contexts"]
    answer = row.get("answer") or ""
    jr: dict = {}
    if not row.get("refused") and answer:
        jr["faithfulness"], jr["faithfulness_detail"] = judge.faithfulness(q.question, answer, ctx)
    if q.answerable:
        jr["answer_relevancy"], jr["relevancy_detail"] = judge.answer_relevancy(q.question, answer)
        jr["context_precision"], jr["context_recall"], jr["context_detail"] = judge.context_scores(
            q.question, q.ground_truth, ctx)
    row["judge"] = jr
    return row


def summarize(rows: list[dict], top_k: int) -> dict:
    ok = [r for r in rows if "error" not in r]
    ans = [r for r in ok if r["answerable_truth"]]
    summary: dict = {"n_questions": len(rows), "n_errors": len(rows) - len(ok), "n_answerable": len(ans)}
    for k in KS:
        summary[f"hit@{k}"] = m.mean(r[f"hit@{k}"] for r in ans)
    summary["mrr@10"] = m.mean(r["rr@10"] for r in ans)
    summary["ci95"] = {  # bootstrap over questions - read differences against these intervals
        "hit@1": m.bootstrap_ci([r["hit@1"] for r in ans]),
        "hit@5": m.bootstrap_ci([r["hit@5"] for r in ans]),
        "mrr@10": m.bootstrap_ci([r["rr@10"] for r in ans]),
    }
    summary["doc_recall@5"] = m.mean(r["doc_recall@5"] for r in ans)
    summary["evidence@k"] = m.mean(r.get("evidence@k") for r in ans)
    summary["ci95"]["evidence@k"] = m.bootstrap_ci([r["evidence@k"] for r in ans if r.get("evidence@k") is not None])
    summary["citation_hit"] = m.mean(float(r["citation_hit"]) for r in ans if r.get("citation_hit") is not None)

    outcomes = Counter(r["refusal_outcome"] for r in ok if "refusal_outcome" in r)
    if outcomes:
        unans = outcomes["correct_refusal"] + outcomes["hallucinated_answer"]
        answ = outcomes["answered"] + outcomes["false_refusal"]
        summary["refusal"] = {
            **dict(outcomes),
            "correct_refusal_rate": round(outcomes["correct_refusal"] / unans, 4) if unans else None,
            "false_refusal_rate": round(outcomes["false_refusal"] / answ, 4) if answ else None,
            "accuracy": round((outcomes["correct_refusal"] + outcomes["answered"]) / sum(outcomes.values()), 4),
        }

    judged = [r["judge"] for r in ok if r.get("judge")]
    if judged:
        for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            summary[metric] = m.mean(j.get(metric) for j in judged)
            summary["ci95"][metric] = m.bootstrap_ci([j[metric] for j in judged if j.get(metric) is not None])

    timing_keys = ("retrieve", "rerank", "generate", "total", "wall")
    summary["latency_ms"] = {
        key: {"p50": m.percentile([r["timings_ms"].get(key, 0) for r in ok], 0.5),
              "p95": m.percentile([r["timings_ms"].get(key, 0) for r in ok], 0.95)}
        for key in timing_keys
    }
    summary["context_tokens_mean"] = m.mean(r.get("context_tokens") for r in ok)
    summary["tokens_mean"] = {
        "input": m.mean(r["usage"].get("input_tokens", 0) for r in ok),
        "output": m.mean(r["usage"].get("output_tokens", 0) for r in ok),
    }
    by_cat: dict[str, list] = defaultdict(list)
    for r in ans:
        by_cat[r["category"]].append(r["hit@5"])
    summary["hit@5_by_category"] = {c: m.mean(v) for c, v in sorted(by_cat.items())}
    summary["misses@10"] = [r["id"] for r in ans if r["first_relevant_rank"] is None]
    return summary


def run_experiment(name: str, client, questions: list[EvalQuestion], top_k: int, generate: bool, judge,
                   workers: int, info: dict, dataset_path: Path, tag: str | None = None) -> Path:
    exp = EXPERIMENTS[name]
    log.info("== %s: %s (%d questions, generate=%s, judge=%s)", name, exp["label"], len(questions), generate,
             judge.model_id if judge else None)
    t0 = time.perf_counter()
    done = [0]

    def ask_one(q: EvalQuestion) -> dict:
        row = evaluate_question(client, q, exp, top_k, generate)
        done[0] += 1
        if done[0] % 10 == 0 or done[0] == len(questions):
            log.info("   %d/%d questions (%.0fs)", done[0], len(questions), time.perf_counter() - t0)
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(ask_one, questions))
    usage_before = dict(judge.usage) if judge else {}
    if judge is not None and generate:
        log.info("   judging %d answers with %s", len(rows), judge.model_id)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(lambda rq: judge_row(rq[0], rq[1], judge), zip(rows, questions, strict=True)))
    for r in rows:
        r.pop("_contexts", None)
    summary = summarize(rows, top_k)
    settings = get_settings()
    coll = (info.get("collections") or {}).get(exp["chunking"], {})
    result = {
        "experiment": name,
        "tag": tag,
        "label": exp["label"],
        "config": {**exp, "top_k": top_k, "generate": generate},
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "duration_s": round(time.perf_counter() - t0, 1),
        "docs_commit_sha": coll.get("commit_sha"),
        "embedding_model": coll.get("embedding_model"),
        "llm_model": info.get("llm", {}).get("model") if generate else None,
        "judge_model": judge.model_id if judge else None,
        # Per-experiment share of the judge usage (the Judge object is shared across experiments).
        "judge_usage": {k: v - usage_before.get(k, 0) for k, v in judge.usage.items()} if judge else None,
        "dataset": {"path": f"eval/{dataset_path.name}", "n": len(questions),
                    "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest()[:16]},
        "settings": settings.fingerprint(),
        "summary": summary,
        "questions": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"__{tag}" if tag else ""
    out = RESULTS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}_{name}{suffix}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    s = summary
    log.info("   hit@5=%s mrr@10=%s faithfulness=%s refusal=%s -> %s", s["hit@5"], s["mrr@10"],
             s.get("faithfulness"), (s.get("refusal") or {}).get("accuracy"), out)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", default="main", help="comma list, 'main' (0-3) or 'all'")
    parser.add_argument("--api-url", help="evaluate through the HTTP API instead of in-process")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--top-k", type=int, default=get_settings().default_top_k)
    parser.add_argument("--no-generate", action="store_true", help="retrieval metrics only (no LLM calls)")
    parser.add_argument("--judge", action="store_true", help="compute LLM-judge metrics (needs generation)")
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    parser.add_argument("--ids", help="comma list of question ids")
    parser.add_argument("--workers", type=int, default=1, help="parallel questions (API/LLM rate limits!)")
    parser.add_argument("--tag", help="suffix for the result file, e.g. the model being compared")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "rag", "app.embeddings"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    names = MAIN if args.experiments == "main" else list(EXPERIMENTS) if args.experiments == "all" \
        else [x.strip() for x in args.experiments.split(",")]
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        parser.error(f"unknown experiments {unknown}; choose from {list(EXPERIMENTS)}")

    questions = load_questions(args.dataset)
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q.id in wanted]
    if args.limit:
        questions = questions[: args.limit]

    generate = not args.no_generate
    client = ApiClient(args.api_url) if args.api_url else InProcessClient(with_llm=generate)
    info = client.info()

    judge = None
    if args.judge:
        if not generate:
            parser.error("--judge needs generated answers; drop --no-generate")
        from app.embeddings import make_dense
        from app.llm import make_llm
        from eval.judge import Judge

        s = get_settings()
        judge_llm = make_llm(s, provider=s.resolved_judge_provider, model=s.resolved_judge_model)
        if judge_llm is None:
            parser.error("the 'extractive' provider cannot judge - set JUDGE_PROVIDER/JUDGE_MODEL")
        judge = Judge(judge_llm, make_dense(s))

    for name in names:
        print(run_experiment(name, client, questions, args.top_k, generate, judge, args.workers, info, args.dataset,
                             tag=args.tag))


if __name__ == "__main__":
    main()

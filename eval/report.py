"""Turn eval/results/*.json into the results tables of docs/DESIGN.md.

Usage:
    python -m eval.report                 # print the table, write eval/results/summary.md
    python -m eval.report --update-docs   # also replace the tables between the markers in docs/DESIGN.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.run_eval import COMPARE, EXPERIMENTS, MAIN, RESULTS_DIR

DOC = Path(__file__).resolve().parent.parent / "docs" / "DESIGN.md"
START, END = "<!-- EVAL_TABLE_START -->", "<!-- EVAL_TABLE_END -->"
AGENT_START, AGENT_END = "<!-- AGENT_TABLE_START -->", "<!-- AGENT_TABLE_END -->"
MAIN_DATASET, MULTIHOP_DATASET = "questions.jsonl", "questions_multihop.jsonl"


def _is_main_run(data: dict) -> bool:
    return data.get("tag") != "compare" and EXPERIMENTS[data["experiment"]].get("kind") != "agent"


def _is_compare_run(data: dict) -> bool:
    # /ask rows only from comparison runs (same LLM as the agent), never from the main-table runs.
    return data["experiment"] in COMPARE and (data.get("tag") is not None or data["experiment"] == "agent")


def latest_results(prefer_judged: bool = True, dataset: str = MAIN_DATASET, keep=_is_main_run) -> dict[str, dict]:
    """Most recent result per experiment on one dataset (a judged run beats a newer retrieval-only run)."""
    best: dict[str, dict] = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        name = data.get("experiment")
        if name not in EXPERIMENTS:
            continue
        if Path((data.get("dataset") or {}).get("path", "")).name != dataset or not keep(data):
            continue
        cur = best.get(name)
        judged = data.get("judge_model") is not None
        if cur is None or (judged or not (prefer_judged and cur.get("judge_model"))):
            data["_file"] = path.name
            best[name] = data
    return best


def _f(v, pct: bool = False) -> str:
    if v is None:
        return "–"
    return f"{v * 100:.0f}%" if pct else f"{v:.2f}"


def _ci(s: dict, key: str) -> str:
    ci = (s.get("ci95") or {}).get(key)
    return f" <sub>[{ci[0]:.2f}-{ci[1]:.2f}]</sub>" if ci else ""


def main_table(results: dict[str, dict]) -> str:
    head = ("| # | Configuration | Hit@1 | Hit@5 | MRR@10 | Evidence@6 | Faithfulness | Answer rel. | "
            "Ctx precision | Ctx recall | Refusal acc. | p50 latency |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for name in MAIN:
        i = name.split("_")[0]
        r = results.get(name)
        if r is None:
            lines.append(f"| {i} | {EXPERIMENTS[name]['label']} | " + " | ".join(["–"] * 10) + " |")
            continue
        s = r["summary"]
        refusal = (s.get("refusal") or {}).get("accuracy")
        lat = (s.get("latency_ms") or {}).get("total", {}).get("p50")
        lines.append(
            f"| {i} | {r['label']} | {_f(s.get('hit@1'))} | {_f(s.get('hit@5'))} | "
            f"{_f(s.get('mrr@10'))}{_ci(s, 'mrr@10')} | {_f(s.get('evidence@k'))} | "
            f"{_f(s.get('faithfulness'))} | {_f(s.get('answer_relevancy'))} | {_f(s.get('context_precision'))} | "
            f"{_f(s.get('context_recall'))} | {_f(refusal, pct=True)} | "
            f"{'–' if lat is None else f'{lat / 1000:.1f} s'} |"
        )
    return "\n".join(lines)


def ablation_table(results: dict[str, dict]) -> str:
    lines = ["| Configuration | Hit@1 | Hit@5 | Hit@10 | MRR@10 | p50 retrieval |", "|---|---|---|---|---|---|"]
    for name, exp in EXPERIMENTS.items():
        r = results.get(name)
        if r is None:
            continue
        s = r["summary"]
        lat = s["latency_ms"]
        retr = (lat["retrieve"]["p50"] or 0) + (lat["rerank"]["p50"] or 0)
        lines.append(f"| {exp['label']} | {_f(s.get('hit@1'))} | {_f(s.get('hit@5'))} | {_f(s.get('hit@10'))} | "
                     f"{_f(s.get('mrr@10'))} | {retr:.0f} ms |")
    return "\n".join(lines)


def provenance(results: dict[str, dict]) -> str:
    results = {k: v for k, v in results.items() if k in MAIN} or results  # describe the main table
    any_r = next(iter(results.values()), None)
    if any_r is None:
        return ""
    judges = sorted({r.get("judge_model") or "none" for r in results.values()})
    llms = sorted({r.get("llm_model") or "none (retrieval only)" for r in results.values()})
    sha = any_r.get("docs_commit_sha") or "?"
    ds = any_r["dataset"]
    return (f"Dataset: `eval/{Path(ds['path']).name}` ({ds['n']} questions, sha256 `{ds['sha256']}`) · docs commit `{sha[:12]}` · "
            f"embedding `{any_r.get('embedding_model')}` · reranker `{any_r['settings'].get('rerank_model')}` · "
            f"generator {', '.join(f'`{x}`' for x in llms)} · judge {', '.join(f'`{x}`' for x in judges)}")


# USD per million tokens (input, output) - Anthropic list prices as of 2026-10.
CLAUDE_PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00), "claude-opus-5-5": (4.00, 20.00)}


def cost_section(results: dict[str, dict], name: str = "2b_hybrid_weighted") -> str:
    r = results.get(name)
    if r is None or not r["config"].get("generate"):
        return "_needs a run with generation_"
    s = r["summary"]
    lat = s["latency_ms"]
    rows = ["| Stage | p50 | p95 |", "|---|---|---|"]
    for key, label in (("retrieve", "retrieval (embed query + dense/BM25 + fusion)"),
                       ("rerank", "cross-encoder confidence (6 chunks)"),
                       ("generate", f"generation ({r.get('llm_model')}, local GPU)"), ("total", "total")):
        rows.append(f"| {label} | {lat[key]['p50'] / 1000:.2f} s | {lat[key]['p95'] / 1000:.2f} s |")
    tin, tout = s["tokens_mean"]["input"] or 0, s["tokens_mean"]["output"] or 0
    lines = ["\n".join(rows), "",
             f"Mean tokens per question: **{tin:.0f} input / {tout:.0f} output** "
             f"(context: {s.get('context_tokens_mean') or 0:.0f} tokens in 6 chunks). "
             "Estimated API cost per question for the same prompt with Claude:", "",
             "| Model | Cost / question | Cost / 1,000 questions |", "|---|---|---|"]
    for model, (pin, pout) in CLAUDE_PRICES.items():
        c = tin / 1e6 * pin + tout / 1e6 * pout
        lines.append(f"| `{model}` | ${c:.4f} | ${c * 1000:.2f} |")
    lines += ["", "Token counts come from the local model's tokenizer; Claude's tokenizer and any adaptive-thinking "
              "tokens shift these numbers, so treat them as an order of magnitude. Embeddings, BM25 and the "
              "reranker run locally (free); embedding the whole corpus with `text-embedding-3-small` instead "
              "would cost about 1.1 M tokens = $0.02."]
    return "\n".join(lines)


def _ci_pct(s: dict, key: str) -> str:
    ci = (s.get("ci95") or {}).get(key)
    return f" <sub>[{ci[0] * 100:.0f}-{ci[1] * 100:.0f}]</sub>" if ci else ""


def cost_per_question(r: dict) -> float | None:
    """Mean API cost of the generating model per question (judge excluded), from measured tokens."""
    price = CLAUDE_PRICES.get(r.get("llm_model") or "")
    tokens = r["summary"].get("tokens_mean") or {}
    if price is None or tokens.get("input") is None:
        return None
    return (tokens["input"] or 0) / 1e6 * price[0] + (tokens["output"] or 0) / 1e6 * price[1]


def comparison_table(results: dict[str, dict]) -> str:
    """Single search (/ask) vs. the agent on the same questions with the same LLM."""
    head = ("| Mode | All parts found | Parts found | Key facts in context | Key facts in answer | Parts cited | "
            "Faithfulness | Refusal acc. | Tool calls | Tokens in / out | Cost / question | p50 latency |")
    lines = [head, "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in COMPARE:
        r = results.get(name)
        if r is None:
            continue
        s = r["summary"]
        mode = "Agent (`/agent`)" if name == "agent" else "Single search (`/ask`)"
        tools = (s.get("agent") or {}).get("tool_calls_mean")
        tok = s.get("tokens_mean") or {}
        cost = cost_per_question(r)
        lat = (s.get("latency_ms") or {}).get("total", {}).get("p50")
        lines.append(
            f"| {mode} | {_f(s.get('all_hops'), pct=True)}{_ci_pct(s, 'all_hops')} | {_f(s.get('hop_coverage'), pct=True)} | "
            f"{_f(s.get('evidence@k'), pct=True)} | {_f(s.get('answer_evidence'), pct=True)}{_ci_pct(s, 'answer_evidence')} | "
            f"{_f(s.get('cited_hop_coverage'), pct=True)} | {_f(s.get('faithfulness'))} | "
            f"{_f((s.get('refusal') or {}).get('accuracy'), pct=True)} | {'1 search' if tools is None else f'{tools:.1f}'} | "
            f"{tok.get('input') or 0:,.0f} / {tok.get('output') or 0:,.0f} | {'–' if cost is None else f'${cost:.4f}'} | "
            f"{'–' if lat is None else f'{lat / 1000:.1f} s'} |"
        )
    any_r = next(iter(results.values()))
    ds = any_r["dataset"]
    models = sorted({r.get("llm_model") or "?" for r in results.values()})
    judges = sorted({r.get("judge_model") or "none" for r in results.values()})
    lines += ["", f"Dataset: `eval/{Path(ds['path']).name}` ({ds['n']} questions, sha256 `{ds['sha256']}`) · "
                  f"LLM {', '.join(f'`{m}`' for m in models)} · judge {', '.join(f'`{j}`' for j in judges)}"]
    return "\n".join(lines)


def comparison_section() -> str:
    parts = []
    for dataset, title in ((MULTIHOP_DATASET, "**Multi-hop questions** (each needs 2 different doc pages):"),
                           (MAIN_DATASET, "**Original single-topic questions:**")):
        results = latest_results(dataset=dataset, keep=_is_compare_run)
        if results:
            parts += [title, comparison_table(results)]
    return "\n\n".join(parts) or "_no comparison runs yet - run `python -m eval.run_eval --experiments compare`_"


def _replace_block(text: str, start: str, end: str, body: str) -> str:
    if start not in text or end not in text:
        return text
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    return f"{before}{start}\n{body}\n{end}{after}"


def render(results: dict[str, dict]) -> str:
    return "\n\n".join([main_table(results), provenance(results), "**Ablations (retrieval only):**",
                        ablation_table(results)])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update-docs", action="store_true")
    args = parser.parse_args()
    results = latest_results()
    if not results:
        sys.exit("no results in eval/results - run `python -m eval.run_eval` first")
    table = render(results)
    compare = comparison_section()
    (RESULTS_DIR / "summary.md").write_text(f"{table}\n\n## Agent vs. single search\n\n{compare}\n", encoding="utf-8")
    print(f"{table}\n\nAgent vs. single search:\n\n{compare}")
    if args.update_docs:
        text = DOC.read_text(encoding="utf-8")
        if START not in text or END not in text:
            sys.exit(f"{DOC} is missing the {START} / {END} markers")
        text = _replace_block(text, START, END, table)
        text = _replace_block(text, "<!-- COST_START -->", "<!-- COST_END -->", cost_section(results))
        text = _replace_block(text, AGENT_START, AGENT_END, compare)
        DOC.write_text(text, encoding="utf-8")
        print(f"\nupdated {DOC.name}")


if __name__ == "__main__":
    main()

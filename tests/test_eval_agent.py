"""Eval pieces for the agent comparison: multi-hop dataset, hop metrics, agent rows, report table."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval import metrics as m
from eval.dataset import load_questions
from eval.report import comparison_table, cost_per_question
from eval.run_eval import COMPARE, evaluate_question, hops_of, summarize

MULTIHOP = Path(__file__).resolve().parent.parent / "eval" / "questions_multihop.jsonl"


# ---------------------------------------------------------------- dataset


def test_multihop_dataset_is_valid():
    qs = load_questions(MULTIHOP)
    answerable = [q for q in qs if q.answerable]
    assert len(answerable) >= 12 and sum(not q.answerable for q in qs) >= 2
    for q in answerable:
        hops = q.extra["hops"]
        assert len(hops) >= 2, q.id  # multi-hop: at least two different parts
        assert all(hop for hop in hops), q.id
        assert {d for hop in hops for d in hop} == set(q.relevant_docs), q.id
        assert q.evidence, q.id
    for q in qs:
        if not q.answerable:
            assert q.relevant_docs == [] and not q.extra.get("hops")


def test_single_hop_questions_count_any_relevant_doc_as_one_hop():
    q = next(q for q in load_questions() if len(q.relevant_docs) > 1)
    assert hops_of(q) == [q.relevant_docs]


# ---------------------------------------------------------------- metrics


def test_hop_coverage():
    hops = [["a.md", "b.md"], ["c.md"]]
    assert m.hop_coverage(["b.md", "x.md"], hops) == 0.5  # first hop via its alternative
    assert m.hop_coverage(["a.md", "c.md"], hops) == 1.0
    assert m.hop_coverage([], hops) == 0.0
    assert m.hop_coverage(["a.md"], []) is None


# ---------------------------------------------------------- agent rows


class FakeAgentClient:
    def __init__(self, response: dict):
        self.response = response
        self.bodies: list[dict] = []

    def agent(self, body):
        self.bodies.append(body)
        return self.response


def _question(qid="m01"):
    return next(q for q in load_questions(MULTIHOP) if q.id == qid)


def _agent_response(q, docs, answer, cited, answerable=True):
    return {
        "answer": answer, "answerable": answerable,
        "citations": [{"n": i, "doc_path": d, "url": f"https://x/{d}"} for i, d in enumerate(cited, 1)],
        "tool_calls": 3, "model_calls": 3, "stopped_by": "answer",
        "timings_ms": {"model": 4000.0, "tools": 300.0, "total": 4400.0},
        "usage": {"input_tokens": 9000, "output_tokens": 600},
        "steps": [{"step": 1, "tool": "search_docs", "args": {"query": "x"}, "sources": [1, 2], "ms": 100.0}],
        "sources": [{"n": i, "chunk_id": f"c{i}", "doc_path": d, "title": "T", "url": "u", "heading": None,
                     "read": i == 1, "text": f"text of {d} custom-fields.xml"} for i, d in enumerate(docs, 1)],
    }


def test_agent_row_has_hop_and_answer_metrics():
    q = _question("m01")
    first, second = q.extra["hops"][0][0], q.extra["hops"][1][0]
    client = FakeAgentClient(_agent_response(q, [first, "other.md", second],
                                             "Declare it in custom-fields.xml and print "
                                             "page.product.translated.customFields.x via sw_extends [1][2].",
                                             [first]))
    row = evaluate_question(client, q, {"kind": "agent"}, top_k=6, generate=True)

    assert client.bodies == [{"question": q.question, "include_trace": True, "include_sources": True}]
    assert row["hop_coverage"] == 1.0 and row["all_hops"] == 1.0
    assert row["answer_evidence"] == 1.0
    assert row["evidence@k"] == pytest.approx(1 / 3)  # the sources only contain custom-fields.xml
    assert row["cited_hop_coverage"] == 0.5  # only the first part is backed by a citation
    assert row["first_relevant_rank"] == 1 and row["tool_calls"] == 3
    assert row["timings_ms"]["retrieve"] == 300.0 and row["timings_ms"]["generate"] == 4000.0
    assert row["refusal_outcome"] == "answered"
    assert row["sources_seen"][0]["read"] is True


def test_agent_row_missing_a_hop():
    q = _question("m05")
    client = FakeAgentClient(_agent_response(q, [q.extra["hops"][0][0]], "Implement AsyncMessageInterface [1].",
                                             [q.extra["hops"][0][0]]))
    row = evaluate_question(client, q, {"kind": "agent"}, top_k=6, generate=True)
    assert row["hop_coverage"] == 0.5 and row["all_hops"] == 0.0
    assert row["answer_evidence"] == pytest.approx(1 / 3)


def test_agent_refusal_on_unanswerable():
    q = next(q for q in load_questions(MULTIHOP) if not q.answerable)
    client = FakeAgentClient(_agent_response(q, [], "I could not find this in the Shopware documentation.", [],
                                             answerable=False))
    row = evaluate_question(client, q, {"kind": "agent"}, top_k=6, generate=True)
    assert row["refusal_outcome"] == "correct_refusal" and row["all_hops"] is None


def test_summary_and_comparison_table():
    q = _question("m01")
    hop1, hop2 = q.extra["hops"][0][0], q.extra["hops"][1][0]
    full = evaluate_question(FakeAgentClient(_agent_response(q, [hop1, hop2], "custom-fields.xml [1]", [hop1, hop2])),
                             q, {"kind": "agent"}, 6, True)
    half = evaluate_question(FakeAgentClient(_agent_response(q, [hop1], "custom-fields.xml [1]", [hop1])),
                             q, {"kind": "agent"}, 6, True)
    rows = [{**full}, {**half, "id": "m01b"}]
    for r in rows:
        r.pop("_contexts")
    s = summarize(rows, top_k=6)
    assert s["all_hops"] == 0.5 and s["hop_coverage"] == 0.75
    assert s["agent"]["tool_calls_mean"] == 3 and s["incomplete_hops"] == ["m01b"]

    dataset = {"path": "eval/questions_multihop.jsonl", "n": 18, "sha256": "abc"}
    ask_summary = {**s, "agent": None, "all_hops": 0.25}
    results = {
        "agent": {"summary": s, "llm_model": "claude-haiku-5-5", "dataset": dataset, "judge_model": None},
        "2b_hybrid_weighted": {"summary": ask_summary, "llm_model": "claude-haiku-5-5", "dataset": dataset,
                               "judge_model": None},
    }
    table = comparison_table(results)
    assert table.index("Single search") < table.index("Agent (`/agent`)")  # COMPARE order
    assert "| 50%" in table and "| 25%" in table and "1 search" in table and "3.0" in table
    assert cost_per_question(results["agent"]) == pytest.approx(9000 / 1e6 * 0.10 + 600 / 1e6 * 0.50)
    assert COMPARE == ["2b_hybrid_weighted", "agent"]

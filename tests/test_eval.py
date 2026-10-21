from eval import metrics as m
from eval.calibrate import _accuracy, best_threshold
from eval.dataset import load_questions


def test_retrieval_metrics():
    ranked = ["a.md", "b.md", "a.md", "c.md"]
    assert m.first_relevant_rank(ranked, ["c.md"]) == 4
    assert m.first_relevant_rank(ranked, ["z.md"]) is None
    assert m.hit_at_k(4, 5) == 1.0 and m.hit_at_k(4, 3) == 0.0 and m.hit_at_k(None, 5) == 0.0
    assert m.reciprocal_rank(2) == 0.5 and m.reciprocal_rank(11, k=10) == 0.0
    assert m.doc_recall_at_k(ranked, ["a.md", "c.md"], 2) == 0.5


def test_average_precision_rewards_good_ordering():
    assert m.average_precision([True, True, False]) == 1.0
    assert m.average_precision([False, True]) == 0.5
    assert m.average_precision([False, False]) == 0.0


def test_refusal_outcomes():
    assert m.refusal_outcome(False, True) == "correct_refusal"
    assert m.refusal_outcome(False, False) == "hallucinated_answer"
    assert m.refusal_outcome(True, True) == "false_refusal"
    assert m.refusal_outcome(True, False) == "answered"


def test_threshold_calibration():
    rows = [(5.0, True), (4.0, True), (1.0, False), (0.5, False), (3.0, True)]
    t = best_threshold(rows)
    assert 1.0 < t < 3.0
    assert _accuracy(rows, t)[0] == 1.0


def test_dataset_is_valid_and_balanced():
    qs = load_questions()
    assert 30 <= len(qs) <= 50
    cats = {q.category for q in qs}
    assert {"plugin-dev", "dal", "storefront", "api", "cli-config", "unanswerable"} <= cats
    assert sum(not q.answerable for q in qs) >= 4
    assert all(q.relevant_docs for q in qs if q.answerable)


def test_evidence_recall():
    assert m.evidence_recall(["Tag with kernel.event_subscriber"], ["KERNEL.event_subscriber", "missing"]) == 0.5
    assert m.evidence_recall(["x"], []) is None


def test_every_answerable_question_has_evidence():
    qs = [q for q in load_questions() if q.answerable]
    assert all(q.evidence for q in qs)

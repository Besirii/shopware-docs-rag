from app.generation import build_user_message, extractive_answer, snippet, validate_citations
from app.retrieval import Candidate, Retriever, rrf_fuse
from tests.conftest import FakeReranker


def test_rrf_formula_and_determinism():
    fused = dict(rrf_fuse([["a", "b", "c"], ["c", "a"]], k=60))
    assert fused["a"] == 1 / 61 + 1 / 62
    assert fused["c"] == 1 / 63 + 1 / 61
    assert fused["b"] == 1 / 62
    order = [pid for pid, _ in rrf_fuse([["a", "b", "c"], ["c", "a"]], k=60)]
    assert order == ["a", "c", "b"]
    # Ties are broken by id, so repeated runs give identical rankings.
    assert [p for p, _ in rrf_fuse([["x"], ["y"]])] == ["x", "y"]


def test_weighted_rrf():
    order = [p for p, _ in rrf_fuse([["a"], ["b"]], k=60, weights=[1.0, 2.0])]
    assert order == ["b", "a"]


def test_citation_validation_strips_invented_numbers():
    answer = "Tag it with kernel.event_subscriber [1]. Compile the theme [9]. See [1, 3] and [link](http://x)."
    clean, cites = validate_citations(answer, [3, 7, 2], n_contexts=3)
    assert "[9]" not in clean and "[1][3]" in clean
    assert "[link](http://x)" in clean  # markdown links are not citations
    assert cites == [1, 3, 2]  # inline order first, then valid extras from the list; 7 dropped


def test_snippet_is_plain_and_short():
    s = snippet("## Title\n\nSome **bold** text with [a link](x.md).\n\n```php\ncode();\n```\n" + "more " * 100, 80)
    assert "**" not in s and "[a link]" not in s and len(s) <= 80


def test_user_message_numbers_excerpts():
    cands = [Candidate("1", {"title": "A", "breadcrumb": ["G", "A"], "url": "u1", "text": "alpha"}, 1.0),
             Candidate("2", {"title": "B", "breadcrumb": ["G", "B"], "url": "u2", "text": "beta"}, 0.5)]
    msg = build_user_message("What?", cands)
    assert '<excerpt n="1" title="A" section="G > A" url="u1">' in msg and '<excerpt n="2"' in msg
    assert msg.endswith("Question: What?")
    gen = extractive_answer(cands)
    assert gen.answer == "alpha [1]" and gen.citation_numbers == [1]


def test_retriever_modes(settings, indexed):
    client, dense, sparse = indexed
    r = Retriever(settings, client, dense, sparse, FakeReranker())
    q = "How do I tag a subscriber with kernel.event_subscriber?"
    for mode in ("dense", "bm25", "hybrid", "hybrid_rerank"):
        res = r.retrieve(q, mode, "heading")
        assert res.candidates, mode
        assert res.candidates[0].doc_path == "guides/events.md", mode
    hybrid = r.retrieve(q, "hybrid", "heading").candidates
    assert all(c.dense_rank or c.sparse_rank for c in hybrid)
    reranked = r.retrieve(q, "hybrid_rerank", "heading")
    scores = [c.rerank_score for c in reranked.candidates]
    assert scores == sorted(scores, reverse=True)
    assert reranked.top_rerank_score == scores[0]
    # The baseline collection is queried when chunking="fixed".
    assert r.retrieve(q, "dense", "fixed").candidates[0].payload["strategy"] == "fixed"


def test_python_and_qdrant_fusion_agree_on_top_hit(settings, indexed):
    client, dense, sparse = indexed
    q = "theme:compile after changing SCSS"
    py = Retriever(settings, client, dense, sparse).retrieve(q, "hybrid").candidates
    settings.fusion_backend = "qdrant"
    native = Retriever(settings, client, dense, sparse).retrieve(q, "hybrid").candidates
    assert py[0].doc_path == native[0].doc_path == "guides/theme.md"


def test_repair_markdown_fixes_small_model_artefacts():
    from app.generation import repair_markdown

    broken = r"Register it:\n```php\n$services->set(A::class);\n``"  # literal backslash-n, as the model emits
    fixed = repair_markdown(broken)
    assert "\\n" not in fixed and "\n" in fixed
    assert fixed.count("```") == 2 and fixed.endswith("```")
    assert repair_markdown("Fine\n```\ncode\n```") == "Fine\n```\ncode\n```"
    # "Swag\basicExample" in JSON decodes \b to a backspace character.
    assert repair_markdown("use Swag\x08asicExample;") == "use Swag\\basicExample;"


def test_extractive_answer_skips_headings():
    c = Candidate("1", {"text": "## Prerequisites\n\nYou need a plugin first.\n\n## Next\n\nMore text here."}, 1.0)
    gen = extractive_answer([c])
    assert gen.answer.startswith("You need a plugin first.") and "## Prerequisites" not in gen.answer
    assert gen.answer.endswith("[1]")

import itertools

import pytest

from ingest.chunking import FixedChunker, HeadingChunker
from ingest.markdown import code_line_ranges
from tests.conftest import make_doc

LONG_CODE = "```php\n" + "\n".join(f"$line{i} = 'value {i} with some padding words';" for i in range(120)) + "\n```"


def _doc(text: str, title: str = "Plugin Base Guide"):
    return make_doc("guides/plugins/plugin-base-guide.md", text, title=title, parents=["Guides", "Plugins"])


def test_breadcrumb_prefix_and_section_anchor():
    text = ("# Plugin Base Guide\n\n" + "Intro paragraph that is long enough to be useful on its own. " * 3
            + "\n\n## Install\n\n" + "Run the install command and activate the plugin afterwards. " * 30)
    chunks = HeadingChunker(max_tokens=300, min_tokens=20, overlap_tokens=0).chunk(_doc(text))
    install = next(c for c in chunks if c.headings == ["Install"])
    assert install.breadcrumb == ["Guides", "Plugins", "Plugin Base Guide", "Install"]
    assert install.embed_text.startswith("Plugin Base Guide\nGuides > Plugins > Plugin Base Guide > Install\n\n")
    assert install.url.endswith("#install")
    intro = chunks[0]
    assert "# Plugin Base Guide" not in intro.text  # H1 == title is carried by the prefix instead
    assert "#" not in intro.url  # intro links to the page top


def test_code_block_is_never_split():
    text = "# T\n\n## Example\n\nRegister it like this:\n\n" + LONG_CODE + "\n\nAfterwards it works.\n"
    chunker = HeadingChunker(max_tokens=200, min_tokens=20, overlap_tokens=40)
    chunks = chunker.chunk(_doc(text, title="T"))
    code = text[text.index("```php"): text.index("```\n\nAfter") + 3]
    holders = [c for c in chunks if code in c.text]
    assert len(holders) == 1  # whole block, exactly once
    # The lead-in line ending with ":" is glued to the code it introduces.
    assert "Register it like this:" in holders[0].text
    for c in chunks:  # and no chunk ever contains half a fence
        assert c.text.count("```") % 2 == 0


def test_small_sections_are_merged_and_long_ones_split_with_overlap():
    text = ("# T\n\n## Tiny\n\nOne line.\n\n## Big\n\n"
            + "\n\n".join(f"Paragraph {i}. " + "Shopware explains a concept here. " * 12 for i in range(12)))
    chunker = HeadingChunker(max_tokens=200, min_tokens=60, overlap_tokens=60)
    chunks = chunker.chunk(_doc(text, title="T"))
    assert not any(c.text.strip() == "## Tiny\n\nOne line." for c in chunks)  # merged, not alone
    assert all(c.token_count <= 200 * 1.25 for c in chunks)
    big = [c for c in chunks if "Paragraph" in c.text]
    assert len(big) > 2
    # Overlap: each continuation starts with the tail of the previous chunk.
    for prev, nxt in itertools.pairwise(big):
        last_para = prev.text.split("\n\n")[-1]
        assert nxt.text.startswith(last_para) or last_para.split(". ")[-1] in nxt.text


def test_chunk_ids_are_stable_and_content_addressed():
    text = "# T\n\n## A\n\n" + "Some content about plugins and services. " * 20
    a = HeadingChunker().chunk(_doc(text, title="T"))
    b = HeadingChunker().chunk(_doc(text, title="T"))
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    changed = HeadingChunker().chunk(_doc(text + "\nNew sentence.", title="T"))
    assert changed[-1].chunk_id != a[-1].chunk_id
    assert len({c.point_id for c in a}) == len(a)


def test_table_split_repeats_header():
    rows = "\n".join(f"| key{i} | some value describing option {i} in detail |" for i in range(150))
    text = f"# T\n\n## Options\n\n| Key | Description |\n| --- | --- |\n{rows}\n"
    chunks = HeadingChunker(max_tokens=250, min_tokens=20, overlap_tokens=0).chunk(_doc(text, title="T"))
    table_chunks = [c for c in chunks if "| key" in c.text]
    assert len(table_chunks) > 1
    assert all("| Key | Description |" in c.text for c in table_chunks)


def test_fixed_baseline_ignores_structure():
    text = "# T\n\n" + "word " * 1200
    chunks = FixedChunker(chunk_tokens=512, overlap_tokens=64).chunk(_doc(text, title="T"))
    assert len(chunks) == 3
    assert all(c.token_count <= 512 for c in chunks)
    assert chunks[0].embed_text == chunks[0].text  # no title/breadcrumb prefix


def test_fixed_baseline_splits_code_but_heading_chunker_does_not():
    text = "# T\n\nIntro.\n\n" + LONG_CODE + "\n"
    fixed = FixedChunker(chunk_tokens=200, overlap_tokens=20).chunk(_doc(text, title="T"))
    assert any(c.text.count("```") % 2 == 1 for c in fixed)  # the baseline cuts the fence
    heading = HeadingChunker(max_tokens=200, min_tokens=20).chunk(_doc(text, title="T"))
    assert all(c.text.count("```") % 2 == 0 for c in heading)


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        HeadingChunker(max_tokens=100, overlap_tokens=100)
    with pytest.raises(ValueError):
        FixedChunker(chunk_tokens=100, overlap_tokens=150)


def test_code_ranges_include_fences_in_lists():
    text = "1. Step one:\n\n   ```bash\n   bin/console x\n   ```\n2. Step two\n"
    assert code_line_ranges(text) == [(2, 5)]

from ingest import markdown as md


def test_front_matter_removed_and_title_kept():
    raw = "---\nnav:\n  title: Short\n  position: 1\n---\n\n# Long Title\n\nBody text.\n"
    meta, body = md.parse_front_matter(raw)
    assert meta["nav"]["title"] == "Short"
    assert body.startswith("# Long Title")
    assert md.doc_title(meta, body, "a/b.md") == "Long Title"  # H1 beats nav.title
    assert md.nav_title(meta, body, "a/b.md") == "Short"


def test_front_matter_title_wins():
    meta, body = md.parse_front_matter("---\ntitle: ADR title\n---\n# Other\n")
    assert md.doc_title(meta, body, "x.md") == "ADR title"


def test_title_falls_back_to_file_name():
    assert md.doc_title({}, "no heading", "guides/my-cool-page.md") == "My Cool Page"
    assert md.doc_title({}, "", "guides/caching/index.md") == "Caching"


def test_headings_inside_code_are_not_headings():
    text = "# Real\n\n```bash\n# not a heading\n```\n\n## Also real\n"
    assert [h.text for h in md.headings(text)] == ["Real", "Also real"]


def test_vitepress_cleanup_keeps_meaning_and_never_touches_code():
    body = (
        ":::info Heads up\nUse the CLI.\n:::\n\n"
        "<Tabs>\n<Tab title=\"PHP\">\n\n```php\n<!-- keep me -->\n:::info not a container\n```\n\n</Tab>\n</Tabs>\n\n"
        "<PageRef page=\"../x.md\" title=\"X guide\" />\n\n![screenshot](img.png)\n<!-- drop me -->\n"
    )
    out = md.clean_vitepress(body)
    assert "**Note:** Heads up" in out
    assert "**PHP:**" in out
    assert "See also: [X guide](../x.md)" in out
    assert "<!-- keep me -->" in out and ":::info not a container" in out  # code untouched
    assert "drop me" not in out and "img.png" not in out and "<Tabs>" not in out


def test_include_and_snippet_resolution():
    files = {
        "snippets/guide/part.md": "---\nx: 1\n---\nIncluded **text**.",
        "snippets/config/app.xml": "<manifest/>",
    }
    body = "Intro\n<!--@include: ../snippets/guide/part.md-->\n\n<<< @/docs/snippets/config/app.xml [Manifest]\n"
    out = md.resolve_snippet_imports(md.resolve_includes(body, "guides/page.md", files.get), files.get)
    assert "Included **text**." in out and "x: 1" not in out
    assert "```xml\n<manifest/>\n```" in out and "**Manifest**" in out


def test_navigation_page_detection():
    nav = "# Section\n\n* [A](a.md)\n* [B](b.md)\n* [C](c.md)\n"
    real = "# Guide\n\n" + "This explains how plugins register services in the container. " * 5
    assert md.is_navigation_page(nav, 30)
    assert not md.is_navigation_page(real, 30)
    assert not md.is_navigation_page(nav + "\n```bash\nbin/console x\n```\n", 30)  # code = content


def test_url_mapping_and_slugify():
    base = "https://developer.shopware.com/docs"
    assert md.path_to_url("guides/plugins/plugins/plugin-base-guide.md", base) == \
        f"{base}/guides/plugins/plugins/plugin-base-guide.html"
    assert md.path_to_url("guides/plugins/index.md", base) == f"{base}/guides/plugins/"
    # Same algorithm as VitePress (verified against the live site's anchors).
    assert md.slugify("Decorating the service") == "decorating-the-service"
    assert md.slugify("Using the `EntityRepository`") == "using-the-entityrepository"
    assert md.slugify("2. Install & activate") == "_2-install-activate"

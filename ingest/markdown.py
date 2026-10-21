"""Markdown helpers specific to the Shopware docs (VitePress flavour).

Everything here is a pure function of strings, so it is unit-tested without git
or a network. Two rules drive the design:

1. Code is sacred. In Shopware docs the answer is often the code itself
   (services.xml, a subscriber class, a bin/console command), so no transform
   ever touches the inside of a fenced code block. Fences are located with a
   real CommonMark parser (markdown-it) instead of regexes, which also handles
   fences nested inside list items.
2. Don't lose content hidden behind site features. VitePress pulls text in via
   `<!--@include: file.md-->` and `<<< @/docs/snippets/file.xml`. Left as-is,
   that content would simply not exist for retrieval.
"""

from __future__ import annotations

import posixpath
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

import frontmatter
from markdown_it import MarkdownIt

_md = MarkdownIt("commonmark").enable("table")

ReadFn = Callable[[str], str | None]

# ----------------------------------------------------------------- structure


@dataclass(frozen=True)
class Heading:
    level: int
    text: str
    line: int  # 0-based line index in the parsed text


def parse_blocks(text: str):
    """markdown-it block tokens (a flat stream that includes nested blocks)."""
    return _md.parse(text)


def code_line_ranges(text: str) -> list[tuple[int, int]]:
    """[start, end) line ranges of fenced and indented code blocks."""
    ranges = [
        (t.map[0], t.map[1])
        for t in parse_blocks(text)
        if t.type in ("fence", "code_block") and t.map
    ]
    return sorted(ranges)


def headings(text: str) -> list[Heading]:
    tokens = parse_blocks(text)
    out: list[Heading] = []
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open" and tok.map:
            inline = tokens[i + 1]
            out.append(Heading(level=int(tok.tag[1]), text=plain_inline(inline.content), line=tok.map[0]))
    return out


def plain_inline(s: str) -> str:
    """Inline markdown -> plain text (for headings / titles)."""
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("`", "").replace("**", "").replace("__", "")
    s = re.sub(r"\{#[^}]*\}\s*$", "", s)  # explicit anchors: "## Title {#anchor}"
    return re.sub(r"\s+", " ", s).strip()


def split_code_segments(text: str) -> list[tuple[bool, str]]:
    """Split text into (is_code, chunk) segments along code-block line ranges."""
    lines = text.split("\n")
    segments: list[tuple[bool, str]] = []
    cursor = 0
    for start, end in code_line_ranges(text):
        if start < cursor:  # overlapping ranges should not happen; be defensive
            continue
        if start > cursor:
            segments.append((False, "\n".join(lines[cursor:start])))
        segments.append((True, "\n".join(lines[start:end])))
        cursor = end
    if cursor < len(lines):
        segments.append((False, "\n".join(lines[cursor:])))
    return segments


def map_prose(text: str, fn: Callable[[str], str]) -> str:
    """Apply `fn` to every non-code segment; code passes through untouched."""
    return "\n".join(seg if is_code else fn(seg) for is_code, seg in split_code_segments(text))


# -------------------------------------------------------------- front matter


def parse_front_matter(raw: str) -> tuple[dict, str]:
    raw = raw.replace("\r\n", "\n").lstrip("﻿")
    try:
        post = frontmatter.loads(raw)
        return dict(post.metadata), post.content
    except Exception:  # malformed YAML: keep the body, drop the block
        body = re.sub(r"\A---\n.*?\n---\n", "", raw, flags=re.S)
        return {}, body


def doc_title(meta: dict, body: str, path: str) -> str:
    """Front matter `title` > first H1 > `nav.title` > file name."""
    title = meta.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    for h in headings(body):
        if h.level == 1 and h.text:
            return h.text
    nav = meta.get("nav")
    if isinstance(nav, dict) and isinstance(nav.get("title"), str) and nav["title"].strip():
        return nav["title"].strip()
    name = posixpath.splitext(posixpath.basename(path))[0]
    if name in ("index", "README"):
        name = posixpath.basename(posixpath.dirname(path)) or name
    return name.replace("-", " ").replace("_", " ").title()


def nav_title(meta: dict, body: str, path: str) -> str:
    """Sidebar label - used to build breadcrumbs for child pages."""
    nav = meta.get("nav")
    if isinstance(nav, dict) and isinstance(nav.get("title"), str) and nav["title"].strip():
        return nav["title"].strip()
    return doc_title(meta, body, path)


# ------------------------------------------------------- includes & snippets

_INCLUDE_RE = re.compile(r"<!--\s*@include:\s*([^\s>{#]+)\s*(?:\{(\d*),?(\d*)\})?\s*-->")
_SNIPPET_RE = re.compile(
    r"^[ \t]*<<<\s+@/docs/(\S+?)(?:\{[^}]*\})?(?:\s+\[([^\]]+)\])?[ \t]*$", re.M
)

_EXT_LANG = {
    ".xml": "xml", ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".php": "php",
    ".js": "javascript", ".ts": "typescript", ".twig": "twig", ".sh": "bash",
    ".html": "html", ".scss": "scss", ".css": "css", ".vue": "vue", ".sql": "sql",
}


def resolve_includes(body: str, doc_path: str, read: ReadFn, depth: int = 0, max_depth: int = 3) -> str:
    """Inline `<!--@include: rel.md-->` (VitePress markdown include), recursively."""
    if depth >= max_depth:
        return body

    def repl(m: re.Match[str]) -> str:
        target = posixpath.normpath(posixpath.join(posixpath.dirname(doc_path), m.group(1)))
        raw = read(target)
        if raw is None:
            return ""
        _, inc = parse_front_matter(raw)
        if m.group(2) or m.group(3):  # optional {start,end} line range, 1-based inclusive
            lines = inc.split("\n")
            start = int(m.group(2) or 1) - 1
            end = int(m.group(3)) if m.group(3) else len(lines)
            inc = "\n".join(lines[start:end])
        return resolve_includes(inc.strip("\n"), target, read, depth + 1, max_depth)

    return map_prose(body, lambda seg: _INCLUDE_RE.sub(repl, seg))


def resolve_snippet_imports(body: str, read: ReadFn) -> str:
    """Turn `<<< @/docs/snippets/x.xml [Label]` into an inline fenced code block."""

    def repl(m: re.Match[str]) -> str:
        path = m.group(1).split("#")[0]
        content = read(path)
        if content is None:
            return ""
        name = posixpath.basename(path)
        lang = _EXT_LANG.get(posixpath.splitext(path)[1].lower(), "")
        label = f"**{m.group(2).strip()}** (`{name}`):" if m.group(2) else f"`{name}`:"
        fence = "````" if "```" in content else "```"
        return f"{label}\n\n{fence}{lang}\n{content.rstrip()}\n{fence}"

    return map_prose(body, lambda seg: _SNIPPET_RE.sub(repl, seg))


# --------------------------------------------------------- VitePress cleanup

_CONTAINER_LABELS = {
    "info": "Note", "tip": "Tip", "note": "Note", "warning": "Warning",
    "danger": "Caution", "caution": "Caution", "details": "Details",
}
_CONTAINER_RE = re.compile(r"^[ \t]*:{3,}[ \t]*([a-zA-Z-]+)?[ \t]*(.*?)[ \t]*$", re.M)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_TABS_RE = re.compile(r"^[ \t]*</?Tabs\b[^>]*>[ \t]*$", re.M)
_TAB_OPEN_RE = re.compile(r"^[ \t]*<Tab\b[^>]*?title=\"([^\"]*)\"[^>]*>[ \t]*$", re.M)
_TAB_CLOSE_RE = re.compile(r"^[ \t]*</Tab>[ \t]*$", re.M)
_PAGEREF_RE = re.compile(r"<PageRef\b([^>]*?)/?>", re.S)
_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
_IMG_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_IMG_HTML_RE = re.compile(r"<(img|iframe|video|source)\b[^>]*/?>(?:</\1>)?", re.I | re.S)
_COMPONENT_LINE_RE = re.compile(r"^[ \t]*</?(ClientOnly|YoutubeRef|Badge)\b[^>]*>[ \t]*$", re.M)
_BLANKS_RE = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")


def _container(m: re.Match[str]) -> str:
    kind = (m.group(1) or "").lower()
    title = (m.group(2) or "").strip()
    if not kind or kind in ("code-group", "raw", "v-pre"):
        return ""  # closing ':::' or a purely visual wrapper
    label = _CONTAINER_LABELS.get(kind, kind.title())
    return f"**{label}:** {title}".rstrip()


def _pageref(m: re.Match[str]) -> str:
    attrs = dict(_ATTR_RE.findall(m.group(1)))
    title, page = attrs.get("title", "").strip(), attrs.get("page", "").strip()
    return f"See also: [{title or page}]({page})" if page else ""


def clean_vitepress(body: str) -> str:
    """Strip site-only syntax while keeping its textual meaning. Code is untouched."""

    def clean(seg: str) -> str:
        seg = _COMMENT_RE.sub("", seg)
        seg = _CONTAINER_RE.sub(_container, seg)
        seg = _TABS_RE.sub("", seg)
        seg = _TAB_OPEN_RE.sub(lambda m: f"**{m.group(1)}:**", seg)
        seg = _TAB_CLOSE_RE.sub("", seg)
        seg = _PAGEREF_RE.sub(_pageref, seg)
        seg = _IMG_MD_RE.sub("", seg)
        seg = _IMG_HTML_RE.sub("", seg)
        seg = _COMPONENT_LINE_RE.sub("", seg)
        return _BLANKS_RE.sub("\n\n", seg)

    return map_prose(body, clean).strip("\n") + "\n"


# --------------------------------------------------- navigation-page filter


def prose_stats(body: str) -> tuple[int, int]:
    """(prose words outside links/headings/code, number of links)."""
    words = links = 0
    for is_code, seg in split_code_segments(body):
        if is_code:
            continue
        for line in seg.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            links += len(re.findall(r"\[[^\]]*\]\([^)]*\)", stripped))
            no_links = re.sub(r"\[[^\]]*\]\([^)]*\)", "", stripped)
            no_links = re.sub(r"<[^>]+>", "", no_links)
            words += len(re.findall(r"\w+", no_links))
    return words, links


def is_navigation_page(body: str, max_prose_words: int) -> bool:
    """A page that is essentially a list of links (or empty) carries no answer."""
    if any(is_code for is_code, _ in split_code_segments(body)):
        return False
    words, links = prose_stats(body)
    return words < max_prose_words and (links >= 2 or words < 10)


# ------------------------------------------------------------------- links


def path_to_url(path: str, base_url: str) -> str:
    """guides/a/b.md -> {base}/guides/a/b.html ; guides/a/index.md -> {base}/guides/a/"""
    base = base_url.rstrip("/")
    stem = path[:-3] if path.endswith(".md") else path
    name = posixpath.basename(stem)
    if name in ("index", "README"):
        directory = posixpath.dirname(stem)
        return f"{base}/{directory}/" if directory else f"{base}/"
    return f"{base}/{stem}.html"


_SLUG_SPECIAL = re.compile(r"[\s~`!@#$%^&*()\-_+=\[\]{}|\\;:\"'“”‘’<>,.?/]+")


def slugify(heading: str) -> str:
    """VitePress' default heading-anchor slugify, so citations deep-link to the section."""
    s = unicodedata.normalize("NFKD", heading)
    s = re.sub(r"[̀-ͯ]", "", s)
    s = re.sub(r"[\u0000-\u001f]", "", s)
    s = _SLUG_SPECIAL.sub("-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    s = re.sub(r"^(\d)", r"_\1", s)
    return s.lower()

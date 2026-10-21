"""Turn the git checkout into clean `Document` objects.

Files are read from the git object store at HEAD (`git cat-file --batch`), not
from the working tree. Two reasons:

* Reproducibility - the corpus is *exactly* the pinned commit, even if somebody
  edited a file in data/raw by accident.
* Portability - Windows without LongPathsEnabled cannot open some of these
  paths through the filesystem, but git can always hand us the blob.
"""

from __future__ import annotations

import fnmatch
import logging
import posixpath
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from ingest import markdown as md

log = logging.getLogger(__name__)


@dataclass
class Document:
    path: str  # repo-relative, e.g. guides/plugins/plugins/plugin-base-guide.md
    url: str
    title: str
    parents: list[str]  # sidebar titles of ancestor sections, e.g. ["Guides", "Extensions", "Plugins"]
    text: str  # cleaned markdown body
    commit_sha: str
    meta: dict = field(default_factory=dict)

    @property
    def breadcrumb(self) -> list[str]:
        return [*self.parents, self.title]


class GitTreeReader:
    """Read files of one commit straight from git's object database."""

    def __init__(self, repo: Path, rev: str = "HEAD") -> None:
        self.repo = repo
        self.rev = rev
        self._cache: dict[str, str | None] = {}
        self._proc: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> GitTreeReader:
        self._proc = subprocess.Popen(
            ["git", "-C", str(self.repo), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        return self

    def __exit__(self, *exc: object) -> None:
        if self._proc and self._proc.stdin:
            self._proc.stdin.close()
            self._proc.wait(timeout=10)
        self._proc = None

    def commit_sha(self) -> str:
        return subprocess.run(
            ["git", "-C", str(self.repo), "rev-parse", self.rev], capture_output=True, text=True, check=True
        ).stdout.strip()

    def list_files(self, prefixes: list[str]) -> list[str]:
        out = subprocess.run(
            ["git", "-C", str(self.repo), "ls-tree", "-r", "--name-only", self.rev, "--", *prefixes],
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
        ).stdout
        return sorted(p for p in out.split("\n") if p)

    def read(self, path: str) -> str | None:
        if path in self._cache:
            return self._cache[path]
        assert self._proc and self._proc.stdin and self._proc.stdout, "use GitTreeReader as a context manager"
        self._proc.stdin.write(f"{self.rev}:{path}\n".encode())
        self._proc.stdin.flush()
        header = self._proc.stdout.readline().decode().split()
        if len(header) < 3 or header[1] != "blob":  # "<obj> missing"
            self._cache[path] = None
            return None
        size = int(header[2])
        data = self._proc.stdout.read(size)
        self._proc.stdout.read(1)  # trailing LF
        text = data.decode("utf-8", errors="replace")
        self._cache[path] = text
        return text


def _excluded(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(posixpath.basename(path), pat) for pat in patterns)


@dataclass
class CorpusStats:
    files_seen: int = 0
    excluded: int = 0
    navigation_pages: list[str] = field(default_factory=list)
    documents: int = 0


def iter_documents(settings: Settings, stats: CorpusStats | None = None) -> Iterator[Document]:
    stats = stats if stats is not None else CorpusStats()
    with GitTreeReader(settings.docs_dir) as reader:
        sha = reader.commit_sha()
        paths = [p for p in reader.list_files(settings.docs_include) if p.endswith(".md")]
        stats.files_seen = len(paths)

        parent_titles = _section_titles(reader, paths)

        for path in paths:
            if _excluded(path, settings.docs_exclude):
                stats.excluded += 1
                continue
            raw = reader.read(path)
            if raw is None:
                continue
            meta, body = md.parse_front_matter(raw)
            title = md.doc_title(meta, body, path)
            body = md.resolve_includes(body, path, reader.read)
            body = md.resolve_snippet_imports(body, reader.read)
            body = md.clean_vitepress(body)
            if md.is_navigation_page(body, settings.nav_page_max_prose_words):
                stats.navigation_pages.append(path)
                continue
            stats.documents += 1
            yield Document(
                path=path,
                url=md.path_to_url(path, settings.docs_base_url),
                title=title,
                parents=_parents(path, parent_titles),
                text=body,
                commit_sha=sha,
                meta={k: v for k, v in meta.items() if k in ("area", "tags", "date")},
            )


def _section_titles(reader: GitTreeReader, paths: list[str]) -> dict[str, str]:
    """Directory -> sidebar title, taken from that directory's index.md."""
    titles: dict[str, str] = {}
    for path in paths:
        if posixpath.basename(path) in ("index.md", "README.md"):
            raw = reader.read(path)
            if raw is None:
                continue
            meta, body = md.parse_front_matter(raw)
            titles[posixpath.dirname(path)] = md.nav_title(meta, body, path)
    return titles


def _parents(path: str, titles: dict[str, str]) -> list[str]:
    directory = posixpath.dirname(path)
    is_index = posixpath.basename(path) in ("index.md", "README.md")
    if is_index:  # an index page's own directory is the page itself
        directory = posixpath.dirname(directory)
    parts = directory.split("/") if directory else []
    crumbs = []
    for i in range(1, len(parts) + 1):
        d = "/".join(parts[:i])
        crumbs.append(titles.get(d) or parts[i - 1].replace("-", " ").title())
    return crumbs


def load_documents(settings: Settings) -> tuple[list[Document], CorpusStats]:
    stats = CorpusStats()
    docs = list(iter_documents(settings, stats))
    log.info(
        "Corpus: %d files, %d excluded, %d navigation pages skipped, %d documents",
        stats.files_seen, stats.excluded, len(stats.navigation_pages), stats.documents,
    )
    return docs, stats

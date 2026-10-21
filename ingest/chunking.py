"""Chunking strategies.

`HeadingChunker` (production) - structure-aware:
  * splits a page at H1/H2/H3 boundaries and keeps the heading chain as a
    breadcrumb ("Guides > Extensions > Plugins > Plugin Base Guide > Install");
  * builds chunks from *atoms* - a fenced code block, or a blank-line separated
    paragraph/list/table - and never cuts inside an atom, so a code block is
    never split (an oversized one becomes its own chunk);
  * glues a lead-in line ending with ":" and a heading line to the atom that
    follows, so "Register the subscriber in services.xml:" stays with the XML;
  * merges sections that are too short to stand alone with their neighbour;
  * overlaps consecutive chunks of a long section by whole trailing atoms;
  * prefixes title + breadcrumb to the text that gets embedded, so the vector
    "knows" where the text lives.

`FixedChunker` (baseline, experiment #0) - fixed token windows with overlap over
the same cleaned text, no prefix, no structure. It exists so the README can show
a measured "before" for the heading-aware strategy.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import tiktoken

from ingest import markdown as md
from ingest.corpus import Document

SPLIT_LEVEL = 3  # H1-H3 start a new section; H4+ stay inside their parent section
MIN_WORDS = 8  # chunks with fewer words carry no answer


@lru_cache(maxsize=4)
def get_encoder(name: str = "cl100k_base") -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


@dataclass
class Chunk:
    chunk_id: str
    doc_path: str
    url: str
    title: str
    headings: list[str]  # section path inside the page, e.g. ["Install", "Via CLI"]
    breadcrumb: list[str]  # parents + title + headings
    text: str  # chunk body shown to the LLM / user
    token_count: int
    chunk_index: int
    strategy: str
    commit_sha: str
    section_anchor: str | None = None

    @property
    def point_id(self) -> str:
        """Qdrant point ids must be UUIDs or ints - derive one from the content hash."""
        return str(uuid.UUID(hex=self.chunk_id[:32]))

    @property
    def embed_text(self) -> str:
        if self.strategy == "fixed":
            return self.text
        return f"{self.title}\n{' > '.join(self.breadcrumb)}\n\n{self.text}"

    def payload(self) -> dict:
        data = asdict(self)
        data["embed_text"] = self.embed_text
        return data


def _common_prefix(paths: list[tuple[str, ...]]) -> list[str]:
    if not paths:
        return []
    prefix = list(paths[0])
    for p in paths[1:]:
        n = 0
        while n < min(len(prefix), len(p)) and prefix[n] == p[n]:
            n += 1
        prefix = prefix[:n]
    return prefix


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# ------------------------------------------------------------------ atoms


@dataclass
class Atom:
    text: str
    tokens: int
    is_code: bool = False
    glue_next: bool = False  # never end a chunk right after this atom


@dataclass
class Section:
    headings: list[str]
    anchor: str | None
    atoms: list[Atom] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(a.tokens for a in self.atoms)


class HeadingChunker:
    strategy = "heading"

    def __init__(self, max_tokens: int = 600, min_tokens: int = 120, overlap_tokens: int = 80,
                 encoding: str = "cl100k_base") -> None:
        if not 0 <= overlap_tokens < max_tokens:
            raise ValueError("overlap must be smaller than max_tokens")
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.enc = get_encoder(encoding)

    def count(self, text: str) -> int:
        return len(self.enc.encode(text, disallowed_special=()))

    # -- sections -----------------------------------------------------------
    def sections(self, doc: Document) -> list[Section]:
        lines = doc.text.split("\n")
        code_ranges = md.code_line_ranges(doc.text)
        heads = [h for h in md.headings(doc.text) if h.level <= SPLIT_LEVEL]

        boundaries = [(h.line, h) for h in heads]
        sections: list[Section] = []
        stack: list[tuple[int, str]] = []  # (level, text)

        def build(start: int, end: int, head: md.Heading | None) -> None:
            if head is not None:
                while stack and stack[-1][0] >= head.level:
                    stack.pop()
                stack.append((head.level, head.text))
            # An H1 that repeats the page title is noise inside the breadcrumb.
            path = [t for lvl, t in stack if not (lvl == 1 and t == doc.title)]
            seg_lines = lines[start:end]
            if head is not None and head.level == 1 and head.text == doc.title:
                seg_lines = seg_lines[1:]  # drop the "# Title" line itself, the prefix carries it
            ranges = [(s - start, e - start) for s, e in code_ranges if s >= start and e <= end]
            if head is not None and head.level == 1 and head.text == doc.title:
                ranges = [(s - 1, e - 1) for s, e in ranges]
            atoms = self._atoms(seg_lines, ranges)
            is_title = head is not None and head.level == 1 and head.text == doc.title
            anchor = md.slugify(head.text) if head is not None and not is_title else None
            if atoms:
                sections.append(Section(headings=path, anchor=anchor, atoms=atoms))

        first = boundaries[0][0] if boundaries else len(lines)
        if first > 0:
            build(0, first, None)
        for i, (line, head) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines)
            build(line, end, head)
        return sections

    def _atoms(self, lines: list[str], code_ranges: list[tuple[int, int]]) -> list[Atom]:
        """Split section lines into atoms: code blocks and blank-line separated prose."""
        raw: list[tuple[bool, list[str]]] = []
        i = 0
        ranges = sorted(code_ranges)
        r = 0
        buf: list[str] = []

        def flush() -> None:
            if any(line.strip() for line in buf):
                raw.append((False, buf.copy()))
            buf.clear()

        while i < len(lines):
            if r < len(ranges) and i == ranges[r][0]:
                flush()
                s, e = ranges[r]
                raw.append((True, lines[s:e]))
                i = e
                r += 1
                continue
            line = lines[i]
            if not line.strip():
                flush()
            elif line.lstrip().startswith("#") and re.match(r"^\s{0,3}#{1,6}\s", line):
                flush()  # a heading line is its own (glued) atom
                raw.append((False, [line]))
            else:
                buf.append(line)
            i += 1
        flush()

        atoms: list[Atom] = []
        for is_code, block in raw:
            text = "\n".join(block).strip("\n")
            if not text.strip():
                continue
            is_heading = not is_code and len(block) == 1 and re.match(r"^\s{0,3}#{1,6}\s", block[0]) is not None
            glue = is_heading or (not is_code and text.rstrip().endswith(":"))
            tokens = self.count(text)
            if not is_code and tokens > self.max_tokens:
                atoms.extend(self._split_prose(block))
            else:
                atoms.append(Atom(text=text, tokens=tokens, is_code=is_code, glue_next=glue))
        return atoms

    def _split_prose(self, block: list[str]) -> list[Atom]:
        """Fallback for a giant paragraph/table: split by lines, repeat table headers."""
        header: list[str] = []
        if len(block) > 2 and block[0].lstrip().startswith("|") and re.match(r"^\s*\|?\s*:?-{3,}", block[1]):
            header = block[:2]
            body = block[2:]
        else:
            body = block
        header_tokens = self.count("\n".join(header)) if header else 0
        out: list[Atom] = []
        cur: list[str] = []
        cur_tokens = header_tokens
        for line in body:
            t = self.count(line) + 1
            if cur and cur_tokens + t > self.max_tokens:
                text = "\n".join(header + cur)
                out.append(Atom(text=text, tokens=self.count(text)))
                cur, cur_tokens = [], header_tokens
            cur.append(line)
            cur_tokens += t
        if cur:
            text = "\n".join(header + cur)
            out.append(Atom(text=text, tokens=self.count(text)))
        return out

    # -- packing ------------------------------------------------------------
    @staticmethod
    def _top(sec: Section) -> str | None:
        return sec.headings[0] if sec.headings else None

    def _group_sections(self, sections: list[Section]) -> list[list[Section]]:
        """Decide which consecutive sections may share a chunk.

        * A group smaller than `min_tokens` always absorbs the next section - a
          lone heading or a one-line section is useless as a retrieval unit. If
          that makes the group too big, `_pack` splits it again at atom borders.
        * Sibling subsections under the same top-level heading are packed
          together up to `max_tokens`, so chunks follow the page's own topics.
        * A new top-level heading starts a new group (topic boundary).
        """
        groups: list[list[Section]] = []
        cur: list[Section] = []
        for sec in sections:
            if not cur:
                cur = [sec]
                continue
            cur_tokens = sum(s.tokens for s in cur)
            same_topic = self._top(sec) is not None and self._top(sec) == self._top(cur[0])
            if cur_tokens < self.min_tokens or (same_topic and cur_tokens + sec.tokens <= self.max_tokens):
                cur.append(sec)
            else:
                groups.append(cur)
                cur = [sec]
        if cur:
            # A too-small last group joins the previous one; `_pack` re-splits if needed.
            if groups and sum(s.tokens for s in cur) < self.min_tokens:
                groups[-1].extend(cur)
            else:
                groups.append(cur)
        return groups

    def _pack(self, atoms: list[tuple[Section, Atom]]) -> list[list[tuple[Section, Atom]]]:
        """Greedy packing of atoms into <= max_tokens chunks with atom-level overlap."""
        chunks: list[list[tuple[Section, Atom]]] = []
        cur: list[tuple[Section, Atom]] = []
        cur_tokens = 0
        i = 0
        while i < len(atoms):

            # Treat a glued run (lead-in + following atom) as one unit when deciding to break.
            run = [atoms[i]]
            j = i
            while atoms[j][1].glue_next and j + 1 < len(atoms):
                j += 1
                run.append(atoms[j])
            run_tokens = sum(a.tokens for _, a in run)
            if cur and cur_tokens + run_tokens > self.max_tokens:
                chunks.append(cur)
                cur = self._overlap_tail(cur)
                cur_tokens = sum(a.tokens for _, a in cur)
                if cur_tokens + run_tokens > self.max_tokens:  # overlap would not fit - drop it
                    cur, cur_tokens = [], 0
            cur.extend(run)
            cur_tokens += run_tokens
            i = j + 1
        if cur:
            tail_tokens = sum(a.tokens for _, a in cur)
            prev_tokens = sum(a.tokens for _, a in chunks[-1]) if chunks else 0
            # A tiny remainder is worth more appended to its predecessor (25% slack)
            # than as a near-empty chunk of its own.
            if chunks and tail_tokens < self.min_tokens and prev_tokens + tail_tokens <= self.max_tokens * 1.25:
                overlap = self._overlap_tail(chunks[-1])
                chunks[-1].extend(cur[len(overlap):] if cur[:len(overlap)] == overlap else cur)
            else:
                chunks.append(cur)
        return chunks

    def _overlap_tail(self, chunk: list[tuple[Section, Atom]]) -> list[tuple[Section, Atom]]:
        if self.overlap_tokens <= 0:
            return []
        tail: list[tuple[Section, Atom]] = []
        total = 0
        last_section = chunk[-1][0]
        for sec, atom in reversed(chunk):
            if sec is not last_section or total + atom.tokens > self.overlap_tokens:
                break
            if atom.glue_next and not tail:  # a dangling lead-in/heading is useless as overlap
                break
            tail.insert(0, (sec, atom))
            total += atom.tokens
        if not tail:
            # Last atom is bigger than the overlap budget: carry its last sentences (prose only).
            sec, atom = chunk[-1]
            if not atom.is_code:
                sentences = re.split(r"(?<=[.!?])\s+", atom.text)
                carried: list[str] = []
                for s in reversed(sentences[1:]):
                    if self.count(" ".join([s, *carried])) > self.overlap_tokens:
                        break
                    carried.insert(0, s)
                if carried:
                    text = " ".join(carried)
                    tail = [(sec, Atom(text=text, tokens=self.count(text)))]
        return tail

    # -- public -------------------------------------------------------------
    def chunk(self, doc: Document) -> list[Chunk]:
        out: list[Chunk] = []
        seen: set[str] = set()
        for group in self._group_sections(self.sections(doc)):
            flat = [(sec, atom) for sec in group for atom in sec.atoms]
            for packed in self._pack(flat):
                first_sec = packed[0][0]
                text = "\n\n".join(a.text for _, a in packed).strip()
                if len(re.findall(r"\w+", text)) < MIN_WORDS:
                    continue  # e.g. a caption whose image was stripped
                # The breadcrumb is the heading path shared by everything in the chunk.
                section_paths = list(dict.fromkeys(tuple(sec.headings) for sec, _ in packed))
                common = _common_prefix(section_paths)
                cid = _hash(self.strategy, doc.path, text)
                if cid in seen:
                    cid = _hash(self.strategy, doc.path, text, str(len(out)))
                seen.add(cid)
                anchor = first_sec.anchor
                out.append(Chunk(
                    chunk_id=cid,
                    doc_path=doc.path,
                    url=f"{doc.url}#{anchor}" if anchor else doc.url,
                    title=doc.title,
                    headings=common,
                    breadcrumb=[*doc.breadcrumb, *common],
                    text=text,
                    token_count=self.count(text),
                    chunk_index=len(out),
                    strategy=self.strategy,
                    commit_sha=doc.commit_sha,
                    section_anchor=anchor,
                ))
        return out


class FixedChunker:
    """Baseline: fixed token windows, no structure, no prefix."""

    strategy = "fixed"

    def __init__(self, chunk_tokens: int = 512, overlap_tokens: int = 64, encoding: str = "cl100k_base") -> None:
        if not 0 <= overlap_tokens < chunk_tokens:
            raise ValueError("overlap must be smaller than chunk_tokens")
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.enc = get_encoder(encoding)

    def chunk(self, doc: Document) -> list[Chunk]:
        tokens = self.enc.encode(doc.text, disallowed_special=())
        step = self.chunk_tokens - self.overlap_tokens
        out: list[Chunk] = []
        for start in range(0, max(len(tokens), 1), step):
            window = tokens[start:start + self.chunk_tokens]
            if not window:
                break
            text = self.enc.decode(window).strip()
            if text:
                out.append(Chunk(
                    chunk_id=_hash(self.strategy, doc.path, str(start), text),
                    doc_path=doc.path,
                    url=doc.url,
                    title=doc.title,
                    headings=[],
                    breadcrumb=[*doc.breadcrumb],
                    text=text,
                    token_count=len(window),
                    chunk_index=len(out),
                    strategy=self.strategy,
                    commit_sha=doc.commit_sha,
                ))
            if start + self.chunk_tokens >= len(tokens):
                break
        return out


def make_chunker(strategy: str, settings) -> HeadingChunker | FixedChunker:
    if strategy == "heading":
        return HeadingChunker(settings.chunk_max_tokens, settings.chunk_min_tokens,
                              settings.chunk_overlap_tokens, settings.tokenizer_encoding)
    if strategy == "fixed":
        return FixedChunker(settings.baseline_chunk_tokens, settings.baseline_overlap_tokens,
                            settings.tokenizer_encoding)
    raise ValueError(f"unknown chunking strategy: {strategy!r}")


def chunk_documents(docs: Iterable[Document], chunker: HeadingChunker | FixedChunker) -> list[Chunk]:
    return [c for d in docs for c in chunker.chunk(d)]

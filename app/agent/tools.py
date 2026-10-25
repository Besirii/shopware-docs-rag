"""The agent's tools, run against the same Retriever that powers POST /ask.

Two tools, on purpose:

* search_docs(query)    -> the best sections for one query, numbered, with a short preview
* read_section(source)  -> the full text of one numbered source

Search returns previews, not full sections, so a few searches stay cheap; the
model reads in full only what it is going to use. That split is also what makes
the trace useful: it shows which sources the agent actually read.

Source numbers are given out once per question and never change: if a later
search returns a section the agent has already seen, it keeps its number. The
final answer cites these numbers, and `validate_citations` drops any number the
tools never handed out - the same guarantee /ask gives.

The tools never raise on bad input from the model (unknown tool, missing
argument, unknown source). They return an error text instead, so the model can
correct itself on its next turn. Errors still count against the budget, which
stops a model from looping on broken calls.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field, ValidationError

from app.retrieval import Candidate, Retriever

SEARCH_DOCS = "search_docs"
READ_SECTION = "read_section"
PREVIEW_CHARS = 400


class SearchDocsArgs(BaseModel):
    """Search the Shopware 6 developer documentation (meaning + keyword search). Returns the
    best-matching sections, numbered, with a short preview of each."""

    query: str = Field(min_length=2, max_length=300,
                       description="Short, specific search query in English. Use exact identifiers "
                                   "when you have them, e.g. 'EntityRepository search Criteria filter'.")


class ReadSectionArgs(BaseModel):
    """Read the full text of one numbered source from earlier search results."""

    source: int = Field(ge=1, description="The source number shown in the search results, e.g. 3.")


TOOL_ARGS: dict[str, type[BaseModel]] = {SEARCH_DOCS: SearchDocsArgs, READ_SECTION: ReadSectionArgs}


def tool_specs() -> list[dict]:
    """Tool definitions in the OpenAI function format, which every LangChain chat model accepts."""
    specs = []
    for name, schema in TOOL_ARGS.items():
        spec = convert_to_openai_tool(schema)
        spec["function"]["name"] = name
        specs.append(spec)
    return specs


# A source is a plain dict (not a dataclass) so the whole agent state stays JSON-serialisable:
# that is what lets us store a run as an audit trace later.
Source = dict


def make_source(n: int, c: Candidate) -> Source:
    p = c.payload
    return {
        "n": n,
        "chunk_id": p.get("chunk_id", c.point_id),
        "doc_path": p.get("doc_path", ""),
        "title": p.get("title", ""),
        "url": p.get("url", ""),
        "headings": p.get("headings") or [],
        "breadcrumb": p.get("breadcrumb") or [],
        "text": p.get("text", ""),
        "read": False,
    }


def label(src: Source) -> str:
    return " > ".join(src["breadcrumb"]) or src["title"]


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    text = re.sub(r"^#{1,6} [^\n]*\n+", "", text.strip())  # the heading is already in the label
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " …"


@dataclass
class ToolOutcome:
    content: str  # what the model sees
    sources: dict[str, Source] = field(default_factory=dict)  # registry entries added or changed
    step: dict = field(default_factory=dict)  # one entry of the trace
    error: str | None = None


class DocsTools:
    def __init__(self, retriever: Retriever, top_k: int = 5, mode: str = "hybrid") -> None:
        self.retriever = retriever
        self.top_k = top_k
        self.mode = mode

    def run(self, name: str, args: dict, registry: dict[str, Source]) -> ToolOutcome:
        """Run one tool call. `registry` (chunk_id -> source) is read, never modified here."""
        t0 = time.perf_counter()
        schema = TOOL_ARGS.get(name)
        if schema is None:
            outcome = self._error(f"Unknown tool {name!r}. Available tools: {', '.join(TOOL_ARGS)}.")
        else:
            try:
                parsed = schema.model_validate(args or {})
            except ValidationError as e:
                problems = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'args'}: {err['msg']}" for err in e.errors())
                outcome = self._error(f"Invalid arguments for {name}: {problems}.")
            else:
                if isinstance(parsed, SearchDocsArgs):
                    outcome = self._search(parsed.query, registry)
                else:
                    outcome = self._read(parsed.source, registry)  # type: ignore[attr-defined]
        outcome.step = {"tool": name, "args": args or {}, "sources": outcome.step.get("sources", []),
                        "ms": round((time.perf_counter() - t0) * 1000, 1), "error": outcome.error}
        return outcome

    # ----------------------------------------------------------------- tools
    def _search(self, query: str, registry: dict[str, Source]) -> ToolOutcome:
        hits = self.retriever.retrieve(query, self.mode).candidates[: self.top_k]
        if not hits:
            return ToolOutcome(f'No results for "{query}". Try other words or an exact identifier.',
                               step={"sources": []})
        new: dict[str, Source] = {}
        next_n = len(registry) + 1
        blocks, numbers = [], []
        for c in hits:
            chunk_id = c.payload.get("chunk_id", c.point_id)
            src = registry.get(chunk_id) or new.get(chunk_id)
            if src is None:
                src = new[chunk_id] = make_source(next_n, c)
                next_n += 1
            numbers.append(src["n"])
            blocks.append(f"[{src['n']}] {label(src)}\nurl: {src['url']}\n{preview(src['text'])}")
        content = (f'Results for "{query}":\n\n' + "\n\n".join(blocks)
                   + "\n\nUse read_section to read a source in full.")
        return ToolOutcome(content, sources=new, step={"sources": numbers})

    def _read(self, n: int, registry: dict[str, Source]) -> ToolOutcome:
        src = next((s for s in registry.values() if s["n"] == n), None)
        if src is None:
            known = sorted(s["n"] for s in registry.values())
            hint = f"Known sources: {known}." if known else "Run search_docs first."
            return self._error(f"Unknown source {n}. {hint}")
        content = f"[{n}] {label(src)}\nurl: {src['url']}\n\n{src['text'].strip()}"
        return ToolOutcome(content, sources={src["chunk_id"]: {**src, "read": True}}, step={"sources": [n]})

    @staticmethod
    def _error(message: str) -> ToolOutcome:
        return ToolOutcome(f"Error: {message}", step={"sources": []}, error=message)

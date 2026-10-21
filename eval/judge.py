"""LLM-as-judge implementations of the four RAGAS metrics.

Why not the `ragas` package? Its metrics are a handful of prompts plus simple
arithmetic; implementing them directly keeps the dependency tree small, works
with any provider here (Claude, OpenAI, a local Ollama model), and - most
importantly for learning - makes every number traceable to a prompt you can read.

Definitions (same as RAGAS):
  faithfulness       = supported claims / all claims in the answer
  answer relevancy   = mean cosine(question, questions generated from the answer);
                       0 if the answer is noncommittal ("I don't know")
  context precision  = average precision of "useful" verdicts over the top-k chunks
  context recall     = ground-truth statements attributable to the chunks / all statements

The judge model is fixed per run, recorded in the results file, and every call
is cached on disk, so re-running a report never re-bills the same judgement.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

from app.embeddings import DenseEmbedder
from app.llm import LLMClient, LLMError
from eval.metrics import average_precision

log = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).parent / ".cache" / "judge.jsonl"

JUDGE_SYSTEM = (
    "You are a strict, careful evaluator of answers produced by a documentation assistant "
    "for Shopware 6. Follow the instructions exactly and output only the requested JSON."
)

FAITHFULNESS_PROMPT = """\
You check an ANSWER written by an assistant against the CONTEXT the assistant was given.

Step 1 - list the factual claims made IN THE ANSWER. Only the answer: never list facts from the \
context that the answer does not state. Each claim is a short paraphrase of one statement of the \
answer. Skip citation markers like [1], greetings, and sentences saying something is not covered.
Step 2 - for each claim set "supported" to true only if the CONTEXT states it or it follows \
unambiguously from it. Code, file paths, class names and commands must appear in the CONTEXT.

<answer>
{answer}
</answer>

<context>
{context}
</context>

(The question was: {question})"""

FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"claim": {"type": "string"}, "supported": {"type": "boolean"}},
                "required": ["claim", "supported"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

RELEVANCY_PROMPT = """\
Read the ANSWER below. Write 3 different questions that this answer would be a good, direct \
answer to. Also decide whether the answer is noncommittal: true if it is evasive, vague, or \
says the information is not available.

ANSWER:
{answer}"""

RELEVANCY_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {"type": "array", "items": {"type": "string"}},
        "noncommittal": {"type": "boolean"},
    },
    "required": ["questions", "noncommittal"],
    "additionalProperties": False,
}

CONTEXT_PROMPT = """\
You are given a QUESTION, its REFERENCE ANSWER and numbered CONTEXT passages retrieved by a search engine.

Task 1 - for every passage n, decide if it is useful: it contains information that helps to \
arrive at the reference answer.
Task 2 - split the REFERENCE ANSWER into short factual statements; for each statement decide \
whether it can be attributed to (is supported by) any of the passages.

QUESTION: {question}

REFERENCE ANSWER:
{ground_truth}

CONTEXT:
{context}"""

CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "passages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"n": {"type": "integer"}, "useful": {"type": "boolean"}},
                "required": ["n", "useful"],
                "additionalProperties": False,
            },
        },
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"statement": {"type": "string"}, "attributed": {"type": "boolean"}},
                "required": ["statement", "attributed"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["passages", "statements"],
    "additionalProperties": False,
}


def _numbered(contexts: list[str]) -> str:
    return "\n\n".join(f"[{i}] {c.strip()}" for i, c in enumerate(contexts, start=1))


class Judge:
    def __init__(self, llm: LLMClient, embedder: DenseEmbedder, cache_path: Path = CACHE_PATH,
                 max_tokens: int = 2000) -> None:
        self.llm = llm
        self.embedder = embedder
        self.max_tokens = max_tokens
        self.cache_path = cache_path
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "cached": 0}
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._cache[row["key"]] = row["data"]

    @property
    def model_id(self) -> str:
        return f"{self.llm.provider}:{self.llm.model}"

    def _call(self, prompt: str, schema: dict, name: str) -> dict | None:
        key = hashlib.sha256(f"{self.model_id}\x00{name}\x00{prompt}".encode()).hexdigest()
        if key in self._cache:
            self.usage["cached"] += 1
            return self._cache[key]
        res = None
        for attempt in range(3):  # local servers hiccup (model reloads, sleep/resume); retry before giving up
            try:
                res = self.llm.generate_json(JUDGE_SYSTEM, prompt, schema, name, self.max_tokens)
                break
            except LLMError as e:
                log.warning("judge call %s failed (attempt %d/3): %s", name, attempt + 1, e)
                time.sleep(5 * (attempt + 1))
        if res is None:
            self.usage["failed"] = self.usage.get("failed", 0) + 1
            return None
        with self._lock:
            self.usage["input_tokens"] += res.input_tokens
            self.usage["output_tokens"] += res.output_tokens
            self.usage["calls"] += 1
            self._cache[key] = res.data
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "model": self.model_id, "name": name, "data": res.data}) + "\n")
        return res.data

    # ---------------------------------------------------------------- metrics
    def faithfulness(self, question: str, answer: str, contexts: list[str]) -> tuple[float | None, dict]:
        data = self._call(FAITHFULNESS_PROMPT.format(context=_numbered(contexts), question=question, answer=answer),
                          FAITHFULNESS_SCHEMA, "faithfulness")
        claims = (data or {}).get("claims") or []
        if not claims:
            return None, {"claims": []}
        score = sum(bool(c.get("supported")) for c in claims) / len(claims)
        return round(score, 4), {"claims": claims}

    def answer_relevancy(self, question: str, answer: str) -> tuple[float | None, dict]:
        data = self._call(RELEVANCY_PROMPT.format(answer=answer), RELEVANCY_SCHEMA, "answer_relevancy")
        if data is None:
            return None, {}
        generated = [q for q in data.get("questions", []) if isinstance(q, str) and q.strip()][:3]
        if data.get("noncommittal") or not generated:
            return 0.0, {"noncommittal": bool(data.get("noncommittal")), "questions": generated}
        q = np.array(self.embedder.embed_query(question))
        g = np.array([self.embedder.embed_query(x) for x in generated])
        sims = g @ q / (np.linalg.norm(g, axis=1) * np.linalg.norm(q) + 1e-12)
        return round(float(sims.mean()), 4), {"noncommittal": False, "questions": generated}

    def context_scores(self, question: str, ground_truth: str, contexts: list[str]) -> tuple[float | None, float | None, dict]:
        data = self._call(CONTEXT_PROMPT.format(question=question, ground_truth=ground_truth,
                                                context=_numbered(contexts)), CONTEXT_SCHEMA, "context")
        if data is None:
            return None, None, {}
        useful = {int(p["n"]): bool(p["useful"]) for p in data.get("passages", []) if isinstance(p.get("n"), int)}
        verdicts = [useful.get(i, False) for i in range(1, len(contexts) + 1)]
        precision = round(average_precision(verdicts), 4)
        statements = data.get("statements") or []
        recall = round(sum(bool(s.get("attributed")) for s in statements) / len(statements), 4) if statements else None
        return precision, recall, {"useful": verdicts, "statements": statements}

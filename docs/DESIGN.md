# Design notes and full evaluation

The [README](../README.md) is the short version. This page holds the details: the full pipeline,
every eval number, where the system fails, the design decisions and what it costs.

## Architecture in detail

```mermaid
flowchart LR
    subgraph Ingestion [Ingestion - offline]
        A[shopware/docs<br/>GitHub, pinned SHA] -->|shallow sparse clone| B[corpus.py<br/>read from git objects<br/>resolve includes & snippets<br/>strip VitePress syntax]
        B --> C[chunking.py<br/>split at H1-H3<br/>code blocks never split<br/>title + breadcrumb prefix]
        C --> D[index.py<br/>granite-embedding-small-r2 dense<br/>+ BM25 sparse]
    end
    D --> Q[(Qdrant<br/>shopware_docs<br/>shopware_docs_fixed)]
    subgraph Query [Query - online]
        U[POST /ask] --> R1[dense top 30]
        U --> R2[BM25 top 30]
        R1 & R2 --> F[RRF fusion<br/>k=60]
        F --> X[cross-encoder<br/>rerank or confidence]
        X -->|top score < threshold| N[not found]
        X -->|top-k chunks numbered 1..n| G[LLM, structured JSON<br/>answer + citations + answerable]
        G --> V[citation validation<br/>drop invented n]
    end
    Q --- R1 & R2
    subgraph Eval [Eval - offline]
        E[questions.jsonl<br/>47 questions] --> RE[run_eval.py] -->|calls /ask| U
        RE --> M[Hit@k · MRR · evidence@k<br/>refusal accuracy<br/>LLM judge: faithfulness,<br/>relevancy, ctx precision/recall]
    end
```

| Part | What it does | Code |
|---|---|---|
| Fetch | `git clone --depth 1 --filter=blob:none --sparse` of `guides/`, `concepts/`, `resources/references/` (+ `snippets/` that pages include); SHA saved to `data/manifest.json` | [`ingest/fetch_docs.py`](../ingest/fetch_docs.py) |
| Parse | front matter → title; `<!--@include-->` and `<<< @/docs/snippets/...` inlined; `:::info`, `<Tabs>`, `<PageRef>` turned into text; navigation-only pages dropped; path → URL (+ `#anchor` with VitePress' slug rules) | [`ingest/corpus.py`](../ingest/corpus.py), [`ingest/markdown.py`](../ingest/markdown.py) |
| Chunk | heading-aware (production) and fixed 512-token windows (baseline) | [`ingest/chunking.py`](../ingest/chunking.py) |
| Index | idempotent upserts keyed by content hash; only new/changed chunks are embedded (an interrupted ingest resumed with "652 new, 2240 unchanged"); stale ones deleted | [`ingest/index.py`](../ingest/index.py) |
| Retrieve | dense, BM25, hybrid (RRF), cross-encoder rerank / confidence gate | [`app/retrieval.py`](../app/retrieval.py) |
| Generate | numbered context, provider-native structured output, citation validation | [`app/generation.py`](../app/generation.py), [`app/llm.py`](../app/llm.py) |
| API | `POST /ask`, `GET /health`, `POST /ingest` (admin token) | [`app/main.py`](../app/main.py) |
| Eval | runner, deterministic metrics, LLM judge, threshold calibration, report | [`eval/`](../eval/) |

### API and local development

`POST /ask` request fields: `question`, `top_k` (default 6), `mode` (`dense` | `bm25` | `hybrid` |
`hybrid_rerank`), `chunking` (`heading` | `fixed` - the baseline index), `sparse_weight`, `generate`
(false = retrieval only), `include_contexts`. Every experiment below is just a different request -
no code changes.

Local development without Docker for the app: `uv sync`, `docker compose up -d qdrant`,
`uv run python -m ingest`, `uv run uvicorn app.main:app --reload`, `uv run pytest`.

---

## Evaluation

The eval set ([`eval/questions.jsonl`](../eval/questions.jsonl)) has **47 questions**: 42 answerable
across plugin development (12), DAL (7), storefront/themes (7), Store/Admin API (7), CLI &
configuration (7) and concepts (2), plus **5 the docs do not answer** (GraphQL API, Magento sync,
pricing plans, a non-existent feature, WooCommerce). Each answerable question has a hand-checked
ground truth, the page(s) that answer it and 1-3 *evidence* strings (identifiers/commands that must
be in the context, e.g. `kernel.event_subscriber`). A test verifies every evidence string exists in
its page at the pinned commit.

Metrics: **Hit@k / MRR@10** (first chunk from a relevant page), **evidence@6** (share of evidence
strings present in the 6 chunks the LLM sees - chunk-level and still deterministic), **refusal
accuracy**, and four LLM-judge metrics with RAGAS definitions implemented in
[`eval/judge.py`](../eval/judge.py). Intervals are 95 % bootstrap CIs over questions.

<!-- EVAL_TABLE_START -->
| # | Configuration | Hit@1 | Hit@5 | MRR@10 | Evidence@6 | Faithfulness | Answer rel. | Ctx precision | Ctx recall | Refusal acc. | p50 latency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Baseline: fixed 512-token chunks + dense only | 0.74 | 0.95 | 0.83 <sub>[0.75-0.92]</sub> | 0.94 | 0.91 | 0.87 | 0.80 | 0.90 | 89% | 5.4 s |
| 1 | + heading-aware chunking | 0.79 | 0.98 | 0.87 <sub>[0.78-0.94]</sub> | 0.92 | 0.94 | 0.83 | 0.76 | 0.88 | 89% | 3.6 s |
| 2 | + hybrid BM25 + dense (RRF, k=60) | 0.71 | 1.00 | 0.83 <sub>[0.74-0.91]</sub> | 0.88 | 0.92 | 0.82 | 0.81 | 0.85 | 91% | 3.9 s |
| 2b | + weighted RRF (BM25 x 0.5) | 0.76 | 1.00 | 0.86 <sub>[0.78-0.93]</sub> | 0.90 | 0.91 | 0.78 | 0.80 | 0.87 | 87% | 3.0 s |
| 3 | + cross-encoder reranker | 0.69 | 0.98 | 0.80 <sub>[0.70-0.89]</sub> | 0.85 | 0.90 | 0.80 | 0.76 | 0.81 | 89% | 4.5 s |

Dataset: `eval/questions.jsonl` (47 questions, sha256 `9ccaa05d9097bb10`) · docs commit `7ab78da58458` · embedding `ibm-granite/granite-embedding-small-english-r2` · reranker `Xenova/ms-marco-MiniLM-L-6-v2` · generator `qwen2.5:3b` · judge `ollama:qwen3:4b`

**Ablations (retrieval only):**

| Configuration | Hit@1 | Hit@5 | Hit@10 | MRR@10 | p50 retrieval |
|---|---|---|---|---|---|
| Baseline: fixed 512-token chunks + dense only | 0.74 | 0.95 | 1.00 | 0.83 | 663 ms |
| + heading-aware chunking | 0.79 | 0.98 | 1.00 | 0.87 | 309 ms |
| + hybrid BM25 + dense (RRF, k=60) | 0.71 | 1.00 | 1.00 | 0.83 | 283 ms |
| + weighted RRF (BM25 x 0.5) | 0.76 | 1.00 | 1.00 | 0.86 | 320 ms |
| + cross-encoder reranker | 0.69 | 0.98 | 1.00 | 0.80 | 1245 ms |
| BM25 only (heading chunks) | 0.55 | 0.90 | 0.98 | 0.69 | 244 ms |
| fixed chunks + hybrid | 0.64 | 0.98 | 1.00 | 0.81 | 254 ms |
| fixed chunks + hybrid + reranker | 0.76 | 0.95 | 1.00 | 0.85 | 1220 ms |
<!-- EVAL_TABLE_END -->

<!-- FINDINGS_START -->
### What the numbers say

**Retrieval (deterministic, 42 answerable questions):**

1. **Heading-aware chunking is the clearest win.** MRR 0.83 → 0.87 and Hit@1 0.74 → 0.79, while
   sending the LLM *less* text (1,874 vs 2,627 tokens in the top 6) and cutting zero code blocks
   instead of 9.2 %. The baseline's slightly higher evidence@6 (0.94 vs 0.92) is a size effect:
   six 512-token windows simply contain more text.
2. **Classic hybrid search did not help on its own.** Equal-weight RRF lowered MRR to 0.83
   because BM25 alone is weak on paraphrased questions here (MRR 0.69), and it pulls lexically
   similar but wrong pages up (e.g. "YAML" → the deployment-helper config page). It did reach
   Hit@5 = 1.00 - no question missed in the top 5. **Down-weighting BM25 to 0.5** keeps that and
   recovers MRR to 0.86.
3. **The cross-encoder made ranking worse** (MRR 0.80, evidence@6 0.85): MS MARCO-trained rerankers
   prefer passages that *read* like an answer over the passage that holds the identifier the
   question is about. It is used as a confidence gate instead (next point).
4. **Most differences are inside the 95 % intervals** (shown under MRR). With 42 questions only
   BM25-only is clearly worse; treat the rest as directional and grow the eval set before tuning
   further.

**Answers (generator `qwen2.5:3b`, judge `qwen3:4b`, both local):**

5. **Heading-aware chunks give the most faithful answers for less money.** Faithfulness 0.91 →
   0.94 and citation hit 0.85 → 0.89, with 22 % fewer input tokens (3,286 → 2,575) and generation
   p50 4.7 s → 3.0 s. Answer relevancy and context recall dip slightly (less text in the prompt).
6. **Refusals are the generator's weak spot.** On the fixed baseline the 3B model *invented* answers
   for 3 of the 5 unanswerable questions; with hybrid retrieval it refused all 5 - but it also
   refused 4-6 answerable ones, sometimes with the exact answer in chunk 1. A retrieval-confidence
   gate cannot fix over-refusal (it only adds refusals), so `RERANK_THRESHOLD` is left empty for
   this generator. With the retrieval signal alone the calibrated gate (1.26) would reach 96 %
   (leave-one-out 96 %) - the right setting depends on the LLM, which is why it is calibrated
   per configuration.
7. Judge-based differences between rows are a few points and noisy; the deterministic metrics and
   the refusal counts are the more reliable signal at this sample size.

### Where it fails, and why

- **Invented configuration (u05, baseline).** "How do I configure WooCommerce shipping zones?" -
  the retrieved chunks were about *app shipping methods*, close enough in wording that the 3B model
  answered with a made-up `wc_shipping_zones` manifest key. Hybrid retrieval surfaced less
  convincing context and the model refused. Lesson: lexical similarity can make a hallucination
  *more* likely, not less; refusals need their own metric.
- **Swapped facts (q41, production config).** The context says plugins are self-hosted only and
  apps run everywhere; the answer said the opposite. The judge caught it (faithfulness 0.5).
  Small generators mis-bind facts in comparison questions; a stronger model or an explicit
  "quote, then answer" prompt is the fix.
- **Over-refusal with the answer in front of it (q36).** The first chunk is "Debugging scheduled
  tasks" with `bin/console scheduled-task:run-single`, confidence score 7.9 (unanswerable questions
  never exceed 1.2), and the model still replied "not found". Next step below.
- **Lexical traps in retrieval (q15, q40).** "fetch products by name" ranks a custom-field page
  about *fetching* first; "YAML ... override" ranks the deployment-helper YAML page above the static
  system-config page. Query rewriting or a stronger embedding model would target these.

### Next steps (in order of expected value)

1. **Refusal override:** if the LLM says "not answerable" but retrieval confidence is far above
   anything seen for unanswerable questions, ask once more with "the excerpts were judged relevant;
   answer from them if possible". Would address most false refusals above - validate on new
   questions, not on this set.
2. Run the same eval with Claude as generator (and a stronger judge) - the pipeline is unchanged,
   only `.env`.
3. Grow the eval set with real questions (`eval/generate_questions.py` drafts candidates).
4. Query rewriting for keyword-heavy vs. paraphrased questions; keep it only if the eval improves.
<!-- FINDINGS_END -->

---

## Key decisions and why

**Chunking by headings, code blocks atomic.** In Shopware docs the answer is often the code
(`services.php`, a subscriber class, a `bin/console` command). The chunker builds chunks from
*atoms* - a fenced block, or a blank-line separated paragraph/list/table - so a fence is never
cut (the fixed baseline cuts **9.2 %** of the corpus' 2,900 code blocks; heading-aware: **0 %**).
A lead-in line ending in `:` is glued to the code it introduces, sibling H3s of one H2 are packed
together up to 600 tokens, tiny sections merge with a neighbour, and long sections overlap by whole
atoms. Every chunk is embedded with `title + breadcrumb` prepended
(`Guides > Extensions > Plugins > Plugin Base Guide > Install`).

**Read the corpus from git objects, not the filesystem.** The indexed text is exactly the pinned
commit, and paths longer than Windows' 260-character limit (this corpus has them) never matter.

**Embedding model with an explicit length budget.** Most small embedding models silently truncate
at 512 tokens; some chunks here are 5,000 tokens of code. `granite-embedding-small-english-r2`
(384-d, 8k context, trained with code) is used with an explicit 1,024-token cut, length-sorted
batching (attention memory is quadratic and a batch pads to its longest member) and a truncation
count in the index stats (18 of 2,892 chunks). Benchmarked alternatives:
bge-small (512 tokens), jina-v2-small, nomic-v1.5-Q, bge-base.

**Qdrant, one collection per chunking strategy.** Dense and BM25 sparse vectors live in the same
point; IDF is applied by Qdrant at query time (`Modifier.IDF`), and BM25's `avg_len` is computed
from this corpus (147 stemmed tokens per chunk) instead of fastembed's default 256. Separate
collections keep BM25 statistics clean. Collection metadata records the embedding model and the
API refuses to serve a collection built with a different one.

**RRF in Python by default** (both searches in one `query_batch_points` round trip) so the formula
is explicit and unit-tested; `FUSION_BACKEND=qdrant` switches to Qdrant's native prefetch + RRF
(a test checks both agree).

**Reranker as a confidence classifier.** See the results: on this corpus the cross-encoders tried
made the *ranking* worse (MRR on the full set: MiniLM-L-6 0.80, MiniLM-L-12 0.79, jina-turbo 0.80
vs 0.86 without reranking), and the larger bge-base / bge-v2-m3 took 15-60 s per query on this
laptop CPU while picking the wrong top passage on the spot checks. Removing the title/breadcrumb
prefix from the reranker input made it worse still (0.76). The cross-encoder is still useful as a relevance score for the
"not found" gate: it scores only the chunks that will reach the LLM, and the best score is compared
with a threshold calibrated by [`eval/calibrate.py`](../eval/calibrate.py) (with leave-one-out accuracy,
because 5 unanswerable questions are easy to overfit).

**Structured output per provider, no regex parsing.** Claude: `output_config.format` JSON schema
(current Claude models reject `temperature` and forced `tool_choice`; cost is controlled with
`effort`, refusals with server-side fallbacks). OpenAI: strict `json_schema`. Ollama: grammar
constrained `format` plus `num_ctx=8192` (the default context silently drops the start of a RAG
prompt - i.e. the documentation) and a repeat penalty (3B models occasionally loop until Ollama
aborts). The backend then strips any `[n]` that is not in the context. The compose file also
tunes Ollama for a 4 GB GPU (flash attention, 8-bit KV cache, one resident model) and disables
llama.cpp's host-RAM prompt cache (`LLAMA_ARG_CACHE_RAM=0`): with thousands of distinct prompts it
grew the runner to 7 GB, exhausted Docker's VM and slowed the judge from ~8 s per call to minutes.

**Own LLM judge instead of the `ragas` package.** Four prompts and some arithmetic; any provider
works, every number is traceable to a prompt, and judgements are cached on disk so re-running a
report costs nothing. The judge model is fixed per run and stored in the result file.

---

## Agent vs. single search

`/ask` runs one hybrid search and answers from the top 6 sections. `/agent` lets the LLM run its
own searches (`search_docs`), read the sections it will use in full (`read_section`) and stop when
it has enough, within a budget of 8 tool calls. Both use the same index, the same hybrid search and,
in these runs, the same LLM, so the difference is the agentic loop alone.

The comparison uses a second question set, `eval/questions_multihop.jsonl`: 15 questions that each
need **two different doc pages** (e.g. "add a custom field *and* show it in the storefront", "create
a Store API route *and* rate-limit it") plus 3 unanswerable ones. Every question lists its `hops`
(one or more interchangeable pages per part), and no single page contains all of its key facts -
checked against the pinned docs commit. The metrics are deterministic:

- **All parts found** - share of questions where *every* hop reached the LLM (the multi-hop success
  rate); **Parts found** is the average share of hops.
- **Key facts in context / in answer** - share of the question's key identifiers and commands
  (`evidence`) that appear in what the LLM saw, and in what it wrote.
- **Parts cited** - share of hops backed by a citation of one of their pages.

<!-- AGENT_TABLE_START -->
**Multi-hop questions** (each needs 2 different doc pages):

| Mode | All parts found | Parts found | Key facts in context | Key facts in answer | Parts cited | Faithfulness | Refusal acc. | Tool calls | Tokens in / out | Cost / question | p50 latency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Single search (`/ask`) | 40% <sub>[13-67]</sub> | 70% | 59% | 57% <sub>[46-67]</sub> | 70% | – | 100% | 1 search | 5,026 / 712 | $0.0009 | 4.2 s |
| Agent (`/agent`) | 100% <sub>[100-100]</sub> | 100% | 88% | 86% <sub>[75-95]</sub> | 97% | – | 100% | 5.7 | 18,568 / 1,504 | $0.0026 | 10.0 s |

Dataset: `eval/questions_multihop.jsonl` (18 questions, sha256 `d1ec6426d67acb82`) · LLM `claude-haiku-5-5` · judge `none`

**Original single-topic questions:**

| Mode | All parts found | Parts found | Key facts in context | Key facts in answer | Parts cited | Faithfulness | Refusal acc. | Tool calls | Tokens in / out | Cost / question | p50 latency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Single search (`/ask`) | 100% <sub>[100-100]</sub> | 100% | 90% | 87% <sub>[80-93]</sub> | 100% | – | 98% | 1 search | 4,608 / 380 | $0.0007 | 2.5 s |
| Agent (`/agent`) | 100% <sub>[100-100]</sub> | 100% | 92% | 92% <sub>[86-98]</sub> | 98% | – | 100% | 3.6 | 10,429 / 828 | $0.0015 | 5.2 s |

Dataset: `eval/questions.jsonl` (47 questions, sha256 `9ccaa05d9097bb10`) · LLM `claude-haiku-5-5` · judge `none`
<!-- AGENT_TABLE_END -->

### What the numbers say

- **Multi-part questions are where one search breaks.** A single search put both required pages in
  front of the LLM for only 6 of 15 questions. The agent searched once per part and found both
  every time. Typical misses of the single search: for the custom-field question it retrieved the
  *Administration* custom-field page instead of the plugin one, and for "Store API route + rate
  limit" its top 6 had the rate-limiter page but not the Store API route page.
- **More of the answer is right, and backed by a source.** Key identifiers in the answer rose from
  57% to 86%, and the share of parts backed by a citation from 70% to 97%.
- **No regression on simple questions.** On the original 47 questions both find the right page,
  key facts in the answer go from 87% to 92% (overlapping intervals, so no real difference), and the
  agent avoided the one false refusal of the single search (q20). One answer moved away from the
  reference (q41): the agent also read the custom-entity docs and answered "apps can create tables,
  but only for their own custom entities", where the reference says apps cannot change the schema.
- **It costs 2-4x.** On multi-part questions the agent used 5.7 tool calls, 3.7x the input tokens
  and 2.4x the latency (10 s vs 4 s at p50); on simple ones 2.3x tokens and 2x latency. With
  `claude-haiku-5-5` that is $0.0026 instead of $0.0009 per question.
- **Refusals are unchanged.** Both refused all unanswerable questions. The agent opens a refusal with
  the not-found sentence and then says what it searched; the first run scored those as answers
  because only the bare sentence counted as a refusal. That detection was fixed and the agent
  re-run - the table shows the re-run.

**Caveats.** The 15 multi-part questions were written for this comparison, so they show the case the
agent is built for, not average traffic. All metrics are deterministic string checks, there is no
LLM judge in these runs yet, and each configuration ran once.

**Next steps.** A judged run (faithfulness) with a stronger judge; a reviewer node that checks every
claim against the sections the agent read, measured on the same set; and routing - single search
by default, the agent only for multi-part questions or when the first search looks weak - to keep
most of the gain at a fraction of the cost.

---

## Latency and cost per question

<!-- COST_START -->
| Stage | p50 | p95 |
|---|---|---|
| retrieval (embed query + dense/BM25 + fusion) | 0.03 s | 0.08 s |
| cross-encoder confidence (6 chunks) | 0.29 s | 0.37 s |
| generation (qwen2.5:3b, local GPU) | 2.67 s | 10.57 s |
| total | 2.99 s | 10.95 s |

Mean tokens per question: **2657 input / 164 output** (context: 1956 tokens in 6 chunks). Estimated API cost per question for the same prompt with Claude:

| Model | Cost / question | Cost / 1,000 questions |
|---|---|---|
| `claude-haiku-5-5` | $0.0003 | $0.35 |
| `claude-sonnet-5-5` | $0.0070 | $6.96 |
| `claude-opus-5-5` | $0.0139 | $13.91 |

Token counts come from the local model's tokenizer; Claude's tokenizer and any adaptive-thinking tokens shift these numbers, so treat them as an order of magnitude. Embeddings, BM25 and the reranker run locally (free); embedding the whole corpus with `text-embedding-3-small` instead would cost about 1.1 M tokens = $0.02.
<!-- COST_END -->

**Eval cost.** One full judged run (5 configurations × 47 questions) sent the judge ~906 k input /
87 k output tokens (521 calls, plus cache hits). With `claude-sonnet-5-5` as judge that is about
**$2.70**, with `claude-haiku-5-5` about **$0.13**; the retrieval-only metrics (`--no-generate`)
are free and take under a minute. Judgements are cached, so re-running unchanged questions is free.

---

## Known limitations

- **Small eval set.** 42 answerable questions give ±0.08-0.09 confidence intervals on MRR; most
  differences between configurations 0-3 are inside that noise. The questions were written by
  reading the docs, so they lean towards "how do I X" phrasing that matches page titles. Add your own
  real questions (`eval/generate_questions.py` drafts more from random chunks for review).
- **Local judge.** The published judge numbers come from a 4B local model; it is noticeably
  stricter and noisier than a frontier judge. Compare rows with each other, not with RAGAS numbers
  from other projects. Set `JUDGE_PROVIDER=anthropic` for a stronger judge.
- **English only.** The embedding model and reranker are English; questions in other languages
  need a multilingual embedding (`EMBEDDING_MODEL=...`, rebuild with `--recreate`).
- **Doc-level relevance labels.** A question can be answered by pages that are not in
  `relevant_docs` (e.g. a hosting page and a plugin guide both describe `scheduled-task:run`), which
  undercounts Hit@k.
- **No answer streaming, no auth/rate limiting on `/ask`** - it is a reference implementation,
  not a hosted service.

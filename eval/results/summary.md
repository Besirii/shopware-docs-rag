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

## Agent vs. single search

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

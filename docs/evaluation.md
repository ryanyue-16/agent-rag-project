# Evaluation

## Evaluation contract

The retrieval benchmark is fixed and manually labelled:

- 10 synthetic enterprise-policy documents and 40 chunks;
- 55 queries: 50 answerable and 5 unanswerable;
- 11 exact, 20 semantic, 9 confusable and 10 multi-hop answerable cases;
- explicit `gold_chunk_ids` for every answerable query;
- identical corpus, cases and `k = 1, 3, 5` across retrieval modes.

Retrieval MRR, hit rate, precision and recall are calculated over the 50
answerable cases. Recall measures how many labelled evidence chunks are present
in the top-k results, so a two-evidence multi-hop query receives 0.5 when only
one gold chunk is retrieved. The five unanswerable cases are retained for
grounded-response/refusal evaluation but are not assigned artificial retrieval
relevance labels.

Reports store IDs, ranks and scores but omit raw document, chunk, prompt and
answer text.

## Retrieval benchmark

The following values are frozen in
`evals/reports/stage2_ablation_v1_summary.json`.

| Mode | MRR | Recall@5 | Multi-hop Recall@5 | p50 retrieval | p95 retrieval | p50 rerank | p95 rerank |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 1.00 | 0.97 | 0.85 | 665.4 ms | 3,328.4 ms | — | — |
| BM25 | 0.44 | 0.45 | 0.05 | 0.2 ms | 0.2 ms | — | — |
| Hybrid RRF | 0.99 | 0.98 | 0.90 | 949.7 ms | 17,984.0 ms | — | — |
| Hybrid RRF + cross-encoder | 1.00 | 1.00 | 1.00 | 728.0 ms | 22,467.1 ms | 106.2 ms | 605.1 ms |

Against the dense baseline, the accepted hybrid-reranked configuration improved
overall Recall@5 from 0.97 to 1.00 (+3 percentage points) and multi-hop Recall@5
from 0.85 to 1.00 (+15 percentage points). MRR did not improve because the dense
baseline was already 1.00 on this controlled corpus.

Retrieval latency includes the remote query-embedding call. Network instability
caused the high p95 values, so these numbers are end-to-end observations for the
recorded runs—not local FAISS or cross-encoder throughput claims. The BM25
latency is not directly comparable because it makes no embedding request.

## Grounded-response checks

The Stage 4 LangGraph smoke run used four selected cases: two answerable and two
unanswerable, spanning English/Chinese and single-/multi-hop behavior.

| Measure | Recorded result |
|---|---:|
| Cases | 4 |
| Pass rate | 1.00 |
| Citation-valid rate | 1.00 |
| Gold citation recall | 1.00 |
| Answer-group coverage | 1.00 |
| Unanswerable refusal rate | 1.00 |
| p50 end-to-end latency | 23,610.2 ms |
| p95 end-to-end latency | 28,489.4 ms |

This is a smoke test, not a claim of 100% full-corpus answer accuracy. Citation
validity means each cited source, page and chunk ID existed in the evidence for
that request. It is not an independent entailment score.

## Production acceptance

The Stage 6 acceptance run recorded 380 total tokens (155 input, 225 output), a
9,766.879 ms direct-route workflow latency, successful request-ID propagation,
and passing liveness/readiness probes. Cost was `null` because model pricing was
not configured. The final deterministic suite contains 62 passing tests.

Routing decisions are traced, but routing accuracy has not been evaluated on a
labelled routing dataset. No routing-accuracy number should appear in a CV,
README or interview answer until that benchmark exists.

## Reproduction

```bash
python -m evals.run_eval --retrieval-mode dense --ks 1 3 5
python -m evals.run_eval --retrieval-mode bm25 --ks 1 3 5
python -m evals.run_eval --retrieval-mode hybrid --ks 1 3 5
python -m evals.run_eval --retrieval-mode hybrid_rerank --ks 1 3 5
python -m evals.ablation
python -m evals.run_grounded_eval
python -m evals.run_stage6_smoke
```

The retrieval and smoke evaluations call external models and therefore require
a valid OpenAI key, network access and sufficient usage allowance. Unit tests
use injected fakes where possible and do not require an API key.

## Evidence sources

- `evals/reports/dense_enterprise_v1_summary.json`: frozen dense baseline.
- `evals/reports/stage2_ablation_v1_summary.json`: four-mode benchmark.
- `evals/reports/stage4_langgraph_v1_summary.json`: grounded LangGraph smoke.
- `evals/reports/stage5_enterprise_kb_v1_summary.json`: lifecycle/persistence acceptance.
- `evals/reports/stage6_production_v1_summary.json`: production acceptance.


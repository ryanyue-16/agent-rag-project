# Failure Analysis

## Observed retrieval failures

The frozen dense baseline achieved 0.97 Recall@5 but retrieved only one of two
gold evidence chunks for three multi-hop cases:

| Case | Missing dense top-5 evidence | Dense Recall@5 | Hybrid Recall@5 | Hybrid + rerank Recall@5 |
|---|---|---:|---:|---:|
| `remote-after-probation` | `employee-handbook-p3-c0` | 0.50 | 0.50 | 1.00 |
| `incident-evidence-retention` | `incident-response-p3-c0` | 0.50 | 0.50 | 1.00 |
| `candidate-data-rights` | `privacy-notice-p1-c0` | 0.50 | 1.00 | 1.00 |

The failure pattern was coverage, not first-hit ranking: dense MRR was already
1.00 because one relevant chunk appeared first. This is why MRR alone would
have hidden the multi-hop defect. Query decomposition increased candidate
coverage, RRF combined lexical and semantic evidence, and cross-encoder
reranking placed all labelled evidence within the final top five.

BM25 failed 28 of the 50 answerable cases at full Recall@5 and reached only 0.05
multi-hop Recall@5. Its main failure modes were paraphrases and Chinese queries
against an English corpus. This is expected for lexical matching and is the
reason BM25 is used as a complementary signal rather than the production mode.

## Latency diagnosis

The hybrid and reranked runs recorded p95 retrieval latencies of 17.984 s and
22.467 s. Per-query retrieval timing includes the OpenAI embedding request, and
the recorded runs experienced network variability. Cross-encoder p95 reranking
was 605.1 ms, which shows that the large end-to-end tail cannot be attributed to
reranking alone.

Recommended follow-up measurements are warm/cold separation, repeated runs,
local-only FAISS/BM25/reranker timers and a larger corpus. The current report
must not be used as a throughput or service-level claim.

## Grounding and citation risks

The graph reduces unsupported answers through evidence grading, one bounded
rewrite/retry and fail-closed citation validation. Remaining risks include:

- a grader can accept relevant-looking but incomplete evidence;
- a generated claim can overreach even when its citation identifier is valid;
- the four-case grounded smoke is too small to estimate production answer
  accuracy;
- an unanswerable question may retrieve semantically similar distractors;
- page extraction errors can preserve an identifier while degrading its text.

Mitigations already present are a canonical refusal, validated evidence IDs,
bounded retries and raw-text-free reports. A future evaluation should add
claim-level entailment labels and a larger adversarial unanswerable set before
claiming semantic citation correctness.

## Operational failure modes

| Failure | Current behavior | Remaining limitation |
|---|---|---|
| Invalid input | Structured 422 validation error | No client-specific schema versioning |
| Upstream model error | Retry in the OpenAI client, then retryable 502 | No circuit breaker or fallback model |
| Request timeout | Structured retryable 504 | A synchronous worker thread may finish in the background |
| Rate limit exceeded | Structured 429 with `Retry-After` | Counter is process-local |
| Invalid active index | `/ready` returns 503 | No automated repair job |
| Failed index rebuild | Previous immutable version remains active | Rebuild runs in the API process |
| Concurrent same-session requests | Serialized by a session lock | Locks are not distributed |
| Process restart | Documents/indexes reload from disk | Conversation checkpoints are lost |

## Security and scope gaps

Collection IDs constrain retrieval scope but do not constitute authorization.
The API has no identity provider, RBAC, durable audit log, malware scanning or
content moderation layer. It should be described as a deployable portfolio
service, not a production-ready multi-tenant platform.

Only text-based PDFs are supported. OCR, tables, images and layout-sensitive
documents remain outside the evaluated input domain.

## Claims that are not supported

Do not claim any of the following from the current evidence:

- routing accuracy;
- 100% answer accuracy over all 55 questions;
- semantic verification of every cited claim;
- durable conversation memory;
- distributed rate limiting or horizontal scalability;
- OCR support;
- production SLA, QPS or cost savings.


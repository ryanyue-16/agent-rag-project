# Agent RAG Project

An end-to-end enterprise knowledge-base Agentic RAG built with LangGraph,
FastAPI, OpenAI embeddings and hybrid retrieval. The current version loads
text-based PDF files, exposes a bounded single-agent workflow through CLI and
HTTP interfaces, maintains checkpointed per-session conversation state, and
includes offline retrieval and grounded-answer evaluation harnesses.

The service combines FAISS and BM25 retrieval, reciprocal-rank fusion,
multilingual cross-encoder reranking, grounded answer generation, validated
citations, versioned persistent indexes and production request controls. It is
intentionally a single-agent system rather than a multi-agent demonstration.

## Measured results

On the fixed manually labelled benchmark of 55 queries (50 answerable), the
accepted hybrid-reranked configuration produced:

| Metric | Dense baseline | Hybrid + rerank | Change |
|---|---:|---:|---:|
| Recall@5 | 0.97 | 1.00 | +3 percentage points |
| Multi-hop Recall@5 | 0.85 | 1.00 | +15 percentage points |
| MRR | 1.00 | 1.00 | no change |

The deterministic test suite contains 62 passing tests. See the evaluation
document for methodology, latency results and limitations; the four-case
grounded-response smoke is not presented as full-benchmark answer accuracy.

## Project documentation

- [Architecture](docs/architecture.md)
- [Evaluation and benchmark table](docs/evaluation.md)
- [Failure analysis](docs/failure-analysis.md)
- [Resume-ready project description](docs/resume.md)

## Current architecture

```text
                                      +-> FAISS dense --------+
Query -> deterministic decomposition |                       |
                                      +-> BM25 sparse --------+-> RRF
                                                                  |
                                                        Cross-Encoder rerank
                                                                  |
User -> CLI or FastAPI -> contextualize_query -> route_query
                                                |       |       |
                                           calculate  direct  retrieve
                                                              |
                                                       grade_evidence
                                                        |           |
                                                  sufficient   rewrite (max 1)
                                                        |           |
                                                 generate_answer <-+
                                                        |
                                               validate_citations
                                                        |
                                            answer or safe fallback
```

The graph is capped at 12 steps. Its `InMemorySaver` checkpointer isolates
conversation state by `collection_id + session_id`, preventing conversation
context from crossing collection boundaries. Conversation state is not durable
across process restarts yet; documents and indexes are.

Each collection stores immutable index versions containing `faiss.index`,
`bm25.json`, `chunks.json` and `manifest.json`. Rebuilds create a complete new
version before atomically replacing the `CURRENT` pointer, so readers continue
using the previous valid version if a rebuild fails.

## Requirements

- Python 3.12 or 3.13
- An OpenAI API key for real embedding and chat-model calls
- Docker and Docker Compose, optionally

Do not commit `.env`, local virtual environments, generated evaluation reports,
or documents containing confidential information.

## Local setup

Create a fresh environment for the operating system on which the project is
running. Virtual environments copied from another machine or operating system
are not portable.

```bash
python3 -m venv .venv-local
source .venv-local/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then add text-based PDF files under
`data/pdfs/`.

## Run the application

CLI:

```bash
python main.py
```

API:

```bash
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Example requests:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready

curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","collection_id":"default","metadata_filter":{"tags":["policy"]},"question":"文档的主要内容是什么？"}'

curl -X POST http://127.0.0.1:8000/collections \
  -H 'Content-Type: application/json' \
  -d '{"collection_id":"hr","name":"HR policies","metadata":{"region":"uk"}}'

curl -X POST http://127.0.0.1:8000/collections/hr/documents \
  -F 'file=@handbook.pdf' \
  -F 'tags=policy,uk' \
  -F 'metadata={"department":"people"}'

curl http://127.0.0.1:8000/collections/hr/documents

curl -X POST http://127.0.0.1:8000/collections/hr/rebuild

curl -X DELETE \
  http://127.0.0.1:8000/collections/hr/documents/DOCUMENT_ID

curl -X DELETE http://127.0.0.1:8000/sessions/demo
```

## Test

Tests use fake agents where possible and do not require an OpenAI API key:

```bash
python -m pytest
```

Run the same local quality gates used by CI:

```bash
python -m pip install -r requirements-dev.txt
ruff check .
mypy api app agents rag tools evals
pytest -q
```

## Offline evaluation

The default retrieval benchmark uses a fixed synthetic enterprise-policy corpus
with manually assigned gold chunk IDs. Retrieval-only evaluation calls the
embedding API but not the chat model:

```bash
python -m evals.run_eval --retrieval-mode dense --ks 1 3 5
python -m evals.run_eval --retrieval-mode bm25 --ks 1 3 5
python -m evals.run_eval --retrieval-mode hybrid --ks 1 3 5
python -m evals.run_eval --retrieval-mode hybrid_rerank --ks 1 3 5
python -m evals.ablation
```

Run the small real grounded-answer smoke evaluation:

```bash
python -m evals.run_grounded_eval
python -m evals.run_stage5_smoke
python -m evals.run_stage6_smoke
```

Generated JSON, CSV and Markdown reports are written to `evals/reports/` and
are ignored by Git. Raw
chunk text is intentionally excluded from generated reports to reduce the risk
of exposing document contents. `baseline_dense.json` is a sanitized snapshot of
the legacy dense-only baseline and is explicitly not a final benchmark. The
sanitized `dense_enterprise_v1_summary.json` file freezes the stage-1 baseline
used for future ablation comparisons. The sanitized
`stage2_ablation_v1_summary.json` records the accepted four-mode Stage 2 run.
`stage3_grounded_v1_summary.json` records the Stage 3 grounded-response smoke
evaluation without storing generated answers or raw evidence text.
`stage4_langgraph_v1_summary.json` records the accepted Stage 4 graph and
grounded-response smoke evaluation.
`stage5_enterprise_kb_v1_summary.json` records the accepted Stage 5 lifecycle,
persistence and real-PDF smoke checks.
`stage6_production_v1_summary.json` records the accepted Stage 6 production,
observability, concurrency and container checks without storing prompts,
answers or document text.

## Docker

```bash
docker compose up --build
```

The compose configuration mounts `data/pdfs` read-only, persists uploaded
documents and versioned indexes in the `knowledge_indexes` volume, and exposes
the API on port 8000. The image installs CPU-only PyTorch and runs one Uvicorn
worker because graph checkpoints, session locks and rate-limit counters are
currently process-local. Compose uses `/ready` for its container health check.

## Production controls

- Every response includes `X-Request-ID`; a valid inbound value is preserved.
- Application logs are one-line JSON and include request ID, status, latency
  and workflow trace fields where available. Uvicorn lifecycle lines retain
  Uvicorn's native format.
- `/health` is a liveness probe. `/ready` also verifies that all registered
  collections have valid active index manifests.
- Request timeout, OpenAI timeout/retries and a sliding-window rate limit are
  configurable through `.env`. Health probes bypass request limits.
- `/ask` returns route, retrieval/workflow latency, retry count, node count and
  aggregate token usage. Estimated cost is returned after the two per-million
  model rates are configured; otherwise it is `null`.
- API failures use a stable error envelope with code, message, request ID and
  retryability instead of exposing internal exceptions.

## Known limitations

- Query decomposition is deterministic; follow-up contextualization uses
  checkpointed conversation state.
- Authentication, authorization and durable audit storage are not implemented.
- Rate limiting, session locks and graph checkpoints are process-local. A
  multi-worker or multi-replica deployment needs shared external state.
- Request timeout responses cannot forcibly stop Python work already running in
  a worker thread; the per-session lock still prevents overlapping mutations
  for the same session.
- Estimated cost remains unavailable until the selected model's input/output
  rates are configured.
- Sessions and graph checkpoints live in process memory and do not survive a
  restart.
- Citation validation verifies that every returned source, page and chunk ID
  exists in evidence used during that request; semantic claim verification is
  not yet a separate model-based stage.
- Only text-based PDFs are supported; scanned documents require OCR, which is
  intentionally out of scope for v1.

These limitations are intentional baseline boundaries and must not be presented
as completed features in a resume or project description.

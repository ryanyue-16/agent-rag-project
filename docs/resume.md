# Resume-ready Project Description

## Recommended bullets

- Built a stateful enterprise knowledge-base RAG service with LangGraph and
  FastAPI, implementing conditional routing, bounded query rewrite/retry,
  collection-scoped retrieval, multi-turn checkpoints and validated page-level
  citations.
- Implemented FAISS and BM25 hybrid retrieval with reciprocal-rank fusion and a
  multilingual cross-encoder, improving Recall@5 from 97% to 100% and
  multi-hop Recall@5 from 85% to 100% on a manually labelled 55-query benchmark
  (50 answerable queries used for retrieval metrics).
- Developed a reproducible evaluation and observability pipeline covering four
  retrieval modes, grounded-response checks, p50/p95 latency, token usage,
  configurable cost estimation and failure diagnostics, backed by 62 automated
  tests and a CPU-only Docker Compose deployment.

## One-line project summary

Enterprise Agentic RAG service using LangGraph, FastAPI, OpenAI embeddings,
FAISS, BM25/RRF and cross-encoder reranking, with versioned knowledge-base
indexes, grounded citations, evaluation harnesses and production controls.

## Numbers and their provenance

| Resume value | Meaning | Source |
|---|---|---|
| 55 | Total manually labelled benchmark queries | Stage 2 summary |
| 50 | Answerable queries included in retrieval metrics | Stage 2 summary |
| 97% → 100% | Overall Recall@5, dense to hybrid-reranked | Stage 2 summary |
| 85% → 100% | Multi-hop Recall@5, dense to hybrid-reranked | Stage 2 summary |
| 4 | Retrieval modes compared | Stage 2 summary |
| 62 | Passing deterministic tests at Stage 6 acceptance | Stage 6 summary |

The four-case grounded-response smoke achieved a 100% pass and citation-valid
rate, but it is intentionally omitted from the recommended bullets because the
sample is too small for a broad quality claim. Routing accuracy and monetary
cost were not measured and must not be added as numbers.


from __future__ import annotations

import time
from statistics import mean, median
from typing import Any

from evals.dataset import EvalCase
from evals.metrics import (
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def _percentile(
    values: list[float],
    percentile: float,
) -> float:
    if not values:
        raise ValueError("values 不能为空")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile 必须在 0 到 100 之间")

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return (
        ordered[lower] * (1 - weight)
        + ordered[upper] * weight
    )


def evaluate_retrieval_case(
    rag_system: Any,
    case: EvalCase,
    ks: list[int],
) -> dict[str, Any]:
    max_k = max(ks)

    relevant_ids = set(case.gold_chunk_ids)
    corpus_ids = {
        str(chunk["chunk_id"])
        for chunk in rag_system.chunks
    }
    missing_gold = relevant_ids - corpus_ids
    if missing_gold:
        raise ValueError(
            f"case={case.case_id} 包含不存在的 gold chunk："
            f"{sorted(missing_gold)}"
        )

    started = time.perf_counter()
    retrieved = rag_system.retrieve(
        case.query,
        top_k=max_k,
    )
    retrieval_ms = (
        time.perf_counter() - started
    ) * 1000
    reranking_ms = float(
        getattr(
            rag_system,
            "last_retrieval_metrics",
            {},
        ).get("reranking_latency_ms", 0.0)
    )

    retrieved_ids = [
        str(item["chunk_id"])
        for item in retrieved
    ]

    by_k: dict[str, dict[str, float]] = {}
    if case.answerable:
        for k in ks:
            by_k[str(k)] = {
                "hit": hit_at_k(
                    retrieved_ids,
                    relevant_ids,
                    k,
                ),
                "precision": precision_at_k(
                    retrieved_ids,
                    relevant_ids,
                    k,
                ),
                "recall": recall_at_k(
                    retrieved_ids,
                    relevant_ids,
                    k,
                ),
            }

    return {
        "id": case.case_id,
        "query": case.query,
        "category": case.category,
        "answerable": case.answerable,
        "gold_chunk_ids": sorted(relevant_ids),
        "retrieved_chunk_ids": retrieved_ids,
        "mrr": (
            reciprocal_rank(
                retrieved_ids,
                relevant_ids,
            )
            if case.answerable
            else None
        ),
        "retrieval_latency_ms": retrieval_ms,
        "reranking_latency_ms": reranking_ms,
        "metrics": by_k,
        # Reports intentionally omit raw chunk text. Enterprise documents may
        # contain personal or confidential data; source identifiers and ranks
        # are sufficient for aggregate evaluation and failure triage.
        "retrieved": [
            {
                "chunk_id": str(item["chunk_id"]),
                "source": str(item["source"]),
                "page": int(item["page"]),
                "score": float(item["score"]),
                **{
                    key: item[key]
                    for key in (
                        "dense_score",
                        "dense_rank",
                        "bm25_score",
                        "bm25_rank",
                        "rrf_score",
                        "rerank_score",
                        "rerank_full_query_score",
                        "dense_query_rrf_score",
                        "dense_query_match_count",
                        "bm25_query_rrf_score",
                        "bm25_query_match_count",
                        "rerank_promoted",
                    )
                    if item.get(key) is not None
                },
            }
            for item in retrieved
        ],
    }


def aggregate_retrieval(
    results: list[dict[str, Any]],
    ks: list[int],
) -> dict[str, Any]:
    answerable_results = [
        item
        for item in results
        if item["answerable"]
    ]
    if not answerable_results:
        raise ValueError(
            "至少需要一条可回答评测样本"
        )

    latencies = [
        float(item["retrieval_latency_ms"])
        for item in results
    ]
    reranking_latencies = [
        float(item.get("reranking_latency_ms", 0.0))
        for item in results
    ]

    summary: dict[str, Any] = {
        "case_count": len(results),
        "answerable_case_count": len(answerable_results),
        "unanswerable_case_count": (
            len(results) - len(answerable_results)
        ),
        "mrr": mean(
            item["mrr"]
            for item in answerable_results
        ),
        "avg_retrieval_latency_ms": mean(latencies),
        "p50_retrieval_latency_ms": median(latencies),
        "p95_retrieval_latency_ms": _percentile(
            latencies,
            95,
        ),
        "avg_reranking_latency_ms": mean(reranking_latencies),
        "p50_reranking_latency_ms": median(reranking_latencies),
        "p95_reranking_latency_ms": _percentile(
            reranking_latencies,
            95,
        ),
        "by_k": {},
        "by_category": {},
    }

    for k in ks:
        key = str(k)
        summary["by_k"][key] = {
            "hit_rate": mean(
                item["metrics"][key]["hit"]
                for item in answerable_results
            ),
            "precision": mean(
                item["metrics"][key]["precision"]
                for item in answerable_results
            ),
            "recall": mean(
                item["metrics"][key]["recall"]
                for item in answerable_results
            ),
        }

    categories = sorted(
        {
            str(item["category"])
            for item in answerable_results
        }
    )
    for category in categories:
        category_results = [
            item
            for item in answerable_results
            if item["category"] == category
        ]
        summary["by_category"][category] = {
            "case_count": len(category_results),
            "mrr": mean(
                item["mrr"]
                for item in category_results
            ),
            "recall_at_max_k": mean(
                item["metrics"][str(max(ks))]["recall"]
                for item in category_results
            ),
        }

    return summary



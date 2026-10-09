from __future__ import annotations

from typing import Any


def fuse_query_results(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    score_key: str,
    rank_key: str,
    diagnostic_key: str,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """RRF multiple query variants from the same retrieval channel."""
    if rrf_k <= 0:
        raise ValueError("rrf_k 必须大于 0")
    if not ranked_lists:
        return []

    fused: dict[str, dict[str, Any]] = {}
    for results in ranked_lists:
        for fallback_rank, result in enumerate(results, start=1):
            chunk_id = str(result["chunk_id"])
            rank = int(result.get(rank_key, fallback_rank))
            item = fused.setdefault(
                chunk_id,
                {
                    "text": result["text"],
                    "page": result["page"],
                    "source": result["source"],
                    "chunk_id": result["chunk_id"],
                    score_key: float(result[score_key]),
                    diagnostic_key: 0.0,
                    "query_match_count": 0,
                    "metadata": dict(result.get("metadata", {})),
                    "document_id": result.get("document_id"),
                    "collection_id": result.get("collection_id"),
                },
            )
            item[score_key] = max(
                float(item[score_key]),
                float(result[score_key]),
            )
            item[diagnostic_key] += 1.0 / (rrf_k + rank)
            item["query_match_count"] += 1

    ranked = sorted(
        fused.values(),
        key=lambda item: (
            -float(item[diagnostic_key]),
            -float(item[score_key]),
            str(item["chunk_id"]),
        ),
    )
    for rank, item in enumerate(ranked, start=1):
        item[rank_key] = rank
        item["score"] = float(item[diagnostic_key])
    return ranked


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    *,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse ranked results by chunk ID while preserving diagnostics."""
    if rrf_k <= 0:
        raise ValueError("rrf_k 必须大于 0")

    fused: dict[str, dict[str, Any]] = {}
    for source, score_key, rank_key, query_score_key, count_key in (
        (
            dense_results,
            "dense_score",
            "dense_rank",
            "dense_query_rrf_score",
            "dense_query_match_count",
        ),
        (
            bm25_results,
            "bm25_score",
            "bm25_rank",
            "bm25_query_rrf_score",
            "bm25_query_match_count",
        ),
    ):
        for fallback_rank, result in enumerate(source, start=1):
            chunk_id = str(result["chunk_id"])
            rank = int(result.get(rank_key, fallback_rank))
            item = fused.setdefault(
                chunk_id,
                {
                    "text": result["text"],
                    "page": result["page"],
                    "source": result["source"],
                    "chunk_id": result["chunk_id"],
                    "dense_score": None,
                    "dense_rank": None,
                    "bm25_score": None,
                    "bm25_rank": None,
                    "rrf_score": 0.0,
                    "metadata": dict(result.get("metadata", {})),
                    "document_id": result.get("document_id"),
                    "collection_id": result.get("collection_id"),
                },
            )
            item[score_key] = float(
                result.get(score_key, result["score"])
            )
            item[rank_key] = rank
            if query_score_key in result:
                item[query_score_key] = float(result[query_score_key])
                item[count_key] = int(result["query_match_count"])
            item["rrf_score"] += 1.0 / (rrf_k + rank)

    ranked = sorted(
        fused.values(),
        key=lambda item: (
            -float(item["rrf_score"]),
            min(
                rank
                for rank in (
                    item["dense_rank"],
                    item["bm25_rank"],
                )
                if rank is not None
            ),
            str(item["chunk_id"]),
        ),
    )
    for item in ranked:
        item["score"] = float(item["rrf_score"])
    return ranked

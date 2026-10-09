from __future__ import annotations

from evals.ablation import MODES, build_ablation_summary


def _report(
    mode: str,
    *,
    recall: float,
    multi_hop_recall: float,
) -> dict[str, object]:
    return {
        "retrieval_mode": mode,
        "dataset": "cases.json",
        "corpus": "corpus.json",
        "ks": [1, 3, 5],
        "retrieval_candidate_k": 20,
        "rrf_k": 60,
        "embedding_model": "embedding",
        "reranker_model": "reranker",
        "retrieval_summary": {
            "case_count": 55,
            "mrr": 1.0,
            "by_k": {"5": {"recall": recall}},
            "by_category": {
                "multi_hop": {"recall_at_max_k": multi_hop_recall}
            },
            "p50_retrieval_latency_ms": 10.0,
            "p95_retrieval_latency_ms": 20.0,
            "p50_reranking_latency_ms": 0.0,
            "p95_reranking_latency_ms": 0.0,
        },
    }


def test_ablation_acceptance_requires_multi_hop_improvement() -> None:
    reports = [
        _report(
            mode,
            recall=0.97,
            multi_hop_recall=(0.9 if mode == "hybrid_rerank" else 0.85),
        )
        for mode in MODES
    ]

    summary = build_ablation_summary(reports)

    assert summary["acceptance"]["passed"] is True
    assert summary["acceptance"]["improved_modes"] == ["hybrid_rerank"]

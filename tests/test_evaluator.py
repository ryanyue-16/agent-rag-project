from evals.evaluator import (
    _percentile,
    aggregate_retrieval,
)


def test_percentile_uses_linear_interpolation():
    assert _percentile(
        [100.0, 200.0, 300.0],
        50,
    ) == 200.0


def test_aggregate_excludes_unanswerable_from_quality_metrics():
    answerable = {
        "answerable": True,
        "category": "exact",
        "mrr": 1.0,
        "retrieval_latency_ms": 100.0,
        "metrics": {
            "1": {
                "hit": 1.0,
                "precision": 1.0,
                "recall": 1.0,
            }
        },
    }
    unanswerable = {
        "answerable": False,
        "category": "unanswerable",
        "mrr": None,
        "retrieval_latency_ms": 300.0,
        "metrics": {},
    }

    summary = aggregate_retrieval(
        [answerable, unanswerable],
        [1],
    )

    assert summary["case_count"] == 2
    assert summary["answerable_case_count"] == 1
    assert summary["unanswerable_case_count"] == 1
    assert summary["mrr"] == 1.0
    assert summary["p50_retrieval_latency_ms"] == 200.0

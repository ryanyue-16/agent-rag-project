from evals.metrics import (
    answer_group_coverage,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_retrieval_metrics():
    retrieved = ["c3", "c1","c2", "c4"]
    relevent = {"c1", "c2"}

    assert hit_at_k(retrieved, relevent, 1) == 0.0
    assert hit_at_k(retrieved, relevent, 2) == 1.0
    assert precision_at_k(retrieved, relevent, 2) == 0.5
    assert recall_at_k(retrieved, relevent, 2) == 0.5
    assert recall_at_k(retrieved, relevent, 3) == 1.0
    assert reciprocal_rank(retrieved, relevent) == 0.5


def test_answer_group_coverage_accepts_alternatives():
    groups = (
        ("Imperial College London", "帝国理工"),
        ("Electronic and Information Engineering", "电子信息工程"),
    )

    answer = "他在帝国理工学习电子信息工程。"

    assert answer_group_coverage(answer, groups) == 1.0
    




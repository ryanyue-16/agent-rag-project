from __future__ import annotations


def _top_k(
    retrieved_ids: list[str],
    k: int,
) -> list[str]:
    if k <= 0:
        raise ValueError("k 必须大于 0")
    return retrieved_ids[:k]


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Top-K 中有多少比例是相关结果。"""
    top = _top_k(retrieved_ids, k)
    if not top:
        return 0.0
    relevant_count = sum(
        item_id in relevant_ids
        for item_id in top
    )
    return relevant_count / len(top)


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """所有相关结果中，有多少被 Top-K 找回。"""
    if not relevant_ids:
        raise ValueError("relevant_ids 不能为空")
    top = _top_k(retrieved_ids, k)
    found = set(top) & relevant_ids
    return len(found) / len(relevant_ids)


def hit_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Top-K 是否至少命中一个相关结果。"""
    top = _top_k(retrieved_ids, k)
    return float(
        any(item_id in relevant_ids for item_id in top)
    )


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """第一个相关结果排名的倒数；没有命中则为 0。"""
    for rank, item_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def answer_group_coverage(
    answer: str,
    expected_groups: tuple[tuple[str, ...], ...],
) -> float:
    """
    每个 group 表示一组可接受的同义表达。
    group 中任意一个词出现即认为该知识点命中。
    """
    if not expected_groups:
        return 1.0

    normalized = answer.casefold()
    matched = 0

    for group in expected_groups:
        if any(
            alternative.casefold() in normalized
            for alternative in group
        ):
            matched += 1

    return matched / len(expected_groups)

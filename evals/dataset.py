from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    """一条离线评估样本。"""

    case_id: str
    query: str
    gold_chunk_ids: tuple[str, ...]
    category: str
    answerable: bool = True
    expected_answer_groups: tuple[tuple[str, ...], ...] = ()


def _require_non_empty_string(
    item: dict[str, Any],
    key: str,
    case_index: int,
) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"第 {case_index + 1} 条评估样本的 {key} 必须是非空字符串"
        )
    return value.strip()


def _read_string_list(
    value: Any,
    *,
    field_name: str,
    case_index: int,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(
            f"第 {case_index + 1} 条评估样本的 {field_name} 必须是字符串数组"
        )

    cleaned: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(
                f"第 {case_index + 1} 条评估样本的 {field_name} 只能包含非空字符串"
            )
        cleaned.append(entry.strip())

    if not cleaned and not allow_empty:
        raise ValueError(
            f"第 {case_index + 1} 条评估样本的 {field_name} 不能为空"
        )

    return tuple(cleaned)


def _read_answer_groups(
    value: Any,
    *,
    case_index: int,
) -> tuple[tuple[str, ...], ...]:
    if value is None:
        return ()

    if not isinstance(value, list):
        raise ValueError(
            f"第 {case_index + 1} 条评估样本的 expected_answer_groups 必须是二维数组"
        )

    groups: list[tuple[str, ...]] = []
    for group_index, group in enumerate(value):
        if not isinstance(group, list):
            raise ValueError(
                f"第 {case_index + 1} 条样本的 expected_answer_groups "
                f"第 {group_index + 1} 组必须是字符串数组"
            )
        parsed = _read_string_list(
            group,
            field_name=(
                f"expected_answer_groups[{group_index}]"
            ),
            case_index=case_index,
        )
        groups.append(parsed)

    return tuple(groups)


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    """从 JSON 文件读取并校验评估集。"""
    case_path = Path(path)
    if not case_path.exists():
        raise FileNotFoundError(
            f"评估集不存在：{case_path}"
        )

    with case_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    if not isinstance(raw, list) or not raw:
        raise ValueError("评估集必须是非空 JSON 数组")

    cases: list[EvalCase] = []
    seen_ids: set[str] = set()

    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(
                f"第 {index + 1} 条评估样本必须是 JSON object"
            )

        case_id = _require_non_empty_string(
            item,
            "id",
            index,
        )
        if case_id in seen_ids:
            raise ValueError(
                f"评估样本 id 重复：{case_id}"
            )
        seen_ids.add(case_id)

        query = _require_non_empty_string(
            item,
            "query",
            index,
        )
        answerable = item.get(
            "answerable",
            True,
        )
        if not isinstance(answerable, bool):
            raise ValueError(
                f"第 {index + 1} 条评估样本的 answerable 必须是布尔值"
            )

        category = _require_non_empty_string(
            item,
            "category",
            index,
        )

        gold_chunk_ids = _read_string_list(
            item.get("gold_chunk_ids"),
            field_name="gold_chunk_ids",
            case_index=index,
            allow_empty=not answerable,
        )
        if not answerable and gold_chunk_ids:
            raise ValueError(
                f"第 {index + 1} 条不可回答样本的 gold_chunk_ids 必须为空"
            )
        answer_groups = _read_answer_groups(
            item.get("expected_answer_groups"),
            case_index=index,
        )

        cases.append(
            EvalCase(
                case_id=case_id,
                query=query,
                gold_chunk_ids=gold_chunk_ids,
                category=category,
                answerable=answerable,
                expected_answer_groups=answer_groups,
            )
        )

    return cases

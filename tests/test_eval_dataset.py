import json

import pytest

from evals.dataset import load_eval_cases


def test_load_eval_cases(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "school",
                    "query": "学校是什么",
                    "gold_chunk_ids": ["school-p1-c0"],
                    "category": "semantic",
                    "expected_answer_groups": [
                        ["Imperial College London", "帝国理工"]
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding = "utf-8",
    )

    cases = load_eval_cases(path)

    assert len(cases) == 1
    assert cases[0].case_id == "school"
    assert cases[0].gold_chunk_ids == (
        "school-p1-c0",
    )
    assert cases[0].category == "semantic"
    assert cases[0].answerable is True


def test_duplicate_case_id_is_rejected(tmp_path):
    path = tmp_path / "cases.json"
    payload = [
        {
            "id": "same",
            "query": "q1",
            "gold_chunk_ids": ["a"],
            "category": "exact",
        },
        {
            "id": "same",
            "query": "q2",
            "gold_chunk_ids": ["b"],
            "category": "exact",
        },
    ]
    path.write_text(
        json.dumps(payload),
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match="id 重复"):
        load_eval_cases(path)


def test_unanswerable_case_accepts_empty_gold(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "unknown",
                    "query": "文档没有的问题",
                    "gold_chunk_ids": [],
                    "category": "unanswerable",
                    "answerable": False,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    case = load_eval_cases(path)[0]

    assert case.answerable is False
    assert case.gold_chunk_ids == ()

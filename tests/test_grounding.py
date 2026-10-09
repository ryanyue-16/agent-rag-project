from __future__ import annotations

import json
from types import SimpleNamespace

from rag.grounding import (
    SAFE_UNANSWERABLE,
    SAFE_VALIDATION_FAILURE,
    build_search_result,
    normalize_agent_result,
)

EVIDENCE = {
    "query": "What is the leave allowance?",
    "evidence": [
        {
            "source": "handbook.pdf",
            "page": 2,
            "chunk_id": "handbook-p2-c0",
            "text": "Employees receive 25 days of annual leave.",
        }
    ],
}


def _knowledge_result(output: dict[str, object]) -> dict[str, object]:
    action = SimpleNamespace(tool="SearchKnowledgeBase")
    return {
        "output": json.dumps(output),
        "intermediate_steps": [(action, json.dumps(EVIDENCE))],
    }


def test_grounded_response_accepts_only_exact_evidence_citation() -> None:
    response = normalize_agent_result(
        _knowledge_result(
            {
                "answer": "员工有 25 天年假。",
                "grounded": True,
                "citations": [
                    {
                        "source": "handbook.pdf",
                        "page": 2,
                        "chunk_id": "handbook-p2-c0",
                    }
                ],
            }
        )
    )

    assert response.answer == "员工有 25 天年假。"
    assert response.citations == [
        {
            "source": "handbook.pdf",
            "page": 2,
            "chunk_id": "handbook-p2-c0",
        }
    ]


def test_forged_citation_fails_closed() -> None:
    response = normalize_agent_result(
        _knowledge_result(
            {
                "answer": "伪造回答",
                "grounded": True,
                "citations": [
                    {
                        "source": "handbook.pdf",
                        "page": 99,
                        "chunk_id": "invented",
                    }
                ],
            }
        )
    )

    assert response.answer == SAFE_VALIDATION_FAILURE
    assert response.citations == []


def test_unsupported_knowledge_answer_uses_canonical_refusal() -> None:
    response = normalize_agent_result(
        _knowledge_result(
            {
                "answer": "模型不应控制这段拒答文本",
                "grounded": False,
                "citations": [],
            }
        )
    )

    assert response.answer == SAFE_UNANSWERABLE
    assert response.citations == []


def test_search_payload_contains_retrieval_evidence_only() -> None:
    class RetrievalOnlyRAG:
        def retrieve(self, query: str, top_k: int):
            assert query == "annual leave"
            assert top_k == 3
            return [
                {
                    "source": "handbook.pdf",
                    "page": 2,
                    "chunk_id": "handbook-p2-c0",
                    "text": "25 days annual leave",
                    "score": 0.9,
                }
            ]

        def answer(self, query: str, top_k: int):
            raise AssertionError("Search tool must not call answer generation")

    rag = RetrievalOnlyRAG()
    payload = build_search_result(
        "annual leave",
        rag.retrieve("annual leave", top_k=3),
    )

    assert payload["evidence"] == [
        {
            "source": "handbook.pdf",
            "page": 2,
            "chunk_id": "handbook-p2-c0",
            "text": "25 days annual leave",
        }
    ]

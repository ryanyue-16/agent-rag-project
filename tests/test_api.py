from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from api.server import create_app
from api.session_store import SessionStore


class FakeAgent:
    """
    测试专用 Agent。

    不读取 PDF，不调用 OpenAI，
    只根据输入和历史返回固定结果。
    """

    def invoke(
        self,
        inputs: dict[str, Any],
    ) -> dict[str, str]:
        question = str(
            inputs["input"]
        )

        history = inputs.get(
            "chat_history",
            [],
        )

        history_text = "\n".join(
            str(
                getattr(
                    message,
                    "content",
                    "",
                )
            )
            for message in history
        )

        if question == (
            "我告诉你，他喜欢足球。"
        ):
            return {
                "output": "知道了。"
            }

        if question == (
            "他喜欢足球吗？"
        ):
            remembered = (
                "他喜欢足球"
                in history_text
            )

            return {
                "output": (
                    "喜欢。"
                    if remembered
                    else "不知道。"
                )
            }

        return {
            "output": (
                f"测试回答：{question}"
            )
        }


def build_test_app():
    """创建不访问 OpenAI 的测试应用。"""
    return create_app(
        workflow=FakeAgent(),
        session_store=SessionStore(
            max_turns=6,
        ),
    )


def test_health_api() -> None:
    app = build_test_app()

    with TestClient(app) as client:
        response = client.get(
            "/health"
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }


def test_same_session_keeps_memory() -> None:
    app = build_test_app()

    with TestClient(app) as client:
        first = client.post(
            "/ask",
            json={
                "session_id": "user-a",
                "question": (
                    "我告诉你，他喜欢足球。"
                ),
            },
        )

        second = client.post(
            "/ask",
            json={
                "session_id": "user-a",
                "question": (
                    "他喜欢足球吗？"
                ),
            },
        )

    assert first.status_code == 200
    assert first.json()["answer"] == (
        "知道了。"
    )

    assert second.status_code == 200
    assert second.json()["answer"] == (
        "喜欢。"
    )


def test_different_sessions_are_isolated() -> None:
    app = build_test_app()

    with TestClient(app) as client:
        client.post(
            "/ask",
            json={
                "session_id": "user-a",
                "question": (
                    "我告诉你，他喜欢足球。"
                ),
            },
        )

        response = client.post(
            "/ask",
            json={
                "session_id": "user-b",
                "question": (
                    "他喜欢足球吗？"
                ),
            },
        )

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "不知道。"
    )


def test_clear_session_removes_memory() -> None:
    app = build_test_app()

    with TestClient(app) as client:
        client.post(
            "/ask",
            json={
                "session_id": "user-a",
                "question": (
                    "我告诉你，他喜欢足球。"
                ),
            },
        )

        clear_response = client.delete(
            "/sessions/user-a"
        )

        ask_response = client.post(
            "/ask",
            json={
                "session_id": "user-a",
                "question": (
                    "他喜欢足球吗？"
                ),
            },
        )

    assert clear_response.status_code == 200
    assert clear_response.json() == {
        "session_id": "user-a",
        "cleared": True,
    }

    assert ask_response.status_code == 200
    assert ask_response.json()["answer"] == (
        "不知道。"
    )


def test_ask_rejects_blank_question() -> None:
    app = build_test_app()

    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={
                "session_id": "user-a",
                "question": "   ",
            },
        )

    assert response.status_code == 422


def test_api_returns_validated_citations() -> None:
    class GroundedFakeAgent:
        def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
            evidence = {
                "query": inputs["input"],
                "evidence": [
                    {
                        "source": "policy.pdf",
                        "page": 3,
                        "chunk_id": "policy-p3-c0",
                        "text": "The deadline is 30 days.",
                    }
                ],
            }
            output = {
                "answer": "期限是 30 天。",
                "grounded": True,
                "citations": [
                    {
                        "source": "policy.pdf",
                        "page": 3,
                        "chunk_id": "policy-p3-c0",
                    }
                ],
            }
            return {
                "output": json.dumps(output),
                "intermediate_steps": [
                    (
                        SimpleNamespace(tool="SearchKnowledgeBase"),
                        json.dumps(evidence),
                    )
                ],
            }

    app = create_app(
        workflow=GroundedFakeAgent(),
        session_store=SessionStore(max_turns=6),
    )
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={"session_id": "grounded", "question": "期限？"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "grounded"
    assert payload["answer"] == "期限是 30 天。"
    assert payload["citations"] == [
            {
                "source": "policy.pdf",
                "page": 3,
                "chunk_id": "policy-p3-c0",
            }
        ]
    assert payload["metrics"]["route"] == "unknown"


def test_api_passes_session_id_to_stateful_workflow_and_clears_it() -> None:
    class StatefulWorkflow:
        supports_thread_state = True

        def __init__(self) -> None:
            self.thread_ids: list[str] = []
            self.deleted: list[str] = []

        def invoke(
            self,
            inputs: dict[str, Any],
            config: dict[str, Any],
        ) -> dict[str, Any]:
            self.thread_ids.append(config["configurable"]["thread_id"])
            return {
                "answer": f"图回答：{inputs['input']}",
                "citations": [],
                "validated_response": True,
            }

        def delete_thread(self, thread_id: str) -> None:
            self.deleted.append(thread_id)

    workflow = StatefulWorkflow()
    app = create_app(
        workflow=workflow,
        session_store=SessionStore(max_turns=6),
    )
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={"session_id": "graph-a", "question": "你好"},
        )
        client.post(
            "/ask",
            json={
                "session_id": "graph-a",
                "collection_id": "other",
                "question": "另一个集合",
            },
        )
        cleared = client.delete("/sessions/graph-a")

    assert response.status_code == 200
    assert response.json()["answer"] == "图回答：你好"
    assert cleared.json()["cleared"] is True
    assert workflow.thread_ids == ["default:graph-a", "other:graph-a"]
    assert "default:graph-a" in workflow.deleted
    assert "other:graph-a" in workflow.deleted

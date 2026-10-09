from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from agents.langgraph_workflow import AgenticRAGWorkflow
from rag.grounding import SAFE_UNANSWERABLE

EVIDENCE = {
    "source": "handbook.pdf",
    "page": 2,
    "chunk_id": "handbook-p2-c0",
    "text": "Employees receive 25 days of annual leave.",
    "score": 0.9,
}


class FakeRAG:
    def __init__(self, results: list[list[dict[str, Any]]]) -> None:
        self.results = list(results)
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        assert top_k == 3
        self.queries.append(query)
        return self.results.pop(0) if self.results else []


def workflow(
    rag: FakeRAG,
    *,
    route: str = "knowledge",
    grades: list[bool] | None = None,
    answer: dict[str, Any] | None = None,
    context_calls: list[Any] | None = None,
) -> AgenticRAGWorkflow:
    grade_values = list(grades or [True])

    def contextualize(payload: Any) -> dict[str, str]:
        if context_calls is not None:
            context_calls.append(payload)
        return {"query": "standalone follow-up"}

    def grade(_: Any) -> dict[str, bool]:
        return {"sufficient": grade_values.pop(0)}

    return AgenticRAGWorkflow(
        rag_system=rag,
        top_k=3,
        document_names=["handbook.pdf"],
        max_retries=1,
        max_steps=12,
        contextualizer=contextualize,
        router=lambda _: {"route": route},
        grader=grade,
        rewriter=lambda _: {"query": "annual leave entitlement"},
        answer_generator=lambda _: answer
        or {
            "answer": "员工有 25 天年假。",
            "grounded": True,
            "citations": [
                {
                    "source": "handbook.pdf",
                    "page": 2,
                    "chunk_id": "handbook-p2-c0",
                }
            ],
        },
        direct_generator=lambda _: AIMessage(content="你好！"),
    )


def invoke(graph: AgenticRAGWorkflow, query: str, thread: str = "t1"):
    return graph.invoke(
        {"input": query},
        config={"configurable": {"thread_id": thread}},
    )


def test_graph_contains_exact_required_nodes() -> None:
    graph = workflow(FakeRAG([[EVIDENCE]]))
    assert graph.node_names == {
        "contextualize_query",
        "route_query",
        "retrieve",
        "grade_evidence",
        "rewrite_query",
        "generate_answer",
        "validate_citations",
        "calculate",
        "direct_response",
        "fallback",
    }


def test_grounded_path_returns_only_validated_citations() -> None:
    result = invoke(workflow(FakeRAG([[EVIDENCE]])), "年假有多少天？")
    assert result["answer"] == "员工有 25 天年假。"
    assert result["citations"][0]["chunk_id"] == "handbook-p2-c0"
    assert result["node_trace"] == [
        "contextualize_query",
        "route_query",
        "retrieve",
        "grade_evidence",
        "generate_answer",
        "validate_citations",
    ]


def test_insufficient_evidence_rewrites_once_then_succeeds() -> None:
    rag = FakeRAG([[], [EVIDENCE]])
    result = invoke(workflow(rag, grades=[True]), "年假？")
    assert rag.queries == ["年假？", "annual leave entitlement"]
    assert result["retry_count"] == 1
    assert result["answer"] == "员工有 25 天年假。"


def test_retry_cap_falls_back_without_unbounded_loop() -> None:
    rag = FakeRAG([[], []])
    result = invoke(workflow(rag), "不存在的信息？")
    assert len(rag.queries) == 2
    assert result["retry_count"] == 1
    assert result["answer"] == SAFE_UNANSWERABLE
    assert result["node_trace"][-1] == "fallback"


def test_forged_citation_routes_to_fallback() -> None:
    bad_answer = {
        "answer": "伪造回答",
        "grounded": True,
        "citations": [
            {"source": "handbook.pdf", "page": 99, "chunk_id": "fake"}
        ],
    }
    result = invoke(
        workflow(FakeRAG([[EVIDENCE]]), answer=bad_answer), "年假？"
    )
    assert result["answer"] == SAFE_UNANSWERABLE
    assert result["citations"] == []
    assert result["validation_error"] == "citation_validation_failed"
    assert result["node_trace"][-2:] == ["validate_citations", "fallback"]


def test_calculator_and_direct_routes_skip_retrieval() -> None:
    calculator_rag = FakeRAG([])
    calculated = invoke(workflow(calculator_rag), "(12 + 8) * 3")
    assert calculated["answer"] == "60"
    assert calculated["node_trace"][-1] == "calculate"
    assert calculator_rag.queries == []

    direct_rag = FakeRAG([])
    direct = invoke(workflow(direct_rag, route="direct"), "你好")
    assert direct["answer"] == "你好！"
    assert direct["node_trace"][-1] == "direct_response"
    assert direct_rag.queries == []


def test_checkpointer_keeps_threads_isolated_and_supports_clear() -> None:
    calls: list[Any] = []
    graph = workflow(FakeRAG([]), route="direct", context_calls=calls)
    invoke(graph, "第一问", "session-a")
    invoke(graph, "追问", "session-a")
    invoke(graph, "另一个会话", "session-b")

    assert len(calls) == 1
    graph.delete_thread("session-a")
    invoke(graph, "清空后", "session-a")
    assert len(calls) == 1


def test_retrieve_forwards_collection_scope_and_metadata_filter() -> None:
    class ScopedRAG:
        def __init__(self) -> None:
            self.call: dict[str, Any] = {}

        def retrieve_scoped(self, query: str, **kwargs):
            self.call = {"query": query, **kwargs}
            return [EVIDENCE]

    rag = ScopedRAG()
    graph = AgenticRAGWorkflow(
        rag_system=rag,
        top_k=3,
        document_names=["handbook.pdf"],
        contextualizer=lambda _: {"query": "standalone"},
        router=lambda _: {"route": "knowledge"},
        grader=lambda _: {"sufficient": True},
        rewriter=lambda _: {"query": "rewrite"},
        answer_generator=lambda _: {
            "answer": "25 days",
            "grounded": True,
            "citations": [
                {
                    "source": "handbook.pdf",
                    "page": 2,
                    "chunk_id": "handbook-p2-c0",
                }
            ],
        },
        direct_generator=lambda _: AIMessage(content="direct"),
    )
    graph.invoke(
        {
            "input": "leave?",
            "collection_id": "hr",
            "metadata_filter": {"tags": ["uk"]},
        },
        config={"configurable": {"thread_id": "hr:s1"}},
    )

    assert rag.call == {
        "query": "leave?",
        "top_k": 3,
        "collection_id": "hr",
        "metadata_filter": {"tags": ["uk"]},
    }

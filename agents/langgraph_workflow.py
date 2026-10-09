from __future__ import annotations

import logging
import re
import time
from typing import Annotated, Any, Callable, Literal, TypedDict, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.observability import UsageMetricsCallback
from rag.grounding import (
    SAFE_UNANSWERABLE,
    SAFE_VALIDATION_FAILURE,
    build_search_result,
    validate_grounded_payload,
)
from tools.calculator import calculator

logger = logging.getLogger(__name__)


class Citation(BaseModel):
    source: str
    page: int
    chunk_id: str


class RouteDecision(BaseModel):
    route: Literal["knowledge", "calculator", "direct"]


class EvidenceGrade(BaseModel):
    sufficient: bool
    reason: str = ""


class RewrittenQuery(BaseModel):
    query: str


class StandaloneQuery(BaseModel):
    query: str


class GroundedAnswer(BaseModel):
    answer: str
    grounded: bool
    citations: list[Citation] = Field(default_factory=list)


class WorkflowState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    contextualized_query: str
    route: Literal["knowledge", "calculator", "direct"]
    evidence: list[dict[str, Any]]
    retry_count: int
    candidate_answer: dict[str, Any]
    answer: str
    citations: list[dict[str, Any]]
    grounded: bool
    validation_error: str
    validation_passed: bool
    validated_response: bool
    node_trace: list[str]
    collection_id: str
    metadata_filter: dict[str, Any]
    retrieval_latency_ms: float


Dependency = Callable[..., Any]


class AgenticRAGWorkflow:
    """Single-agent LangGraph workflow with bounded recovery and citations."""

    node_names = {
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
    supports_thread_state = True

    def __init__(
        self,
        *,
        rag_system: Any,
        top_k: int,
        document_names: list[str],
        llm: Any | None = None,
        max_retries: int = 1,
        max_steps: int = 12,
        memory_max_turns: int = 6,
        contextualizer: Dependency | None = None,
        router: Dependency | None = None,
        grader: Dependency | None = None,
        rewriter: Dependency | None = None,
        answer_generator: Dependency | None = None,
        direct_generator: Dependency | None = None,
        checkpointer: Any | None = None,
        default_collection_id: str = "default",
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries 不能小于 0")
        if max_steps <= 0 or memory_max_turns <= 0:
            raise ValueError("max_steps 和 memory_max_turns 必须大于 0")

        self.rag_system = rag_system
        self.top_k = top_k
        self.document_names = document_names
        self.max_retries = max_retries
        self.max_steps = max_steps
        self.memory_max_turns = memory_max_turns
        self.checkpointer = checkpointer or InMemorySaver()
        self.default_collection_id = default_collection_id
        self.knowledge_base: Any | None = None
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million

        self.contextualizer = contextualizer or self._structured(
            llm, StandaloneQuery
        )
        self.router = router or self._structured(llm, RouteDecision)
        self.grader = grader or self._structured(llm, EvidenceGrade)
        self.rewriter = rewriter or self._structured(llm, RewrittenQuery)
        self.answer_generator = answer_generator or self._structured(
            llm, GroundedAnswer
        )
        self.direct_generator = direct_generator or llm

        if any(
            dependency is None
            for dependency in (
                self.contextualizer,
                self.router,
                self.grader,
                self.rewriter,
                self.answer_generator,
                self.direct_generator,
            )
        ):
            raise ValueError("必须提供 llm 或全部节点依赖")

        builder = StateGraph(WorkflowState)
        builder.add_node("contextualize_query", self._contextualize_query)
        builder.add_node("route_query", self._route_query)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("grade_evidence", self._grade_evidence)
        builder.add_node("rewrite_query", self._rewrite_query)
        builder.add_node("generate_answer", self._generate_answer)
        builder.add_node("validate_citations", self._validate_citations)
        builder.add_node("calculate", self._calculate)
        builder.add_node("direct_response", self._direct_response)
        builder.add_node("fallback", self._fallback)

        builder.add_edge(START, "contextualize_query")
        builder.add_edge("contextualize_query", "route_query")
        builder.add_conditional_edges(
            "route_query",
            lambda state: state["route"],
            {
                "knowledge": "retrieve",
                "calculator": "calculate",
                "direct": "direct_response",
            },
        )
        builder.add_edge("retrieve", "grade_evidence")
        builder.add_conditional_edges(
            "grade_evidence",
            self._after_grade,
            {
                "answer": "generate_answer",
                "rewrite": "rewrite_query",
                "fallback": "fallback",
            },
        )
        builder.add_edge("rewrite_query", "retrieve")
        builder.add_edge("generate_answer", "validate_citations")
        builder.add_conditional_edges(
            "validate_citations",
            lambda state: "done" if state["validation_passed"] else "fallback",
            {"done": END, "fallback": "fallback"},
        )
        builder.add_edge("calculate", END)
        builder.add_edge("direct_response", END)
        builder.add_edge("fallback", END)
        self.graph = builder.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _structured(llm: Any | None, schema: type[BaseModel]) -> Any | None:
        return None if llm is None else llm.with_structured_output(schema)

    @staticmethod
    def _value(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, BaseModel):
            return getattr(value, name, default)
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    @staticmethod
    def _call(dependency: Any, payload: Any) -> Any:
        if hasattr(dependency, "invoke"):
            return dependency.invoke(payload)
        return dependency(payload)

    @staticmethod
    def _trace(state: WorkflowState, node: str) -> list[str]:
        return [*state.get("node_trace", []), node]

    def _history(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        return messages[-(self.memory_max_turns * 2 + 1) :]

    def _contextualize_query(self, state: WorkflowState) -> dict[str, Any]:
        messages = state.get("messages", [])
        current = next(
            (
                str(message.content)
                for message in reversed(messages)
                if isinstance(message, HumanMessage)
            ),
            "",
        ).strip()
        prior = messages[:-1]
        contextualized = current
        if prior:
            prompt = [
                (
                    "system",
                    "结合对话历史，把最后一个用户问题改写为可独立理解的检索问题。"
                    "只改写，不回答；不要补充历史中不存在的事实。",
                ),
                *self._history(messages),
            ]
            result = self._call(self.contextualizer, prompt)
            contextualized = str(self._value(result, "query", current)).strip() or current

        return {
            "query": current,
            "contextualized_query": contextualized,
            "evidence": [],
            "retry_count": 0,
            "candidate_answer": {},
            "answer": "",
            "citations": [],
            "grounded": False,
            "validation_error": "",
            "validation_passed": False,
            "validated_response": False,
            "node_trace": ["contextualize_query"],
        }

    @staticmethod
    def _looks_like_calculation(query: str) -> bool:
        clean = query.strip().rstrip("=?？ ")
        return bool(clean) and bool(
            re.fullmatch(r"[\d\s.()+\-*/%]+", clean)
        )

    def _route_query(self, state: WorkflowState) -> dict[str, Any]:
        query = state["contextualized_query"]
        if self._looks_like_calculation(query):
            route = "calculator"
        else:
            prompt = [
                (
                    "system",
                    "把请求路由为 knowledge、calculator 或 direct。"
                    "涉及本地文档、制度、政策、项目资料的事实问题选 knowledge；"
                    "纯算术选 calculator；无需文档的一般对话选 direct。"
                    f"可用文档：{', '.join(self.document_names) or '未命名文档'}。",
                ),
                ("human", query),
            ]
            result = self._call(self.router, prompt)
            route = str(self._value(result, "route", "knowledge"))
            if route not in {"knowledge", "calculator", "direct"}:
                route = "knowledge"
        return {"route": route, "node_trace": self._trace(state, "route_query")}

    def _retrieve(self, state: WorkflowState) -> dict[str, Any]:
        query = state["contextualized_query"]
        started = time.perf_counter()
        if hasattr(self.rag_system, "retrieve_scoped"):
            retrieved = self.rag_system.retrieve_scoped(
                query,
                top_k=self.top_k,
                collection_id=state.get(
                    "collection_id", self.default_collection_id
                ),
                metadata_filter=state.get("metadata_filter") or None,
            )
        else:
            retrieved = self.rag_system.retrieve(query, top_k=self.top_k)
        evidence = build_search_result(query, retrieved)["evidence"]
        return {
            "evidence": evidence,
            "retrieval_latency_ms": (time.perf_counter() - started) * 1000,
            "node_trace": self._trace(state, "retrieve"),
        }

    def _grade_evidence(self, state: WorkflowState) -> dict[str, Any]:
        evidence = state.get("evidence", [])
        if not evidence:
            sufficient = False
        else:
            prompt = [
                (
                    "system",
                    "判断证据是否足以完整回答问题。只有所有关键结论都能由证据明确支持时才为 true。",
                ),
                (
                    "human",
                    f"问题：{state['contextualized_query']}\n证据：{evidence}",
                ),
            ]
            result = self._call(self.grader, prompt)
            sufficient = bool(self._value(result, "sufficient", False))
        return {
            "grounded": sufficient,
            "node_trace": self._trace(state, "grade_evidence"),
        }

    def _after_grade(self, state: WorkflowState) -> str:
        if state.get("grounded"):
            return "answer"
        if state.get("retry_count", 0) < self.max_retries:
            return "rewrite"
        return "fallback"

    def _rewrite_query(self, state: WorkflowState) -> dict[str, Any]:
        prompt = [
            (
                "system",
                "根据原问题和不足的证据，改写为更适合文档检索的查询。只输出改写查询，不回答。",
            ),
            (
                "human",
                f"原问题：{state['query']}\n当前查询：{state['contextualized_query']}\n"
                f"证据：{state.get('evidence', [])}",
            ),
        ]
        result = self._call(self.rewriter, prompt)
        rewritten = str(
            self._value(result, "query", state["contextualized_query"])
        ).strip()
        return {
            "contextualized_query": rewritten or state["contextualized_query"],
            "retry_count": state.get("retry_count", 0) + 1,
            "node_trace": self._trace(state, "rewrite_query"),
        }

    def _generate_answer(self, state: WorkflowState) -> dict[str, Any]:
        prompt = [
            (
                "system",
                "只能依据给定证据回答。每个事实结论必须引用证据中完全一致的 "
                "source、page、chunk_id。必须完整回答问题中的每个子问题；数字、期限、"
                "专有名词和角色名称要保留证据原文，使用其他语言回答时在译文后用括号"
                "保留原文术语。证据不足时 grounded=false 且 citations=[]。",
            ),
            (
                "human",
                f"问题：{state['query']}\n证据：{state['evidence']}",
            ),
        ]
        result = self._call(self.answer_generator, prompt)
        if isinstance(result, BaseModel):
            candidate = result.model_dump()
        elif isinstance(result, dict):
            candidate = result
        else:
            candidate = {}
        return {
            "candidate_answer": candidate,
            "node_trace": self._trace(state, "generate_answer"),
        }

    def _validate_citations(self, state: WorkflowState) -> dict[str, Any]:
        response = validate_grounded_payload(
            state.get("candidate_answer", {}),
            state.get("evidence", []),
            knowledge_used=True,
        )
        passed = response.answer != SAFE_VALIDATION_FAILURE
        updates: dict[str, Any] = {
            "answer": response.answer,
            "citations": response.citations,
            "validation_passed": passed,
            "validated_response": passed,
            "validation_error": "" if passed else "citation_validation_failed",
            "node_trace": self._trace(state, "validate_citations"),
        }
        if passed:
            updates["messages"] = [AIMessage(content=response.answer)]
        return updates

    @staticmethod
    def _extract_expression(query: str) -> str:
        matches = re.findall(r"[\d\s.()+\-*/%]+", query)
        return max(matches, key=len).strip() if matches else query.strip()

    def _calculate(self, state: WorkflowState) -> dict[str, Any]:
        answer = calculator(self._extract_expression(state["query"]))
        return {
            "answer": answer,
            "citations": [],
            "grounded": True,
            "validation_passed": True,
            "validated_response": True,
            "messages": [AIMessage(content=answer)],
            "node_trace": self._trace(state, "calculate"),
        }

    def _direct_response(self, state: WorkflowState) -> dict[str, Any]:
        prompt = [
            ("system", "简洁、准确地回答一般问题。"),
            *self._history(state.get("messages", [])),
        ]
        result = self._call(self.direct_generator, prompt)
        content = self._value(result, "content", result)
        answer = str(content).strip()
        return {
            "answer": answer,
            "citations": [],
            "grounded": True,
            "validation_passed": True,
            "validated_response": True,
            "messages": [AIMessage(content=answer)],
            "node_trace": self._trace(state, "direct_response"),
        }

    def _fallback(self, state: WorkflowState) -> dict[str, Any]:
        return {
            "answer": SAFE_UNANSWERABLE,
            "citations": [],
            "grounded": False,
            "validation_passed": True,
            "validated_response": True,
            "messages": [AIMessage(content=SAFE_UNANSWERABLE)],
            "node_trace": self._trace(state, "fallback"),
        }

    def invoke(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        question = str(inputs.get("input", "")).strip()
        if not question:
            raise ValueError("input 不能为空")
        runtime_config = dict(config or {})
        started = time.perf_counter()
        usage = UsageMetricsCallback(
            input_cost_per_million=self.input_cost_per_million,
            output_cost_per_million=self.output_cost_per_million,
        )
        callbacks = list(runtime_config.get("callbacks", []))
        callbacks.append(usage)
        runtime_config["callbacks"] = callbacks
        configurable = dict(runtime_config.get("configurable", {}))
        configurable.setdefault("thread_id", "default")
        runtime_config["configurable"] = configurable
        runtime_config.setdefault("recursion_limit", self.max_steps)
        graph_input: WorkflowState = {
            "messages": [HumanMessage(content=question)],
            "collection_id": str(
                inputs.get("collection_id", self.default_collection_id)
            ),
            "metadata_filter": dict(inputs.get("metadata_filter") or {}),
        }
        result = cast(
            dict[str, Any],
            self.graph.invoke(
                graph_input,
                cast(RunnableConfig, runtime_config),
            ),
        )
        result["metrics"] = {
            **usage.snapshot(),
            "workflow_latency_ms": round(
                (time.perf_counter() - started) * 1000, 3
            ),
            "retrieval_latency_ms": round(
                float(result.get("retrieval_latency_ms", 0.0)), 3
            ),
            "retry_count": int(result.get("retry_count", 0)),
            "route": str(result.get("route", "unknown")),
            "node_count": len(result.get("node_trace", [])),
        }
        return result

    def delete_thread(self, thread_id: str) -> None:
        self.checkpointer.delete_thread(thread_id)

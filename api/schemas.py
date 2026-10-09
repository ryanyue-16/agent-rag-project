from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class AskRequest(BaseModel):
    """POST /ask 的请求体。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    session_id: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "会话标识。相同 session_id "
            "共享多轮对话历史。"
        ),
        examples=["demo-user-001"],
    )

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="提交给 Agent 的问题。",
        examples=[
            "Yuhao Yue 的简历里写了什么？"
        ],
    )

    collection_id: str = Field(
        default="default",
        pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$",
        description="本次请求唯一允许访问的 collection。",
    )

    metadata_filter: dict[str, str | list[str]] = Field(
        default_factory=dict,
        description="可选 document metadata 精确过滤条件。",
    )


class Citation(BaseModel):
    """Validated source reference returned with a grounded answer."""

    source: str
    page: int = Field(gt=0)
    chunk_id: str


class TraceMetrics(BaseModel):
    workflow_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None
    retry_count: int = 0
    route: str = "unknown"
    node_count: int = 0


class AskResponse(BaseModel):
    """POST /ask 的成功响应。"""

    session_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    metrics: TraceMetrics = Field(default_factory=TraceMetrics)


class ClearSessionResponse(BaseModel):
    """清空指定会话后的响应。"""

    session_id: str
    cleared: bool


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str


class ReadyResponse(BaseModel):
    status: str
    workflow: bool
    knowledge_base: bool


class CollectionCreateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    collection_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    name: str = Field(min_length=1, max_length=200)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionResponse(BaseModel):
    collection_id: str
    name: str
    metadata: dict[str, Any]
    created_at: str
    document_count: int | None = None
    active_version: str | None = None


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    title: str
    tags: list[str]
    metadata: dict[str, Any]
    sha256: str
    page_count: int
    created_at: str


class DeleteDocumentResponse(BaseModel):
    collection_id: str
    document_id: str
    deleted: bool


class RebuildResponse(BaseModel):
    collection_id: str
    version_id: str
    document_count: int
    chunk_count: int

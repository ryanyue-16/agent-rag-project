from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Path,
    Request,
    UploadFile,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas import (
    AskRequest,
    AskResponse,
    ClearSessionResponse,
    CollectionCreateRequest,
    CollectionResponse,
    DeleteDocumentResponse,
    DocumentResponse,
    HealthResponse,
    ReadyResponse,
    RebuildResponse,
    TraceMetrics,
)
from api.schemas import (
    Citation as CitationSchema,
)
from api.session_store import SessionStore
from app.config import (
    LOG_LEVEL,
    MEMORY_MAX_TURNS,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
)
from app.errors import STATUS_FAILURES, FailureCode
from app.logging_config import request_id_context, setup_logging
from app.rate_limit import SlidingWindowRateLimiter
from rag.grounding import normalize_agent_result

logger = logging.getLogger(__name__)


def create_app(
    workflow: Any | None = None,
    session_store: SessionStore | None = None,
    rate_limiter: SlidingWindowRateLimiter | None = None,
    request_timeout_seconds: float | None = None,
) -> FastAPI:
    """
    创建 FastAPI 应用。

    测试可以注入 FakeAgent 和 SessionStore，
    避免读取 PDF 和调用 OpenAI。
    """

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ):
        setup_logging(LOG_LEVEL)

        if workflow is None:
            from app.application import (
                build_application,
            )

            logger.info(
                "FastAPI 服务启动，"
                "开始初始化 Agent"
            )

            executor = (
                build_application()
            )

            logger.info(
                "Agent 初始化完成，"
                "API 可以接收请求"
            )
        else:
            executor = workflow

        store = (
            session_store
            if session_store is not None
            else SessionStore(
                max_turns=(
                    MEMORY_MAX_TURNS
                ),
            )
        )

        app.state.workflow = executor
        app.state.session_store = store

        try:
            yield
        finally:
            if getattr(executor, "supports_thread_state", False):
                for thread_id in store.all_thread_ids():
                    executor.delete_thread(thread_id)
            store.clear_all()

            app.state.workflow = None
            app.state.session_store = None

            logger.info(
                "FastAPI 服务已关闭，"
                "进程内会话已清空"
            )

    app = FastAPI(
        title="Agent + RAG API",
        description=(
            "通过 HTTP JSON 接口调用 "
            "LangGraph Agent、Calculator "
            "和本地知识库检索，并返回经过验证的 citations。"
        ),
        version="1.3.0",
        lifespan=lifespan,
    )
    limiter = rate_limiter or SlidingWindowRateLimiter(
        limit=RATE_LIMIT_REQUESTS,
        window_seconds=RATE_LIMIT_WINDOW_SECONDS,
    )
    timeout_seconds = request_timeout_seconds or REQUEST_TIMEOUT_SECONDS

    def error_response(
        *,
        request: Request,
        status_code: int,
        code: FailureCode,
        message: str,
        retryable: bool,
        headers: Mapping[str, str] | None = None,
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "code": code.value,
                    "message": message,
                    "request_id": request_id,
                    "retryable": retryable,
                }
            },
            headers=headers,
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        code, retryable = STATUS_FAILURES.get(
            exc.status_code, (FailureCode.INTERNAL_ERROR, False)
        )
        return error_response(
            request=request,
            status_code=exc.status_code,
            code=code,
            message=str(exc.detail),
            retryable=retryable,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return error_response(
            request=request,
            status_code=422,
            code=FailureCode.VALIDATION_ERROR,
            message="请求参数校验失败",
            retryable=False,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception(
            "未处理的 API 异常",
            extra={
                "event": "request_failed",
                "failure_code": FailureCode.INTERNAL_ERROR.value,
                "retryable": False,
            },
        )
        return error_response(
            request=request,
            status_code=500,
            code=FailureCode.INTERNAL_ERROR,
            message="服务内部错误",
            retryable=False,
        )

    @app.middleware("http")
    async def production_controls(request: Request, call_next: Any):
        supplied = request.headers.get("X-Request-ID", "")
        request_id = (
            supplied
            if re.fullmatch(r"[A-Za-z0-9._-]{1,128}", supplied)
            else uuid.uuid4().hex
        )
        request.state.request_id = request_id
        token = request_id_context.set(request_id)
        started = time.perf_counter()
        response = None
        try:
            if request.url.path not in {"/health", "/ready"}:
                client = request.client.host if request.client else "unknown"
                allowed, retry_after = limiter.allow(client)
                if not allowed:
                    response = error_response(
                        request=request,
                        status_code=429,
                        code=FailureCode.RATE_LIMITED,
                        message="请求过于频繁",
                        retryable=True,
                        headers={"Retry-After": str(max(1, int(retry_after)))},
                    )
                else:
                    try:
                        response = await asyncio.wait_for(
                            call_next(request), timeout=timeout_seconds
                        )
                    except TimeoutError:
                        response = error_response(
                            request=request,
                            status_code=504,
                            code=FailureCode.TIMEOUT,
                            message="请求处理超时",
                            retryable=True,
                        )
            else:
                response = await call_next(request)
            assert response is not None
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "HTTP 请求完成",
                extra={
                    "event": "request_completed",
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": getattr(response, "status_code", 500),
                    "latency_ms": round(latency_ms, 3),
                },
            )
            request_id_context.reset(token)

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["system"],
    )
    def health() -> HealthResponse:
        """服务健康检查。"""
        return HealthResponse(
            status="ok",
        )

    @app.get(
        "/ready",
        response_model=ReadyResponse,
        tags=["system"],
    )
    def ready(request: Request) -> ReadyResponse:
        runtime = getattr(request.app.state, "workflow", None)
        workflow_ready = runtime is not None
        knowledge_base = getattr(runtime, "knowledge_base", None)
        knowledge_base_ready = (
            True
            if knowledge_base is None and workflow_ready
            else bool(knowledge_base and knowledge_base.is_ready())
        )
        if not workflow_ready or not knowledge_base_ready:
            raise HTTPException(status_code=503, detail="服务尚未就绪")
        return ReadyResponse(
            status="ready",
            workflow=True,
            knowledge_base=True,
        )

    @app.post(
        "/ask",
        response_model=AskResponse,
        tags=["agent"],
    )
    def ask(
        body: AskRequest,
        request: Request,
    ) -> AskResponse:
        """
        向 Agent 提问。

        LangGraph workflow.invoke() 是同步调用，
        所以本路由使用普通 def。
        """
        executor = getattr(
            request.app.state,
            "workflow",
            None,
        )

        store = getattr(
            request.app.state,
            "session_store",
            None,
        )

        if (
            executor is None
            or store is None
        ):
            raise HTTPException(
                status_code=(
                    status.HTTP_503_SERVICE_UNAVAILABLE
                ),
                detail=(
                    "Agent 尚未初始化完成"
                ),
            )

        session = store.get_or_create(
            body.session_id
        )

        knowledge_base = getattr(executor, "knowledge_base", None)
        if knowledge_base is not None:
            try:
                knowledge_base.get_collection(body.collection_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

        # 同一个 session_id 的请求串行处理。
        with session.lock:
            logger.info(
                "收到 API 问题："
                "session_id=%s "
                "question_length=%d "
                "memory_turn_count=%d",
                body.session_id,
                len(body.question),
                len(session.memory),
            )

            try:
                if getattr(executor, "supports_thread_state", False):
                    thread_id = f"{body.collection_id}:{body.session_id}"
                    store.register_thread(body.session_id, thread_id)
                    result = executor.invoke(
                        {
                            "input": body.question,
                            "collection_id": body.collection_id,
                            "metadata_filter": body.metadata_filter,
                        },
                        config={
                            "configurable": {"thread_id": thread_id}
                        },
                    )
                else:
                    result = executor.invoke(
                        {
                            "input": body.question,
                            "chat_history": session.memory.to_messages(),
                        }
                    )
            except Exception as exc:
                logger.exception(
                    "Agent API 调用失败："
                    "session_id=%s "
                    "question_length=%d",
                    body.session_id,
                    len(body.question),
                )

                error_name = type(exc).__name__.casefold()
                dependency_failure = any(
                    marker in error_name
                    for marker in ("openai", "connection", "timeout", "rate")
                )
                raise HTTPException(
                    status_code=502 if dependency_failure else 500,
                    detail=(
                        "上游模型服务暂时不可用"
                        if dependency_failure
                        else "处理问题时发生错误"
                    ),
                ) from exc

            grounded_response = normalize_agent_result(result)
            answer = grounded_response.answer

            if not answer:
                logger.error(
                    "Agent 返回结果中缺少 output："
                    "session_id=%s",
                    body.session_id,
                )

                raise HTTPException(
                    status_code=(
                        status.HTTP_500_INTERNAL_SERVER_ERROR
                    ),
                    detail=(
                        "Agent 没有返回有效答案"
                    ),
                )

            # Legacy injected agents still use SessionStore memory. LangGraph
            # persists messages through its checkpointer instead.
            if not getattr(executor, "supports_thread_state", False):
                session.memory.add_turn(user=body.question, assistant=answer)

        metrics = TraceMetrics.model_validate(result.get("metrics", {}))
        logger.info(
            "Agent 请求完成",
            extra={
                "event": "agent_completed",
                "session_id": body.session_id,
                "collection_id": body.collection_id,
                "route": metrics.route,
                "latency_ms": metrics.workflow_latency_ms,
                "input_tokens": metrics.input_tokens,
                "output_tokens": metrics.output_tokens,
                "estimated_cost_usd": metrics.estimated_cost_usd,
            },
        )
        return AskResponse(
            session_id=body.session_id,
            answer=answer,
            citations=[
                CitationSchema.model_validate(item)
                for item in grounded_response.citations
            ],
            metrics=metrics,
        )

    @app.delete(
        "/sessions/{session_id}",
        response_model=(
            ClearSessionResponse
        ),
        tags=["session"],
    )
    def clear_session(
        request: Request,
        session_id: str = Path(
            min_length=1,
            max_length=100,
            description=(
                "需要清空的会话标识。"
            ),
        ),
    ) -> ClearSessionResponse:
        """清空指定 API 会话的历史。"""
        store = getattr(
            request.app.state,
            "session_store",
            None,
        )

        if store is None:
            raise HTTPException(
                status_code=(
                    status.HTTP_503_SERVICE_UNAVAILABLE
                ),
                detail=(
                    "会话存储尚未初始化完成"
                ),
            )

        thread_ids = store.thread_ids(session_id)
        cleared = store.clear(
            session_id
        )

        executor = getattr(request.app.state, "workflow", None)
        if (
            cleared
            and executor is not None
            and getattr(executor, "supports_thread_state", False)
        ):
            for thread_id in thread_ids:
                executor.delete_thread(thread_id)

        logger.info(
            "清空 API 会话："
            "session_id=%s cleared=%s",
            session_id,
            cleared,
        )

        return ClearSessionResponse(
            session_id=session_id,
            cleared=cleared,
        )

    def require_knowledge_base(request: Request) -> Any:
        runtime = getattr(request.app.state, "workflow", None)
        knowledge_base = getattr(runtime, "knowledge_base", None)
        if knowledge_base is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="知识库管理服务尚未初始化",
            )
        return knowledge_base

    @app.post(
        "/collections",
        response_model=CollectionResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["knowledge-base"],
    )
    def create_collection(
        body: CollectionCreateRequest, request: Request
    ) -> CollectionResponse:
        service = require_knowledge_base(request)
        try:
            created = service.create_collection(
                body.collection_id,
                name=body.name,
                metadata=body.metadata,
            )
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return CollectionResponse(**service.get_collection(created["collection_id"]))

    @app.get(
        "/collections",
        response_model=list[CollectionResponse],
        tags=["knowledge-base"],
    )
    def list_collections(request: Request) -> list[CollectionResponse]:
        service = require_knowledge_base(request)
        return [CollectionResponse(**item) for item in service.list_collections()]

    @app.get(
        "/collections/{collection_id}/documents",
        response_model=list[DocumentResponse],
        tags=["knowledge-base"],
    )
    def list_documents(request: Request, collection_id: str) -> list[DocumentResponse]:
        service = require_knowledge_base(request)
        try:
            return [
                DocumentResponse(**item)
                for item in service.list_documents(collection_id)
            ]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post(
        "/collections/{collection_id}/documents",
        response_model=DocumentResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["knowledge-base"],
    )
    async def upload_document(
        request: Request,
        collection_id: str,
        file: Annotated[UploadFile, File()],
        title: Annotated[str | None, Form()] = None,
        tags: Annotated[str, Form()] = "",
        metadata: Annotated[str, Form()] = "{}",
    ) -> DocumentResponse:
        service = require_knowledge_base(request)
        content = await file.read()
        if len(content) > 25 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="PDF 不能超过 25 MB")
        try:
            parsed_metadata = json.loads(metadata)
            if not isinstance(parsed_metadata, dict):
                raise ValueError("metadata 必须是 JSON object")
            created = service.upload_document(
                collection_id,
                filename=file.filename or "document.pdf",
                content=content,
                title=title,
                tags=tags.split(","),
                metadata=parsed_metadata,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return DocumentResponse(**created)

    @app.get(
        "/collections/{collection_id}/documents/{document_id}",
        response_model=DocumentResponse,
        tags=["knowledge-base"],
    )
    def get_document(
        request: Request, collection_id: str, document_id: str
    ) -> DocumentResponse:
        service = require_knowledge_base(request)
        try:
            return DocumentResponse(
                **service.get_document(collection_id, document_id)
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete(
        "/collections/{collection_id}/documents/{document_id}",
        response_model=DeleteDocumentResponse,
        tags=["knowledge-base"],
    )
    def delete_document(
        request: Request, collection_id: str, document_id: str
    ) -> DeleteDocumentResponse:
        service = require_knowledge_base(request)
        try:
            deleted = service.delete_document(collection_id, document_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="document 不存在")
        return DeleteDocumentResponse(
            collection_id=collection_id,
            document_id=document_id,
            deleted=True,
        )

    @app.post(
        "/collections/{collection_id}/rebuild",
        response_model=RebuildResponse,
        tags=["knowledge-base"],
    )
    def rebuild_collection(
        request: Request, collection_id: str
    ) -> RebuildResponse:
        service = require_knowledge_base(request)
        try:
            manifest = service.rebuild(collection_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return RebuildResponse(**manifest)

    return app


app = create_app()

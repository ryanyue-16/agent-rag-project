from __future__ import annotations

import logging
from typing import Any

from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import SecretStr

from agents.langgraph_workflow import AgenticRAGWorkflow
from app.config import (
    CHAT_MODEL,
    DEFAULT_COLLECTION_ID,
    EMBED_MODEL,
    GRAPH_MAX_RETRIES,
    GRAPH_MAX_STEPS,
    INDEX_DIR,
    LLM_INPUT_COST_PER_MILLION,
    LLM_OUTPUT_COST_PER_MILLION,
    LOG_LEVEL,
    MEMORY_MAX_TURNS,
    OPENAI_API_KEY,
    OPENAI_MAX_RETRIES,
    OPENAI_TIMEOUT_SECONDS,
    PDF_DIR,
    RERANKER_MODEL,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_MODE,
    RRF_K,
    TOP_K,
    validate_config,
)
from app.logging_config import setup_logging
from rag.chunking import build_chunks
from rag.grounding import normalize_agent_result
from rag.knowledge_base import KnowledgeBaseService
from rag.loader import load_pdf_documents
from rag.qa_system import RAGQASystem

logger = logging.getLogger(__name__)


def build_rag_system_from_chunks(
    chunks: list[dict[str, Any]],
    *,
    retrieval_mode: str | None = None,
) -> RAGQASystem:
    """Build a configured retriever over an explicit chunk set."""
    validate_config()

    openai_client = OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )

    return RAGQASystem(
        chunks=chunks,
        client=openai_client,
        embed_model=EMBED_MODEL,
        chat_model=CHAT_MODEL,
        retrieval_mode=retrieval_mode or RETRIEVAL_MODE,
        candidate_k=RETRIEVAL_CANDIDATE_K,
        rrf_k=RRF_K,
        reranker_model=RERANKER_MODEL,
    )


def build_rag_system(
    *,
    retrieval_mode: str | None = None,
) -> tuple[RAGQASystem, list[str]]:
    """
    单独构建 RAG 系统，供 Agent 和离线评估共同复用。

    第 11 课增加这个函数后，评估代码不需要复制一套
    PDF 加载、chunking、Embedding 和 FAISS 初始化逻辑。
    """
    validate_config()

    logger.info(
        "RAG 开始初始化："
        "embed_model=%s pdf_dir=%s",
        EMBED_MODEL,
        PDF_DIR,
    )

    documents = load_pdf_documents(
        PDF_DIR
    )

    document_names = sorted(
        {
            str(document["source"])
            for document in documents
        }
    )

    chunks = build_chunks(
        documents,
        size=500,
        overlap=100,
    )

    rag_system = build_rag_system_from_chunks(
        chunks,
        retrieval_mode=retrieval_mode,
    )

    return rag_system, document_names


def build_workflow_for_rag_system(
    rag_system: Any,
    document_names: list[str],
) -> AgenticRAGWorkflow:
    """Build the bounded single-agent LangGraph workflow."""
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0,
        api_key=SecretStr(OPENAI_API_KEY),
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )

    return AgenticRAGWorkflow(
        rag_system=rag_system,
        top_k=TOP_K,
        document_names=document_names,
        llm=llm,
        max_retries=GRAPH_MAX_RETRIES,
        max_steps=GRAPH_MAX_STEPS,
        memory_max_turns=MEMORY_MAX_TURNS,
        input_cost_per_million=LLM_INPUT_COST_PER_MILLION,
        output_cost_per_million=LLM_OUTPUT_COST_PER_MILLION,
    )


def build_application():
    """
    组装 OpenAI Client、RAG 和 LangGraph 工作流。

    CLI 和 FastAPI 都复用这个函数。
    """
    validate_config()

    logger.info(
        "应用开始初始化："
        "chat_model=%s "
        "embed_model=%s "
        "pdf_dir=%s "
        "top_k=%d "
        "memory_max_turns=%d",
        CHAT_MODEL,
        EMBED_MODEL,
        PDF_DIR,
        TOP_K,
        MEMORY_MAX_TURNS,
    )

    openai_client = OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )
    knowledge_base = KnowledgeBaseService(
        root=INDEX_DIR,
        client=openai_client,
        embed_model=EMBED_MODEL,
        chat_model=CHAT_MODEL,
        retrieval_mode=RETRIEVAL_MODE,
        candidate_k=RETRIEVAL_CANDIDATE_K,
        rrf_k=RRF_K,
        reranker_model=RERANKER_MODEL,
    )
    collection_ids = {
        item["collection_id"] for item in knowledge_base.list_collections()
    }
    created_default = DEFAULT_COLLECTION_ID not in collection_ids
    if created_default:
        knowledge_base.create_collection(
            DEFAULT_COLLECTION_ID,
            name="Default knowledge base",
        )
    if created_default:
        knowledge_base.import_directory(DEFAULT_COLLECTION_ID, PDF_DIR)

    workflow = build_workflow_for_rag_system(
        knowledge_base,
        knowledge_base.document_names(DEFAULT_COLLECTION_ID),
    )
    workflow.default_collection_id = DEFAULT_COLLECTION_ID
    workflow.knowledge_base = knowledge_base
    return workflow


def run_cli(workflow: AgenticRAGWorkflow) -> None:
    """运行支持多轮对话的命令行问答循环。"""
    thread_id = "cli"

    print("\nAgent + RAG 已启动。")
    print("可以询问数学问题或 PDF 内容。")
    print(
        f"当前会话最多保留最近 "
        f"{MEMORY_MAX_TURNS} 轮对话。"
    )
    print("输入 /clear 清空当前会话记忆。")
    print("输入 exit 退出。\n")

    while True:
        user_input = input(
            "You: "
        ).strip()

        if user_input.lower() in {
            "exit",
            "quit",
            "退出",
        }:
            logger.info(
                "用户主动退出程序："
                "memory_turn_count=%d",
                0,
            )

            print("程序已退出。")
            break

        if user_input.lower() in {
            "/clear",
            "清空对话",
        }:
            workflow.delete_thread(thread_id)

            logger.info(
                "当前会话记忆已清空"
            )

            print(
                "\nAssistant: "
                "当前会话记忆已清空。\n"
            )
            continue

        if not user_input:
            continue

        logger.info(
            "开始处理用户问题："
            "question_length=%d "
            "memory_turn_count=%d",
            len(user_input),
            0,
        )

        try:
            result = workflow.invoke(
                {"input": user_input},
                config={"configurable": {"thread_id": thread_id}},
            )

            grounded_response = normalize_agent_result(result)
            answer = grounded_response.answer

            if not answer:
                raise ValueError(
                    "Agent 没有返回有效答案"
                )

            print(
                "\nAssistant:",
                answer,
                "\n",
            )
            if grounded_response.citations:
                print("Citations:")
                for citation in grounded_response.citations:
                    print(
                        "- "
                        f"{citation['source']} "
                        f"p{citation['page']} "
                        f"({citation['chunk_id']})"
                    )
                print()

        except Exception:
            logger.exception(
                "Agent 执行失败："
                "query_length=%d",
                len(user_input),
            )

            print(
                "\nAssistant: "
                "处理问题时发生错误，"
                "请查看日志。\n"
            )


def main() -> None:
    setup_logging(LOG_LEVEL)

    try:
        workflow = (
            build_application()
        )
    except Exception as exc:
        logger.exception(
            "应用初始化失败"
        )
        raise SystemExit(1) from exc

    run_cli(workflow)

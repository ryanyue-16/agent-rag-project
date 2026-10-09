from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 始终读取项目根目录中的 .env，
# 不依赖启动命令时所在的工作目录。
load_dotenv(PROJECT_ROOT / ".env")


OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "",
).strip()

CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "gpt-5-mini",
).strip()

EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "text-embedding-3-large",
).strip()

RETRIEVAL_MODE = os.getenv(
    "RETRIEVAL_MODE",
    "hybrid_rerank",
).strip()

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
).strip()

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO",
).upper().strip()


def _read_positive_int(
    name: str,
    default: str,
) -> int:
    """读取必须大于 0 的整数环境变量。"""
    raw_value = os.getenv(
        name,
        default,
    ).strip()

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} 必须是整数"
        ) from exc

    if value <= 0:
        raise ValueError(
            f"{name} 必须大于 0"
        )

    return value


def _read_non_negative_int(name: str, default: str) -> int:
    raw_value = os.getenv(name, default).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if value < 0:
        raise ValueError(f"{name} 不能小于 0")
    return value


def _read_non_negative_float(name: str, default: str) -> float:
    raw_value = os.getenv(name, default).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是数字") from exc
    if value < 0:
        raise ValueError(f"{name} 不能小于 0")
    return value


def _read_positive_float(name: str, default: str) -> float:
    value = _read_non_negative_float(name, default)
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0")
    return value


TOP_K = _read_positive_int(
    "TOP_K",
    "4",
)

MEMORY_MAX_TURNS = _read_positive_int(
    "MEMORY_MAX_TURNS",
    "6",
)

RETRIEVAL_CANDIDATE_K = _read_positive_int(
    "RETRIEVAL_CANDIDATE_K",
    "20",
)

RRF_K = _read_positive_int(
    "RRF_K",
    "60",
)

GRAPH_MAX_RETRIES = _read_positive_int(
    "GRAPH_MAX_RETRIES",
    "1",
)

GRAPH_MAX_STEPS = _read_positive_int(
    "GRAPH_MAX_STEPS",
    "12",
)

DEFAULT_COLLECTION_ID = os.getenv(
    "DEFAULT_COLLECTION_ID", "default"
).strip().lower()

REQUEST_TIMEOUT_SECONDS = _read_positive_float(
    "REQUEST_TIMEOUT_SECONDS", "60"
)
OPENAI_TIMEOUT_SECONDS = _read_positive_float(
    "OPENAI_TIMEOUT_SECONDS", "45"
)
OPENAI_MAX_RETRIES = _read_non_negative_int("OPENAI_MAX_RETRIES", "2")
RATE_LIMIT_REQUESTS = _read_positive_int("RATE_LIMIT_REQUESTS", "60")
RATE_LIMIT_WINDOW_SECONDS = _read_positive_float(
    "RATE_LIMIT_WINDOW_SECONDS", "60"
)
LLM_INPUT_COST_PER_MILLION = _read_non_negative_float(
    "LLM_INPUT_COST_PER_MILLION", "0"
)
LLM_OUTPUT_COST_PER_MILLION = _read_non_negative_float(
    "LLM_OUTPUT_COST_PER_MILLION", "0"
)


def _resolve_project_path(
    raw_path: str,
) -> Path:
    """
    将相对路径解析为以项目根目录为基准的绝对路径。
    """
    path = Path(raw_path).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


PDF_DIR = _resolve_project_path(
    os.getenv(
        "PDF_DIR",
        "data/pdfs",
    ).strip()
)

INDEX_DIR = _resolve_project_path(
    os.getenv("INDEX_DIR", "data/indexes").strip()
)


def validate_config() -> None:
    """在项目启动前检查配置。"""
    if not OPENAI_API_KEY:
        raise ValueError(
            "缺少 OPENAI_API_KEY。"
            "请将 .env.example 复制为 .env，"
            "然后填写真实 API key。"
        )

    valid_levels = {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }

    if LOG_LEVEL not in valid_levels:
        raise ValueError(
            "LOG_LEVEL 必须是 DEBUG、INFO、"
            "WARNING、ERROR 或 CRITICAL"
        )

    valid_retrieval_modes = {
        "dense",
        "bm25",
        "hybrid",
        "hybrid_rerank",
    }
    if RETRIEVAL_MODE not in valid_retrieval_modes:
        raise ValueError(
            "RETRIEVAL_MODE 必须是 dense、bm25、hybrid "
            "或 hybrid_rerank"
        )

    if not RERANKER_MODEL:
        raise ValueError("RERANKER_MODEL 不能为空")

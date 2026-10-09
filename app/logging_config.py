from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

request_id_context: ContextVar[str] = ContextVar(
    "request_id", default="background"
)


class JsonFormatter(logging.Formatter):
    """Small dependency-free JSON formatter for machine-searchable logs."""

    _fields = (
        "event",
        "failure_code",
        "latency_ms",
        "status_code",
        "method",
        "path",
        "session_id",
        "collection_id",
        "route",
        "input_tokens",
        "output_tokens",
        "estimated_cost_usd",
        "retryable",
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(
                record, "request_id", request_id_context.get()
            ),
        }
        for field in self._fields:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str) -> None:
    """Configure one structured JSON handler for the process."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level))
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

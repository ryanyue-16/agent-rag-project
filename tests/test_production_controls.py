from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from api.server import create_app
from api.session_store import SessionStore
from app.logging_config import JsonFormatter, request_id_context
from app.observability import UsageMetricsCallback
from app.rate_limit import SlidingWindowRateLimiter


class FastWorkflow:
    supports_thread_state = True

    def invoke(self, inputs: dict[str, Any], config: dict[str, Any]):
        return {
            "answer": "ok",
            "citations": [],
            "validated_response": True,
            "metrics": {
                "workflow_latency_ms": 12.5,
                "retrieval_latency_ms": 3.2,
                "input_tokens": 10,
                "output_tokens": 4,
                "total_tokens": 14,
                "estimated_cost_usd": 0.001,
                "retry_count": 0,
                "route": "knowledge",
                "node_count": 6,
            },
        }

    def delete_thread(self, thread_id: str) -> None:
        return None


def test_request_id_readiness_metrics_and_validation_taxonomy() -> None:
    app = create_app(
        workflow=FastWorkflow(), session_store=SessionStore(max_turns=2)
    )
    with TestClient(app) as client:
        ready = client.get("/ready", headers={"X-Request-ID": "req-123"})
        answer = client.post(
            "/ask",
            headers={"X-Request-ID": "req-456"},
            json={"session_id": "s1", "question": "hello"},
        )
        invalid = client.post(
            "/ask", json={"session_id": "s1", "question": " "}
        )

    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.headers["X-Request-ID"] == "req-123"
    assert answer.headers["X-Request-ID"] == "req-456"
    assert answer.json()["metrics"]["total_tokens"] == 14
    assert answer.json()["metrics"]["estimated_cost_usd"] == 0.001
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "validation_error"
    assert invalid.json()["error"]["request_id"]


def test_rate_limit_returns_retryable_failure() -> None:
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)
    app = create_app(
        workflow=FastWorkflow(),
        session_store=SessionStore(max_turns=2),
        rate_limiter=limiter,
    )
    with TestClient(app) as client:
        first = client.post(
            "/ask", json={"session_id": "s1", "question": "one"}
        )
        second = client.post(
            "/ask", json={"session_id": "s1", "question": "two"}
        )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "rate_limited"
    assert second.json()["error"]["retryable"] is True
    assert int(second.headers["Retry-After"]) >= 1


def test_request_timeout_has_failure_taxonomy() -> None:
    class SlowWorkflow(FastWorkflow):
        def invoke(self, inputs, config):
            time.sleep(0.08)
            return super().invoke(inputs, config)

    app = create_app(
        workflow=SlowWorkflow(),
        session_store=SessionStore(max_turns=2),
        request_timeout_seconds=0.02,
    )
    with TestClient(app) as client:
        response = client.post(
            "/ask", json={"session_id": "slow", "question": "wait"}
        )

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "timeout"
    assert response.json()["error"]["retryable"] is True


def test_concurrent_sessions_progress_while_each_session_is_serialized() -> None:
    class ConcurrentWorkflow(FastWorkflow):
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.active: dict[str, int] = {}
            self.max_by_thread: dict[str, int] = {}
            self.global_active = 0
            self.max_global = 0

        def invoke(self, inputs, config):
            thread_id = config["configurable"]["thread_id"]
            with self.lock:
                self.active[thread_id] = self.active.get(thread_id, 0) + 1
                self.max_by_thread[thread_id] = max(
                    self.max_by_thread.get(thread_id, 0),
                    self.active[thread_id],
                )
                self.global_active += 1
                self.max_global = max(self.max_global, self.global_active)
            time.sleep(0.05)
            with self.lock:
                self.active[thread_id] -= 1
                self.global_active -= 1
            return super().invoke(inputs, config)

    workflow = ConcurrentWorkflow()
    app = create_app(
        workflow=workflow,
        session_store=SessionStore(max_turns=2),
        rate_limiter=SlidingWindowRateLimiter(limit=100, window_seconds=60),
    )
    with TestClient(app) as client:
        def ask(session_id: str):
            return client.post(
                "/ask",
                json={"session_id": session_id, "question": session_id},
            )

        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(ask, ["same", "same", "a", "b"]))

    assert all(response.status_code == 200 for response in responses)
    assert workflow.max_by_thread["default:same"] == 1
    assert workflow.max_global >= 2


def test_json_logs_and_usage_aggregation() -> None:
    record = logging.LogRecord(
        "test", logging.INFO, __file__, 1, "done", (), None
    )
    record.event = "unit_test"
    token = request_id_context.set("trace-1")
    try:
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_context.reset(token)
    assert payload["request_id"] == "trace-1"
    assert payload["event"] == "unit_test"

    callback = UsageMetricsCallback(
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
    )
    callback.on_llm_end(
        SimpleNamespace(
            llm_output={
                "token_usage": {"prompt_tokens": 100, "completion_tokens": 50}
            },
            generations=[],
        )
    )
    metrics = callback.snapshot()
    assert metrics["total_tokens"] == 150
    assert metrics["estimated_cost_usd"] == 0.0002

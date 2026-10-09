from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.server import create_app
from app.application import build_application


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real Stage 6 readiness and tracing smoke checks."
    )
    parser.add_argument(
        "--output",
        default="evals/reports/stage6_observability_smoke.json",
    )
    args = parser.parse_args()

    workflow = build_application()
    app = create_app(workflow=workflow)
    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        answer = client.post(
            "/ask",
            headers={"X-Request-ID": "stage6-real-smoke"},
            json={
                "session_id": "stage6-smoke",
                "collection_id": "default",
                "question": "你好，请简短回复。",
            },
        )

    body = answer.json() if answer.status_code == 200 else {}
    metrics = body.get("metrics", {})
    checks = {
        "health_ok": health.status_code == 200,
        "ready_ok": ready.status_code == 200,
        "ask_ok": answer.status_code == 200,
        "request_id_propagated": (
            answer.headers.get("X-Request-ID") == "stage6-real-smoke"
        ),
        "workflow_latency_traced": metrics.get("workflow_latency_ms", 0) > 0,
        "token_usage_traced": metrics.get("total_tokens", 0) > 0,
        "route_traced": metrics.get("route") in {
            "knowledge",
            "calculator",
            "direct",
        },
    }
    payload = {
        "stage": 6,
        "status_code": answer.status_code,
        "route": metrics.get("route"),
        "input_tokens": metrics.get("input_tokens", 0),
        "output_tokens": metrics.get("output_tokens", 0),
        "total_tokens": metrics.get("total_tokens", 0),
        "workflow_latency_ms": metrics.get("workflow_latency_ms", 0),
        "estimated_cost_usd": metrics.get("estimated_cost_usd"),
        "checks": checks,
        "passed": all(checks.values()),
        "notes": [
            "No answer text, prompt text or document text is stored.",
            "Cost is null until per-model rates are configured in the environment.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

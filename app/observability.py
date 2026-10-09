from __future__ import annotations

from threading import Lock
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class UsageMetricsCallback(BaseCallbackHandler):
    """Aggregate token usage across every LLM node in one graph invocation."""

    def __init__(
        self,
        *,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
    ) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self._lock = Lock()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        usage = (response.llm_output or {}).get("token_usage", {})
        input_tokens = int(
            usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        )
        output_tokens = int(
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        )
        if not input_tokens and not output_tokens:
            try:
                metadata = response.generations[0][0].message.usage_metadata or {}
                input_tokens = int(metadata.get("input_tokens", 0) or 0)
                output_tokens = int(metadata.get("output_tokens", 0) or 0)
            except (AttributeError, IndexError, TypeError):
                pass
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    def snapshot(self) -> dict[str, Any]:
        total = self.input_tokens + self.output_tokens
        configured = bool(
            self.input_cost_per_million or self.output_cost_per_million
        )
        cost = None
        if configured:
            cost = round(
                self.input_tokens * self.input_cost_per_million / 1_000_000
                + self.output_tokens
                * self.output_cost_per_million
                / 1_000_000,
                8,
            )
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": total,
            "estimated_cost_usd": cost,
        }

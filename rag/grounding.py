from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

SAFE_UNANSWERABLE = "文档中未提及相关信息。"
SAFE_VALIDATION_FAILURE = "无法根据检索证据生成可验证的回答。"


@dataclass(frozen=True)
class GroundedResponse:
    answer: str
    citations: list[dict[str, Any]]


def build_search_result(
    query: str,
    retrieved: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create the only document payload exposed to the final generator."""
    return {
        "query": query,
        "evidence": [
            {
                "chunk_id": str(item["chunk_id"]),
                "source": str(item["source"]),
                "page": int(item["page"]),
                "text": str(item["text"]),
            }
            for item in retrieved
        ],
    }


def serialize_search_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _parse_json_object(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_search_evidence(
    result: dict[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    """Extract evidence from legacy injected tool-step results."""
    knowledge_used = False
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()

    for step in result.get("intermediate_steps", []):
        if not isinstance(step, (tuple, list)) or len(step) != 2:
            continue
        action, observation = step
        if getattr(action, "tool", None) != "SearchKnowledgeBase":
            continue
        knowledge_used = True
        payload = _parse_json_object(observation)
        if payload is None or not isinstance(payload.get("evidence"), list):
            continue
        for item in payload["evidence"]:
            if not isinstance(item, dict):
                continue
            try:
                normalized: dict[str, Any] = {
                    "chunk_id": str(item["chunk_id"]),
                    "source": str(item["source"]),
                    "page": int(item["page"]),
                    "text": str(item["text"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            key: tuple[str, int, str] = (
                str(normalized["source"]),
                int(normalized["page"]),
                str(normalized["chunk_id"]),
            )
            if key not in seen:
                seen.add(key)
                evidence.append(normalized)
    return knowledge_used, evidence


def normalize_agent_result(
    result: dict[str, Any],
) -> GroundedResponse:
    """Parse one final generation and validate citations against evidence."""
    if result.get("validated_response") is True:
        return GroundedResponse(
            str(result.get("answer", "")).strip(),
            list(result.get("citations", [])),
        )

    knowledge_used, evidence = extract_search_evidence(result)
    parsed = _parse_json_object(result.get("output"))

    # Backwards compatibility for direct/calculator responses and injected
    # test agents. Knowledge answers must always use the structured contract.
    if parsed is None:
        if knowledge_used:
            return GroundedResponse(SAFE_VALIDATION_FAILURE, [])
        answer = str(result.get("output", "")).strip()
        return GroundedResponse(answer, [])

    return validate_grounded_payload(
        parsed,
        evidence,
        knowledge_used=knowledge_used,
    )


def validate_grounded_payload(
    parsed: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    knowledge_used: bool,
) -> GroundedResponse:
    """Validate a structured final answer against request-scoped evidence."""
    answer = str(parsed.get("answer", "")).strip()
    grounded = parsed.get("grounded")
    raw_citations = parsed.get("citations", [])

    if not answer:
        return GroundedResponse(SAFE_VALIDATION_FAILURE, [])
    if not knowledge_used:
        return GroundedResponse(answer, [])
    if grounded is False:
        return GroundedResponse(SAFE_UNANSWERABLE, [])
    if grounded is not True or not isinstance(raw_citations, list):
        return GroundedResponse(SAFE_VALIDATION_FAILURE, [])

    allowed: set[tuple[str, int, str]] = {
        (str(item["source"]), int(item["page"]), str(item["chunk_id"]))
        for item in evidence
    }
    citations: list[dict[str, Any]] = []
    seen_citations: set[tuple[str, int, str]] = set()
    for citation in raw_citations:
        if not isinstance(citation, dict):
            return GroundedResponse(SAFE_VALIDATION_FAILURE, [])
        try:
            key = (
                str(citation["source"]),
                int(citation["page"]),
                str(citation["chunk_id"]),
            )
        except (KeyError, TypeError, ValueError):
            return GroundedResponse(SAFE_VALIDATION_FAILURE, [])
        if key not in allowed:
            return GroundedResponse(SAFE_VALIDATION_FAILURE, [])
        if key not in seen_citations:
            seen_citations.add(key)
            citations.append(
                {
                    "source": key[0],
                    "page": key[1],
                    "chunk_id": key[2],
                }
            )

    if not citations:
        return GroundedResponse(SAFE_VALIDATION_FAILURE, [])
    return GroundedResponse(answer, citations)

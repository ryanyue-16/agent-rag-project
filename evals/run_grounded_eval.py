from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

from app.application import (
    build_rag_system_from_chunks,
    build_workflow_for_rag_system,
)
from app.config import CHAT_MODEL, EMBED_MODEL, RERANKER_MODEL
from evals.corpus import load_eval_chunks
from evals.dataset import load_eval_cases
from evals.metrics import answer_group_coverage
from rag.grounding import SAFE_UNANSWERABLE, normalize_agent_result

DEFAULT_CASE_IDS = (
    "annual-leave",
    "incident-evidence-retention",
    "parental-leave",
    "password-length",
)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small real grounded-answer and citation evaluation."
    )
    parser.add_argument("--cases", default="evals/cases.json")
    parser.add_argument(
        "--corpus",
        default="evals/corpus/enterprise_chunks.json",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--output",
        default="evals/reports/stage4_langgraph_smoke.json",
    )
    args = parser.parse_args()

    selected_ids = set(args.case_id or DEFAULT_CASE_IDS)
    cases = [
        case
        for case in load_eval_cases(args.cases)
        if case.case_id in selected_ids
    ]
    if {case.case_id for case in cases} != selected_ids:
        raise ValueError("存在未知 grounded eval case ID")

    chunks = load_eval_chunks(args.corpus)
    rag_system = build_rag_system_from_chunks(
        chunks,
        retrieval_mode="hybrid_rerank",
    )
    document_names = sorted({str(chunk["source"]) for chunk in chunks})
    executor = build_workflow_for_rag_system(rag_system, document_names)

    results: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        raw_result = executor.invoke(
            {"input": case.query},
            config={
                "configurable": {"thread_id": f"eval-{case.case_id}"}
            },
        )
        response = normalize_agent_result(raw_result)
        latency_ms = (time.perf_counter() - started) * 1000
        cited_ids = [
            str(citation["chunk_id"])
            for citation in response.citations
        ]
        gold_ids = set(case.gold_chunk_ids)

        if case.answerable:
            citation_valid = bool(cited_ids) and set(cited_ids).issubset(
                gold_ids
            )
            citation_recall = (
                len(set(cited_ids) & gold_ids) / len(gold_ids)
            )
            coverage = answer_group_coverage(
                response.answer,
                case.expected_answer_groups,
            )
            passed = citation_valid and citation_recall == 1.0 and coverage == 1.0
        else:
            citation_valid = not cited_ids
            citation_recall = 1.0
            coverage = 1.0
            passed = response.answer == SAFE_UNANSWERABLE and citation_valid

        results.append(
            {
                "id": case.case_id,
                "answerable": case.answerable,
                "gold_chunk_ids": sorted(gold_ids),
                "cited_chunk_ids": cited_ids,
                "citation_valid": citation_valid,
                "citation_recall": citation_recall,
                "answer_group_coverage": coverage,
                "passed": passed,
                "latency_ms": latency_ms,
            }
        )

    latencies = [float(item["latency_ms"]) for item in results]
    payload = {
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBED_MODEL,
        "reranker_model": RERANKER_MODEL,
        "dataset": args.cases,
        "corpus": args.corpus,
        "case_count": len(results),
        "pass_rate": mean(float(item["passed"]) for item in results),
        "citation_valid_rate": mean(
            float(item["citation_valid"]) for item in results
        ),
        "avg_latency_ms": mean(latencies),
        "p50_latency_ms": median(latencies),
        "p95_latency_ms": _percentile(latencies, 95),
        "cases": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

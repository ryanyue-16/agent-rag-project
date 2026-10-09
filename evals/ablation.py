from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


def build_ablation_summary(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    if {report.get("retrieval_mode") for report in reports} != set(MODES):
        raise ValueError("ablation 必须包含四种且不重复的 retrieval mode")

    reference = reports[0]
    for report in reports[1:]:
        for key in ("dataset", "corpus", "ks"):
            if report.get(key) != reference.get(key):
                raise ValueError(f"ablation 报告的 {key} 不一致")

    max_k = str(max(reference["ks"]))
    rows: dict[str, dict[str, Any]] = {}
    for report in reports:
        mode = str(report["retrieval_mode"])
        summary = report["retrieval_summary"]
        rows[mode] = {
            "mrr": summary["mrr"],
            f"recall_at_{max_k}": summary["by_k"][max_k]["recall"],
            f"multi_hop_recall_at_{max_k}": summary["by_category"][
                "multi_hop"
            ]["recall_at_max_k"],
            "p50_retrieval_latency_ms": summary[
                "p50_retrieval_latency_ms"
            ],
            "p95_retrieval_latency_ms": summary[
                "p95_retrieval_latency_ms"
            ],
            "p50_reranking_latency_ms": summary[
                "p50_reranking_latency_ms"
            ],
            "p95_reranking_latency_ms": summary[
                "p95_reranking_latency_ms"
            ],
        }

    dense = rows["dense"]
    improved_modes = [
        mode
        for mode in ("hybrid", "hybrid_rerank")
        if rows[mode][f"multi_hop_recall_at_{max_k}"]
        > dense[f"multi_hop_recall_at_{max_k}"]
        and rows[mode][f"recall_at_{max_k}"]
        >= dense[f"recall_at_{max_k}"] - 0.01
    ]
    return {
        "dataset": reference["dataset"],
        "corpus": reference["corpus"],
        "ks": reference["ks"],
        "case_count": reference["retrieval_summary"]["case_count"],
        "retrieval_candidate_k": reference["retrieval_candidate_k"],
        "rrf_k": reference["rrf_k"],
        "embedding_model": reference["embedding_model"],
        "reranker_model": reference["reranker_model"],
        "modes": rows,
        "acceptance": {
            "passed": bool(improved_modes),
            "improved_modes": improved_modes,
            "criterion": (
                f"hybrid or hybrid_rerank multi-hop Recall@{max_k} "
                f"> dense, with total Recall@{max_k} no more than 0.01 lower"
            ),
        },
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    max_k = str(max(summary["ks"]))
    lines = [
        "# Retrieval Ablation Summary",
        "",
        f"- Cases: {summary['case_count']}",
        f"- Candidate K: {summary['retrieval_candidate_k']}",
        f"- RRF K: {summary['rrf_k']}",
        f"- Acceptance passed: {summary['acceptance']['passed']}",
        "",
        f"| Mode | MRR | Recall@{max_k} | Multi-hop Recall@{max_k} | "
        "P50 retrieval ms | P95 retrieval ms | P50 rerank ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        row = summary["modes"][mode]
        lines.append(
            f"| {mode} | {row['mrr']:.4f} | "
            f"{row[f'recall_at_{max_k}']:.4f} | "
            f"{row[f'multi_hop_recall_at_{max_k}']:.4f} | "
            f"{row['p50_retrieval_latency_ms']:.1f} | "
            f"{row['p95_retrieval_latency_ms']:.1f} | "
            f"{row['p50_reranking_latency_ms']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a validated four-mode retrieval ablation summary."
    )
    parser.add_argument("--report-dir", default="evals/reports")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    reports = []
    for mode in MODES:
        path = report_dir / f"retrieval_{mode}.json"
        with path.open("r", encoding="utf-8") as file:
            reports.append(json.load(file))

    summary = build_ablation_summary(reports)
    json_path = report_dir / "ablation_summary.json"
    markdown_path = report_dir / "ablation_summary.md"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(markdown_path, summary)
    print(f"Acceptance passed: {summary['acceptance']['passed']}")
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()

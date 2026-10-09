from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from app.application import (
    build_rag_system,
    build_rag_system_from_chunks,
)
from app.config import (
    EMBED_MODEL,
    LOG_LEVEL,
    RERANKER_MODEL,
    RETRIEVAL_CANDIDATE_K,
    RRF_K,
)
from app.logging_config import setup_logging
from evals.corpus import load_eval_chunks
from evals.dataset import load_eval_cases
from evals.evaluator import aggregate_retrieval, evaluate_retrieval_case


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "K 必须大于 0"
        )
    return parsed


def _write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


def _write_csv(
    path: Path,
    retrieval_results: list[dict[str, Any]],
    ks: list[int],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "id",
        "query",
        "category",
        "answerable",
        "mrr",
        "retrieval_latency_ms",
        "reranking_latency_ms",
    ]
    for k in ks:
        fieldnames.extend(
            [
                f"hit@{k}",
                f"precision@{k}",
                f"recall@{k}",
            ]
        )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for item in retrieval_results:
            row: dict[str, Any] = {
                "id": item["id"],
                "query": item["query"],
                "category": item["category"],
                "answerable": item["answerable"],
                "mrr": (
                    round(item["mrr"], 4)
                    if item["mrr"] is not None
                    else ""
                ),
                "retrieval_latency_ms": round(
                    item["retrieval_latency_ms"],
                    2,
                ),
                "reranking_latency_ms": round(
                    item["reranking_latency_ms"],
                    2,
                ),
            }

            for k in ks:
                metrics = item["metrics"].get(
                    str(k)
                )
                if metrics is None:
                    row[f"hit@{k}"] = ""
                    row[f"precision@{k}"] = ""
                    row[f"recall@{k}"] = ""
                else:
                    row[f"hit@{k}"] = round(
                        metrics["hit"],
                        4,
                    )
                    row[f"precision@{k}"] = round(
                        metrics["precision"],
                        4,
                    )
                    row[f"recall@{k}"] = round(
                        metrics["recall"],
                        4,
                    )

            writer.writerow(row)


def _write_markdown(
    path: Path,
    retrieval_summary: dict[str, Any],
    ks: list[int],
    *,
    case_path: str,
    corpus_path: str,
    retrieval_mode: str,
) -> None:
    lines = [
        "# Retrieval Evaluation Report",
        "",
        f"- Embedding model: `{EMBED_MODEL}`",
        f"- Retrieval mode: `{retrieval_mode}`",
        f"- Candidate K: {RETRIEVAL_CANDIDATE_K}",
        f"- RRF K: {RRF_K}",
        f"- Reranker model: `{RERANKER_MODEL}`",
        f"- Dataset: `{case_path}`",
        f"- Corpus: `{corpus_path}`",
        f"- Cases: {retrieval_summary['case_count']} ",
        f"  ({retrieval_summary['answerable_case_count']} answerable, "
        f"{retrieval_summary['unanswerable_case_count']} unanswerable)",
        f"- MRR: {retrieval_summary['mrr']:.4f}",
        "- Retrieval latency: "
        f"p50={retrieval_summary['p50_retrieval_latency_ms']:.1f} ms, "
        f"p95={retrieval_summary['p95_retrieval_latency_ms']:.1f} ms",
        "- Reranking latency: "
        f"p50={retrieval_summary['p50_reranking_latency_ms']:.1f} ms, "
        f"p95={retrieval_summary['p95_reranking_latency_ms']:.1f} ms",
        "",
        "## Metrics by K",
        "",
        "| K | HitRate | Precision | Recall |",
        "|---:|---:|---:|---:|",
    ]

    for k in ks:
        item = retrieval_summary["by_k"][str(k)]
        lines.append(
            f"| {k} | {item['hit_rate']:.4f} | "
            f"{item['precision']:.4f} | {item['recall']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Metrics by category",
            "",
            "| Category | Cases | MRR | Recall at max K |",
            "|---|---:|---:|---:|",
        ]
    )
    for category, item in retrieval_summary[
        "by_category"
    ].items():
        lines.append(
            f"| {category} | {item['case_count']} | "
            f"{item['mrr']:.4f} | {item['recall_at_max_k']:.4f} |"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _print_summary(
    retrieval_summary: dict[str, Any],
    ks: list[int],
) -> None:
    print("\n=== Retrieval Evaluation Summary ===")
    print(
        f"Cases: {retrieval_summary['case_count']}"
    )
    print(
        f"MRR: {retrieval_summary['mrr']:.3f}"
    )
    print(
        "Avg retrieval latency: "
        f"{retrieval_summary['avg_retrieval_latency_ms']:.1f} ms"
    )
    print(
        "P50 / P95 reranking latency: "
        f"{retrieval_summary['p50_reranking_latency_ms']:.1f} / "
        f"{retrieval_summary['p95_reranking_latency_ms']:.1f} ms"
    )
    print(
        "P50 / P95 retrieval latency: "
        f"{retrieval_summary['p50_retrieval_latency_ms']:.1f} / "
        f"{retrieval_summary['p95_retrieval_latency_ms']:.1f} ms"
    )
    print()
    print(
        f"{'K':>4} {'HitRate':>10} "
        f"{'Precision':>10} {'Recall':>10}"
    )

    for k in ks:
        item = retrieval_summary["by_k"][str(k)]
        print(
            f"{k:>4} "
            f"{item['hit_rate']:>10.3f} "
            f"{item['precision']:>10.3f} "
            f"{item['recall']:>10.3f}"
        )

    print("\nBy category:")
    for category, item in retrieval_summary[
        "by_category"
    ].items():
        print(
            f"- {category}: cases={item['case_count']} "
            f"MRR={item['mrr']:.3f} "
            f"Recall@{max(ks)}={item['recall_at_max_k']:.3f}"
        )

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "评估当前 RAG 的检索召回率、命中率、"
            "Precision、MRR，以及可选的答案覆盖率。"
        )
    )
    parser.add_argument(
        "--cases",
        default="evals/cases.json",
        help="评估集 JSON 路径",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="只运行指定 case；可重复传入用于失败分析",
    )
    parser.add_argument(
        "--corpus",
        default="evals/corpus/enterprise_chunks.json",
        help=(
            "固定评测语料 JSON 路径；传入空字符串时使用 data/pdfs"
        ),
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("dense", "bm25", "hybrid", "hybrid_rerank"),
        default="dense",
        help="要评估的检索方案",
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=_positive_int,
        default=[1, 2, 4],
        help="需要评估的 K，例如 --ks 1 2 4",
    )
    parser.add_argument(
        "--report-dir",
        default="evals/reports",
        help="报告输出目录",
    )
    args = parser.parse_args()

    ks = sorted(set(args.ks))

    setup_logging(LOG_LEVEL)
    cases = load_eval_cases(args.cases)
    if args.case_id:
        requested_case_ids = set(args.case_id)
        cases = [
            case
            for case in cases
            if case.case_id in requested_case_ids
        ]
        found_case_ids = {case.case_id for case in cases}
        missing_case_ids = requested_case_ids - found_case_ids
        if missing_case_ids:
            parser.error(
                "未知 case ID：" + ", ".join(sorted(missing_case_ids))
            )
    if args.corpus:
        chunks = load_eval_chunks(
            args.corpus
        )
        rag_system = build_rag_system_from_chunks(
            chunks,
            retrieval_mode=args.retrieval_mode,
        )
        document_names = sorted(
            {
                str(chunk["source"])
                for chunk in chunks
            }
        )
    else:
        rag_system, document_names = build_rag_system(
            retrieval_mode=args.retrieval_mode,
        )

    print(
        "Loaded documents: "
        + ", ".join(document_names)
    )
    print(
        f"Loaded evaluation cases: {len(cases)}"
    )

    retrieval_results = [
        evaluate_retrieval_case(
            rag_system,
            case,
            ks,
        )
        for case in cases
    ]

    retrieval_summary = aggregate_retrieval(
        retrieval_results,
        ks,
    )

    payload = {
        "embedding_model": EMBED_MODEL,
        "retrieval_mode": args.retrieval_mode,
        "retrieval_candidate_k": RETRIEVAL_CANDIDATE_K,
        "rrf_k": RRF_K,
        "reranker_model": RERANKER_MODEL,
        "dataset": str(args.cases),
        "corpus": str(args.corpus),
        "documents": document_names,
        "ks": ks,
        "retrieval_summary": retrieval_summary,
        "retrieval_cases": retrieval_results,
    }

    report_dir = Path(args.report_dir)
    report_stem = f"retrieval_{args.retrieval_mode}"
    json_path = report_dir / f"{report_stem}.json"
    csv_path = report_dir / f"{report_stem}.csv"
    markdown_path = report_dir / f"{report_stem}.md"

    _write_json(json_path, payload)
    _write_csv(
        csv_path,
        retrieval_results,
        ks,
    )
    _write_markdown(
        markdown_path,
        retrieval_summary,
        ks,
        case_path=str(args.cases),
        corpus_path=str(args.corpus),
        retrieval_mode=args.retrieval_mode,
    )

    _print_summary(
        retrieval_summary,
        ks,
    )

    print("\nReports:")
    print(f"- {json_path}")
    print(f"- {csv_path}")
    print(f"- {markdown_path}")


if __name__ == "__main__":
    main()

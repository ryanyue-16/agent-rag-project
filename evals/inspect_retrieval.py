from __future__ import annotations

import argparse

from app.application import build_rag_system
from app.config import LOG_LEVEL
from app.logging_config import setup_logging


def _preview(text: str, limit: int = 220) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="查看某个问题实际召回了哪些 chunks。"
    )
    parser.add_argument(
        "--query",
        required=True,
        help="要测试的检索问题",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="返回多少个 chunk，默认 4",
    )
    args = parser.parse_args()

    setup_logging(LOG_LEVEL)
    rag_system, _ = build_rag_system()

    results = rag_system.retrieve(
        args.query,
        top_k=args.top_k,
    )

    print("\n=== Retrieval Inspection ===")
    print(f"Query: {args.query}")
    print(f"Top-K: {args.top_k}\n")

    for rank, item in enumerate(results, start=1):
        print(
            f"[{rank}] score={item['score']:.4f} "
            f"chunk_id={item['chunk_id']} "
            f"source={item['source']} "
            f"page={item['page']}"
        )
        print(_preview(item["text"]))
        print()


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "chunk_id",
    "source",
    "page",
    "text",
}


def load_eval_chunks(
    path: str | Path,
) -> list[dict[str, Any]]:
    """Load and validate a fixed retrieval-evaluation corpus."""
    corpus_path = Path(path)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"评测语料不存在：{corpus_path}"
        )

    with corpus_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw = json.load(file)

    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "评测语料必须是非空 JSON 数组"
        )

    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(
                f"第 {index + 1} 个 chunk 必须是 JSON object"
            )

        missing = REQUIRED_FIELDS - set(item)
        if missing:
            raise ValueError(
                f"第 {index + 1} 个 chunk 缺少字段：{sorted(missing)}"
            )

        chunk_id = item["chunk_id"]
        source = item["source"]
        text = item["text"]
        page = item["page"]

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(
                f"第 {index + 1} 个 chunk_id 必须是非空字符串"
            )
        if chunk_id in seen_ids:
            raise ValueError(
                f"chunk_id 重复：{chunk_id}"
            )
        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                f"第 {index + 1} 个 source 必须是非空字符串"
            )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"第 {index + 1} 个 text 必须是非空字符串"
            )
        if not isinstance(page, int) or page <= 0:
            raise ValueError(
                f"第 {index + 1} 个 page 必须是正整数"
            )

        seen_ids.add(chunk_id)
        chunks.append(
            {
                "chunk_id": chunk_id.strip(),
                "source": source.strip(),
                "page": page,
                "text": text.strip(),
            }
        )

    return chunks

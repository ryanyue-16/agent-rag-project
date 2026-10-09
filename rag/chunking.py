from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

def chunk_text(
    text: str,
    size: int = 800,
    overlap: int = 150,
) -> list[str]:
    if size <= 0:
        raise ValueError("size 必须大于 0")

    if overlap < 0:
        raise ValueError("overlap 不能小于 0")

    if overlap >= size:
        raise ValueError("overlap 不能大于 size")
    
    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def build_chunks(
    documents: list[dict[str, Any]],
    size: int = 800,
    overlap: int = 150,
) -> list[dict[str, Any]]:
    all_chunks: list[dict[str, Any]] = []

    for document in documents:
        chunks = chunk_text(
            document['text'], 
            size=size, 
            overlap=overlap,
        )

        for chunk_index, chunk in enumerate(chunks):
            document_id = str(
                document.get("document_id", document["source"])
            )
            item: dict[str, Any] = {
                    "text": chunk,
                    "page": document['page'],
                    "source": document['source'],
                    "chunk_id": (
                        f"{document_id}"
                        f"-p{document['page']}"
                        f"-c{chunk_index}"
                    ),
                }
            for key in ("document_id", "collection_id", "metadata"):
                if key in document:
                    item[key] = document[key]
            all_chunks.append(item)

    logger.info(
        "文本切分完成：document_count=%d chunk_count=%d", 
        len(documents),
        len(all_chunks),
    )

    return all_chunks

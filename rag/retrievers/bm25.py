from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

TOKEN_PATTERN = re.compile(
    r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]",
    re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    """Tokenize English words/numbers and CJK text at character level."""
    return TOKEN_PATTERN.findall(text.casefold())


class BM25Retriever:
    """Small deterministic BM25 implementation over immutable chunks."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("chunks 不能为空")
        if k1 <= 0:
            raise ValueError("k1 必须大于 0")
        if not 0 <= b <= 1:
            raise ValueError("b 必须在 0 到 1 之间")

        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._term_frequencies = [
            Counter(tokenize(str(chunk["text"])))
            for chunk in chunks
        ]
        self._document_lengths = [
            sum(frequencies.values())
            for frequencies in self._term_frequencies
        ]
        self._average_document_length = (
            sum(self._document_lengths) / len(chunks)
        )

        document_frequencies: Counter[str] = Counter()
        for frequencies in self._term_frequencies:
            document_frequencies.update(frequencies.keys())

        document_count = len(chunks)
        self._inverse_document_frequencies = {
            term: math.log(
                1
                + (
                    document_count
                    - frequency
                    + 0.5
                )
                / (frequency + 0.5)
            )
            for term, frequency in document_frequencies.items()
        }

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores: list[tuple[float, str, int]] = []
        for index, frequencies in enumerate(
            self._term_frequencies
        ):
            document_length = self._document_lengths[index]
            score = 0.0
            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                inverse_frequency = (
                    self._inverse_document_frequencies.get(term, 0.0)
                )
                length_normalizer = 1 - self.b + self.b * (
                    document_length / self._average_document_length
                )
                score += inverse_frequency * (
                    term_frequency * (self.k1 + 1)
                ) / (
                    term_frequency + self.k1 * length_normalizer
                )

            if score > 0:
                scores.append(
                    (
                        score,
                        str(self.chunks[index]["chunk_id"]),
                        index,
                    )
                )

        scores.sort(key=lambda item: (-item[0], item[1]))
        results: list[dict[str, Any]] = []
        for rank, (score, _, index) in enumerate(
            scores[:top_k],
            start=1,
        ):
            chunk = self.chunks[index]
            results.append(
                {
                    "score": score,
                    "bm25_score": score,
                    "bm25_rank": rank,
                    "text": chunk["text"],
                    "page": chunk["page"],
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "metadata": dict(chunk.get("metadata", {})),
                    "document_id": chunk.get("document_id"),
                    "collection_id": chunk.get("collection_id"),
                }
            )
        return results

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable BM25 index snapshot."""
        return {
            "k1": self.k1,
            "b": self.b,
            "term_frequencies": [dict(item) for item in self._term_frequencies],
            "document_lengths": self._document_lengths,
            "average_document_length": self._average_document_length,
            "inverse_document_frequencies": self._inverse_document_frequencies,
        }

    @classmethod
    def from_dict(
        cls,
        chunks: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> "BM25Retriever":
        """Restore BM25 without recomputing token statistics."""
        instance = cls.__new__(cls)
        instance.chunks = chunks
        instance.k1 = float(state["k1"])
        instance.b = float(state["b"])
        instance._term_frequencies = [
            Counter({str(k): int(v) for k, v in item.items()})
            for item in state["term_frequencies"]
        ]
        instance._document_lengths = [
            int(value) for value in state["document_lengths"]
        ]
        instance._average_document_length = float(
            state["average_document_length"]
        )
        instance._inverse_document_frequencies = {
            str(k): float(v)
            for k, v in state["inverse_document_frequencies"].items()
        }
        if len(instance._term_frequencies) != len(chunks):
            raise ValueError("BM25 snapshot 与 chunks 数量不一致")
        return instance

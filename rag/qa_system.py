from __future__ import annotations

import logging
import time
from typing import Any

from rag.retrievers import (
    BM25Retriever,
    CrossEncoderReranker,
    decompose_query,
    fuse_query_results,
    reciprocal_rank_fusion,
)

logger = logging.getLogger(__name__)

VALID_RETRIEVAL_MODES = {
    "dense",
    "bm25",
    "hybrid",
    "hybrid_rerank",
}


class RAGQASystem:
    """封装 Embedding、FAISS 检索和基于检索内容的问答。"""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        client: Any,
        embed_model: str,
        chat_model: str,
        retrieval_mode: str = "dense",
        candidate_k: int = 20,
        rrf_k: int = 60,
        reranker_model: str = (
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        ),
        reranker: CrossEncoderReranker | None = None,
        dense_index: Any | None = None,
        bm25_retriever: BM25Retriever | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("chunks 不能为空")

        if retrieval_mode not in VALID_RETRIEVAL_MODES:
            raise ValueError(
                "retrieval_mode 必须是 dense、bm25、hybrid "
                "或 hybrid_rerank"
            )
        if candidate_k <= 0:
            raise ValueError("candidate_k 必须大于 0")
        if rrf_k <= 0:
            raise ValueError("rrf_k 必须大于 0")

        self.chunks = chunks
        self.client = client
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.retrieval_mode = retrieval_mode
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.reranker_model = reranker_model
        self.last_retrieval_metrics: dict[str, float] = {
            "reranking_latency_ms": 0.0,
        }
        self._np: Any = None
        self.index: Any = None
        self.bm25: BM25Retriever | None = None
        self.reranker: CrossEncoderReranker | None = None

        if retrieval_mode in {"bm25", "hybrid", "hybrid_rerank"}:
            self.bm25 = bm25_retriever or BM25Retriever(chunks)

        if retrieval_mode == "hybrid_rerank":
            self.reranker = reranker or CrossEncoderReranker(
                reranker_model
            )

        if retrieval_mode == "bm25":
            logger.info(
                "BM25 索引构建完成：chunk_count=%d",
                len(chunks),
            )
            return

        # Dense dependencies are loaded only when the selected mode uses them.
        import faiss
        import numpy as np

        self._np = np

        if dense_index is not None:
            if int(dense_index.ntotal) != len(chunks):
                raise ValueError("FAISS snapshot 与 chunks 数量不一致")
            self.index = dense_index
            logger.info(
                "持久化向量库加载完成：vector_count=%d",
                self.index.ntotal,
            )
            return

        logger.info(
            "开始构建向量库：chunk_count=%d model=%s",
            len(chunks),
            embed_model,
        )

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self._embed_texts(texts)
        embeddings = self._normalize(embeddings)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        logger.info(
            "向量库构建完成：vector_count=%d dimension=%d",
            self.index.ntotal,
            dimension,
        )

    def _normalize(self, embeddings: Any) -> Any:
        if self._np is None:
            raise RuntimeError("当前检索模式未初始化 dense dependencies")
        norms = self._np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        if self._np.any(norms == 0):
            raise ValueError(
                "Embedding 中出现零向量，无法归一化"
            )

        return embeddings / norms

    def _embed_texts(self, texts: list[str]) -> Any:
        if self._np is None:
            raise RuntimeError("当前检索模式未初始化 dense dependencies")
        response = self.client.embeddings.create(
            model=self.embed_model,
            input=texts,
        )

        vectors = [
            self._np.array(
                item.embedding,
                dtype="float32",
            )
            for item in response.data
        ]

        return self._np.vstack(vectors)

    def _embed_query(self, query: str) -> Any:
        embedding = self._embed_texts([query])
        return self._normalize(embedding)

    def _retrieve_dense(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        return self._retrieve_dense_queries(
            [query],
            top_k,
        )[0]

    def _retrieve_dense_queries(
        self,
        queries: list[str],
        top_k: int,
    ) -> list[list[dict[str, Any]]]:
        actual_top_k = min(
            top_k,
            len(self.chunks),
        )

        query_embeddings = self._normalize(
            self._embed_texts(queries)
        )

        scores, indices = self.index.search(
            query_embeddings,
            actual_top_k,
        )

        all_results: list[list[dict[str, Any]]] = []
        for query_scores, query_indices in zip(scores, indices, strict=True):
            results: list[dict[str, Any]] = []
            for rank, (score, index) in enumerate(
                zip(query_scores, query_indices, strict=True),
                start=1,
            ):
                if index < 0:
                    continue
                chunk = self.chunks[index]
                results.append(
                    {
                        "score": float(score),
                        "dense_score": float(score),
                        "dense_rank": rank,
                        "text": chunk["text"],
                        "page": chunk["page"],
                        "source": chunk["source"],
                        "chunk_id": chunk["chunk_id"],
                        "metadata": dict(chunk.get("metadata", {})),
                        "document_id": chunk.get("document_id"),
                        "collection_id": chunk.get("collection_id"),
                    }
                )
            all_results.append(results)
        return all_results

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve with the configured dense/BM25/hybrid strategy."""
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        reranking_latency_ms = 0.0
        search_k = len(self.chunks) if metadata_filter else top_k
        if self.retrieval_mode == "dense":
            results = self._filter_results(
                self._retrieve_dense(query, search_k), metadata_filter
            )[:top_k]
        elif self.retrieval_mode == "bm25":
            if self.bm25 is None:
                raise RuntimeError("BM25 未初始化")
            results = self._filter_results(
                self.bm25.retrieve(query, search_k), metadata_filter
            )[:top_k]
        else:
            candidate_k = min(
                len(self.chunks)
                if metadata_filter
                else max(self.candidate_k, top_k),
                len(self.chunks),
            )
            query_variants = decompose_query(query)
            dense_results = fuse_query_results(
                self._retrieve_dense_queries(
                    query_variants,
                    candidate_k,
                ),
                score_key="dense_score",
                rank_key="dense_rank",
                diagnostic_key="dense_query_rrf_score",
                rrf_k=self.rrf_k,
            )
            if self.bm25 is None:
                raise RuntimeError("BM25 未初始化")
            bm25_results = fuse_query_results(
                [
                    self.bm25.retrieve(
                        variant,
                        candidate_k,
                    )
                    for variant in query_variants
                ],
                score_key="bm25_score",
                rank_key="bm25_rank",
                diagnostic_key="bm25_query_rrf_score",
                rrf_k=self.rrf_k,
            )
            fused = reciprocal_rank_fusion(
                self._filter_results(dense_results, metadata_filter),
                self._filter_results(bm25_results, metadata_filter),
                rrf_k=self.rrf_k,
            )
            if self.retrieval_mode == "hybrid":
                results = fused[:top_k]
            else:
                if self.reranker is None:
                    raise RuntimeError("reranker 未初始化")
                rerank_started = time.perf_counter()
                results = self.reranker.rerank(
                    query,
                    fused,
                    top_k=top_k,
                    query_variants=query_variants,
                    preserve_retrieval_top_k=True,
                )
                reranking_latency_ms = (
                    time.perf_counter() - rerank_started
                ) * 1000

        self.last_retrieval_metrics = {
            "reranking_latency_ms": reranking_latency_ms,
        }
        best_score = (
            results[0]["score"]
            if results
            else 0.0
        )

        logger.info(
            "检索完成：mode=%s query_length=%d requested_top_k=%d "
            "result_count=%d best_score=%.3f reranking_ms=%.1f",
            self.retrieval_mode,
            len(query),
            top_k,
            len(results),
            best_score,
            reranking_latency_ms,
        )

        return results

    @staticmethod
    def _filter_results(
        results: list[dict[str, Any]],
        metadata_filter: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not metadata_filter:
            return results

        def matches(item: dict[str, Any]) -> bool:
            values = {
                **dict(item.get("metadata", {})),
                "document_id": item.get("document_id"),
                "collection_id": item.get("collection_id"),
                "source": item.get("source"),
            }
            for key, expected in metadata_filter.items():
                actual = values.get(key)
                expected_values = expected if isinstance(expected, list) else [expected]
                if isinstance(actual, list):
                    if not set(map(str, actual)) & set(map(str, expected_values)):
                        return False
                elif str(actual) not in {str(value) for value in expected_values}:
                    return False
            return True

        return [item for item in results if matches(item)]


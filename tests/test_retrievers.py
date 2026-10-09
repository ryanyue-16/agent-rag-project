from __future__ import annotations

import pytest

from rag.qa_system import RAGQASystem
from rag.retrievers.bm25 import BM25Retriever, tokenize
from rag.retrievers.fusion import (
    fuse_query_results,
    reciprocal_rank_fusion,
)
from rag.retrievers.query import decompose_query
from rag.retrievers.reranker import CrossEncoderReranker


def _chunk(chunk_id: str, text: str) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "source": "policy.pdf",
        "page": 1,
    }


def test_bm25_ranks_rare_keyword_and_records_diagnostics() -> None:
    retriever = BM25Retriever(
        [
            _chunk("common", "employees use approved devices"),
            _chunk("rare", "quasar exception requires director approval"),
            _chunk("other", "employees require manager approval"),
        ]
    )

    results = retriever.retrieve("quasar approval", top_k=2)

    assert results[0]["chunk_id"] == "rare"
    assert results[0]["bm25_rank"] == 1
    assert results[0]["bm25_score"] > 0


def test_bm25_empty_match_returns_no_results() -> None:
    retriever = BM25Retriever([_chunk("one", "annual leave")])

    assert retriever.retrieve("unmatched-token", top_k=5) == []
    assert tokenize("远程 Work 3-days") == ["远", "程", "work", "3", "days"]


def test_rrf_deduplicates_and_preserves_rank_and_score() -> None:
    dense = [
        {
            **_chunk("a", "alpha"),
            "score": 0.9,
            "dense_score": 0.9,
            "dense_rank": 1,
        },
        {
            **_chunk("b", "beta"),
            "score": 0.8,
            "dense_score": 0.8,
            "dense_rank": 2,
        },
    ]
    sparse = [
        {
            **_chunk("b", "beta"),
            "score": 4.0,
            "bm25_score": 4.0,
            "bm25_rank": 1,
        },
        {
            **_chunk("c", "gamma"),
            "score": 3.0,
            "bm25_score": 3.0,
            "bm25_rank": 2,
        },
    ]

    results = reciprocal_rank_fusion(dense, sparse, rrf_k=60)

    assert [item["chunk_id"] for item in results] == ["b", "a", "c"]
    assert len(results) == 3
    assert results[0]["dense_rank"] == 2
    assert results[0]["bm25_rank"] == 1
    assert results[0]["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert results[0]["score"] == results[0]["rrf_score"]


def test_query_decomposition_handles_multi_hop_chinese_structures() -> None:
    assert decompose_query(
        "安全事故证据日志最长保留多久，事故记录由谁维护？"
    ) == [
        "安全事故证据日志最长保留多久，事故记录由谁维护？",
        "安全事故证据日志最长保留多久",
        "事故记录由谁维护",
    ]
    assert decompose_query(
        "新员工需要等待多久才能申请每周三天远程办公？"
    ) == [
        "新员工需要等待多久才能申请每周三天远程办公？",
        "新员工需要等待多久",
        "申请每周三天远程办公",
    ]


def test_query_result_fusion_rewards_evidence_across_variants() -> None:
    first = [
        {**_chunk("a", "alpha"), "score": 0.9, "dense_score": 0.9},
        {**_chunk("b", "beta"), "score": 0.8, "dense_score": 0.8},
    ]
    second = [
        {**_chunk("b", "beta"), "score": 0.95, "dense_score": 0.95},
        {**_chunk("c", "gamma"), "score": 0.7, "dense_score": 0.7},
    ]

    results = fuse_query_results(
        [first, second],
        score_key="dense_score",
        rank_key="dense_rank",
        diagnostic_key="dense_query_rrf_score",
        rrf_k=60,
    )

    assert [item["chunk_id"] for item in results] == ["b", "a", "c"]
    assert results[0]["dense_rank"] == 1
    assert results[0]["query_match_count"] == 2


def test_reranker_uses_fake_scores_without_loading_model() -> None:
    seen_pairs: list[tuple[str, str]] = []

    def fake_score(
        pairs: list[tuple[str, str]],
    ) -> list[float]:
        seen_pairs.extend(pairs)
        return [0.1, 0.9]

    reranker = CrossEncoderReranker(
        "fake-model",
        scoring_function=fake_score,
    )
    results = reranker.rerank(
        "query",
        [_chunk("a", "alpha"), _chunk("b", "beta")],
        top_k=2,
    )

    assert seen_pairs == [("query", "alpha"), ("query", "beta")]
    assert [item["chunk_id"] for item in results] == ["b", "a"]
    assert results[0]["rerank_score"] == 0.9
    assert results[0]["score"] == 0.9


def test_reranker_uses_best_clause_score_for_multi_hop_evidence() -> None:
    def fake_score(
        pairs: list[tuple[str, str]],
    ) -> list[float]:
        assert pairs == [
            ("full", "alpha"),
            ("full", "beta"),
            ("clause", "alpha"),
            ("clause", "beta"),
        ]
        return [0.9, 0.1, 0.2, 0.8]

    reranker = CrossEncoderReranker(
        "fake-model",
        scoring_function=fake_score,
    )
    results = reranker.rerank(
        "full",
        [_chunk("a", "alpha"), _chunk("b", "beta")],
        query_variants=["full", "clause"],
    )

    assert [item["chunk_id"] for item in results] == ["a", "b"]
    assert results[1]["rerank_score"] == 0.8
    assert results[1]["rerank_full_query_score"] == 0.1


def test_reranker_preserves_base_evidence_and_promotes_positive_match() -> None:
    def fake_score(
        pairs: list[tuple[str, str]],
    ) -> list[float]:
        return [3.0, -2.0, -4.0, 1.0]

    candidates = [
        _chunk("primary", "primary"),
        _chunk("base-evidence", "base evidence"),
        _chunk("weak", "weak"),
        _chunk("new-evidence", "new evidence"),
    ]
    reranker = CrossEncoderReranker(
        "fake-model",
        scoring_function=fake_score,
    )

    results = reranker.rerank(
        "query",
        candidates,
        top_k=3,
        preserve_retrieval_top_k=True,
    )

    assert {item["chunk_id"] for item in results} == {
        "primary",
        "base-evidence",
        "new-evidence",
    }
    promoted = next(
        item for item in results if item["chunk_id"] == "new-evidence"
    )
    assert promoted["rerank_promoted"] is True


def test_retrieval_mode_validation_and_bm25_skips_embeddings() -> None:
    with pytest.raises(ValueError, match="retrieval_mode"):
        RAGQASystem(
            [_chunk("a", "alpha")],
            client=object(),
            embed_model="unused",
            chat_model="unused",
            retrieval_mode="unknown",
        )

    system = RAGQASystem(
        [_chunk("a", "alpha")],
        client=object(),
        embed_model="unused",
        chat_model="unused",
        retrieval_mode="bm25",
    )
    assert system.retrieve("alpha", top_k=1)[0]["chunk_id"] == "a"

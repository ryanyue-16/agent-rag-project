from __future__ import annotations

import atexit
import multiprocessing
import sys
from threading import Lock
from typing import Any, Callable, ClassVar, Sequence

ScoringFunction = Callable[[list[tuple[str, str]]], Sequence[float]]


def _cross_encoder_worker(
    connection: Any,
    model_name: str,
) -> None:
    """Keep PyTorch outside the FAISS process on macOS."""
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name)
        connection.send(("ready", None))
        while True:
            message = connection.recv()
            if message is None:
                break
            scores = model.predict(message)
            connection.send(
                ("scores", [float(score) for score in scores])
            )
    except EOFError:
        pass
    except Exception as exc:
        connection.send(("error", repr(exc)))
    finally:
        connection.close()


class _MacOSCrossEncoderProcess:
    def __init__(self, model_name: str) -> None:
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe()
        self._connection = parent_connection
        self._process = context.Process(
            target=_cross_encoder_worker,
            args=(child_connection, model_name),
            daemon=True,
        )
        self._process.start()
        child_connection.close()
        status, payload = self._connection.recv()
        if status != "ready":
            raise RuntimeError(
                f"Cross-Encoder worker 初始化失败：{payload}"
            )

    def predict(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[float]:
        self._connection.send(pairs)
        status, payload = self._connection.recv()
        if status != "scores":
            raise RuntimeError(
                f"Cross-Encoder worker 评分失败：{payload}"
            )
        return payload

    def close(self) -> None:
        if self._process.is_alive():
            self._connection.send(None)
            self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
        self._connection.close()


class CrossEncoderReranker:
    """Lazy, process-cached CrossEncoder with an injectable test scorer."""

    _models: ClassVar[dict[str, Any]] = {}
    _model_lock: ClassVar[Lock] = Lock()

    def __init__(
        self,
        model_name: str,
        *,
        scoring_function: ScoringFunction | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name 不能为空")
        self.model_name = model_name
        self._scoring_function = scoring_function

    @classmethod
    def _load_model(cls, model_name: str) -> Any:
        model = cls._models.get(model_name)
        if model is not None:
            return model
        with cls._model_lock:
            model = cls._models.get(model_name)
            if model is None:
                if sys.platform == "darwin":
                    model = _MacOSCrossEncoderProcess(model_name)
                    atexit.register(model.close)
                else:
                    from sentence_transformers import CrossEncoder

                    model = CrossEncoder(model_name)
                cls._models[model_name] = model
        return model

    def _score(
        self,
        pairs: list[tuple[str, str]],
    ) -> Sequence[float]:
        if self._scoring_function is not None:
            return self._scoring_function(pairs)
        return self._load_model(self.model_name).predict(pairs)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        top_k: int | None = None,
        query_variants: list[str] | None = None,
        preserve_retrieval_top_k: bool = False,
        promotion_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not candidates:
            return []

        variants = query_variants or [query]
        if not variants or variants[0] != query:
            variants = [query, *variants]
        pairs = [
            (variant, str(candidate["text"]))
            for variant in variants
            for candidate in candidates
        ]
        scores = list(self._score(pairs))
        if len(scores) != len(pairs):
            raise ValueError("reranker 返回的分数数量与候选数量不一致")

        candidate_count = len(candidates)
        score_groups = [
            scores[offset:offset + candidate_count]
            for offset in range(0, len(scores), candidate_count)
        ]
        aggregated_scores = [
            max(float(group[index]) for group in score_groups)
            for index in range(candidate_count)
        ]

        ranked = []
        for original_rank, (candidate, score) in enumerate(
            zip(candidates, aggregated_scores, strict=True),
            start=1,
        ):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            item["rerank_full_query_score"] = float(
                score_groups[0][original_rank - 1]
            )
            item["_original_rank"] = original_rank
            ranked.append(item)

        ranked.sort(
            key=lambda item: (
                -float(item["rerank_score"]),
                int(item["_original_rank"]),
                str(item["chunk_id"]),
            )
        )
        for item in ranked:
            item.pop("_original_rank")
            item["score"] = float(item["rerank_score"])

        if not preserve_retrieval_top_k or top_k is None:
            return ranked[:top_k]

        # Preserve the retrieval channel's Top-K evidence, but allow a
        # Cross-Encoder candidate with a positive relevance logit to replace
        # the weakest selected item. This avoids discarding complementary
        # multi-hop evidence solely because its full-query logit is low.
        selected = [dict(item) for item in candidates[:top_k]]
        scores_by_id = {
            str(item["chunk_id"]): item
            for item in ranked
        }
        for item in selected:
            scored = scores_by_id[str(item["chunk_id"])]
            item["rerank_score"] = scored["rerank_score"]
            item["rerank_full_query_score"] = scored[
                "rerank_full_query_score"
            ]
            item["rerank_promoted"] = False

        selected_ids = {
            str(item["chunk_id"])
            for item in selected
        }
        promotions = [
            item
            for item in ranked
            if str(item["chunk_id"]) not in selected_ids
            and float(item["rerank_score"]) > promotion_threshold
        ]
        for promotion in promotions:
            weakest_index = min(
                range(len(selected)),
                key=lambda index: float(
                    selected[index]["rerank_score"]
                ),
            )
            if float(promotion["rerank_score"]) <= float(
                selected[weakest_index]["rerank_score"]
            ):
                continue
            promoted = dict(promotion)
            promoted["rerank_promoted"] = True
            selected[weakest_index] = promoted

        selected.sort(
            key=lambda item: (
                -float(item["rerank_score"]),
                str(item["chunk_id"]),
            )
        )
        for item in selected:
            item["score"] = float(item["rerank_score"])
        return selected

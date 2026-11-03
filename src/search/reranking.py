from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

from src.core.models import Evidence


class CrossEncoderReranker:
    """Re-score query/source pairs; optionally fuse several model rankings via RRF."""

    def __init__(
        self,
        model_name: str | Sequence[str],
        device: str = "cpu",
        model: Any | Sequence[Any] | None = None,
        model_weights: Sequence[float] | None = None,
        rrf_k: int = 60,
    ):
        names = (model_name,) if isinstance(model_name, str) else tuple(model_name)
        self.model_names = tuple(str(name).strip() for name in names if str(name).strip())
        if not self.model_names:
            raise ValueError("재순위 모델이 하나 이상 필요합니다.")
        if rrf_k < 0:
            raise ValueError("rrf_k는 0 이상이어야 합니다.")
        self.rrf_k = rrf_k
        self.device = device
        weights = tuple(float(value) for value in model_weights) if model_weights is not None else (1.0,) * len(self.model_names)
        if len(weights) != len(self.model_names):
            raise ValueError("재순위 모델 수와 RERANKER_WEIGHTS 개수가 일치해야 합니다.")
        if any(not math.isfinite(weight) or weight < 0 for weight in weights) or not any(weights):
            raise ValueError("재순위 가중치는 0 이상의 유한한 수여야 하며 하나는 0보다 커야 합니다.")
        self.model_weights = weights
        if model is None:
            self._models: list[Any | None] = [None] * len(self.model_names)
        elif isinstance(model, Sequence) and not isinstance(model, (str, bytes)):
            self._models = list(model)
        else:
            self._models = [model]
        if len(self._models) != len(self.model_names):
            raise ValueError("주입한 재순위 모델 수와 모델 이름 개수가 일치해야 합니다.")
        self.model_name = self.model_names[0] if len(self.model_names) == 1 else ",".join(self.model_names)

    def _get_model(self, index: int) -> Any:
        if self._models[index] is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("CrossEncoder 사용에 sentence-transformers가 필요합니다.") from exc
            self._models[index] = CrossEncoder(self.model_names[index], device=self.device)
        return self._models[index]

    @staticmethod
    def _sort_key(item: Evidence) -> tuple[float, str, str]:
        return (-item.score, item.date or "99999999", item.id)

    def rerank(self, query: str, evidence: list[Evidence], limit: int) -> list[Evidence]:
        if not evidence or limit < 1:
            return []
        pairs = [[query, item.text[:4000]] for item in evidence]
        active_model_scores = [
            (
                [float(score) for score in self._get_model(index).predict(
                    pairs, batch_size=8, show_progress_bar=False
                )],
                weight,
            )
            for index, weight in enumerate(self.model_weights)
            if weight > 0
        ]
        if len(active_model_scores) == 1:
            scores, _ = active_model_scores[0]
            ranked = [
                Evidence(**{**item.__dict__, "score": score})
                for item, score in zip(evidence, scores)
            ]
            ranked.sort(key=self._sort_key)
            return ranked[:limit]

        fused_scores: dict[str, float] = {item.id: 0.0 for item in evidence}
        for scores, weight in active_model_scores:
            order = sorted(
                range(min(len(evidence), len(scores))),
                key=lambda index: (-scores[index], evidence[index].date or "99999999", evidence[index].id),
            )
            for rank, index in enumerate(order, start=1):
                fused_scores[evidence[index].id] += weight / (self.rrf_k + rank)
        ranked = [
            Evidence(**{**item.__dict__, "score": fused_scores[item.id]})
            for item in evidence
        ]
        ranked.sort(key=self._sort_key)
        return ranked[:limit]

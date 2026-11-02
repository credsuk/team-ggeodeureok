from __future__ import annotations

from typing import Any

from src.models import Evidence


class CrossEncoderReranker:
    """Re-score query/source pairs with a multilingual local cross-encoder."""

    def __init__(self, model_name: str, device: str = "cpu", model: Any | None = None):
        self.model_name = model_name
        self.device = device
        self._model = model

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("CrossEncoder 사용에 sentence-transformers가 필요합니다.") from exc
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, evidence: list[Evidence], limit: int) -> list[Evidence]:
        if not evidence or limit < 1:
            return []
        model = self._get_model()
        pairs = [[query, item.text[:4000]] for item in evidence]
        scores = model.predict(pairs, batch_size=8, show_progress_bar=False)
        ranked = [
            Evidence(**{**item.__dict__, "score": float(score)})
            for item, score in zip(evidence, scores)
        ]
        ranked.sort(key=lambda item: (-item.score, item.date or "99999999", item.id))
        return ranked[:limit]

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import math
from typing import Any

from src.core.models import RankedHit


def reciprocal_rank_fusion(
    dense: Sequence[dict[str, Any]],
    lexical: Sequence[dict[str, Any]],
    k: int = 60,
    *,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
) -> list[RankedHit]:
    """Merge dense and lexical result lists by stable ID using weighted RRF."""
    if k < 0:
        raise ValueError("k는 0 이상이어야 합니다.")
    weights = (float(dense_weight), float(lexical_weight))
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("검색 가중치는 유한한 0 이상의 수여야 합니다.")
    if not any(weights):
        raise ValueError("dense_weight와 lexical_weight 중 하나는 0보다 커야 합니다.")

    totals: dict[str, float] = defaultdict(float)
    dense_scores: dict[str, float] = {}
    lexical_scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for results, scores, weight in (
        (dense, dense_scores, weights[0]),
        (lexical, lexical_scores, weights[1]),
    ):
        if weight == 0:
            continue
        for rank, item in enumerate(results, start=1):
            source_id = str(item.get("id") or item.get("chroma_id") or "").strip()
            if not source_id:
                continue
            totals[source_id] += weight / (k + rank)
            scores[source_id] = float(item.get("score", 0.0) or 0.0)
            payloads.setdefault(source_id, {}).update(item)
    return [
        RankedHit(
            id=source_id,
            rrf_score=score,
            dense_score=dense_scores.get(source_id, 0.0),
            lexical_score=lexical_scores.get(source_id, 0.0),
            payload=payloads[source_id],
        )
        for source_id, score in sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))
    ]

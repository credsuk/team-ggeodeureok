from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from src.models import Evidence, RankedHit


def reciprocal_rank_fusion(
    dense: Sequence[dict[str, Any]],
    lexical: Sequence[dict[str, Any]],
    k: int = 60,
) -> list[RankedHit]:
    """Merge result lists by stable source ID; lower K values are useful in tests."""
    if k < 0:
        raise ValueError("k는 0 이상이어야 합니다.")
    totals: dict[str, float] = defaultdict(float)
    dense_scores: dict[str, float] = {}
    lexical_scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for results, scores in ((dense, dense_scores), (lexical, lexical_scores)):
        for rank, item in enumerate(results, start=1):
            source_id = str(item.get("id") or item.get("chroma_id") or "").strip()
            if not source_id:
                continue
            totals[source_id] += 1.0 / (k + rank)
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


def validate_citations(
    citations: Sequence[str],
    evidence: Sequence[Evidence],
    *,
    fallback_to_first: bool = True,
) -> list[str]:
    """Keep retrieved source IDs; optionally fall back to the first evidence ID."""
    allowed = {item.id for item in evidence if item.id and item.text.strip()}
    valid: list[str] = []
    for citation in citations:
        if citation in allowed and citation not in valid:
            valid.append(citation)
    if not valid and evidence and fallback_to_first:
        first = next((item.id for item in evidence if item.id and item.text.strip()), None)
        if first:
            valid.append(first)
    return valid

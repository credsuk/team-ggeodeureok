"""Compatibility exports for ranking helpers used by existing search code."""

from collections.abc import Sequence

from model.ensemble.rank_fusion import reciprocal_rank_fusion
from src.core.models import Evidence


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


__all__ = ["reciprocal_rank_fusion", "validate_citations"]

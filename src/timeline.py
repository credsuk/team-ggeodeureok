from __future__ import annotations

from src.models import Evidence


def build_timeline(evidence: list[Evidence]) -> tuple[Evidence | None, list[Evidence]]:
    """Sort unique retrieved evidence chronologically; first is corpus-search first observed."""
    by_id: dict[str, Evidence] = {}
    for item in evidence:
        if item.id and item.text.strip():
            by_id.setdefault(item.id, item)
    ordered = sorted(by_id.values(), key=lambda item: (item.date or "99999999", item.assembly, item.id))
    return (ordered[0] if ordered else None, ordered)

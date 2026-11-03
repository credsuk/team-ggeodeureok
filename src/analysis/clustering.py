from __future__ import annotations

from collections import defaultdict

from src.core.models import Evidence


def cluster_evidence(evidence: list[Evidence]) -> list[dict[str, object]]:
    """Group retrieved records by the canonical core_issue produced in metadata JSONL."""
    groups: dict[str, list[Evidence]] = defaultdict(list)
    labels: dict[str, str] = {}
    for item in evidence:
        raw = str(item.metadata.get("core_issue") or item.metadata.get("policy_category") or "미분류").strip()
        key = " ".join(raw.casefold().split()) or "미분류"
        labels.setdefault(key, raw or "미분류")
        groups[key].append(item)
    clusters = []
    for key, items in groups.items():
        ordered = sorted(items, key=lambda x: (x.date or "99999999", x.assembly, x.id))
        dates = [item.date for item in ordered if item.date]
        clusters.append({
            "issue": labels[key],
            "count": len(ordered),
            "assemblies": sorted({item.assembly for item in ordered if item.assembly}),
            "date_start": min(dates) if dates else "",
            "date_end": max(dates) if dates else "",
            "source_ids": [item.id for item in ordered],
        })
    return sorted(clusters, key=lambda item: (-int(item["count"]), str(item["issue"])))

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.models import Evidence, QueryIntent


def select_spread_evidence(
    evidence: list[Evidence],
    excluded_assemblies: set[str],
    limit: int = 8,
) -> list[Evidence]:
    """Select relevant records from other councils with council diversity first."""
    candidates = [
        item for item in evidence
        if item.id and item.text.strip() and item.assembly and item.assembly not in excluded_assemblies
    ]
    if limit <= 0:
        return []
    selected: list[Evidence] = []
    seen_ids: set[str] = set()
    seen_assemblies: set[str] = set()
    for item in candidates:
        if item.assembly in seen_assemblies:
            continue
        selected.append(item)
        seen_ids.add(item.id)
        seen_assemblies.add(item.assembly)
        if len(selected) >= limit:
            return selected
    for item in candidates:
        if item.id in seen_ids:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _date_digits(value: str) -> str:
    return "".join(character for character in str(value or "") if character.isdigit())[:8]


def build_regional_spread(
    intent: QueryIntent,
    evidence: list[Evidence],
    summarized_ids: set[str] | None = None,
    *,
    other_council_search_performed: bool = False,
) -> dict[str, Any]:
    """Summarize cross-council observations without inferring policy transfer."""
    groups: dict[str, list[Evidence]] = defaultdict(list)
    allowed = summarized_ids if summarized_ids is not None else None
    for item in evidence:
        assembly = " ".join(str(item.assembly or "").split())
        if not assembly or not item.id or (allowed is not None and item.id not in allowed):
            continue
        groups[assembly].append(item)

    requested = set(intent.assemblies)
    scope_limited = bool(requested) and not other_council_search_performed
    origin_groups = {name: rows for name, rows in groups.items() if name in requested}
    if requested:
        target_groups = {name: rows for name, rows in groups.items() if name not in requested}
        has_comparable_target = bool(target_groups)
    else:
        target_groups = groups
        has_comparable_target = len(target_groups) >= 2

    def summarize_groups(source_groups: dict[str, list[Evidence]]) -> list[dict[str, Any]]:
        rows_out: list[dict[str, Any]] = []
        for assembly, rows in sorted(source_groups.items()):
            dated = sorted(
                (date for item in rows if (date := _date_digits(item.date))),
            )
            ordered = sorted(rows, key=lambda item: (_date_digits(item.date) or "99999999", item.id))
            rows_out.append({
                "assembly": assembly,
                "record_count": len(rows),
                "date_range": {"start": dated[0], "end": dated[-1]} if dated else None,
                "source_ids": list(dict.fromkeys(item.id for item in ordered))[:3],
            })
        return rows_out

    origin_councils = summarize_groups(origin_groups)
    observed_councils = summarize_groups(target_groups)
    origin_record_count = sum(row["record_count"] for row in origin_councils)

    if has_comparable_target:
        names = ", ".join(row["assembly"] for row in observed_councils)
        basis_count = sum(row["record_count"] for row in observed_councils)
        if requested:
            origin_names = ", ".join(row["assembly"] for row in origin_councils) or ", ".join(sorted(requested))
            outlook = (
                f"기준 의회 {origin_names} 관련 요약 근거 {origin_record_count}건과 비교해 "
                f"{names}에서도 관련 논의가 확인됩니다({basis_count}건). "
                "이는 여러 의회에서 관련 주제가 관측됐다는 뜻일 뿐, 한 의회에서 다른 의회로 정책이 전파됐다는 관계를 입증하지 않습니다. "
                "추가 확산은 조건부 가능성으로만 볼 수 있으며, 현재 자료만으로 확산 확률이나 영향 방향을 산정할 수 없습니다."
            )
        else:
            outlook = (
                f"요약 가능한 관련 검색 근거에서 {names}의 논의가 확인됩니다({basis_count}건). "
                "이는 여러 의회에서 관련 주제가 관측됐다는 뜻일 뿐, 한 의회에서 다른 의회로 정책이 전파됐다는 관계를 입증하지 않습니다. "
                "추가 확산은 조건부 가능성으로만 볼 수 있으며, 현재 자료만으로 확산 확률이나 영향 방향을 산정할 수 없습니다."
            )
        status = "observed_related_discussions"
    else:
        basis_count = sum(row["record_count"] for row in observed_councils)
        if scope_limited:
            outlook = (
                "질문에 지정한 의회 범위로 검색이 제한되어 다른 시·도 확산 가능성을 판단할 수 없습니다. "
                "이는 다른 의회에서 관련 논의가 없다는 뜻은 아닙니다."
            )
        elif requested:
            origin_names = ", ".join(row["assembly"] for row in origin_councils) or ", ".join(sorted(requested))
            outlook = (
                f"기준 의회 {origin_names}의 요약 근거는 있으나, 타 시·도 검색 후보에서 전망에 활용할 요약 근거를 확보하지 못했습니다. "
                "확산 가능성은 판단할 수 없으며, 이는 전체 corpus에 관련 논의가 없다는 뜻은 아닙니다."
            )
        else:
            names = ", ".join(row["assembly"] for row in observed_councils) or "확인 가능한 의회"
            outlook = (
                f"현재 요약 가능한 검색 근거에서는 {names} 외의 다른 의회 논의를 확인하지 못했습니다. "
                "검색된 후보 범위만으로는 타 시·도 확산 가능성을 판단할 수 없으며, 이는 전체 corpus에 관련 기록이 없다는 뜻은 아닙니다."
            )
        status = "insufficient_evidence"

    return {
        "status": status,
        "scope_limited": scope_limited,
        "cross_council_search_performed": other_council_search_performed,
        "evidence_scope": "요약 가능한 검색·재순위화 근거의 일부이며 전체 corpus가 아닙니다.",
        "basis_record_count": basis_count,
        "origin_record_count": origin_record_count,
        "origin_councils": origin_councils,
        "observed_councils": observed_councils,
        "outlook": outlook,
        "causality_established": False,
        "probability_estimated": False,
    }

"""광역의회 기준 정보: 지도 좌표와 통합 규칙 (config/council_merges.json).

  · 전남광주통합특별시의회 = 광주광역시의회 + 전라남도의회 (2026-07-01부터)
  · 시점(as_of)이 시행일 이후면 전신 의회를 통합 의회 하나로 표시하고, 이전이면 전신 의회를 그대로 쓴다
  · 외부 자료의 통합 지자체 기관명('전남광주통합특별시 순천시')은 하위 지역으로 원래 소속을 되찾는다
  · COUNCIL_COORDINATES는 지도 payload(24·33)가 함께 쓰는 단일 좌표표 — 통합 의회 좌표 포함
"""

from __future__ import annotations

import json
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "config" / "council_merges.json"


@lru_cache(maxsize=1)
def groups() -> tuple[dict[str, Any], ...]:
    if not CONFIG.is_file():
        return ()
    return tuple(json.loads(CONFIG.read_text(encoding="utf-8")).get("merges") or [])


def _day(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value or "")[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def find_group(council: str) -> dict[str, Any] | None:
    """통합 의회 또는 그 전신 의회가 속한 통합 규칙"""
    for group in groups():
        if council == group["council"] or council in member_councils(group):
            return group
    return None


def member_councils(group: dict[str, Any]) -> list[str]:
    return [member["council"] for member in group["members"]]


def members_of(council: str) -> list[str]:
    """통합 의회면 전신 의회 목록, 아니면 [자기 자신]"""
    group = find_group(council)
    return member_councils(group) if group and council == group["council"] else [council]


def is_effective(group: dict[str, Any], as_of: Any) -> bool:
    day = _day(as_of)
    return bool(day and day >= _day(group["effective_date"]))


def display_council(council: str, as_of: Any) -> str:
    """시점 기준 표시 의회: 시행일 이후의 전신 의회 → 통합 의회"""
    group = find_group(council)
    if group and council in member_councils(group) and is_effective(group, as_of):
        return group["council"]
    return council


def region_forms() -> dict[str, list[str]]:
    """통합 의회 → 통합 지자체 표기 목록 (긴 표기 우선)"""
    return {group["council"]: sorted(group["region_names"], key=len, reverse=True) for group in groups()}


def org_council(org: str) -> tuple[str | None, str | None]:
    """통합 지자체 기관명 → (의회, 단계). 본청·교육청은 통합 의회 광역, 하위 시·군·구는 전신 의회 기초"""
    text = (org or "").strip()
    for group in groups():
        for form in sorted(group["region_names"], key=len, reverse=True):
            if not text.startswith(form):
                continue
            rest = text[len(form):].strip()
            if rest in ("", "교육청"):
                return group["council"], "광역"
            sub = rest.split()[0]
            for member in group["members"]:
                if sub in member.get("sub_regions") or []:
                    return member["council"], "기초"
            return group["council"], "기초"
    return None, None


def coordinates(council: str) -> tuple[float, float] | None:
    for group in groups():
        point = group.get("coordinates") or {}
        if council == group["council"] and point.get("latitude") is not None:
            return float(point["latitude"]), float(point["longitude"])
    return None


COUNCIL_COORDINATES = {
    "서울특별시의회": (37.5665, 126.9780),
    "부산광역시의회": (35.1796, 129.0756),
    "대구광역시의회": (35.8714, 128.6014),
    "인천광역시의회": (37.4563, 126.7052),
    "광주광역시의회": (35.1595, 126.8526),
    "대전광역시의회": (36.3504, 127.3845),
    "울산광역시의회": (35.5384, 129.3114),
    "세종특별자치시의회": (36.4800, 127.2890),
    "경기도의회": (37.4138, 127.5183),
    "강원특별자치도의회": (37.8228, 128.1555),
    "충청북도의회": (36.6357, 127.4917),
    "충청남도의회": (36.6588, 126.6728),
    "전북특별자치도의회": (35.8203, 127.1088),
    "전라북도의회": (35.8203, 127.1088),
    "전라남도의회": (34.8679, 126.9910),
    "경상북도의회": (36.5760, 128.5056),
    "경상남도의회": (35.4606, 128.2132),
    "제주특별자치도의회": (33.4996, 126.5312),
}
# 통합 의회 좌표 — 통합 이후 관측이 들어와도 지도에서 빠지지 않게
COUNCIL_COORDINATES.update({
    group["council"]: coordinates(group["council"]) for group in groups() if coordinates(group["council"])
})

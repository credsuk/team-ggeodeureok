#!/usr/bin/env python3
"""검색어와 지역·위원회 필터의 제한적인 별칭을 관리한다.

별칭은 제공된 회의록의 표기 차이를 함께 조회하기 위한 용도다. 외부 시점의
정보를 추가하지 않으며, 원본 council 값은 변경하지 않는다.
"""

from __future__ import annotations

from collections.abc import Iterable


# 전남/전라남도 표기와 특별자치도 전후 표기를 같은 검색 그룹으로 취급한다.
# DB에 실제로 존재하는 값만 검색 결과로 반환되므로 존재하지 않는 별칭을
# 포함해도 새로운 데이터를 생성하지 않는다.
COUNCIL_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("전라남도", "전남", "전라남도의회", "전남도의회"),
    (
        "전라북도", "전북", "전라북도의회",
        "전북특별자치도", "전북특별자치도의회",
    ),
    (
        "강원도", "강원도의회",
        "강원특별자치도", "강원특별자치도의회",
    ),
)


# 위원회 개편 전후 명칭을 필터에서 함께 조회할 수 있게 한다.
COMMITTEE_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("행정자치위원회", "기획행정위원회"),
)


def _normalize(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def _alias_values(
    value: str | None,
    groups: Iterable[tuple[str, ...]],
) -> tuple[str, ...]:
    normalized = _normalize(value)
    if not normalized:
        return ()
    for group in groups:
        if normalized in group:
            return tuple(dict.fromkeys(group))
    return (normalized,)


def council_alias_values(value: str | None) -> tuple[str, ...]:
    """전남/전라남도 등 같은 지역의 실제 DB 필터 후보를 반환한다."""
    return _alias_values(value, COUNCIL_ALIAS_GROUPS)


def committee_alias_values(value: str | None) -> tuple[str, ...]:
    """위원회 개편 전후 명칭의 DB 필터 후보를 반환한다."""
    return _alias_values(value, COMMITTEE_ALIAS_GROUPS)


def expand_query_aliases(query: str) -> str:
    """질의에 등장한 지역·위원회 표기의 같은 그룹 표현을 BM25용으로 덧붙인다."""
    normalized = _normalize(query)
    additions: list[str] = []
    for group in COUNCIL_ALIAS_GROUPS:
        if any(alias in normalized for alias in group):
            additions.extend(alias for alias in group if alias not in normalized)
    for group in COMMITTEE_ALIAS_GROUPS:
        if any(alias in normalized for alias in group):
            additions.extend(alias for alias in group if alias not in normalized)
    return " ".join([normalized, *dict.fromkeys(additions)]).strip()


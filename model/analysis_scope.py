"""Enforce the competition's configured temporal and council analysis scope."""
from copy import copy
from datetime import datetime
import re

ANALYSIS_START_YEAR = "2024"
ANALYSIS_END_YEAR = "2025"
METROPOLITAN_COUNCILS = frozenset({
    "서울특별시의회", "부산광역시의회", "대구광역시의회", "인천광역시의회",
    "광주광역시의회", "대전광역시의회", "울산광역시의회", "세종특별자치시의회",
    "경기도의회", "강원특별자치도의회", "충청북도의회", "충청남도의회",
    "전북특별자치도의회", "전라남도의회", "경상북도의회", "경상남도의회",
    "제주특별자치도의회",
})


def _normalize(value):
    value = str(value or "").strip()
    value = value.replace("강원도의회", "강원특별자치도의회")
    value = value.replace("전라북도의회", "전북특별자치도의회")
    return re.sub(r"[\W_]+", "", value.lower(), flags=re.UNICODE)


_COUNCIL_NAMES = {_normalize(name): name for name in METROPOLITAN_COUNCILS}
_COUNCIL_NAMES[_normalize("강원도의회")] = "강원특별자치도의회"
_COUNCIL_NAMES[_normalize("전라북도의회")] = "전북특별자치도의회"


def normalize_council(value):
    """Return a canonical council name for supported aliases, else normalized text."""
    raw = str(value or "").strip()
    raw = raw.replace("강원도의회", "강원특별자치도의회")
    raw = raw.replace("전라북도의회", "전북특별자치도의회")
    normalized = _normalize(raw)
    return _COUNCIL_NAMES.get(normalized, normalized)


def apply_analysis_period(params):
    """Intersect user-requested years with the fixed 2024–2025 project period."""
    result = dict(params)
    if params.get("start_year") and params.get("end_year") and params["start_year"] > params["end_year"]:
        result["start_year"] = "1"
        result["end_year"] = "0"
    else:
        result["start_year"] = max(ANALYSIS_START_YEAR, params.get("start_year") or ANALYSIS_START_YEAR)
        result["end_year"] = min(ANALYSIS_END_YEAR, params.get("end_year") or ANALYSIS_END_YEAR)
    return result


def in_analysis_scope(metadata, params=None):
    """Whether a record has a valid date and council inside the fixed project scope."""
    params = params or {}
    date = str(metadata.get("date") or metadata.get("MTG_DE") or "")
    digits = "".join(character for character in date if character.isdigit())
    if len(digits) != 8 or not (
        date.isdigit()
        or len(digits) == 8 and date.count("-") == 2
        or len(digits) == 8 and date.count("/") == 2
    ):
        return False
    try:
        meeting_date = datetime.strptime(digits[:8], "%Y%m%d")
    except ValueError:
        return False
    year = digits[:4]
    if not (ANALYSIS_START_YEAR <= year <= ANALYSIS_END_YEAR):
        return False
    start = params.get("start_year") or ANALYSIS_START_YEAR
    end = params.get("end_year") or ANALYSIS_END_YEAR
    if year < start or year > end:
        return False
    council = normalize_council(metadata.get("assembly") or metadata.get("RASMBLY_NM"))
    if council not in {_normalize(name) for name in METROPOLITAN_COUNCILS}:
        return False
    requested_councils = params.get("councils") or []
    if requested_councils and council not in {normalize_council(name) for name in requested_councils}:
        return False
    return True

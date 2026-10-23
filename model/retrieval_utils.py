"""Small pure helpers for reconstructing chunked rows and ordered date scans."""
from collections import defaultdict

TOPIC_TERMS = ("지역 소멸", "지방소멸", "지방소멸대응기금")


def expand_topic_terms(topic, query, terms):
    """Add known equivalent search phrases when the query names one of them."""
    normalized = "".join(f"{topic or ''}{query or ''}".split()).lower()
    aliases = tuple("".join(term.split()).lower() for term in TOPIC_TERMS)
    expanded = list(terms or [])
    if any(alias in normalized for alias in aliases):
        expanded.extend(TOPIC_TERMS)
    return list(dict.fromkeys(expanded))


def merge_overlapping_chunks(chunks, overlap):
    """Join (chunk_index, text) pairs while dropping repeated chunk boundaries."""
    ordered = sorted(chunks, key=lambda item: item[0])
    if not ordered:
        return ""

    def chunk_text(item):
        return item[2] if len(item) >= 3 else item[1]

    text = chunk_text(ordered[0]) or ""
    previous_chunk = text
    overlap = max(0, int(overlap))
    for item in ordered[1:]:
        chunk = chunk_text(item) or ""
        limit = min(overlap, len(previous_chunk), len(chunk))
        matched = 0
        for size in range(limit, 0, -1):
            if previous_chunk[-size:] == chunk[:size]:
                matched = size
                break
        text += chunk[matched:]
        previous_chunk = chunk
    return text


def _date_key(value):
    digits = "".join(character for character in value if character.isdigit())
    return digits[:8] if len(digits) >= 8 else value


def normalize_date_key(value):
    """Normalize common meeting date strings for reliable chronological ordering."""
    return _date_key(str(value or ""))


def group_candidates_by_date(candidates):
    """Group lexical chunks by normalized date and deduplicate to source rows."""
    grouped = defaultdict(dict)
    for candidate in candidates:
        metadata = candidate.get("metadata") or {}
        date = _date_key(str(metadata.get("date") or metadata.get("MTG_DE") or ""))
        if not date:
            continue
        row_id = metadata.get("row_id")
        dataset_key = metadata.get("dataset_key")
        key = (str(dataset_key), str(row_id)) if row_id not in (None, "") else ("id", str(candidate.get("id")))
        grouped[date].setdefault(key, candidate)
    return [(date, list(rows.values())) for date, rows in sorted(grouped.items())]


def first_day_and_limited_followups(candidates, check_day, followup_limit=100):
    """Find the first relevant day and collect bounded relevant rows after it."""
    first_date = None
    first_rows = []
    later_rows = []
    for date, day_candidates in group_candidates_by_date(candidates):
        relevant = check_day(day_candidates)
        if not relevant:
            continue
        if first_date is None:
            first_date, first_rows = date, relevant
            continue
        later_rows.extend(relevant)
        if len(later_rows) >= followup_limit:
            break
    return first_date, first_rows, later_rows

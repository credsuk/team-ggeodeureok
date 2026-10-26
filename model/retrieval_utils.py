"""Pure helpers for reconstructing chunks and deriving ranked-candidate timelines."""

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


def first_observation_from_candidates(rows, followup_limit=100):
    """Derive timeline rows from already retrieved/reranked source rows; never access DB."""
    dated_rows = []
    for row in rows or []:
        metadata = row.get("metadata") or {}
        date = normalize_date_key(str(metadata.get("date") or metadata.get("MTG_DE") or ""))
        if date:
            dated_rows.append((date, row))
    dated_rows.sort(key=lambda item: item[0])
    if not dated_rows:
        return None, [], []

    first_date = dated_rows[0][0]
    first_rows = [row for date, row in dated_rows if date == first_date]
    limit = max(0, int(followup_limit))
    later_rows = [row for date, row in dated_rows if date > first_date][:limit]
    return first_date, first_rows, later_rows

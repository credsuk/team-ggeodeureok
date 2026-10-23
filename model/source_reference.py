"""Stable citation fields mapped from source-row metadata."""


def reference_metadata(metadata):
    """Return meeting title, agenda, and source-row identifiers without invention."""
    metadata = metadata or {}
    row_id = metadata.get("row_id")
    try:
        row_id = int(row_id) if row_id not in (None, "") else None
    except (TypeError, ValueError):
        row_id = None
    return {
        "meetingTitle": str(metadata.get("MINTS_NM") or metadata.get("meeting_title") or ""),
        "agendaTitle": str(metadata.get("agenda_title") or metadata.get("agenda") or metadata.get("MTGNM") or ""),
        "datasetKey": str(metadata.get("dataset_key") or ""),
        "originalRowId": row_id,
        "sourceId": f"{metadata.get('dataset_key')}:{row_id}" if metadata.get("dataset_key") and row_id is not None else "",
    }

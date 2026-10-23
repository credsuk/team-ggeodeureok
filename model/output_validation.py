"""Pure validation for source-linked claims in the serialized response."""


def _validate_reference(reference, known_source_ids):
    source_id = reference.get("sourceId")
    if source_id not in known_source_ids:
        raise ValueError(f"timeline sourceId is not present in evidence: {source_id}")
    dataset_key = reference.get("datasetKey")
    row_id = reference.get("originalRowId")
    if not dataset_key or not isinstance(row_id, int):
        raise ValueError(f"source-row provenance is missing: {source_id}")
    expected_source_id = f"{dataset_key}:{row_id}"
    if source_id != expected_source_id:
        raise ValueError(f"sourceId does not match source-row provenance: {source_id}")


def validate_output_evidence(output):
    """Raise ValueError when citations or source-row identifiers are inconsistent."""
    for result in output.get("results", []):
        evidence = result.get("evidence", [])
        known = {item.get("sourceId") for item in evidence if item.get("sourceId")}
        for reference in evidence:
            _validate_reference(reference, known)

        claims = list(result.get("summaryEvidenceIds", []))
        claims.extend((result.get("forecast") or {}).get("evidenceIds", []))
        missing = [source_id for source_id in claims if source_id not in known]
        if missing:
            raise ValueError(f"claim evidence IDs are not present in evidence: {missing[:5]}")
        for reference in [*result.get("firstObserved", []), *result.get("subsequentDiscussions", [])]:
            _validate_reference(reference, known)
    return True

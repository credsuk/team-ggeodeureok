"""Inspect configured transcript/metadata inputs against the declared corpus scope."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

from analysis_scope import ANALYSIS_START_YEAR, ANALYSIS_END_YEAR, METROPOLITAN_COUNCILS, normalize_council
from _constant import METADATA_ROOT, TRANSCRIPT_ROOT


def _load_metadata(path):
    rows = {}
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except (ValueError, TypeError):
                raise ValueError(f"{path}:{line_number}: invalid JSONL record") from None
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: metadata record must be an object")
            row_id = item.get("row_id")
            if not isinstance(row_id, int) or row_id < 1 or row_id in rows:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate row_id {row_id!r}")
            rows[row_id] = item
    return rows


def _paired_files(metadata_root, transcript_root):
    metadata = {path.stem: path for path in metadata_root.rglob("*.jsonl")}
    transcripts = {path.stem: path for path in transcript_root.rglob("*.txt")}
    if len(metadata) != len(list(metadata_root.rglob("*.jsonl"))):
        raise ValueError("duplicate JSONL stems; cannot pair inputs safely")
    if len(transcripts) != len(list(transcript_root.rglob("*.txt"))):
        raise ValueError("duplicate TSV stems; cannot pair inputs safely")
    if metadata.keys() != transcripts.keys():
        raise ValueError(
            "metadata/transcript pair mismatch: "
            f"missing transcripts={sorted(metadata.keys() - transcripts.keys())[:10]}, "
            f"missing metadata={sorted(transcripts.keys() - metadata.keys())[:10]}"
        )
    if not metadata:
        raise ValueError(f"no JSONL/TXT pairs found: {metadata_root}, {transcript_root}")
    return [(key, metadata[key], transcripts[key]) for key in sorted(metadata)]


def inspect_input_coverage(metadata_root=METADATA_ROOT, transcript_root=TRANSCRIPT_ROOT):
    metadata_root, transcript_root = Path(metadata_root), Path(transcript_root)
    if not metadata_root.is_dir() or not transcript_root.is_dir():
        raise FileNotFoundError(f"input roots missing: {metadata_root}, {transcript_root}")
    pairs = _paired_files(metadata_root, transcript_root)
    years, councils, observed_dates, rows_by_dataset, checks = set(), set(), [], 0, []
    observed_council_years = set()
    for key, metadata_path, transcript_path in pairs:
        metadata = _load_metadata(metadata_path)
        found, count = set(), 0
        with transcript_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            if reader.fieldnames not in (
                ["MTG_DE", "RASMBLY_NM", "MTGNM", "MINTS_NM", "agenda_title", "speaker_role", "speaker_name", "speaker_type", "utterance"],
                ["date", "assembly", "agenda_title", "speaker_role", "speaker_name", "speaker_type", "utterance"],
            ):
                raise ValueError(f"{transcript_path}: transcript header mismatch: {reader.fieldnames}")
            for physical_row, row in enumerate(reader, 1):
                if not row or not any(str(value or "").strip() for value in row.values()):
                    continue
                if None in row or any(row.get(field) is None for field in reader.fieldnames):
                    raise ValueError(f"{key}: transcript row {physical_row + 1} has malformed field count")
                count += 1
                item = metadata.get(count)
                if item is None:
                    raise ValueError(f"{key}: transcript data row {count} lacks metadata")
                found.add(count)
                tdate = row.get("MTG_DE") or row.get("date") or ""
                mdate = item.get("MTG_DE") or item.get("date") or ""
                if not tdate.strip():
                    raise ValueError(f"{key}: row {count} blank MTG_DE in transcript (metadata fallback is unsafe for indexing)")
                if not (row.get("RASMBLY_NM") or row.get("assembly") or "").strip():
                    raise ValueError(f"{key}: row {count} blank RASMBLY_NM in transcript (metadata fallback is unsafe for indexing)")
                if mdate and tdate and str(mdate) != str(tdate):
                    raise ValueError(f"{key}: row {count} date mismatch")
                tcouncil = row.get("RASMBLY_NM") or row.get("assembly") or ""
                mcouncil = item.get("RASMBLY_NM") or item.get("assembly") or ""
                if mcouncil and tcouncil and normalize_council(mcouncil) != normalize_council(tcouncil):
                    raise ValueError(f"{key}: row {count} council mismatch")
                date_digits = "".join(ch for ch in str(tdate or mdate) if ch.isdigit())
                normalized_council = normalize_council(tcouncil)
                year = date_digits[:4] if len(date_digits) >= 4 else ""
                if normalized_council in {normalize_council(name) for name in METROPOLITAN_COUNCILS} and year:
                    observed_council_years.add((normalized_council, year))
                if len(date_digits) >= 8:
                    try:
                        normalized_date = datetime.strptime(date_digits[:8], "%Y%m%d").date().isoformat()
                    except ValueError as exc:
                        raise ValueError(f"{key}: row {count} has invalid meeting date {tdate!r}") from exc
                    observed_dates.append(normalized_date)
                if len(date_digits) >= 4:
                    years.add(date_digits[:4])
                if tcouncil or mcouncil:
                    councils.add(str(tcouncil or mcouncil).strip())
        missing = sorted(set(metadata) - found)
        if missing:
            raise ValueError(f"{key}: {len(missing)} metadata rows lack transcript records")
        checks.append({"dataset": key, "rows": count, "metadata_rows": len(metadata)})

    expected_years = {str(year) for year in range(int(ANALYSIS_START_YEAR), int(ANALYSIS_END_YEAR) + 1)}
    expected_councils = {normalize_council(name): name for name in METROPOLITAN_COUNCILS}
    observed_councils = {council for council, _ in observed_council_years}
    missing_councils = sorted(expected_councils[key] for key in expected_councils.keys() - observed_councils)
    missing_council_year_combinations = [
        {"council": expected_councils[council], "year": year}
        for council in sorted(expected_councils)
        for year in sorted(expected_years)
        if (council, year) not in observed_council_years
    ]
    missing_years = sorted(expected_years - years)
    return {
        "metadata_root": str(metadata_root), "transcript_root": str(transcript_root),
        "paired_datasets": len(pairs), "source_rows": sum(item["rows"] for item in checks),
        "years_present": sorted(years), "councils_present": sorted(councils),
        "date_min": min(observed_dates) if observed_dates else None,
        "date_max": max(observed_dates) if observed_dates else None,
        "missing_years": missing_years,
        "missing_councils": missing_councils,
        "missing_council_year_combinations": missing_council_year_combinations,
        "pairs": checks,
        "scope_complete": not missing_council_year_combinations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT)
    parser.add_argument("--transcript-root", type=Path, default=TRANSCRIPT_ROOT)
    parser.add_argument("--output", type=Path, help="Optional JSON report destination")
    args = parser.parse_args()
    report = inspect_input_coverage(args.metadata_root, args.transcript_root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["scope_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

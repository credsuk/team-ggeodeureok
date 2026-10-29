#!/usr/bin/env python3
"""회의록 TSV와 LLM metadata JSONL의 (source_key, row_id) 결합을 검증한다."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_PATTERN = re.compile(r"^cleaned_(20\d{2})_(.+?)\.(?:txt|jsonl|ndjson)$")
SPACE_PATTERN = re.compile(r"\s+")
DATE_DIGITS = re.compile(r"\D+")
csv.field_size_limit(min(sys.maxsize, 100_000_000))
TEXT_COLUMN_ALIASES = {
    "date": ("MTG_DE", "date", "meeting_date"),
    "assembly": ("RASMBLY_NM", "assembly", "council"),
    "committee": ("MTGNM", "committee"),
    "meeting_name": ("MINTS_NM", "meeting_name"),
    "agenda": ("agenda_title", "agenda"),
    "speaker": ("speaker_name", "speaker"),
    "utterance": ("utterance", "text"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "input")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "restart_v2" / "row_id_validation",
    )
    parser.add_argument("--max-mismatch-samples", type=int, default=20)
    return parser.parse_args()


def source_key(path: Path) -> str | None:
    match = FILE_PATTERN.match(path.name)
    return f"{match.group(1)}_{match.group(2)}" if match else None


def normalize_text(value: Any) -> str:
    return SPACE_PATTERN.sub(" ", "" if value is None else str(value)).strip()


def normalize_date(value: Any) -> str:
    digits = DATE_DIGITS.sub("", "" if value is None else str(value))
    return digits[:8] if len(digits) >= 8 else digits


def parse_row_id(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    text = "" if value is None else str(value).strip()
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    return None


def resolve_text_columns(headers: list[str]) -> dict[str, str | None]:
    header_set = set(headers)
    return {
        canonical: next((name for name in aliases if name in header_set), None)
        for canonical, aliases in TEXT_COLUMN_ALIASES.items()
    }


def open_tsv(path: Path):
    file = path.open("r", encoding="utf-8-sig", errors="replace", newline="")
    first_line = file.readline()
    file.seek(0)
    delimiter = "\t" if first_line.count("\t") >= first_line.count(",") else ","
    reader = csv.DictReader(file, delimiter=delimiter)
    return file, reader, delimiter


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, None, f"{type(exc).__name__}: {exc}"
                continue
            if not isinstance(value, dict):
                yield line_number, None, "JSON object가 아님"
                continue
            yield line_number, value, None


def validate_pair(
    key: str,
    text_path: Path,
    metadata_path: Path,
    mismatch_limit: int,
) -> dict[str, Any]:
    text_file, text_reader, delimiter = open_tsv(text_path)
    headers = text_reader.fieldnames or []
    text_columns = resolve_text_columns(headers)
    missing_text_columns = sorted(
        canonical for canonical in ("date", "assembly", "utterance")
        if text_columns[canonical] is None
    )

    text_rows = 0
    metadata_rows = 0
    invalid_json_rows = 0
    missing_row_id = 0
    duplicate_row_id = 0
    non_integer_row_id = 0
    order_breaks = 0
    date_mismatches = 0
    assembly_mismatches = 0
    source_key_mismatches = 0
    agenda_exact_matches = 0
    speaker_exact_matches = 0
    replacement_chars_text = 0
    replacement_chars_metadata = 0
    mismatch_samples: list[dict[str, Any]] = []
    seen_row_ids: set[str] = set()
    numeric_ids: list[int] = []
    previous_numeric_id: int | None = None

    try:
        pairs = zip_longest(text_reader, iter_jsonl(metadata_path), fillvalue=None)
        for pair_index, (text_row, metadata_item) in enumerate(pairs):
            if text_row is not None:
                text_rows += 1
                replacement_chars_text += sum(str(value).count("\ufffd") for value in text_row.values())
            if metadata_item is not None:
                metadata_rows += 1
                line_number, metadata_row, error = metadata_item
            else:
                line_number, metadata_row, error = None, None, None

            if error:
                invalid_json_rows += 1
            if text_row is None or metadata_row is None:
                if len(mismatch_samples) < mismatch_limit:
                    mismatch_samples.append({
                        "pair_index": pair_index,
                        "metadata_line": line_number,
                        "reason": error or ("회의록 행 부족" if text_row is None else "metadata 행 부족"),
                    })
                continue

            replacement_chars_metadata += sum(
                str(value).count("\ufffd") for value in metadata_row.values()
            )
            row_id_raw = metadata_row.get("row_id")
            if row_id_raw is None or not str(row_id_raw).strip():
                missing_row_id += 1
            else:
                row_id_key = str(row_id_raw).strip()
                if row_id_key in seen_row_ids:
                    duplicate_row_id += 1
                else:
                    seen_row_ids.add(row_id_key)
                numeric_id = parse_row_id(row_id_raw)
                if numeric_id is None:
                    non_integer_row_id += 1
                else:
                    numeric_ids.append(numeric_id)
                    if previous_numeric_id is not None and numeric_id != previous_numeric_id + 1:
                        order_breaks += 1
                    previous_numeric_id = numeric_id

            text_date = normalize_date(text_row.get(text_columns["date"]))
            metadata_date = normalize_date(metadata_row.get("date"))
            if text_date != metadata_date:
                date_mismatches += 1
                if len(mismatch_samples) < mismatch_limit:
                    mismatch_samples.append({
                        "pair_index": pair_index,
                        "row_id": row_id_raw,
                        "reason": "date 불일치",
                        "text_date": text_date,
                        "metadata_date": metadata_date,
                    })

            text_assembly = normalize_text(text_row.get(text_columns["assembly"]))
            metadata_assembly = normalize_text(metadata_row.get("assembly"))
            if text_assembly != metadata_assembly:
                assembly_mismatches += 1
                if len(mismatch_samples) < mismatch_limit:
                    mismatch_samples.append({
                        "pair_index": pair_index,
                        "row_id": row_id_raw,
                        "reason": "assembly 불일치",
                    })

            metadata_source_key = normalize_text(metadata_row.get("source_key"))
            if metadata_source_key and metadata_source_key != key:
                source_key_mismatches += 1
                if len(mismatch_samples) < mismatch_limit:
                    mismatch_samples.append({
                        "pair_index": pair_index,
                        "row_id": row_id_raw,
                        "reason": "source_key 불일치",
                        "expected": key,
                        "actual": metadata_source_key,
                    })

            if normalize_text(text_row.get(text_columns["agenda"])) == normalize_text(metadata_row.get("agenda")):
                agenda_exact_matches += 1
            if normalize_text(text_row.get(text_columns["speaker"])) == normalize_text(metadata_row.get("speaker")):
                speaker_exact_matches += 1
    finally:
        text_file.close()

    row_id_basis = "non_integer"
    if numeric_ids:
        minimum, maximum = min(numeric_ids), max(numeric_ids)
        if minimum == 0 and maximum == metadata_rows - 1 and order_breaks == 0:
            row_id_basis = "zero_based_sequential"
        elif minimum == 1 and maximum == metadata_rows and order_breaks == 0:
            row_id_basis = "one_based_sequential"
        else:
            row_id_basis = "integer_nonsequential"
    else:
        minimum, maximum = None, None

    blocking_errors = []
    if missing_text_columns:
        blocking_errors.append("회의록 필수 열 누락")
    if text_rows != metadata_rows:
        blocking_errors.append("행 수 불일치")
    if invalid_json_rows:
        blocking_errors.append("JSONL 오류")
    if missing_row_id or duplicate_row_id:
        blocking_errors.append("row_id 누락/중복")
    if date_mismatches or assembly_mismatches:
        blocking_errors.append("date/assembly 행 정렬 불일치")
    if source_key_mismatches:
        blocking_errors.append("source_key 불일치")

    return {
        "source_key": key,
        "text_file": str(text_path),
        "metadata_file": str(metadata_path),
        "delimiter": "TAB" if delimiter == "\t" else "COMMA",
        "text_headers": headers,
        "resolved_text_columns": text_columns,
        "missing_text_columns": missing_text_columns,
        "text_rows": text_rows,
        "metadata_rows": metadata_rows,
        "row_count_match": text_rows == metadata_rows,
        "invalid_json_rows": invalid_json_rows,
        "missing_row_id": missing_row_id,
        "duplicate_row_id": duplicate_row_id,
        "non_integer_row_id": non_integer_row_id,
        "row_id_basis": row_id_basis,
        "row_id_min": minimum,
        "row_id_max": maximum,
        "row_id_order_breaks": order_breaks,
        "date_mismatches": date_mismatches,
        "assembly_mismatches": assembly_mismatches,
        "source_key_mismatches": source_key_mismatches,
        "agenda_exact_match_ratio": round(agenda_exact_matches / metadata_rows, 6) if metadata_rows else None,
        "speaker_exact_match_ratio": round(speaker_exact_matches / metadata_rows, 6) if metadata_rows else None,
        "replacement_chars_text": replacement_chars_text,
        "replacement_chars_metadata": replacement_chars_metadata,
        "blocking_errors": blocking_errors,
        "safe_to_join": not blocking_errors,
        "mismatch_samples": mismatch_samples,
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    text_paths = sorted(
        path for path in input_dir.rglob("*.txt")
        if path.parent.name in {"2024_광역 회의록", "2025_광역 회의록"}
        and ":Zone.Identifier" not in path.name
    )
    metadata_root = input_dir / "metadata_jsonl_python"
    metadata_paths = sorted(
        path for path in metadata_root.rglob("*.jsonl")
        if ":Zone.Identifier" not in path.name
    )
    text_map = {source_key(path): path for path in text_paths if source_key(path)}
    metadata_map = {source_key(path): path for path in metadata_paths if source_key(path)}

    missing_metadata = sorted(set(text_map) - set(metadata_map))
    extra_metadata = sorted(set(metadata_map) - set(text_map))
    paired_keys = sorted(set(text_map) & set(metadata_map))
    reports = []
    for index, key in enumerate(paired_keys, start=1):
        print(f"[{index}/{len(paired_keys)}] {key}")
        reports.append(validate_pair(
            key,
            text_map[key],
            metadata_map[key],
            args.max_mismatch_samples,
        ))

    total_text_rows = sum(report["text_rows"] for report in reports)
    total_metadata_rows = sum(report["metadata_rows"] for report in reports)
    failed = [report["source_key"] for report in reports if not report["safe_to_join"]]
    row_id_basis_counts = Counter(report["row_id_basis"] for report in reports)
    summary = {
        "meeting_files": len(text_map),
        "metadata_files": len(metadata_map),
        "paired_files": len(paired_keys),
        "missing_metadata_files": missing_metadata,
        "extra_metadata_files": extra_metadata,
        "paired_text_rows": total_text_rows,
        "paired_metadata_rows": total_metadata_rows,
        "row_id_basis_counts": dict(row_id_basis_counts),
        "total_duplicate_row_ids_within_file": sum(report["duplicate_row_id"] for report in reports),
        "total_date_mismatches": sum(report["date_mismatches"] for report in reports),
        "total_assembly_mismatches": sum(report["assembly_mismatches"] for report in reports),
        "total_source_key_mismatches": sum(report["source_key_mismatches"] for report in reports),
        "failed_pair_keys": failed,
        "safe_to_direct_join": not failed and not missing_metadata and not extra_metadata,
        "safe_to_left_join": not failed and not extra_metadata,
        "join_key": ["source_key", "row_id"],
        "expected_policy": "현재 수정 TXT의 각 데이터 행과 JSONL 한 행이 1:1 대응",
        "files": reports,
    }
    json_path = output_dir / "row_id_validation_report.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / "row_id_validation_files.csv"
    csv_fields = [
        "source_key", "text_rows", "metadata_rows", "row_count_match", "row_id_basis",
        "duplicate_row_id", "row_id_order_breaks", "date_mismatches", "assembly_mismatches",
        "source_key_mismatches",
        "agenda_exact_match_ratio", "speaker_exact_match_ratio", "replacement_chars_text",
        "replacement_chars_metadata", "safe_to_join", "blocking_errors",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        for report in reports:
            writer.writerow({field: report.get(field) for field in csv_fields})

    print("\n=== row_id 결합 검증 완료 ===")
    print(f"대응 파일: {len(paired_keys)} | metadata 누락: {missing_metadata or '없음'}")
    print(f"회의록 행: {total_text_rows:,} | metadata 행: {total_metadata_rows:,}")
    print(f"row_id 방식: {dict(row_id_basis_counts)}")
    print(f"파일 내부 row_id 중복: {summary['total_duplicate_row_ids_within_file']:,}")
    print(f"date 불일치: {summary['total_date_mismatches']:,}")
    print(f"assembly 불일치: {summary['total_assembly_mismatches']:,}")
    print(f"source_key 불일치: {summary['total_source_key_mismatches']:,}")
    print(f"검증 실패 파일: {failed or '없음'}")
    print(f"직접 JOIN 가능: {summary['safe_to_direct_join']}")
    print(f"보고서: {json_path}")
    print(f"파일별 요약: {csv_path}")


if __name__ == "__main__":
    main()

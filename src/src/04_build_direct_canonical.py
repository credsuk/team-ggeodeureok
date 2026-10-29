#!/usr/bin/env python3
"""수정 TXT와 신규 LLM metadata를 1:1 결합해 canonical utterances를 생성한다."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_PATTERN = re.compile(r"^cleaned_(20\d{2})_(.+?)\.(?:txt|jsonl|ndjson)$")
SPACE_PATTERN = re.compile(r"\s+")
NON_DIGIT_PATTERN = re.compile(r"\D+")
csv.field_size_limit(min(sys.maxsize, 100_000_000))

OUTPUT_SCHEMA = pa.schema([
    pa.field("utterance_id", pa.string(), nullable=False),
    pa.field("document_id", pa.string(), nullable=False),
    pa.field("source_key", pa.string(), nullable=False),
    pa.field("source_row_id", pa.int64(), nullable=False),
    pa.field("source_file", pa.string(), nullable=False),
    pa.field("metadata_file", pa.string(), nullable=False),
    pa.field("council", pa.string(), nullable=False),
    pa.field("meeting_date", pa.string(), nullable=False),
    pa.field("year", pa.int16(), nullable=False),
    pa.field("committee", pa.string()),
    pa.field("meeting_name", pa.string()),
    pa.field("agenda_title", pa.string()),
    pa.field("speaker_name", pa.string()),
    pa.field("speaker_role", pa.string()),
    pa.field("speaker_type", pa.string()),
    pa.field("utterance", pa.string(), nullable=False),
    pa.field("llm_row_id", pa.int64(), nullable=False),
    pa.field("llm_agenda", pa.string()),
    pa.field("llm_speaker", pa.string()),
    pa.field("core_issue", pa.string()),
    pa.field("keywords_json", pa.string()),
    pa.field("policy_category", pa.string()),
    pa.field("summary", pa.string()),
    pa.field("is_substantive", pa.bool_()),
    pa.field("parse_status", pa.string()),
    pa.field("metadata_status", pa.string(), nullable=False),
    pa.field("text_sha256", pa.string(), nullable=False),
    pa.field("char_length", pa.int32(), nullable=False),
    pa.field("replacement_char_count", pa.int32(), nullable=False),
    pa.field("embedding_text_raw", pa.string(), nullable=False),
    pa.field("embedding_text_enriched", pa.string(), nullable=False),
    pa.field("canonical_version", pa.string(), nullable=False),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "input")
    parser.add_argument(
        "--metadata-root", type=Path,
        default=PROJECT_ROOT / "data" / "input" / "metadata_jsonl_python",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "canonical_v3")
    parser.add_argument(
        "--report-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "restart_v3" / "canonical_build",
    )
    parser.add_argument("--batch-rows", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_string(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", clean_string(value)).replace("\u00a0", " ")
    return SPACE_PATTERN.sub(" ", text).strip()


def normalize_date(value: Any) -> str:
    digits = NON_DIGIT_PATTERN.sub("", clean_string(value))
    return digits[:8] if len(digits) >= 8 else digits


def iso_date(value: Any) -> str:
    digits = normalize_date(value)
    if len(digits) != 8:
        raise ValueError(f"잘못된 날짜: {value!r}")
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def stable_id(prefix: str, *values: Any) -> str:
    payload = "\x1f".join(clean_string(value) for value in values)
    return f"{prefix}_{digest(payload)[:32]}"


def source_key(path: Path) -> str | None:
    match = FILE_PATTERN.match(path.name)
    return f"{match.group(1)}_{match.group(2)}" if match else None


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "예", "실질", "substantive"}:
        return True
    if text in {"false", "0", "no", "n", "아니오", "비실질", "non-substantive"}:
        return False
    return None


def render_keywords(value: Any) -> tuple[str | None, str]:
    if value is None:
        return None, ""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None, ""
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [item.strip() for item in re.split(r"[,|]", text) if item.strip()]
    else:
        parsed = value
    encoded = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    if isinstance(parsed, list):
        rendered = ", ".join(map(str, parsed))
    else:
        rendered = normalize_text(parsed)
    return encoded, rendered


def discover_files(input_dir: Path, metadata_root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    text_paths = sorted(
        path for path in input_dir.rglob("*.txt")
        if path.parent.name in {"2024_광역 회의록", "2025_광역 회의록"}
        and ":Zone.Identifier" not in path.name
    )
    metadata_paths = sorted(
        path for path in metadata_root.rglob("*.jsonl")
        if ":Zone.Identifier" not in path.name
    )
    text_map = {source_key(path): path for path in text_paths if source_key(path)}
    metadata_map = {source_key(path): path for path in metadata_paths if source_key(path)}
    return text_map, metadata_map


def iter_metadata(path: Path) -> Iterator[tuple[int, dict[str, Any] | None, str | None]]:
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


def raw_embedding_text(agenda: str | None, utterance: str) -> str:
    return f"[안건] {agenda}\n[발언] {utterance}" if agenda else f"[발언] {utterance}"


def enriched_embedding_text(raw: str, metadata: dict[str, Any]) -> tuple[str, str | None]:
    keywords_encoded, keywords_rendered = render_keywords(metadata.get("keywords"))
    parts = []
    for label, value in (
        ("정책분야", metadata.get("policy_category")),
        ("핵심이슈", metadata.get("core_issue")),
        ("키워드", keywords_rendered),
        ("요약", metadata.get("summary")),
    ):
        rendered = normalize_text(value)
        if rendered:
            parts.append(f"[{label}] {rendered}")
    parts.append(raw)
    return "\n".join(parts), keywords_encoded


def build_row(
    key: str,
    row_id: int,
    text_row: dict[str, Any],
    metadata: dict[str, Any],
    text_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    council = normalize_text(text_row.get("RASMBLY_NM"))
    meeting_date = iso_date(text_row.get("MTG_DE"))
    committee = normalize_text(text_row.get("MTGNM")) or None
    meeting_name = normalize_text(text_row.get("MINTS_NM")) or None
    agenda = normalize_text(text_row.get("agenda_title")) or None
    speaker_name = normalize_text(text_row.get("speaker_name")) or None
    speaker_role = normalize_text(text_row.get("speaker_role")) or None
    speaker_type = normalize_text(text_row.get("speaker_type")) or None
    utterance = normalize_text(text_row.get("utterance"))
    text_hash = digest(utterance)
    document_id = stable_id("doc", council, meeting_date, committee, meeting_name)
    utterance_id = stable_id("utt", key, row_id, text_hash)
    raw_text = raw_embedding_text(agenda, utterance)
    enriched_text, keywords_encoded = enriched_embedding_text(raw_text, metadata)
    parse_status = normalize_text(metadata.get("parse_status")) or "success"
    return {
        "utterance_id": utterance_id,
        "document_id": document_id,
        "source_key": key,
        "source_row_id": row_id,
        "source_file": str(text_path),
        "metadata_file": str(metadata_path),
        "council": council,
        "meeting_date": meeting_date,
        "year": int(meeting_date[:4]),
        "committee": committee,
        "meeting_name": meeting_name,
        "agenda_title": agenda,
        "speaker_name": speaker_name,
        "speaker_role": speaker_role,
        "speaker_type": speaker_type,
        "utterance": utterance,
        "llm_row_id": int(metadata["row_id"]),
        "llm_agenda": normalize_text(metadata.get("agenda")) or None,
        "llm_speaker": normalize_text(metadata.get("speaker")) or None,
        "core_issue": normalize_text(metadata.get("core_issue")) or None,
        "keywords_json": keywords_encoded,
        "policy_category": normalize_text(metadata.get("policy_category")) or None,
        "summary": normalize_text(metadata.get("summary")) or None,
        "is_substantive": parse_bool(metadata.get("is_substantive")),
        "parse_status": parse_status,
        "metadata_status": "direct_row_id_match" if parse_status == "success" else f"direct_{parse_status}",
        "text_sha256": text_hash,
        "char_length": len(utterance),
        "replacement_char_count": utterance.count("\ufffd"),
        "embedding_text_raw": raw_text,
        "embedding_text_enriched": enriched_text,
        "canonical_version": "direct_v3",
    }


def process_pair(
    key: str,
    text_path: Path,
    metadata_path: Path,
    writer: pq.ParquetWriter,
    batch_rows: int,
) -> dict[str, Any]:
    text_count = 0
    metadata_count = 0
    invalid_json = 0
    row_id_mismatches = 0
    date_mismatches = 0
    assembly_mismatches = 0
    source_key_mismatches = 0
    empty_utterances = 0
    replacement_chars = 0
    substantive_counts: Counter[str] = Counter()
    parse_status_counts: Counter[str] = Counter()
    buffer: list[dict[str, Any]] = []

    with text_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as text_file:
        reader = csv.DictReader(text_file, delimiter="\t")
        required = {
            "MTG_DE", "RASMBLY_NM", "MTGNM", "MINTS_NM", "agenda_title",
            "speaker_role", "speaker_name", "speaker_type", "utterance",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{text_path.name} 필수 열 누락: {missing}")

        for expected_row_id, pair in enumerate(
            zip_longest(reader, iter_metadata(metadata_path), fillvalue=None), start=1
        ):
            text_row, metadata_item = pair
            if text_row is not None:
                text_count += 1
            if metadata_item is not None:
                metadata_count += 1
                _, metadata, error = metadata_item
            else:
                metadata, error = None, None
            if error:
                invalid_json += 1
            if text_row is None or metadata is None:
                continue

            try:
                metadata_row_id = int(metadata.get("row_id"))
            except (TypeError, ValueError):
                metadata_row_id = -1
            row_id_mismatches += int(metadata_row_id != expected_row_id)
            date_mismatches += int(
                normalize_date(text_row.get("MTG_DE")) != normalize_date(metadata.get("date"))
            )
            assembly_mismatches += int(
                normalize_text(text_row.get("RASMBLY_NM")) != normalize_text(metadata.get("assembly"))
            )
            metadata_source_key = normalize_text(metadata.get("source_key"))
            source_key_mismatches += int(bool(metadata_source_key) and metadata_source_key != key)

            output_row = build_row(
                key, expected_row_id, text_row, metadata, text_path, metadata_path
            )
            empty_utterances += int(not output_row["utterance"])
            replacement_chars += output_row["replacement_char_count"]
            substantive_counts[str(output_row["is_substantive"])] += 1
            parse_status_counts[output_row["parse_status"]] += 1
            buffer.append(output_row)
            if len(buffer) >= batch_rows:
                writer.write_table(pa.Table.from_pylist(buffer, schema=OUTPUT_SCHEMA))
                buffer.clear()

    if buffer:
        writer.write_table(pa.Table.from_pylist(buffer, schema=OUTPUT_SCHEMA))

    blocking_errors = []
    if text_count != metadata_count:
        blocking_errors.append("행 수 불일치")
    if invalid_json:
        blocking_errors.append("JSONL 오류")
    if row_id_mismatches:
        blocking_errors.append("row_id 불일치")
    if date_mismatches:
        blocking_errors.append("date 불일치")
    if assembly_mismatches:
        blocking_errors.append("assembly 불일치")
    if source_key_mismatches:
        blocking_errors.append("source_key 불일치")
    return {
        "source_key": key,
        "text_rows": text_count,
        "metadata_rows": metadata_count,
        "invalid_json_rows": invalid_json,
        "row_id_mismatches": row_id_mismatches,
        "date_mismatches": date_mismatches,
        "assembly_mismatches": assembly_mismatches,
        "source_key_mismatches": source_key_mismatches,
        "empty_utterances": empty_utterances,
        "replacement_characters": replacement_chars,
        "is_substantive_counts": dict(substantive_counts),
        "parse_status_counts": dict(parse_status_counts),
        "blocking_errors": blocking_errors,
        "success": not blocking_errors,
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    metadata_root = args.metadata_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "utterances.parquet"
    partial_path = output_dir / "utterances.partial.parquet"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"이미 존재합니다. 교체하려면 --overwrite 사용: {output_path}")
    if partial_path.exists():
        partial_path.unlink()

    text_map, metadata_map = discover_files(input_dir, metadata_root)
    missing_metadata = sorted(set(text_map) - set(metadata_map))
    extra_metadata = sorted(set(metadata_map) - set(text_map))
    if missing_metadata or extra_metadata:
        raise RuntimeError(
            f"파일 대응 실패 | metadata 누락={missing_metadata} | 회의록 누락={extra_metadata}"
        )
    keys = sorted(text_map)
    if len(keys) != 34:
        raise RuntimeError(f"예상 파일은 34개인데 {len(keys)}개 발견")

    reports = []
    writer = pq.ParquetWriter(partial_path, OUTPUT_SCHEMA, compression="zstd")
    try:
        for index, key in enumerate(keys, start=1):
            print(f"[{index}/{len(keys)}] {key}")
            report = process_pair(
                key, text_map[key], metadata_map[key], writer, args.batch_rows
            )
            reports.append(report)
            print(
                f"  rows={report['text_rows']:,} "
                f"empty={report['empty_utterances']:,} "
                f"replacement={report['replacement_characters']:,} "
                f"success={report['success']}"
            )
    finally:
        writer.close()

    failed = [report["source_key"] for report in reports if not report["success"]]
    total_rows = sum(report["text_rows"] for report in reports)
    summary = {
        "canonical_version": "direct_v3",
        "file_count": len(keys),
        "total_rows": total_rows,
        "total_metadata_rows": sum(report["metadata_rows"] for report in reports),
        "total_empty_utterances": sum(report["empty_utterances"] for report in reports),
        "total_replacement_characters": sum(report["replacement_characters"] for report in reports),
        "failed_files": failed,
        "success": not failed,
        "join_key": ["source_key", "row_id"],
        "output": str(output_path),
        "files": reports,
    }
    report_path = report_dir / "canonical_build_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = report_dir / "canonical_build_files.csv"
    fields = [
        "source_key", "text_rows", "metadata_rows", "invalid_json_rows",
        "row_id_mismatches", "date_mismatches", "assembly_mismatches",
        "source_key_mismatches", "empty_utterances", "replacement_characters",
        "success", "blocking_errors",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        csv_writer = csv.DictWriter(file, fieldnames=fields)
        csv_writer.writeheader()
        for report in reports:
            csv_writer.writerow({field: report.get(field) for field in fields})

    if failed:
        print(f"\n검증 실패로 partial 파일만 남김: {partial_path}")
        raise RuntimeError(f"canonical 생성 실패: {failed}")
    if output_path.exists() and args.overwrite:
        output_path.unlink()
    partial_path.replace(output_path)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "canonical_version": "direct_v3",
        "file": str(output_path),
        "row_count": total_rows,
        "schema": OUTPUT_SCHEMA.names,
        "join_key": ["source_key", "row_id"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== direct canonical 생성 완료 ===")
    print(f"파일: {len(keys)}")
    print(f"행: {total_rows:,}")
    print(f"빈 발언: {summary['total_empty_utterances']:,}")
    print(f"깨진 문자: {summary['total_replacement_characters']:,}")
    print(f"canonical: {output_path}")
    print(f"보고서: {report_path}")


if __name__ == "__main__":
    main()

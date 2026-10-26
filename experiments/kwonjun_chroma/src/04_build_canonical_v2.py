#!/usr/bin/env python3
"""구 canonical을 다리로 사용해 수정 TXT와 LLM metadata를 안전하게 결합한다.

1) 구 corpus.parquet 2,028,420행과 JSONL metadata를 파일별 row_id로 결합한다.
2) 구 원문과 수정 TXT 원문을 날짜·의회·정규화 본문 해시로 다시 매칭한다.
3) 수정 TXT에만 존재하는 신규 행도 보존하고 metadata_status로 구분한다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_PATTERN = re.compile(r"cleaned_(20\d{2})_(.+?)\.(?:txt|jsonl|ndjson)$")
SPACE_PATTERN = re.compile(r"\s+")
NON_DIGIT_PATTERN = re.compile(r"\D+")
csv.field_size_limit(min(sys.maxsize, 100_000_000))

OLD_COLUMNS = [
    "chunk_id", "council", "meeting_date", "agenda_title", "speaker",
    "speaker_role", "speaker_type", "source_file", "source_line_no", "text",
]

OUTPUT_SCHEMA = pa.schema([
    pa.field("utterance_id", pa.string(), nullable=False),
    pa.field("document_id", pa.string(), nullable=False),
    pa.field("source_key", pa.string(), nullable=False),
    pa.field("source_row_id", pa.int64(), nullable=False),
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
    pa.field("llm_row_id", pa.int64()),
    pa.field("llm_agenda", pa.string()),
    pa.field("core_issue", pa.string()),
    pa.field("keywords_json", pa.string()),
    pa.field("policy_category", pa.string()),
    pa.field("summary", pa.string()),
    pa.field("is_substantive", pa.bool_()),
    pa.field("metadata_status", pa.string(), nullable=False),
    pa.field("old_chunk_id", pa.string()),
    pa.field("text_sha256", pa.string(), nullable=False),
    pa.field("replacement_char_count", pa.int32(), nullable=False),
    pa.field("embedding_text_raw", pa.string(), nullable=False),
    pa.field("embedding_text_enriched", pa.string(), nullable=False),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-corpus", type=Path, default=PROJECT_ROOT / "data" / "canonical" / "corpus.parquet")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "input")
    parser.add_argument(
        "--metadata-root", type=Path,
        default=PROJECT_ROOT / "data" / "input" / "metadata_jsonl_python",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "canonical_v2")
    parser.add_argument(
        "--report-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "restart_v2" / "canonical_v2",
    )
    parser.add_argument("--key", help="예: 2024_강원특별자치도의회. 지정하면 해당 파일만 검증")
    parser.add_argument("--validate-only", action="store_true", help="결합률만 검사하고 canonical을 쓰지 않음")
    parser.add_argument("--rebuild-staging", action="store_true")
    parser.add_argument("--min-match-ratio", type=float, default=0.98)
    parser.add_argument("--batch-rows", type=int, default=10_000)
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
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}" if len(digits) == 8 else digits


def source_key_from_value(value: Any) -> str | None:
    match = FILE_PATTERN.search(Path(clean_string(value)).name)
    return f"{match.group(1)}_{match.group(2)}" if match else None


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def bridge_fingerprint(date: Any, council: Any, agenda: Any, text: Any) -> str:
    payload = "\x1f".join((
        normalize_date(date), normalize_text(council),
        normalize_text(agenda), normalize_text(text),
    ))
    return digest_text(payload)


def relaxed_bridge_fingerprint(date: Any, council: Any, text: Any) -> str:
    """안건명이 수정된 경우를 위한 2차 키. 원문은 반드시 정확히 같아야 한다."""
    payload = "\x1f".join((
        normalize_date(date), normalize_text(council), normalize_text(text),
    ))
    return digest_text(payload)


def stable_id(prefix: str, *values: Any) -> str:
    payload = "\x1f".join(clean_string(value) for value in values)
    return f"{prefix}_{digest_text(payload)[:32]}"


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


def keywords_json(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        except json.JSONDecodeError:
            return json.dumps([part.strip() for part in re.split(r"[,|]", text) if part.strip()], ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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
    text_map = {source_key_from_value(path): path for path in text_paths if source_key_from_value(path)}
    metadata_map = {source_key_from_value(path): path for path in metadata_paths if source_key_from_value(path)}
    return text_map, metadata_map


def partition_old_corpus(old_corpus: Path, staging_dir: Path, rebuild: bool) -> dict[str, Path]:
    manifest_path = staging_dir / "manifest.json"
    expected_signature = {
        "path": str(old_corpus.resolve()),
        "size": old_corpus.stat().st_size,
        "mtime_ns": old_corpus.stat().st_mtime_ns,
    }
    if not rebuild and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source") == expected_signature:
            paths = {path.stem: path for path in staging_dir.glob("*.parquet")}
            if paths:
                print(f"구 canonical staging 재사용: {len(paths)}개")
                return paths

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(old_corpus)
    writers: dict[str, pq.ParquetWriter] = {}
    paths: dict[str, Path] = {}
    counts: Counter[str] = Counter()
    try:
        for batch_index, batch in enumerate(parquet.iter_batches(batch_size=32_768, columns=OLD_COLUMNS), start=1):
            rows_by_key: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in batch.to_pylist():
                key = source_key_from_value(row.get("source_file"))
                if key:
                    rows_by_key[key].append(row)
            for key, rows in rows_by_key.items():
                table = pa.Table.from_pylist(rows, schema=batch.schema)
                if key not in writers:
                    path = staging_dir / f"{key}.parquet"
                    paths[key] = path
                    writers[key] = pq.ParquetWriter(path, table.schema, compression="zstd")
                writers[key].write_table(table)
                counts[key] += len(rows)
            if batch_index % 10 == 0:
                print(f"구 canonical 분할 중: {sum(counts.values()):,}행")
    finally:
        for writer in writers.values():
            writer.close()
    manifest_path.write_text(json.dumps({
        "source": expected_signature,
        "counts": dict(counts),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def load_old_rows(path: Path) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    rows.sort(key=lambda row: (int(row.get("source_line_no") or 0), clean_string(row.get("chunk_id"))))
    return rows


def load_old_rows_for_key(old_corpus: Path, key: str) -> list[dict[str, Any]]:
    result = []
    parquet = pq.ParquetFile(old_corpus)
    for batch in parquet.iter_batches(batch_size=32_768, columns=OLD_COLUMNS):
        for row in batch.to_pylist():
            if source_key_from_value(row.get("source_file")) == key:
                result.append(row)
    result.sort(key=lambda row: (int(row.get("source_line_no") or 0), clean_string(row.get("chunk_id"))))
    return result


def load_metadata(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    invalid = 0
    with path.open("r", encoding="utf-8-sig", errors="replace") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if isinstance(value, dict):
                rows.append(value)
            else:
                invalid += 1
    rows.sort(key=lambda row: int(row.get("row_id") or 0))
    return rows, invalid


def make_bridge(old_rows: list[dict[str, Any]], metadata_rows: list[dict[str, Any]]) -> tuple[dict[str, deque], dict[str, Any]]:
    bridge: defaultdict[str, deque] = defaultdict(deque)
    date_mismatches = 0
    council_mismatches = 0
    row_id_mismatches = 0
    for index, (old, metadata) in enumerate(zip(old_rows, metadata_rows), start=1):
        if int(metadata.get("row_id") or 0) != index:
            row_id_mismatches += 1
        if normalize_date(old.get("meeting_date")) != normalize_date(metadata.get("date")):
            date_mismatches += 1
        if normalize_text(old.get("council")) != normalize_text(metadata.get("assembly")):
            council_mismatches += 1
        fingerprint = bridge_fingerprint(
            old.get("meeting_date"), old.get("council"),
            old.get("agenda_title"), old.get("text"),
        )
        bridge[fingerprint].append((old, metadata))
    return bridge, {
        "old_rows": len(old_rows),
        "metadata_rows": len(metadata_rows),
        "old_metadata_count_match": len(old_rows) == len(metadata_rows),
        "row_id_mismatches": row_id_mismatches,
        "date_mismatches_old_metadata": date_mismatches,
        "council_mismatches_old_metadata": council_mismatches,
    }


def iter_new_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required = {"MTG_DE", "RASMBLY_NM", "MTGNM", "MINTS_NM", "agenda_title", "speaker_role", "speaker_name", "speaker_type", "utterance"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path.name} 필수 열 누락: {missing}")
        for row_id, row in enumerate(reader, start=1):
            yield row_id, row


def enriched_text(row: dict[str, Any], metadata: dict[str, Any] | None) -> tuple[str, str]:
    agenda = normalize_text(row.get("agenda_title"))
    utterance = normalize_text(row.get("utterance"))
    raw = f"[안건] {agenda}\n[발언] {utterance}" if agenda else f"[발언] {utterance}"
    if not metadata:
        return raw, raw
    parts = []
    for label, field in (
        ("정책분야", "policy_category"),
        ("핵심이슈", "core_issue"),
        ("키워드", "keywords"),
        ("요약", "summary"),
    ):
        value = metadata.get(field)
        if value:
            rendered = ", ".join(map(str, value)) if isinstance(value, list) else normalize_text(value)
            parts.append(f"[{label}] {rendered}")
    parts.append(raw)
    return raw, "\n".join(parts)


def canonical_row(key: str, source_row_id: int, row: dict[str, Any], matched: tuple | None) -> dict[str, Any]:
    old, metadata, match_status = matched if matched else (None, None, "new_row_without_llm_metadata")
    council = normalize_text(row.get("RASMBLY_NM"))
    meeting_date = iso_date(row.get("MTG_DE"))
    committee = normalize_text(row.get("MTGNM")) or None
    meeting_name = normalize_text(row.get("MINTS_NM")) or None
    agenda = normalize_text(row.get("agenda_title")) or None
    speaker_name = normalize_text(row.get("speaker_name")) or None
    speaker_role = normalize_text(row.get("speaker_role")) or None
    speaker_type = normalize_text(row.get("speaker_type")) or None
    utterance = normalize_text(row.get("utterance"))
    text_hash = digest_text(utterance)
    document_id = stable_id("doc", council, meeting_date, committee, meeting_name)
    utterance_id = stable_id("utt", key, source_row_id, text_hash)
    raw_text, enriched = enriched_text(row, metadata)
    return {
        "utterance_id": utterance_id,
        "document_id": document_id,
        "source_key": key,
        "source_row_id": source_row_id,
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
        "llm_row_id": int(metadata.get("row_id")) if metadata else None,
        "llm_agenda": normalize_text(metadata.get("agenda")) or None if metadata else None,
        "core_issue": normalize_text(metadata.get("core_issue")) or None if metadata else None,
        "keywords_json": keywords_json(metadata.get("keywords")) if metadata else None,
        "policy_category": normalize_text(metadata.get("policy_category")) or None if metadata else None,
        "summary": normalize_text(metadata.get("summary")) or None if metadata else None,
        "is_substantive": parse_bool(metadata.get("is_substantive")) if metadata else None,
        "metadata_status": match_status,
        "old_chunk_id": clean_string(old.get("chunk_id")) or None if old else None,
        "text_sha256": text_hash,
        "replacement_char_count": utterance.count("\ufffd"),
        "embedding_text_raw": raw_text,
        "embedding_text_enriched": enriched,
    }


def process_key(
    key: str,
    new_path: Path,
    metadata_path: Path,
    old_rows: list[dict[str, Any]],
    writer: pq.ParquetWriter | None,
    batch_rows: int,
) -> dict[str, Any]:
    metadata_rows, invalid_json = load_metadata(metadata_path)
    bridge, stats = make_bridge(old_rows, metadata_rows)
    matches_by_row_id: dict[int, tuple] = {}
    unmatched_new_rows: list[tuple[int, dict[str, Any]]] = []
    strict_matched_count = 0
    relaxed_matched_count = 0
    new_count = 0
    agenda_matches = 0
    speaker_matches = 0

    def record_match(source_row_id: int, row: dict[str, Any], matched_pair: tuple, status: str) -> None:
        nonlocal agenda_matches, speaker_matches
        old, metadata = matched_pair
        matches_by_row_id[source_row_id] = (old, metadata, status)
        agenda_matches += int(
            normalize_text(row.get("agenda_title")) == normalize_text(old.get("agenda_title"))
        )
        new_speaker = normalize_text(row.get("speaker_name"))
        old_speaker = normalize_text(old.get("speaker"))
        speaker_matches += int(
            new_speaker == old_speaker
            or new_speaker.endswith(old_speaker)
            or old_speaker.endswith(new_speaker)
        )

    # 1차: 날짜 + 의회 + 안건 + 원문이 모두 같은 경우만 연결한다.
    for source_row_id, row in iter_new_rows(new_path):
        new_count += 1
        fingerprint = bridge_fingerprint(
            row.get("MTG_DE"), row.get("RASMBLY_NM"),
            row.get("agenda_title"), row.get("utterance"),
        )
        matched = bridge[fingerprint].popleft() if bridge.get(fingerprint) else None
        if matched:
            strict_matched_count += 1
            record_match(source_row_id, row, matched, "matched_strict")
        else:
            unmatched_new_rows.append((source_row_id, row))

    # 2차: 1차에서 남은 행끼리 날짜 + 의회 + 원문이 정확히 같은 경우만 연결한다.
    # 안건 표기 변경은 허용하지만 원문의 퍼지 매칭은 하지 않아 잘못된 연결을 막는다.
    relaxed_bridge: defaultdict[str, deque] = defaultdict(deque)
    leftover_pairs = [pair for queue in bridge.values() for pair in queue]
    leftover_pairs.sort(key=lambda pair: int(pair[0].get("source_line_no") or 0))
    for old, metadata in leftover_pairs:
        fingerprint = relaxed_bridge_fingerprint(
            old.get("meeting_date"), old.get("council"), old.get("text")
        )
        relaxed_bridge[fingerprint].append((old, metadata))

    for source_row_id, row in unmatched_new_rows:
        fingerprint = relaxed_bridge_fingerprint(
            row.get("MTG_DE"), row.get("RASMBLY_NM"), row.get("utterance")
        )
        matched = relaxed_bridge[fingerprint].popleft() if relaxed_bridge.get(fingerprint) else None
        if matched:
            relaxed_matched_count += 1
            record_match(source_row_id, row, matched, "matched_relaxed_agenda")

    matched_count = strict_matched_count + relaxed_matched_count

    # 출력 시에는 수정 TXT의 원래 행 순서를 유지하기 위해 두 번째로 순회한다.
    if writer:
        buffer = []
        for source_row_id, row in iter_new_rows(new_path):
            buffer.append(canonical_row(key, source_row_id, row, matches_by_row_id.get(source_row_id)))
            if len(buffer) >= batch_rows:
                writer.write_table(pa.Table.from_pylist(buffer, schema=OUTPUT_SCHEMA))
                buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=OUTPUT_SCHEMA))

    old_unmatched = sum(len(queue) for queue in relaxed_bridge.values())
    stats.update({
        "source_key": key,
        "invalid_metadata_json_rows": invalid_json,
        "new_rows": new_count,
        "strict_matched_rows": strict_matched_count,
        "relaxed_agenda_matched_rows": relaxed_matched_count,
        "matched_metadata_rows": matched_count,
        "new_rows_without_metadata": new_count - matched_count,
        "old_metadata_rows_not_found_in_new": old_unmatched,
        "old_to_new_match_ratio": round(matched_count / len(old_rows), 8) if old_rows else 0.0,
        "new_metadata_coverage": round(matched_count / new_count, 8) if new_count else 0.0,
        "agenda_exact_ratio_on_matches": round(agenda_matches / matched_count, 8) if matched_count else None,
        "speaker_compatible_ratio_on_matches": round(speaker_matches / matched_count, 8) if matched_count else None,
    })
    return stats


def main() -> None:
    args = parse_args()
    old_corpus = args.old_corpus.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    metadata_root = args.metadata_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    if not old_corpus.is_file():
        raise FileNotFoundError(old_corpus)

    text_map, metadata_map = discover_files(input_dir, metadata_root)
    common_keys = sorted(set(text_map) & set(metadata_map))
    if args.key:
        if args.key not in common_keys:
            raise KeyError(f"입력 또는 metadata에 없는 key: {args.key}")
        keys = [args.key]
        old_paths = None
    else:
        keys = common_keys
        staging_dir = report_dir / "old_corpus_by_source"
        old_paths = partition_old_corpus(old_corpus, staging_dir, args.rebuild_staging)

    writer = None
    output_path = output_dir / "utterances.parquet"
    if not args.validate_only and not args.key:
        output_dir.mkdir(parents=True, exist_ok=True)
        partial_path = output_dir / "utterances.partial.parquet"
        writer = pq.ParquetWriter(partial_path, OUTPUT_SCHEMA, compression="zstd")
    else:
        partial_path = None

    reports = []
    try:
        for index, key in enumerate(keys, start=1):
            print(f"[{index}/{len(keys)}] {key}")
            old_rows = load_old_rows_for_key(old_corpus, key) if args.key else load_old_rows(old_paths[key])
            report = process_key(
                key, text_map[key], metadata_map[key], old_rows, writer, args.batch_rows
            )
            reports.append(report)
            print(
                f"  old={report['old_rows']:,} new={report['new_rows']:,} "
                f"matched={report['matched_metadata_rows']:,} "
                f"ratio={report['old_to_new_match_ratio']:.4%}"
            )
    finally:
        if writer:
            writer.close()

    failed = [
        report["source_key"] for report in reports
        if not report["old_metadata_count_match"]
        or report["row_id_mismatches"]
        or report["date_mismatches_old_metadata"]
        or report["council_mismatches_old_metadata"]
        or report["old_to_new_match_ratio"] < args.min_match_ratio
    ]
    summary = {
        "old_corpus": str(old_corpus),
        "validated_keys": keys,
        "validate_only": args.validate_only or bool(args.key),
        "minimum_required_match_ratio": args.min_match_ratio,
        "total_old_rows": sum(report["old_rows"] for report in reports),
        "total_metadata_rows": sum(report["metadata_rows"] for report in reports),
        "total_new_rows": sum(report["new_rows"] for report in reports),
        "total_matched_metadata_rows": sum(report["matched_metadata_rows"] for report in reports),
        "total_new_rows_without_metadata": sum(report["new_rows_without_metadata"] for report in reports),
        "total_old_metadata_rows_not_found_in_new": sum(report["old_metadata_rows_not_found_in_new"] for report in reports),
        "failed_keys": failed,
        "safe_to_build": not failed,
        "files": reports,
    }
    report_path = report_dir / (f"bridge_{args.key}.json" if args.key else "canonical_v2_build_report.json")
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if partial_path:
        if failed:
            print(f"주의: 기준 미달로 partial 파일을 최종 파일로 확정하지 않음: {partial_path}")
        else:
            partial_path.replace(output_path)
            manifest_path = output_dir / "manifest.json"
            manifest_path.write_text(json.dumps({
                "version": "canonical_v2",
                "utterances": str(output_path),
                "row_count": summary["total_new_rows"],
                "metadata_matched": summary["total_matched_metadata_rows"],
                "metadata_missing": summary["total_new_rows_without_metadata"],
                "join_method": "old corpus + exact normalized date/council/agenda/text fingerprint",
            }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== canonical v2 bridge 결과 ===")
    print(f"구 canonical: {summary['total_old_rows']:,}")
    print(f"LLM metadata: {summary['total_metadata_rows']:,}")
    print(f"수정 TXT: {summary['total_new_rows']:,}")
    print(f"정확 매칭: {summary['total_matched_metadata_rows']:,}")
    print(f"신규/미매칭 행: {summary['total_new_rows_without_metadata']:,}")
    print(f"구 metadata 미발견: {summary['total_old_metadata_rows_not_found_in_new']:,}")
    print(f"기준 미달 파일: {failed or '없음'}")
    print(f"전체 생성 가능: {summary['safe_to_build']}")
    print(f"보고서: {report_path}")
    if partial_path and not failed:
        print(f"canonical: {output_path}")


if __name__ == "__main__":
    main()

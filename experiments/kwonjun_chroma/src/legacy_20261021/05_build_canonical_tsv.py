#!/usr/bin/env python3
"""34개 팀 정제 TSV를 DB 공용 canonical Parquet으로 변환한다.

임베딩은 수행하지 않는다. ChromaDB와 PostgreSQL+pgvector가 완전히 같은
행, ID, 메타데이터, embedding_text를 사용하도록 단일 기준 파일을 만든다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_PATTERN = re.compile(r"^cleaned_(?P<year>20\d{2})_(?P<council>.+)\.txt$")

CANONICAL_SCHEMA = pa.schema([
    ("chunk_id", pa.string()),
    ("document_id", pa.string()),
    ("parent_chunk_id", pa.string()),
    ("council", pa.string()),
    ("meeting_date", pa.string()),
    ("year", pa.int16()),
    ("committee", pa.string()),
    ("meeting_name", pa.string()),
    ("agenda_title", pa.string()),
    ("speaker", pa.string()),
    ("speaker_role", pa.string()),
    ("speaker_type", pa.string()),
    ("chunk_type", pa.string()),
    ("priority", pa.string()),
    ("source_file", pa.string()),
    ("source_line_no", pa.int64()),
    ("text", pa.string()),
    ("embedding_text", pa.string()),
    ("text_sha256", pa.string()),
    ("char_length", pa.int32()),
    ("replacement_char_count", pa.int16()),
    ("metadata_source", pa.string()),
    ("metadata_version", pa.string()),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "input")
    parser.add_argument(
        "--column-map",
        type=Path,
        default=PROJECT_ROOT / "config" / "tsv_column_map.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "canonical" / "corpus.parquet",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "canonical" / "corpus_manifest.json",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "canonical" / "canonical_quality_report.json",
    )
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--compression", choices=("zstd", "snappy"), default="zstd")
    parser.add_argument("--max-invalid-date-ratio", type=float, default=0.001)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_space(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"[ \t\r\f\v]+", " ", value).strip()
    return cleaned or None


def normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t\f\v]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or None


def normalize_date(value: str | None) -> str | None:
    if not value:
        return None
    digits = re.sub(r"[^0-9]", "", value)
    if len(digits) < 8:
        return None
    try:
        return datetime.strptime(digits[:8], "%Y%m%d").date().isoformat()
    except ValueError:
        return None


def stable_id(prefix: str, *parts: str | None) -> str:
    payload = "\x1f".join(part or "" for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:32]}"


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def find_text_files(input_dir: Path) -> list[tuple[str, int, str, Path]]:
    found = []
    seen_keys = set()
    for path in sorted(input_dir.rglob("cleaned_*.txt")):
        if ":Zone.Identifier" in path.name:
            continue
        match = TEXT_PATTERN.match(path.name)
        if not match:
            continue
        year = int(match.group("year"))
        council = match.group("council")
        key = f"{year}_{council}"
        if key in seen_keys:
            raise ValueError(f"동일 연도·의회 파일이 중복되었습니다: {key}")
        seen_keys.add(key)
        found.append((key, year, council, path))
    if not found:
        raise FileNotFoundError(f"cleaned_*.txt 파일이 없습니다: {input_dir}")
    return found


def raise_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    config_path = args.column_map.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    quality_path = args.quality_report.expanduser().resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if args.batch_size < 1:
        raise ValueError("--batch-size는 1 이상이어야 합니다.")
    if not 0 <= args.max_invalid_date_ratio <= 1:
        raise ValueError("--max-invalid-date-ratio는 0~1 사이여야 합니다.")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"출력이 이미 존재합니다. 덮어쓰려면 --overwrite: {output_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_header = list(config["expected_header"])
    if expected_header != [
        "date", "assembly", "agenda_title", "speaker_role",
        "speaker_name", "speaker_type", "utterance",
    ]:
        raise ValueError(f"지원하지 않는 열 매핑입니다: {expected_header}")
    metadata_version = str(config["schema_version"])
    files = find_text_files(input_dir)
    raise_csv_limit()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(output_path.name + ".building")
    if temporary_output.exists():
        temporary_output.unlink()

    writer = pq.ParquetWriter(
        temporary_output,
        CANONICAL_SCHEMA,
        compression=args.compression,
        use_dictionary=True,
        write_statistics=True,
    )
    buffer: list[dict] = []
    source_reports = []
    totals = Counter()

    def flush() -> None:
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer, schema=CANONICAL_SCHEMA)
        writer.write_table(table, row_group_size=args.batch_size)
        buffer.clear()

    try:
        for file_index, (source_key, filename_year, filename_council, path) in enumerate(files, start=1):
            print(f"[{file_index}/{len(files)}] {source_key}")
            counters = Counter()
            normalized_digest = hashlib.sha256()

            with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as file:
                reader = csv.reader(file, delimiter="\t")
                try:
                    header = next(reader)
                except StopIteration as exc:
                    raise ValueError(f"빈 파일입니다: {path}") from exc
                if header != expected_header:
                    raise ValueError(
                        f"헤더 불일치: {path}\nexpected={expected_header}\nactual={header}"
                    )

                for source_line_no, values in enumerate(reader, start=2):
                    counters["source_rows"] += 1
                    totals["source_rows"] += 1
                    if values == expected_header:
                        counters["repeated_header_rows"] += 1
                        totals["repeated_header_rows"] += 1
                        continue
                    if len(values) != len(expected_header):
                        counters["invalid_width_rows"] += 1
                        totals["invalid_width_rows"] += 1
                        continue

                    row = dict(zip(expected_header, values))
                    meeting_date = normalize_date(row["date"])
                    source_assembly = normalize_space(row["assembly"])
                    agenda_title = normalize_space(row["agenda_title"])
                    speaker_role = normalize_space(row["speaker_role"])
                    speaker = normalize_space(row["speaker_name"])
                    speaker_type = normalize_space(row["speaker_type"])
                    text = normalize_text(row["utterance"])

                    normalized_digest.update("\x1f".join(value or "" for value in (
                        meeting_date, source_assembly, agenda_title, speaker_role,
                        speaker, speaker_type, text,
                    )).encode("utf-8"))
                    normalized_digest.update(b"\n")

                    if not text:
                        counters["empty_text_rows"] += 1
                        totals["empty_text_rows"] += 1
                        continue
                    if not meeting_date:
                        counters["invalid_date_rows"] += 1
                        totals["invalid_date_rows"] += 1
                        continue
                    if int(meeting_date[:4]) != filename_year:
                        counters["year_mismatch_rows"] += 1
                        totals["year_mismatch_rows"] += 1
                    if source_assembly and source_assembly != filename_council:
                        counters["assembly_mismatch_rows"] += 1
                        totals["assembly_mismatch_rows"] += 1

                    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    document_id = stable_id(
                        "doc", filename_council, meeting_date, agenda_title, source_key,
                    )
                    chunk_id = stable_id(
                        "chunk", source_key, str(source_line_no), filename_council,
                        meeting_date, agenda_title, speaker_role, speaker,
                        speaker_type, text,
                    )

                    embedding_text = (
                        f"[안건] {agenda_title}\n[발언] {text}"
                        if agenda_title else f"[발언] {text}"
                    )
                    replacement_count = text.count("\ufffd")
                    counters["replacement_chars"] += replacement_count
                    totals["replacement_chars"] += replacement_count
                    buffer.append({
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "parent_chunk_id": None,
                        "council": filename_council,
                        "meeting_date": meeting_date,
                        "year": int(meeting_date[:4]),
                        "committee": None,
                        "meeting_name": None,
                        "agenda_title": agenda_title,
                        "speaker": speaker,
                        "speaker_role": speaker_role,
                        "speaker_type": speaker_type,
                        "chunk_type": None,
                        "priority": None,
                        "source_file": str(path.relative_to(input_dir)),
                        "source_line_no": source_line_no,
                        "text": text,
                        "embedding_text": embedding_text,
                        "text_sha256": text_sha256,
                        "char_length": len(text),
                        "replacement_char_count": replacement_count,
                        "metadata_source": "team_cleaned_tsv",
                        "metadata_version": metadata_version,
                    })
                    counters["written_rows"] += 1
                    totals["written_rows"] += 1
                    if len(buffer) >= args.batch_size:
                        flush()
                    if totals["written_rows"] and totals["written_rows"] % 100_000 == 0:
                        print(f"  canonical 생성 중: {totals['written_rows']:,}행")

            source_reports.append({
                "source_key": source_key,
                "relative_path": str(path.relative_to(input_dir)),
                "file_size_bytes": path.stat().st_size,
                "normalized_content_sha256": normalized_digest.hexdigest(),
                **dict(counters),
                "raw_text_in_report": False,
            })

        flush()
    except Exception:
        writer.close()
        if temporary_output.exists():
            temporary_output.unlink()
        raise
    else:
        writer.close()

    source_rows = totals["source_rows"]
    invalid_date_ratio = totals["invalid_date_rows"] / source_rows if source_rows else 0.0
    quality = {
        "schema_version": metadata_version,
        "input_file_count": len(files),
        "source_rows": source_rows,
        "written_rows": totals["written_rows"],
        "excluded_rows": source_rows - totals["written_rows"],
        "invalid_date_ratio": round(invalid_date_ratio, 8),
        "totals": dict(totals),
        "files": source_reports,
        "raw_text_in_report": False,
    }
    quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")

    if invalid_date_ratio > args.max_invalid_date_ratio:
        temporary_output.unlink()
        raise ValueError(
            f"유효하지 않은 날짜 비율이 허용치를 초과했습니다: "
            f"{invalid_date_ratio:.6%} > {args.max_invalid_date_ratio:.6%}. "
            f"품질 보고서 확인: {quality_path}"
        )
    if totals["invalid_width_rows"]:
        temporary_output.unlink()
        raise ValueError(
            f"7열이 아닌 데이터 행이 있습니다: {totals['invalid_width_rows']:,}행. "
            f"품질 보고서 확인: {quality_path}"
        )

    temporary_output.replace(output_path)
    output_hash = file_sha256(output_path)
    manifest = {
        "schema_version": metadata_version,
        "created_by": Path(__file__).name,
        "column_map": str(config_path.relative_to(PROJECT_ROOT)),
        "input_file_count": len(files),
        "row_count": totals["written_rows"],
        "output_file": str(output_path.relative_to(PROJECT_ROOT)),
        "output_size_bytes": output_path.stat().st_size,
        "output_sha256": output_hash,
        "canonical_columns": CANONICAL_SCHEMA.names,
        "id_policy": config["id_policy"],
        "embedding_text_policy": config["embedding_text"],
        "database_policy": "ChromaDB와 pgvector 모두 이 파일과 동일 임베딩을 사용",
        "raw_data_external_transfer": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Canonical Parquet 생성 완료 ===")
    print(f"입력 파일: {len(files)} | 원본 행: {source_rows:,}")
    print(f"저장 행: {totals['written_rows']:,} | 제외 행: {source_rows - totals['written_rows']:,}")
    print(f"날짜 오류: {totals['invalid_date_rows']:,} | 빈 본문: {totals['empty_text_rows']:,}")
    print(f"의회명 불일치: {totals['assembly_mismatch_rows']:,}")
    print(f"corpus: {output_path}")
    print(f"manifest: {manifest_path}")
    print(f"quality: {quality_path}")


if __name__ == "__main__":
    main()

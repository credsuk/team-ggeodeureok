#!/usr/bin/env python3
"""팀 공용 데이터의 스키마와 품질을 원문 노출 없이 검사한다."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.json as pajson
import pyarrow.parquet as pq
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = PROJECT_ROOT / "config" / "data_contract.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="검사할 parquet/csv/jsonl 파일")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "inventory")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args()


def iter_batches(path: Path, batch_size: int) -> Iterator[pa.RecordBatch]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        yield from pq.ParquetFile(path).iter_batches(batch_size=batch_size)
        return
    if suffix == ".csv":
        reader = pacsv.open_csv(path, read_options=pacsv.ReadOptions(block_size=16 * 1024 * 1024))
        yield from reader
        return
    if suffix in {".jsonl", ".ndjson"}:
        reader = pajson.open_json(path, read_options=pajson.ReadOptions(block_size=16 * 1024 * 1024))
        yield from reader
        return
    raise ValueError("지원 형식은 .parquet, .csv, .jsonl, .ndjson입니다. JSON 배열은 먼저 Parquet로 변환하세요.")


def load_contract(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_aliases(columns: list[str], contract: dict) -> dict[str, str | None]:
    lowered = {column.lower(): column for column in columns}
    resolved: dict[str, str | None] = {}
    for canonical, aliases in contract.get("aliases", {}).items():
        matched = next((lowered[a.lower()] for a in aliases if a.lower() in lowered), None)
        resolved[canonical] = matched
    return resolved


def normalize_date(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    text = str(value).strip()
    if not text:
        return None
    digits = re.sub(r"[^0-9]", "", text)
    if len(digits) >= 8:
        try:
            return datetime.strptime(digits[:8], "%Y%m%d").date().isoformat()
        except ValueError:
            return None
    return None


def is_empty(value: object) -> bool:
    return value is not None and isinstance(value, str) and not value.strip()


def safe_percentile(values: list[int], percentile: float) -> float | None:
    return round(float(np.percentile(values, percentile)), 2) if values else None


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    contract = load_contract(args.contract)
    output_dir.mkdir(parents=True, exist_ok=True)

    batches = iter_batches(input_path, args.batch_size)
    try:
        first_batch = next(batches)
    except StopIteration as exc:
        raise ValueError("입력 파일에 행이 없습니다.") from exc

    columns = first_batch.schema.names
    alias_map = resolve_aliases(columns, contract)
    text_column = alias_map.get("text")
    id_column = alias_map.get("chunk_id")
    date_column = alias_map.get("meeting_date")

    null_counts = Counter({column: 0 for column in columns})
    empty_counts = Counter({column: 0 for column in columns})
    row_count = 0
    text_lengths: list[int] = []
    invalid_date_count = 0
    date_min: str | None = None
    date_max: str | None = None
    seen_ids: set[str] = set()
    duplicate_id_count = 0
    missing_id_count = 0
    seen_text_hashes: set[bytes] = set()
    duplicate_text_count = 0
    text_hashes_for_digest: list[bytes] = []

    def process(batch: pa.RecordBatch) -> None:
        nonlocal row_count, invalid_date_count, date_min, date_max
        nonlocal duplicate_id_count, missing_id_count, duplicate_text_count
        data = batch.to_pydict()
        batch_rows = batch.num_rows
        row_count += batch_rows

        for column in columns:
            values = data[column]
            null_counts[column] += sum(value is None for value in values)
            empty_counts[column] += sum(is_empty(value) for value in values)

        if id_column:
            for value in data[id_column]:
                if value is None or not str(value).strip():
                    missing_id_count += 1
                    continue
                key = str(value).strip()
                if key in seen_ids:
                    duplicate_id_count += 1
                else:
                    seen_ids.add(key)

        if date_column:
            for value in data[date_column]:
                parsed = normalize_date(value)
                if parsed is None:
                    if value is not None and str(value).strip():
                        invalid_date_count += 1
                    continue
                date_min = parsed if date_min is None or parsed < date_min else date_min
                date_max = parsed if date_max is None or parsed > date_max else date_max

        if text_column:
            for value in data[text_column]:
                text = "" if value is None else str(value)
                text_lengths.append(len(text))
                digest = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
                if digest in seen_text_hashes:
                    duplicate_text_count += 1
                else:
                    seen_text_hashes.add(digest)
                text_hashes_for_digest.append(digest)

    process(first_batch)
    for index, batch in enumerate(batches, start=2):
        process(batch)
        if index % 20 == 0:
            print(f"검사 중: {row_count:,}행")

    required = list(contract.get("required_columns", {}).keys())
    missing_required = [name for name in required if not alias_map.get(name)]
    schema_rows = []
    for field in first_batch.schema:
        schema_rows.append({
            "column": field.name,
            "arrow_type": str(field.type),
            "null_count": null_counts[field.name],
            "empty_string_count": empty_counts[field.name],
            "null_ratio": round(null_counts[field.name] / row_count, 6) if row_count else math.nan,
        })

    corpus_digest = hashlib.sha256(b"".join(sorted(text_hashes_for_digest))).hexdigest() if text_hashes_for_digest else None
    summary = {
        "report_version": "1.0.0",
        "input_file_name": input_path.name,
        "input_size_bytes": input_path.stat().st_size,
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "resolved_aliases": alias_map,
        "missing_required_columns": missing_required,
        "chunk_id_quality": {
            "column": id_column,
            "missing_count": missing_id_count if id_column else None,
            "duplicate_count": duplicate_id_count if id_column else None,
            "unique_count": len(seen_ids) if id_column else None,
        },
        "meeting_date_quality": {
            "column": date_column,
            "min": date_min,
            "max": date_max,
            "invalid_nonempty_count": invalid_date_count if date_column else None,
        },
        "text_quality": {
            "column": text_column,
            "empty_or_null_count": (
                (null_counts[text_column] + empty_counts[text_column]) if text_column else None
            ),
            "duplicate_sha256_count": duplicate_text_count if text_column else None,
            "length_min": min(text_lengths) if text_lengths else None,
            "length_p25": safe_percentile(text_lengths, 25),
            "length_p50": safe_percentile(text_lengths, 50),
            "length_p75": safe_percentile(text_lengths, 75),
            "length_p95": safe_percentile(text_lengths, 95),
            "length_p99": safe_percentile(text_lengths, 99),
            "length_max": max(text_lengths) if text_lengths else None,
            "length_mean": round(float(np.mean(text_lengths)), 2) if text_lengths else None,
        },
        "corpus_text_multiset_sha256": corpus_digest,
        "safe_to_share": "원문 텍스트와 실제 metadata 값은 포함하지 않음",
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    schema_path = output_dir / "schema_quality.csv"
    with schema_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(schema_rows[0]))
        writer.writeheader()
        writer.writerows(schema_rows)

    print("\n=== 팀 공용 데이터 검사 완료 ===")
    print(f"행: {row_count:,} | 열: {len(columns):,}")
    print(f"인식한 열: {json.dumps(alias_map, ensure_ascii=False)}")
    print(f"필수 열 누락: {missing_required or '없음'}")
    print(f"보고서: {summary_path}")
    print(f"열 품질: {schema_path}")
    print("\n다음 단계: summary.json과 schema_quality.csv를 기준으로 canonical 열 매핑을 확정합니다.")


if __name__ == "__main__":
    main()


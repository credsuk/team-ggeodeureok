#!/usr/bin/env python3
"""Canonical Parquet의 임베딩 적합성 분포를 원문 노출 없이 검사한다.

전체 임베딩 전에 짧은 응답, 절차성 발언, 긴 발언, 메타데이터 결측과
embedding_text 생성 규칙을 측정한다. 이 스크립트는 데이터를 제거하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = (
    "chunk_id", "council", "meeting_date", "year", "agenda_title",
    "speaker", "speaker_role", "speaker_type", "text", "embedding_text",
    "char_length", "replacement_char_count",
)
NULL_PROFILE_COLUMNS = (
    "agenda_title", "speaker", "speaker_role", "speaker_type",
    "committee", "meeting_name", "chunk_type", "priority",
)
TRIVIAL_RESPONSES = {
    "네", "예", "아니요", "알겠습니다", "그렇습니다", "맞습니다",
    "감사합니다", "이상입니다", "동의합니다", "없습니다", "있습니다",
    "수고하셨습니다", "잘알겠습니다", "예알겠습니다",
}
PROCEDURAL_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"개의하겠습니다", r"개회를?\s*선포", r"산회를?\s*선포",
        r"정회를?\s*선포", r"속개하겠습니다", r"의사일정\s*제?\d+항",
        r"이의\s*(가|는)?\s*없으", r"가결되었음을\s*선포",
        r"회의를\s*(시작|마치)", r"자리를\s*정돈",
    )
)
CHAR_THRESHOLDS = (10, 20, 30, 50, 100, 200, 500, 1000, 2000, 4000, 8000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "canonical" / "corpus.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "canonical_profile" / "canonical_profile.json",
    )
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--reservoir-size", type=int, default=200_000)
    return parser.parse_args()


def percentile(sorted_values: list[int], q: float) -> int | None:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, max(0, math.ceil(len(sorted_values) * q) - 1))
    return sorted_values[index]


def compact_text(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", value)


def update_reservoir(sample: list[int], value: int, seen: int, limit: int, rng: random.Random) -> None:
    if len(sample) < limit:
        sample.append(value)
        return
    position = rng.randrange(seen)
    if position < limit:
        sample[position] = value


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if args.batch_size < 1 or args.reservoir_size < 1:
        raise ValueError("--batch-size와 --reservoir-size는 1 이상이어야 합니다.")

    parquet = pq.ParquetFile(input_path)
    schema_columns = parquet.schema_arrow.names
    missing = sorted(set(REQUIRED_COLUMNS) - set(schema_columns))
    if missing:
        raise ValueError(f"필수 canonical 열이 없습니다: {missing}")

    available_null_columns = [name for name in NULL_PROFILE_COLUMNS if name in schema_columns]
    scan_columns = list(dict.fromkeys((*REQUIRED_COLUMNS, *available_null_columns)))
    counters = Counter()
    null_counts = Counter()
    year_counts = Counter()
    council_counts = Counter()
    year_council_counts = Counter()
    char_below_counts = Counter()
    char_above_counts = Counter()
    length_sample: list[int] = []
    rng = random.Random(20261021)
    min_date: str | None = None
    max_date: str | None = None
    min_chars: int | None = None
    max_chars = 0
    total_chars = 0

    for batch_index, batch in enumerate(
        parquet.iter_batches(batch_size=args.batch_size, columns=scan_columns), start=1
    ):
        data = batch.to_pydict()
        for index in range(batch.num_rows):
            counters["rows"] += 1
            row_number = counters["rows"]
            text = data["text"][index] or ""
            agenda = data["agenda_title"][index]
            embedding_text = data["embedding_text"][index] or ""
            stored_length = data["char_length"][index]
            actual_length = len(text)
            total_chars += actual_length
            min_chars = actual_length if min_chars is None else min(min_chars, actual_length)
            max_chars = max(max_chars, actual_length)
            update_reservoir(length_sample, actual_length, row_number, args.reservoir_size, rng)

            if stored_length != actual_length:
                counters["char_length_mismatch"] += 1
            if not text:
                counters["empty_text"] += 1
            if "\ufffd" in text:
                counters["rows_with_replacement_char"] += 1
            counters["replacement_chars"] += text.count("\ufffd")

            for threshold in CHAR_THRESHOLDS:
                if actual_length < threshold:
                    char_below_counts[str(threshold)] += 1
                if actual_length >= threshold:
                    char_above_counts[str(threshold)] += 1

            compact = compact_text(text)
            trivial = compact in TRIVIAL_RESPONSES
            procedural = any(pattern.search(text) for pattern in PROCEDURAL_PATTERNS)
            counters["trivial_response_rows"] += int(trivial)
            counters["procedural_signal_rows"] += int(procedural)
            counters["candidate_min_15_no_trivial"] += int(actual_length >= 15 and not trivial)
            counters["candidate_min_30_no_trivial"] += int(actual_length >= 30 and not trivial)
            counters["candidate_min_50_no_trivial"] += int(actual_length >= 50 and not trivial)
            counters["needs_long_text_review_ge_4000"] += int(actual_length >= 4000)

            expected_embedding_text = (
                f"[안건] {agenda}\n[발언] {text}" if agenda else f"[발언] {text}"
            )
            if embedding_text != expected_embedding_text:
                counters["embedding_text_rule_mismatch"] += 1

            year = data["year"][index]
            council = data["council"][index]
            meeting_date = data["meeting_date"][index]
            year_counts[str(year)] += 1
            council_counts[str(council)] += 1
            year_council_counts[f"{year}|{council}"] += 1
            if meeting_date:
                min_date = meeting_date if min_date is None else min(min_date, meeting_date)
                max_date = meeting_date if max_date is None else max(max_date, meeting_date)

            for column in available_null_columns:
                value = data[column][index]
                if value is None or (isinstance(value, str) and not value.strip()):
                    null_counts[column] += 1

        if batch_index % 5 == 0:
            print(f"프로파일링 중: {counters['rows']:,}행")

    length_sample.sort()
    total_rows = counters["rows"]
    report = {
        "input_file": str(input_path.relative_to(PROJECT_ROOT))
        if input_path.is_relative_to(PROJECT_ROOT) else input_path.name,
        "input_size_bytes": input_path.stat().st_size,
        "row_count": total_rows,
        "date_range": {"min": min_date, "max": max_date},
        "year_counts": dict(sorted(year_counts.items())),
        "council_counts": dict(sorted(council_counts.items())),
        "year_council_counts": dict(sorted(year_council_counts.items())),
        "text_length": {
            "min": min_chars,
            "p50_sample": percentile(length_sample, 0.50),
            "p90_sample": percentile(length_sample, 0.90),
            "p95_sample": percentile(length_sample, 0.95),
            "p99_sample": percentile(length_sample, 0.99),
            "p99_9_sample": percentile(length_sample, 0.999),
            "max": max_chars,
            "mean": round(total_chars / total_rows, 3) if total_rows else 0.0,
            "sample_size": len(length_sample),
            "rows_below_chars": dict(char_below_counts),
            "rows_at_or_above_chars": dict(char_above_counts),
        },
        "null_counts": {
            column: {
                "count": null_counts[column],
                "ratio": round(null_counts[column] / total_rows, 8) if total_rows else 0.0,
            }
            for column in available_null_columns
        },
        "content_shape": {
            key: counters[key]
            for key in (
                "empty_text", "trivial_response_rows", "procedural_signal_rows",
                "needs_long_text_review_ge_4000", "rows_with_replacement_char",
                "replacement_chars", "char_length_mismatch",
                "embedding_text_rule_mismatch",
            )
        },
        "candidate_scenarios": {
            "all_rows": total_rows,
            "min_15_chars_no_trivial": counters["candidate_min_15_no_trivial"],
            "min_30_chars_no_trivial": counters["candidate_min_30_no_trivial"],
            "min_50_chars_no_trivial": counters["candidate_min_50_no_trivial"],
        },
        "interpretation": {
            "profile_only": True,
            "rows_removed": 0,
            "recommended_next_step": "검수 후 동일 기준으로 소규모 임베딩 표본 생성",
            "procedural_policy": "현재는 측정만 하며 자동 제외하지 않음",
            "long_text_policy": "4,000자 이상 행은 모델 토큰 길이 검사 후 분할 여부 결정",
        },
        "raw_text_in_report": False,
        "speaker_names_in_report": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Canonical 임베딩 적합성 프로파일 완료 ===")
    print(f"행: {total_rows:,} | 날짜: {min_date} ~ {max_date}")
    print(
        "길이(표본): "
        f"p50={report['text_length']['p50_sample']}, "
        f"p95={report['text_length']['p95_sample']}, "
        f"p99={report['text_length']['p99_sample']}, max={max_chars}"
    )
    print(f"짧은 단답: {counters['trivial_response_rows']:,}")
    print(f"절차 신호: {counters['procedural_signal_rows']:,}")
    print(f"4,000자 이상: {counters['needs_long_text_review_ge_4000']:,}")
    print(f"embedding_text 규칙 불일치: {counters['embedding_text_rule_mismatch']:,}")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()

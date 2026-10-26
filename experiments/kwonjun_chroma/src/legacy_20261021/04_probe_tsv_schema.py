#!/usr/bin/env python3
"""정제 TXT의 TSV 열 구조를 원문 노출 없이 통계적으로 판별한다.

이 스크립트는 원문/고유 필드값을 결과 파일에 기록하지 않는다. 각 파일의
열 개수 분포와 열별 길이·반복도·날짜형·숫자형 비율만 집계하여 후속 파서가
사용할 열 인덱스를 결정하는 데 사용한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_PATTERN = re.compile(r"^cleaned_(?P<year>20\d{2})_(?P<council>.+)\.txt$")
DATE_PATTERNS = (
    re.compile(r"^20\d{2}[01]\d[0-3]\d$"),
    re.compile(r"^20\d{2}[-./][01]?\d[-./][0-3]?\d$"),
)
INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "input")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "tsv_probe")
    parser.add_argument(
        "--max-unique-hashes",
        type=int,
        default=100_000,
        help="열별 고유도 추정을 위해 보관할 해시의 최대 수",
    )
    parser.add_argument(
        "--reservoir-size",
        type=int,
        default=20_000,
        help="백분위 길이 계산용 열별 reservoir 크기",
    )
    return parser.parse_args()


def percentile(sorted_values: list[int], q: float) -> int | None:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, max(0, math.ceil(len(sorted_values) * q) - 1))
    return sorted_values[index]


@dataclass
class ColumnStats:
    column_index: int
    max_unique_hashes: int
    reservoir_size: int
    seed: int
    row_count: int = 0
    nonempty_count: int = 0
    total_chars: int = 0
    min_chars: int | None = None
    max_chars: int = 0
    short_count: int = 0
    long_count: int = 0
    date_like_count: int = 0
    integer_like_count: int = 0
    replacement_char_count: int = 0
    unique_hashes: set[str] = field(default_factory=set)
    unique_hashes_capped: bool = False
    length_sample: list[int] = field(default_factory=list)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def update(self, value: str) -> None:
        self.row_count += 1
        length = len(value)
        self.total_chars += length
        self.min_chars = length if self.min_chars is None else min(self.min_chars, length)
        self.max_chars = max(self.max_chars, length)
        self.short_count += int(length <= 50)
        self.long_count += int(length >= 200)
        self.replacement_char_count += value.count("\ufffd")

        if length:
            self.nonempty_count += 1
            stripped = value.strip()
            self.date_like_count += int(any(pattern.fullmatch(stripped) for pattern in DATE_PATTERNS))
            self.integer_like_count += int(bool(INTEGER_PATTERN.fullmatch(stripped)))
            if not self.unique_hashes_capped:
                digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()
                self.unique_hashes.add(digest)
                if len(self.unique_hashes) >= self.max_unique_hashes:
                    self.unique_hashes_capped = True

        if len(self.length_sample) < self.reservoir_size:
            self.length_sample.append(length)
        else:
            position = self._rng.randrange(self.row_count)
            if position < self.reservoir_size:
                self.length_sample[position] = length

    def as_dict(self) -> dict:
        sample = sorted(self.length_sample)
        unique_count = len(self.unique_hashes)
        unique_ratio = None
        if self.nonempty_count and not self.unique_hashes_capped:
            unique_ratio = unique_count / self.nonempty_count
        return {
            "column_index": self.column_index,
            "row_count": self.row_count,
            "nonempty_count": self.nonempty_count,
            "nonempty_ratio": round(self.nonempty_count / self.row_count, 6) if self.row_count else 0.0,
            "length": {
                "min": self.min_chars,
                "p50_sample": percentile(sample, 0.50),
                "p95_sample": percentile(sample, 0.95),
                "p99_sample": percentile(sample, 0.99),
                "max": self.max_chars,
                "mean": round(self.total_chars / self.row_count, 3) if self.row_count else 0.0,
            },
            "short_le_50_ratio": round(self.short_count / self.row_count, 6) if self.row_count else 0.0,
            "long_ge_200_ratio": round(self.long_count / self.row_count, 6) if self.row_count else 0.0,
            "date_like_ratio": round(self.date_like_count / self.nonempty_count, 6) if self.nonempty_count else 0.0,
            "integer_like_ratio": round(self.integer_like_count / self.nonempty_count, 6) if self.nonempty_count else 0.0,
            "unique_hash_count": unique_count,
            "unique_ratio": round(unique_ratio, 6) if unique_ratio is not None else None,
            "unique_hashes_capped": self.unique_hashes_capped,
            "replacement_char_count": self.replacement_char_count,
            "raw_values_in_report": False,
        }


def infer_candidates(columns: list[dict], dominant_width: int) -> dict:
    usable = [column for column in columns if column["column_index"] < dominant_width]
    if not usable:
        return {
            "utterance_column": None,
            "speaker_column": None,
            "agenda_column": None,
            "automatic_mapping_is_final": False,
        }

    utterance = max(
        usable,
        key=lambda column: (
            column["length"]["mean"],
            column["length"]["p95_sample"] or 0,
        ),
    )
    remaining = [column for column in usable if column["column_index"] != utterance["column_index"]]

    speaker = None
    if remaining:
        # 발화자는 일반적으로 짧고 반복된다. 고유도 추정이 cap에 걸린 열은 뒤로 보낸다.
        speaker = min(
            remaining,
            key=lambda column: (
                column["unique_hashes_capped"],
                column["length"]["mean"],
                column["unique_ratio"] if column["unique_ratio"] is not None else 1.0,
            ),
        )

    agenda_pool = [
        column for column in remaining
        if speaker is None or column["column_index"] != speaker["column_index"]
    ]
    agenda = max(agenda_pool, key=lambda column: column["length"]["mean"]) if agenda_pool else None

    return {
        "utterance_column": utterance["column_index"],
        "speaker_column": speaker["column_index"] if speaker else None,
        "agenda_column": agenda["column_index"] if agenda else None,
        "automatic_mapping_is_final": False,
        "note": "길이·반복도 기반 후보이며 다음 단계의 표본 검증 후 고정해야 합니다.",
    }


def inspect_file(path: Path, key: str, max_unique_hashes: int, reservoir_size: int) -> dict:
    column_count_distribution: Counter[int] = Counter()
    columns: list[ColumnStats] = []
    line_count = 0
    blank_line_count = 0

    with path.open("r", encoding="utf-8-sig", errors="strict") as file:
        for line in file:
            line_count += 1
            row = line.rstrip("\r\n")
            if not row:
                blank_line_count += 1
            values = row.split("\t")
            column_count_distribution[len(values)] += 1

            while len(columns) < len(values):
                index = len(columns)
                columns.append(
                    ColumnStats(
                        column_index=index,
                        max_unique_hashes=max_unique_hashes,
                        reservoir_size=reservoir_size,
                        seed=20261021 + index,
                    )
                )
            for index, value in enumerate(values):
                columns[index].update(value)

    dominant_width = 0
    dominant_count = 0
    if column_count_distribution:
        dominant_width, dominant_count = column_count_distribution.most_common(1)[0]
    column_reports = [column.as_dict() for column in columns]

    return {
        "key": key,
        "relative_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "line_count": line_count,
        "blank_line_count": blank_line_count,
        "column_count_distribution": {
            str(width): count for width, count in sorted(column_count_distribution.items())
        },
        "dominant_column_count": dominant_width,
        "dominant_shape_ratio": round(dominant_count / line_count, 6) if line_count else 0.0,
        "columns": column_reports,
        "candidate_mapping": infer_candidates(column_reports, dominant_width),
        "raw_values_in_report": False,
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if args.max_unique_hashes < 1 or args.reservoir_size < 1:
        raise ValueError("--max-unique-hashes와 --reservoir-size는 1 이상이어야 합니다.")

    text_files: dict[str, Path] = {}
    for path in sorted(input_dir.rglob("cleaned_*.txt")):
        if ":Zone.Identifier" in path.name:
            continue
        match = TEXT_PATTERN.match(path.name)
        if match:
            key = f"{match.group('year')}_{match.group('council')}"
            if key in text_files:
                raise ValueError(f"동일 연도·의회 파일이 중복되었습니다: {key}")
            text_files[key] = path
    if not text_files:
        raise FileNotFoundError(f"cleaned_*.txt 파일이 없습니다: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    file_reports = []
    for index, key in enumerate(sorted(text_files), start=1):
        print(f"[{index}/{len(text_files)}] {key}")
        report = inspect_file(
            text_files[key],
            key,
            max_unique_hashes=args.max_unique_hashes,
            reservoir_size=args.reservoir_size,
        )
        report["relative_path"] = str(text_files[key].relative_to(input_dir))
        file_reports.append(report)

    dominant_width_counts = Counter(report["dominant_column_count"] for report in file_reports)
    mapping_counts = {
        role: dict(Counter(report["candidate_mapping"][role] for report in file_reports))
        for role in ("agenda_column", "speaker_column", "utterance_column")
    }
    inconsistent_shape_files = [
        report["key"] for report in file_reports if report["dominant_shape_ratio"] < 0.999
    ]
    summary = {
        "file_count": len(file_reports),
        "total_lines": sum(report["line_count"] for report in file_reports),
        "dominant_column_count_by_file": dict(sorted(dominant_width_counts.items())),
        "candidate_mapping_counts": mapping_counts,
        "files_below_99_9_percent_dominant_shape": inconsistent_shape_files,
        "safe_to_build_tsv_parser": (
            len(dominant_width_counts) == 1 and not inconsistent_shape_files
        ),
        "automatic_mapping_is_final": False,
        "files": file_reports,
        "raw_values_in_report": False,
    }
    output_path = output_dir / "tsv_schema_report.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== TSV 열 구조 검사 완료 ===")
    print(f"파일: {len(file_reports)} | 전체 행: {summary['total_lines']:,}")
    print(f"파일별 주 열 개수: {dict(dominant_width_counts)}")
    print(f"후보 열 매핑 빈도: {mapping_counts}")
    print(f"형태 일관성 기준 통과: {summary['safe_to_build_tsv_parser']}")
    print("주의: 후보 열 매핑은 아직 확정값이 아닙니다.")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()

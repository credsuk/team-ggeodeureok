#!/usr/bin/env python3
"""Split a council transcript TSV into small, numbered review batches.

The source format is expected to have these seven columns:
date, assembly, agenda_title, speaker_role, speaker_name, speaker_type, utterance

Each output file includes a stable, global row_id column so metadata extracted
from separate batches can later be combined without losing source row numbers.
Only Python's standard library is used.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterator
import _constant

SOURCE_COLUMNS = [
    "date",
    "assembly",
    "agenda_title",
    "speaker_role",
    "speaker_name",
    "speaker_type",
    "utterance",
]
OUTPUT_COLUMNS = ["row_id", *SOURCE_COLUMNS]

# Paths are relative to the directory where this script is launched.
# Run from C:\project\mini to use these defaults as written.
DEFAULT_INPUT = _constant.SPLIT_DEFAULT_INPUT
DEFAULT_OUTPUT_DIR = _constant.SPLIT_DEFAULT_OUTPUT_DIR
DEFAULT_ROWS_PER_FILE = _constant.SPLIT_DEFAULT_ROWS_PER_FILE
DEFAULT_MAX_CHARS = _constant.SPLIT_DEFAULT_MAX_CHARS



def records(path: Path) -> Iterator[tuple[int, list[str]]]:
    """Yield (global row_id, fields), validating the expected source schema."""
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("입력 파일이 비어 있습니다.")

        if header != SOURCE_COLUMNS:
            raise ValueError(
                "입력 헤더가 예상과 다릅니다.\n"
                f"예상: {SOURCE_COLUMNS}\n실제: {header}"
            )

        row_id = 0
        for line_number, row in enumerate(reader, start=2):
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if len(row) != len(SOURCE_COLUMNS):
                raise ValueError(
                    f"{line_number}번째 파일 행에 열이 {len(row)}개 있습니다. "
                    f"예상 열 수는 {len(SOURCE_COLUMNS)}개입니다."
                )
            row_id += 1
            yield row_id, row


def write_batch(path: Path, batch: list[tuple[int, list[str]]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(OUTPUT_COLUMNS)
        for row_id, fields in batch:
            writer.writerow([row_id, *fields])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="회의록 TSV를 메타데이터 수동 추출용 작은 TSV 파일들로 나눕니다."
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"원본 TSV 파일 (기본값: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"분할 파일 저장 폴더 (기본값: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=DEFAULT_ROWS_PER_FILE,
        help=f"파일당 최대 데이터 행 수 (기본값: {DEFAULT_ROWS_PER_FILE})",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f"파일당 최대 문자 수 근사치 (기본값: {DEFAULT_MAX_CHARS}; 긴 발언은 행 수가 줄어듦)",
    )
    args = parser.parse_args()

    if args.rows_per_file < 1:
        parser.error("--rows-per-file은 1 이상이어야 합니다.")
    if args.max_chars < 1:
        parser.error("--max-chars는 1 이상이어야 합니다.")
    if not args.input.is_file():
        parser.error(f"입력 파일을 찾을 수 없습니다: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch: list[tuple[int, list[str]]] = []
    batch_chars = 0
    batch_number = 0
    total_rows = 0

    def flush() -> None:
        nonlocal batch, batch_chars, batch_number
        if not batch:
            return
        batch_number += 1
        output_path = args.output_dir / f"part_{batch_number:05d}.tsv"
        write_batch(output_path, batch)
        print(f"저장: {output_path} ({len(batch)}행)")
        batch = []
        batch_chars = 0

    try:
        for row_id, row in records(args.input):
            row_chars = sum(len(field) for field in row)
            if batch and (
                len(batch) >= args.rows_per_file
                or batch_chars + row_chars > args.max_chars
            ):
                flush()
            batch.append((row_id, row))
            batch_chars += row_chars
            total_rows += 1
        flush()
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        parser.exit(2, f"오류: {exc}\n")

    print(
        f"완료: {total_rows:,}개 데이터 행을 {batch_number:,}개 파일로 분할했습니다.\n"
        f"row_id는 원본 첫 데이터 행을 1로 시작하는 연속 번호입니다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



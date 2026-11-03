#!/usr/bin/env python3
"""1단계: 정제된 회의록 TSV를 LLM 메타데이터 추출용 소형 배치(part_*.tsv)로 분할 (split_council_tsv.py).

[주요 기능]
1. 정제된 회의록 텍스트(TSV 형식)를 읽어 행 수 및 문자 수 한도에 맞춰 작은 단위로 분할합니다.
2. 각 행에 전역 고유 식별자(`row_id`)를 부여하여, 여러 배치 파일로 분할되어도
   원본 데이터의 행 번호와 일대일 정합성을 완벽하게 유지합니다.
3. 영문 컬럼명(date, assembly...) 및 한글 컬럼명(MTG_DE, RASMBLY_NM...)을 모두 유연하게 지원합니다.
4. Python 표준 라이브러리(csv, pathlib, argparse)만으로 경량 동작합니다.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterator, List, Tuple
import _constant

# ---------------------------------------------------------------------------
# 스키마 정의 (지원 컬럼 규격)
# ---------------------------------------------------------------------------
# 영문 표준 컬럼 규격
ENGLISH_SOURCE_COLUMNS = [
    "date",
    "assembly",
    "agenda_title",
    "speaker_role",
    "speaker_name",
    "speaker_type",
    "utterance",
]

# 4_to_txt.py가 생성하는 한글 표준 컬럼 규격
KOREAN_SOURCE_COLUMNS = [
    "MTG_DE",
    "RASMBLY_NM",
    "MTGNM",
    "MINTS_NM",
    "agenda_title",
    "speaker_role",
    "speaker_name",
    "speaker_type",
    "utterance",
]

# 기본 경로 및 파라미터 (_constant.py 연동)
DEFAULT_INPUT = _constant.SPLIT_DEFAULT_INPUT
DEFAULT_OUTPUT_DIR = _constant.SPLIT_DEFAULT_OUTPUT_DIR
DEFAULT_ROWS_PER_FILE = _constant.SPLIT_DEFAULT_ROWS_PER_FILE
DEFAULT_MAX_CHARS = _constant.SPLIT_DEFAULT_MAX_CHARS


def parse_and_validate_header(reader: csv.reader) -> List[str]:
    """TSV 파일 헤더를 읽어 유효한 스키마인지 검증하고 컬럼 목록을 반환합니다."""
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("입력 파일이 비어 있습니다.")

    # 공백 및 BOM 정리
    header = [col.strip().lstrip("\ufeff") for col in header]

    if header == ENGLISH_SOURCE_COLUMNS or header == KOREAN_SOURCE_COLUMNS:
        return header
    
    # 필수 핵심 컬럼(발언문, 발언자, 의회명 등)이 포함되어 있으면 허용
    if "utterance" in header and ("speaker_name" in header or "speaker" in header):
        return header

    raise ValueError(
        f"입력 헤더 형식이 예상과 다릅니다.\n"
        f"지원 형식 1: {ENGLISH_SOURCE_COLUMNS}\n"
        f"지원 형식 2: {KOREAN_SOURCE_COLUMNS}\n"
        f"실제 헤더: {header}"
    )


def records(path: Path) -> Iterator[Tuple[int, List[str], List[str]]]:
    """입력 TSV 파일에서 (row_id, fields, header)를 생성하는 제너레이터 함수."""
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, delimiter="\t")
        header = parse_and_validate_header(reader)

        row_id = 0
        for line_number, row in enumerate(reader, start=2):
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if len(row) != len(header):
                # 탭 구분자 불일치나 줄바꿈 포함 행 건너뛰기 또는 예외 처리
                if len(row) > len(header):
                    row = row[:len(header)]
                else:
                    continue
            row_id += 1
            yield row_id, row, header


def write_batch(path: Path, header: List[str], batch: List[Tuple[int, List[str]]]) -> None:
    """단일 배치 TSV 파일에 row_id와 원본 필드들을 저장합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(["row_id", *header])
        for row_id, fields in batch:
            writer.writerow([row_id, *fields])


def split_tsv(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    rows_per_file: int = DEFAULT_ROWS_PER_FILE,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> int:
    """회의록 TSV 파일을 지정된 행 수 및 문자 수 기준에 따라 분할합니다."""
    if not input_path.is_file():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    batch: List[Tuple[int, List[str]]] = []
    batch_chars = 0
    batch_number = 0
    total_rows = 0
    active_header: List[str] = []

    def flush() -> None:
        nonlocal batch, batch_chars, batch_number
        if not batch or not active_header:
            return
        batch_number += 1
        output_path = output_dir / f"part_{batch_number:05d}.tsv"
        write_batch(output_path, active_header, batch)
        print(f"  [저장 완료] {output_path.name} ({len(batch)}행, 약 {batch_chars:,}자)")
        batch = []
        batch_chars = 0

    for row_id, row, header in records(input_path):
        active_header = header
        row_chars = sum(len(field) for field in row)
        if batch and (len(batch) >= rows_per_file or batch_chars + row_chars > max_chars):
            flush()
        batch.append((row_id, row))
        batch_chars += row_chars
        total_rows += 1

    flush()

    print(
        f"\n[분할 완료] 총 {total_rows:,}개 데이터 행을 {batch_number:,}개 배치 파일로 분할했습니다.\n"
        f"저장 위치: {output_dir.resolve()}\n"
        f"각 파일의 'row_id'는 원본 데이터의 1번 행부터 부여된 고유 번호입니다."
    )
    return total_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="회의록 TSV를 메타데이터 추출용 소형 TSV 배치 파일들로 분할합니다."
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"원본 정제 TSV 파일 경로 (기본값: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"분할 파일 저장 폴더 경로 (기본값: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--rows-per-file", type=int, default=DEFAULT_ROWS_PER_FILE,
        help=f"파일당 최대 데이터 행 수 (기본값: {DEFAULT_ROWS_PER_FILE})",
    )
    parser.add_argument(
        "--max-chars", type=int, default=DEFAULT_MAX_CHARS,
        help=f"파일당 최대 문자 수 근사치 (기본값: {DEFAULT_MAX_CHARS})",
    )
    args = parser.parse_args()

    try:
        split_tsv(
            input_path=args.input,
            output_dir=args.output_dir,
            rows_per_file=args.rows_per_file,
            max_chars=args.max_chars,
        )
        return 0
    except Exception as exc:
        print(f"[오류 발생] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

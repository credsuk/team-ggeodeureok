"""
change_text 디렉토리 내의 모든 광역의회 회의록(.txt)을 Kiwi 형태소 분석기를 활용하여
data2 디렉토리에 JSONL 메타데이터 파일로 일괄 변환(Batch Ingest)하는 스크립트.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from extract_kiwi_metadata import (
    KiwiCouncilAnalyzer,
    process_council_file_with_kiwi,
    load_processed_row_ids,
)


def format_time(seconds: float) -> str:
    """초 단위를 보기 쉬운 시:분:초 형식 문자열로 변환"""
    sec = int(seconds)
    if sec < 60:
        return f"{sec}초"
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}시간 {m:02d}분 {s:02d}초"
    return f"{m}분 {s:02d}초"


def get_total_input_lines(file_path: Path) -> int:
    """헤더를 제외한 유효 데이터 행 수 계산"""
    count = 0
    with open(file_path, "r", encoding="utf-8-sig") as f:
        f.readline()  # 헤더
        for _ in f:
            count += 1
    return count


def collect_target_files(
    input_dir: str,
    pattern: str = "*.txt",
    year_filter: str = "",
) -> List[Path]:
    """대상 디렉토리 내의 모든 txt 파일을 수집하고 정렬"""
    base_dir = Path(input_dir)
    if not base_dir.exists():
        print(f"[오류] 입력 디렉토리가 존재하지 않습니다: {input_dir}")
        return []

    # 재귀적으로 모든 txt 파일 검색
    files = list(base_dir.rglob(pattern))

    # 연도 필터링
    if year_filter:
        files = [f for f in files if year_filter in f.name or year_filter in str(f.parent)]

    # 파일명 기준으로 정렬 (연도, 의회명 순)
    files.sort(key=lambda p: (str(p.parent), p.name))
    return files


def run_batch_extraction(
    input_dir: str = "change_text",
    output_dir: str = "data2",
    pattern: str = "*.txt",
    year_filter: str = "",
    sample_per_file: int = 0,
    min_char_len: int = 80,
    num_workers: int = 4,
    preserve_structure: bool = True,
):
    print("=" * 70)
    print(" [Kiwi 회의록 정책 메타데이터 대량 일괄 변환기 (Batch Ingest)] ")
    print("=" * 70)
    print(f" * 입력 루트 디렉토리 : {input_dir}")
    print(f" * 출력 대상 디렉토리 : {output_dir}")
    print(f" * 검색 파일 패턴     : {pattern}")
    if year_filter:
        print(f" * 연도 필터          : {year_filter}")
    print(f" * 파일당 처리 건수   : {'전체 처리' if sample_per_file <= 0 else f'최대 {sample_per_file}건 샘플'}")
    print(f" * 최소 발언 글자수   : {min_char_len}자 미만 제외")
    print(f" * Kiwi 워커 수       : {num_workers}개 스레드")
    print(f" * 출력 폴더 구조     : {'원본 하위 폴더 구조 유지 (기본)' if preserve_structure else '단일 폴더(플랫) 저장'}")
    print("=" * 70)

    # 1. 파일 목록 수집
    txt_files = collect_target_files(input_dir, pattern=pattern, year_filter=year_filter)
    if not txt_files:
        print(f"[알림] '{input_dir}' 경로에서 변환 대상 .txt 파일을 찾지 못했습니다.")
        return

    print(f">> 총 {len(txt_files)}개 변환 대상 회의록 파일 발견!\n")

    # 2. 단일 Kiwi 분석기 인스턴스 초기화 (메모리 절약 및 고속 재사용)
    print(">> Kiwi 형태소 분석기 엔진 초기화 중...")
    analyzer = KiwiCouncilAnalyzer(num_workers=num_workers)
    print(">> Kiwi 엔진 초기화 완료.\n")

    # 3. 일괄 변환 루프
    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    total_files = len(txt_files)
    grand_processed = 0
    grand_success = 0
    skipped_files = 0
    start_batch_time = time.time()

    print("-" * 70)
    print(f" [일괄 변환 시작: 총 {total_files}개 파일]")
    print("-" * 70)

    with tqdm(
        total=total_files,
        desc="[Total Batch Progress]",
        unit="file",
        bar_format="{desc}: {percentage:3.0f}%|{bar:25}| {n_fmt}/{total_fmt} files [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        dynamic_ncols=True,
    ) as total_pbar:
        for file_idx, in_file in enumerate(txt_files, start=1):
            # 출력 경로 계산
            if preserve_structure:
                rel_dir = in_file.parent.relative_to(Path(input_dir))
                target_out_dir = out_base / rel_dir
                target_out_dir.mkdir(parents=True, exist_ok=True)
                out_file = target_out_dir / f"{in_file.stem}.jsonl"
            else:
                out_file = out_base / f"{in_file.stem}.jsonl"

            # 지금까지의 진행 시간 및 남은 시간 추정
            current_elapsed = time.time() - start_batch_time
            if file_idx > 1 and current_elapsed > 0:
                avg_time_per_file = current_elapsed / (file_idx - 1)
                remaining_files = total_files - (file_idx - 1)
                est_remaining_sec = avg_time_per_file * remaining_files
                eta_str = f"약 {format_time(est_remaining_sec)}"
            else:
                eta_str = "계산 중..."

            print(f"\n▶ [{file_idx}/{total_files}] 파일 처리 시작: {in_file.name}")
            print(f"   - 경로: {in_file}")
            print(f"   - 출력: {out_file}")
            print(f"   - 전체 남은 예상 시간: {eta_str}")

            # 기존 결과 파일 확인 (전체 완료 여부 확인)
            if sample_per_file <= 0 and out_file.exists():
                already_done_ids = load_processed_row_ids(str(out_file))
                total_in_lines = get_total_input_lines(in_file)
                if len(already_done_ids) >= total_in_lines and total_in_lines > 0:
                    print(f"   [스킵] 이미 전체 {len(already_done_ids):,}건이 변환 완료되어 있습니다.")
                    grand_processed += len(already_done_ids)
                    skipped_files += 1
                    total_pbar.update(1)
                    continue

            # 단일 파일 변환 실행
            proc_count, succ_count = process_council_file_with_kiwi(
                input_file=str(in_file),
                output_file=str(out_file),
                sample_limit=sample_per_file,
                min_char_len=min_char_len,
                num_workers=num_workers,
                analyzer=analyzer,
            )

            grand_processed += proc_count
            grand_success += succ_count

            total_pbar.update(1)
            total_pbar.set_postfix({"rows": f"{grand_processed:,}"})

    elapsed_total = time.time() - start_batch_time
    avg_speed = grand_processed / elapsed_total if elapsed_total > 0 else 0

    # 4. 최종 종합 보고서
    print("\n" + "=" * 70)
    print(" [전체 일괄 변환 완료 종합 보고서] ")
    print("=" * 70)
    print(f" * 총 대상 파일   : {total_files}개")
    print(f" * 건너뛴 파일    : {skipped_files}개 (이미 완료됨)")
    print(f" * 변환된 파일    : {total_files - skipped_files}개")
    print(f" * 이번 총 처리 건: {grand_processed:,}건")
    print(f" * 유효 정책 발언 : {grand_success:,}건")
    print(f" * 총 소요 시간   : {elapsed_total:.2f}초 ({elapsed_total / 60:.1f}분)")
    print(f" * 평균 처리 속도 : 초당 {avg_speed:.1f}건")
    print(f" * 결과 저장 위치 : {out_base.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="change_text 내의 모든 txt 회의록을 data2 폴더의 jsonl 파일로 일괄 변환하는 스크립트"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="change_text",
        help="입력 회의록 루트 디렉토리 (기본: change_text)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data2",
        help="출력 JSONL 저장 루트 디렉토리 (기본: data2)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.txt",
        help="검색할 파일 패턴 (기본: *.txt)",
    )
    parser.add_argument(
        "--year",
        type=str,
        default="",
        help="특정 연도만 선택 변환 (예: 2024 또는 2025)",
    )
    parser.add_argument(
        "--sample-per-file",
        type=int,
        default=0,
        help="각 파일당 처리할 샘플 수 (0이면 전체 처리, 기본: 0)",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=80,
        help="최소 발언 길이 필터링 (기본: 80자)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Kiwi 병렬 워커 수 (기본: 4)",
    )
    parser.add_argument(
        "--preserve-structure",
        dest="preserve_structure",
        action="store_true",
        default=True,
        help="하위 폴더(예: 2024_광역 회의록/) 구조를 그대로 유지하여 저장 (기본값: True)",
    )
    parser.add_argument(
        "--flat",
        dest="preserve_structure",
        action="store_false",
        help="하위 폴더를 만들지 않고 data2 바로 아래에 플랫하게 저장",
    )

    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    run_batch_extraction(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        year_filter=args.year,
        sample_per_file=args.sample_per_file,
        min_char_len=args.min_len,
        num_workers=args.workers,
        preserve_structure=args.preserve_structure,
    )

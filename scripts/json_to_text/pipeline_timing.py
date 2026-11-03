"""지방의회 회의록 전처리 파이프라인 (1~4단계) 소요 시간 측정 및 로깅 모듈.

각 단계(1: 샘플추출, 2: 규칙생성, 3: 규칙통합, 4: 텍스트변환)별로 파일 처리 시간과
누적 소요 시간을 CSV 파일로 기록하고 콘솔에 요약 출력합니다.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import _constant as constant

# 타이밍 로그 저장 디렉토리 및 파일 경로 설정
TIMING_DIR = Path(constant.rules_output_dir) / "timing"
LOG_PATH = TIMING_DIR / "pipeline_timing.csv"
RUN_PATH = TIMING_DIR / "active_run.json"

# CSV 로그 필드 스키마 정의
FIELDS = [
    "run_id",           # 파이프라인 실행 고유 ID (YYYYMMDD_HHMMSS)
    "row_type",         # 행 유형: file (단일 파일), step_total (단계 합계), pipeline_total (누계)
    "step",             # 파이프라인 단계 (1, 2, 3, 4)
    "source_json",      # 대상 회의록 JSON 명칭
    "source_path",      # 실제 처리 대상 파일 경로
    "elapsed_seconds",  # 소요 시간 (초)
    "status",           # 처리 상태 (success, error, fallback 등)
    "item_count",       # 처리 레코드/행 수
    "started_at",       # 시작 타임스탬프 (ISO 8601)
    "finished_at",      # 종료 타임스탬프 (ISO 8601)
]


def now_iso() -> str:
    """현재 시각을 ISO 8601 문자열(초 단위)로 반환합니다."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def start_step(step: str) -> Tuple[str, str]:
    """파이프라인 특정 단계(step) 시작을 기록하고 실행 ID(run_id)와 시작 시각을 반환합니다.
    
    1단계이거나 활성 실행 정보가 없으면 새로운 run_id를 생성하고,
    그 외에는 기존 실행의 run_id를 유지하여 1~4단계 소요 시간을 연속 추적합니다.
    """
    TIMING_DIR.mkdir(parents=True, exist_ok=True)
    started_at = now_iso()
    
    if str(step) == "1" or not RUN_PATH.exists():
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            RUN_PATH.write_text(json.dumps({"run_id": run_id}, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    else:
        try:
            run_id = json.loads(RUN_PATH.read_text(encoding="utf-8")).get("run_id")
        except (OSError, json.JSONDecodeError):
            run_id = None
        if not run_id:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                RUN_PATH.write_text(json.dumps({"run_id": run_id}, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass

    print(f"[시간기록] 실행 ID={run_id}, 단계={step}, 로그 경로={LOG_PATH}", flush=True)
    return run_id, started_at


def source_json_name(path: str | Path, step: str) -> str:
    """파일명에서 접두어(sample_, rule_ 등)를 제거하고 정규화된 원본 JSON 이름을 추출합니다."""
    stem = Path(path).stem
    if str(step) == "2" and stem.startswith("sample_"):
        stem = stem[len("sample_"):]
    elif str(step) == "3":
        for prefix in ("rule_sample_", "rule_"):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
    return stem + ".json"


def _append_log(row: dict) -> None:
    """CSV 로그 파일에 단일 레코드를 스레드/프로세스 안전하게 추가합니다."""
    try:
        TIMING_DIR.mkdir(parents=True, exist_ok=True)
        new_file = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0
        with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
    except OSError as e:
        print(f"[로그 경고] 소요 시간 기록 파일 작성 실패: {e}")


def record_file(
    run_id: str,
    step: str,
    path: str | Path,
    elapsed: float,
    status: str,
    item_count: int = 0,
    started_at: Optional[str] = None,
) -> None:
    """개별 파일 처리 결과를 로그에 추가하고 표준 출력에 진행 상황을 표시합니다."""
    finished_at = now_iso()
    row = {
        "run_id": run_id,
        "row_type": "file",
        "step": str(step),
        "source_json": source_json_name(path, step),
        "source_path": str(path),
        "elapsed_seconds": f"{float(elapsed):.3f}",
        "status": status,
        "item_count": item_count,
        "started_at": started_at or finished_at,
        "finished_at": finished_at,
    }
    _append_log(row)
    print(
        f"[시간][{step}단계] {row['source_json']}: {float(elapsed):.1f}초, "
        f"상태={status}, 건수={item_count}",
        flush=True,
    )


def finish_step(run_id: str, step: str, started_at: Optional[str] = None) -> None:
    """단계를 마감하고 단계별 합계 시간 및 전체 누적 시간을 로그에 기록합니다."""
    rows = []
    if LOG_PATH.exists():
        try:
            with LOG_PATH.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except OSError:
            pass

    file_rows = [r for r in rows if r.get("row_type") == "file" and r.get("run_id") == run_id]
    step_rows = [r for r in file_rows if r.get("step") == str(step)]
    step_total = sum(float(r.get("elapsed_seconds") or 0) for r in step_rows)
    pipeline_total = sum(float(r.get("elapsed_seconds") or 0) for r in file_rows)
    finished_at = now_iso()

    for row_type, label, elapsed in (
        ("step_total", f"{step}단계 합계", step_total),
        ("pipeline_total", "1~현재 단계 누계", pipeline_total),
    ):
        _append_log({
            "run_id": run_id,
            "row_type": row_type,
            "step": str(step),
            "source_json": label,
            "source_path": "",
            "elapsed_seconds": f"{elapsed:.3f}",
            "status": "summary",
            "item_count": len(step_rows) if row_type == "step_total" else len(file_rows),
            "started_at": started_at or finished_at,
            "finished_at": finished_at,
        })

    print(
        f"[시간합계][{step}단계] 파일별 합계={step_total:.1f}초 | 1~현재 단계 누계={pipeline_total:.1f}초",
        flush=True,
    )

    # 4단계 완료 시 전체 파일별 1~4단계 종합 소요 시간 집계
    if str(step) == "4":
        by_source = defaultdict(float)
        stage_counts = defaultdict(int)
        for row in file_rows:
            by_source[row["source_json"]] += float(row.get("elapsed_seconds") or 0)
            stage_counts[row["source_json"]] += 1
        for source_json, elapsed in sorted(by_source.items()):
            _append_log({
                "run_id": run_id,
                "row_type": "source_json_total",
                "step": "1-4",
                "source_json": source_json,
                "source_path": "",
                "elapsed_seconds": f"{elapsed:.3f}",
                "status": "summary",
                "item_count": stage_counts[source_json],
                "started_at": started_at or finished_at,
                "finished_at": finished_at,
            })
            print(f"[시간합계][JSON] {source_json}: 1~4단계 누적 {elapsed:.1f}초", flush=True)

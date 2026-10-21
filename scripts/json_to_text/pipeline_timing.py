"""Shared per-file timing log for pipeline steps 1 through 4."""
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import _constant as constant


TIMING_DIR = Path(constant.rules_output_dir) / "timing"
LOG_PATH = TIMING_DIR / "pipeline_timing.csv"
RUN_PATH = TIMING_DIR / "active_run.json"
FIELDS = [
    "run_id", "row_type", "step", "source_json", "source_path",
    "elapsed_seconds", "status", "item_count", "started_at", "finished_at",
]


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def start_step(step):
    """Start a timing stage. Step 1 begins a new pipeline run; later steps join it."""
    TIMING_DIR.mkdir(parents=True, exist_ok=True)
    started_at = now_iso()
    if str(step) == "1" or not RUN_PATH.exists():
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        RUN_PATH.write_text(json.dumps({"run_id": run_id}, ensure_ascii=False), encoding="utf-8")
    else:
        try:
            run_id = json.loads(RUN_PATH.read_text(encoding="utf-8")).get("run_id")
        except (OSError, json.JSONDecodeError):
            run_id = None
        if not run_id:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            RUN_PATH.write_text(json.dumps({"run_id": run_id}, ensure_ascii=False), encoding="utf-8")
    print(f"[시간기록] 실행 ID={run_id}, 단계={step}, 로그={LOG_PATH}", flush=True)
    return run_id, started_at


def source_json_name(path, step):
    name = Path(path).name
    stem = Path(name).stem
    if str(step) == "2" and stem.startswith("sample_"):
        stem = stem[len("sample_"):]
    elif str(step) == "3":
        for prefix in ("rule_sample_", "rule_"):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
    return stem + ".json"


def _append(row):
    TIMING_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def record_file(run_id, step, path, elapsed, status, item_count=0, started_at=None):
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
    _append(row)
    print(f"[시간][{step}단계] {row['source_json']}: {float(elapsed):.1f}초, 상태={status}, 건수={item_count}", flush=True)


def finish_step(run_id, step, started_at=None):
    """Persist and display per-stage and cumulative processing-time totals."""
    rows = []
    if LOG_PATH.exists():
        with LOG_PATH.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    file_rows = [r for r in rows if r.get("row_type") == "file" and r.get("run_id") == run_id]
    step_rows = [r for r in file_rows if r.get("step") == str(step)]
    step_total = sum(float(r.get("elapsed_seconds") or 0) for r in step_rows)
    pipeline_total = sum(float(r.get("elapsed_seconds") or 0) for r in file_rows)
    finished_at = now_iso()
    for row_type, label, elapsed in (
        ("step_total", f"{step}단계 합계", step_total),
        ("pipeline_total", "1~현재 단계 누계", pipeline_total),
    ):
        _append({
            "run_id": run_id, "row_type": row_type, "step": str(step),
            "source_json": label, "source_path": "", "elapsed_seconds": f"{elapsed:.3f}",
            "status": "summary", "item_count": len(step_rows) if row_type == "step_total" else len(file_rows),
            "started_at": started_at or finished_at, "finished_at": finished_at,
        })
    print(f"[시간합계][{step}단계] 파일별 합계={step_total:.1f}초 | 1~현재 단계 누계={pipeline_total:.1f}초", flush=True)
    if str(step) == "4":
        by_source = defaultdict(float)
        stage_counts = defaultdict(int)
        for row in file_rows:
            by_source[row["source_json"]] += float(row.get("elapsed_seconds") or 0)
            stage_counts[row["source_json"]] += 1
        for source_json, elapsed in sorted(by_source.items()):
            _append({
                "run_id": run_id, "row_type": "source_json_total", "step": "1-4",
                "source_json": source_json, "source_path": "",
                "elapsed_seconds": f"{elapsed:.3f}", "status": "summary",
                "item_count": stage_counts[source_json],
                "started_at": started_at or finished_at, "finished_at": finished_at,
            })
            print(f"[시간합계][JSON] {source_json}: 1~4단계 누적 {elapsed:.1f}초", flush=True)

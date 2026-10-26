"""Run the project's main.py for shared benchmark queries and save baseline evidence.

From repository root, use the project's Windows mini environment or another fully
configured interpreter:
    python model/evaluation/build_main_baseline.py
The existing model/test_history.json is restored after collecting the new entries.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN_SCRIPT = PROJECT_ROOT / "model" / "main.py"
HISTORY_PATH = PROJECT_ROOT / "model" / "test_history.json"
OUTPUT_PATH = PROJECT_ROOT / "model" / "evaluation" / "main_model_baseline.json"
TEST_QUERIES = [
    "다문화 가정",
    "노인 일자리",
    "청년 일자리",
    "지방 소멸",
    "출산 장려",
    "전통시장 활성화",
]
RUN_TIMEOUT_SECONDS = 3600


def make_stdin(queries: list[str]) -> str:
    return "".join(f"{query}\n" for query in queries) + "종료\n"


def merge_query_results(existing_report: dict, refreshed: list[dict], benchmark_queries: list[str]) -> list[dict]:
    """Replace refreshed queries, preserve other baselines, and enforce query order."""
    by_query = {
        item.get("query"): item
        for item in existing_report.get("queries", [])
        if item.get("query") in benchmark_queries
    }
    by_query.update({item.get("query"): item for item in refreshed})
    return [by_query[query] for query in benchmark_queries if query in by_query]


def extract_expected_rows(history: list[dict]) -> dict[str, list[dict]]:
    """Extract evidence rows with real source identifiers from main.py history."""
    rows: dict[str, list[dict]] = {}
    for entry in history:
        query = str(entry.get("input") or "")
        if not query:
            continue
        query_rows = rows.setdefault(query, [])
        output = entry.get("output") or {}
        for result in output.get("results", []):
            for evidence in result.get("evidence", []):
                dataset_key = str(evidence.get("datasetKey") or "")
                row_id = evidence.get("originalRowId")
                if not dataset_key or row_id in (None, ""):
                    continue
                query_rows.append({
                    "source_id": f"{dataset_key}:{row_id}",
                    "dataset_key": dataset_key,
                    "row_id": row_id,
                    "date": evidence.get("date", ""),
                    "council": evidence.get("council", ""),
                    "meeting_title": evidence.get("meetingTitle", ""),
                    "agenda_title": evidence.get("agendaTitle", ""),
                    "excerpt": evidence.get("excerpt", ""),
                })
        # Keep only one instance of a source row while preserving ranked order.
        seen = set()
        rows[query] = [item for item in query_rows
                       if not (item["source_id"] in seen or seen.add(item["source_id"]))]
    return rows


def run_main_queries(queries: list[str] | None = None) -> dict:
    queries = list(queries or TEST_QUERIES)
    unknown_queries = sorted(set(queries) - set(TEST_QUERIES))
    if unknown_queries:
        raise ValueError(f"알 수 없는 benchmark query: {unknown_queries}")
    if not MAIN_SCRIPT.is_file():
        raise FileNotFoundError(f"main 실행 파일을 찾을 수 없습니다: {MAIN_SCRIPT}")
    existing_report = json.loads(OUTPUT_PATH.read_text(encoding="utf-8")) if OUTPUT_PATH.exists() else {}
    original_history = HISTORY_PATH.read_bytes() if HISTORY_PATH.exists() else None
    original_entries = json.loads(original_history.decode("utf-8")) if original_history else []
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    completed = None
    new_entries = []
    try:
        completed = subprocess.run(
            [sys.executable, str(MAIN_SCRIPT)],
            input=make_stdin(queries), capture_output=True, text=True,
            encoding="utf-8", errors="replace", cwd=PROJECT_ROOT, env=environment,
            timeout=RUN_TIMEOUT_SECONDS, check=False,
        )
        if HISTORY_PATH.exists():
            current_entries = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
            new_entries = current_entries[len(original_entries):]
    finally:
        if original_history is None:
            HISTORY_PATH.unlink(missing_ok=True)
        else:
            HISTORY_PATH.write_bytes(original_history)

    entries_by_query = {entry.get("input"): entry for entry in new_entries}
    retrieved = extract_expected_rows(new_entries)
    results = []
    for query in queries:
        entry = entries_by_query.get(query, {})
        results.append({
            "query": query,
            "status": entry.get("status", "not_run"),
            "error": entry.get("error"),
            "timings_seconds": entry.get("timings_seconds", {}),
            "extracted_params": entry.get("extracted_params", {}),
            "retrieved_source_row_count": len(retrieved.get(query, [])),
            "expected_source_rows": retrieved.get(query, []),
        })
    benchmark_results = merge_query_results(existing_report, results, TEST_QUERIES)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "baseline_model": "현재 프로젝트 model/main.py 설정 (model/_constant.py)",
        "main_script": str(MAIN_SCRIPT),
        "command": f"{sys.executable} model/main.py < {'; '.join(queries)}",
        "queries": benchmark_results,
        "process_return_code": completed.returncode if completed else None,
        "run_successfully": bool(completed and completed.returncode == 0
                                  and len(benchmark_results) == len(TEST_QUERIES)
                                  and all(item["status"] == "success" for item in benchmark_results)),
        "interpretation": "main.py 검색 결과 기준선이며, 정답 라벨로 사용하기 전에 사람이 원문을 검수해야 합니다.",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Persist the UTF-8 JSON as the artifact; keep console output ASCII-only so
    # Windows cp949 terminals cannot fail after the report has been saved.
    counts = ", ".join(f"{index + 1}:{len(item['expected_source_rows'])} rows" for index, item in enumerate(results))
    print(f"Baseline saved to {OUTPUT_PATH}. Success={report['run_successfully']}; counts={counts}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query", action="append", choices=TEST_QUERIES,
        help="Run only this query and merge it into the saved benchmark baseline. Repeatable.",
    )
    args = parser.parse_args()
    try:
        report = run_main_queries(args.query or TEST_QUERIES)
    except subprocess.TimeoutExpired as exc:
        print(f"main.py가 {RUN_TIMEOUT_SECONDS}초 안에 끝나지 않았습니다: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"기준선 생성 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(0 if report["run_successfully"] else 1)

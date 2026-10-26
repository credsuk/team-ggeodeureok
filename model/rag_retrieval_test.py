"""CLI 모델 비교: 실행 시간과 출력 데이터의 기대값 일치 여부를 확인합니다.

프로젝트 루트 실행: python model/rag_retrieval_test.py
각 모델의 명령은 argv 배열로 설정하고 {query}를 검색어 위치에 둡니다.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── 설정 상수 ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 모델별 메인 실행 파일과 옵션을 등록합니다. 예:
# {"name": "model-A", "command": ["python", "main.py", "-p", "{query}"],
#  "cwd": PROJECT_ROOT / "other_model"}
# command는 셸 문자열이 아닌 argv 배열로 적고, 검색어 자리에 {query}를 둡니다.
MODEL_COMMANDS = [
    # 아래 예시를 실제 모델의 실행 파일/명령어에 맞게 복사해 설정하세요.
    # {"name": "모델A", "command": ["python", "main.py", "-p", "{query}"], "cwd": PROJECT_ROOT / "model_a"},
]
TEST_QUERIES = ["신혼부부"]
# 각 검색어마다 모델 출력에 있어야 할 데이터 조각과 나오면 안 되는 데이터를 설정합니다.
# 판별 대상 출력은 stdout 및 stderr 텍스트입니다.
EXPECTED_CONTAINS = {"신혼부부": ["검수한 기대 결과 문구를 여기에 입력"]}
EXPECTED_NOT_CONTAINS = {"신혼부부": []}
COMMAND_TIMEOUT_SECONDS = 180
REPORT_PATH = PROJECT_ROOT / "model" / "model_comparison_report.json"
# ────────────────────────────────────────────────────────────────────────────


def _validate_settings() -> None:
    if not MODEL_COMMANDS:
        raise ValueError("MODEL_COMMANDS에 모델별 main 실행 명령을 등록하세요.")
    if not TEST_QUERIES or any(not q.strip() for q in TEST_QUERIES):
        raise ValueError("TEST_QUERIES에 비어 있지 않은 검색어를 입력하세요.")
    for model in MODEL_COMMANDS:
        if not model.get("name") or not model.get("command"):
            raise ValueError("각 모델 설정에는 name과 command가 필요합니다.")
        if not any("{query}" in str(arg) for arg in model["command"]):
            raise ValueError(f"{model['name']}: command에 {{query}} 자리표시자가 필요합니다.")
        cwd = Path(model.get("cwd", PROJECT_ROOT))
        if not cwd.is_dir():
            raise FileNotFoundError(f"{model['name']}: 실행 폴더가 없습니다: {cwd}")
    for query in TEST_QUERIES:
        expected = EXPECTED_CONTAINS.get(query, [])
        if not expected or any("검수한 기대 결과" in value for value in expected):
            raise ValueError(f"{query!r}의 EXPECTED_CONTAINS에 원천 데이터에서 검수한 기대 문구를 넣으세요.")


def _run_model(model: dict, query: str) -> dict:
    cwd = Path(model.get("cwd", PROJECT_ROOT))
    argv = [str(arg).replace("{query}", query) for arg in model["command"]]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=COMMAND_TIMEOUT_SECONDS, check=False,
        )
        elapsed = time.perf_counter() - started
        output = completed.stdout + ("\n" + completed.stderr if completed.stderr else "")
        checks = [{"text": item, "passed": item in output} for item in EXPECTED_CONTAINS.get(query, [])]
        negatives = [{"text": item, "passed": item not in output} for item in EXPECTED_NOT_CONTAINS.get(query, [])]
        return {
            "model": model["name"], "command": shlex.join(argv), "cwd": str(cwd),
            "return_code": completed.returncode, "elapsed_seconds": round(elapsed, 6),
            "passed": completed.returncode == 0 and all(x["passed"] for x in checks + negatives),
            "expected_checks": checks, "unexpected_data_checks": negatives, "output": output,
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        stdout, stderr = exc.stdout or "", exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            "model": model["name"], "command": shlex.join(argv), "cwd": str(cwd),
            "return_code": None, "elapsed_seconds": round(elapsed, 6), "passed": False,
            "error": f"명령 시간 초과 ({COMMAND_TIMEOUT_SECONDS}초)",
            "output": (stdout + "\n" + stderr).strip(),
        }


def run_comparison() -> dict:
    _validate_settings()
    results = []
    for query in TEST_QUERIES:
        for model in MODEL_COMMANDS:
            result = _run_model(model, query)
            result["query"] = query
            results.append(result)
    report = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "execution_target": "MODEL_COMMANDS에 설정한 각 모델의 main 실행 명령",
        "expected_data_source": "사용자가 검수해 EXPECTED_CONTAINS/EXPECTED_NOT_CONTAINS에 입력한 공통 원천 데이터 기대값",
        "passed": all(result["passed"] for result in results), "results": results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n보고서 저장: {REPORT_PATH}")
    return report


if __name__ == "__main__":
    try:
        result = run_comparison()
    except Exception as exc:
        print(f"비교 실행 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(0 if result["passed"] else 1)

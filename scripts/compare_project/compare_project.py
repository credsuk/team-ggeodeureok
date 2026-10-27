#!/usr/bin/env python3
"""Run the colocated main.py and write one readable evaluation report."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# =============================================================================
# 사용 방법 (다른 프로젝트에서 실행할 때)
#
# 1. 실행할 main.py, 이 파일, 기준 JSON(compare_project.json)을 같은 폴더에 둡니다.
#    예: project/model/main.py
#        project/model/compare_project.py
#        project/model/compare_project.json
#
# 2. main.py를 직접 실행하는 명령에 맞춰 MAIN_ARGUMENTS_TEMPLATE 또는
#    MAIN_STDIN_TEMPLATE 중 하나를 설정합니다. {query}가 각 테스트 질문으로 바뀝니다.
#
#    옵션형 CLI 예시: python main.py -d "다문화 가정"
#      MAIN_ARGUMENTS_TEMPLATE = ("-d", "{query}")
#
#    긴 옵션 CLI 예시: python main.py --query "다문화 가정"
#      MAIN_ARGUMENTS_TEMPLATE = ("--query", "{query}")
#
#    위치 인자 CLI 예시: python main.py "다문화 가정"
#      MAIN_ARGUMENTS_TEMPLATE = ("{query}",)
#
#    대화형 입력 예시: 질문을 입력받고 '종료'로 끝나는 main.py
#      MAIN_ARGUMENTS_TEMPLATE = ()
#      MAIN_STDIN_TEMPLATE = "{query}\n종료\n"
#    Python 문자열의 \n 표기는 실제 줄바꿈 문자입니다. 두 설정을 동시에 사용하면 안 됩니다.
#
# 3. TEST_QUERIES를 compare_project.json의 queries와 글자 및 순서까지 똑같이 맞춥니다.
#    일치하지 않으면 기준과 다른 질문을 비교하지 않도록 실행을 중단합니다.
#
# 4. 대상 프로젝트가 사용하는 Python 환경에서 실행합니다.
#      cd project/model
#      python compare_project.py
#
# 5. model 폴더에 두었다면 결과는 project/evaluation/
#    answer_comparison_<실행시각>/evaluation_report.md 한 파일로 저장됩니다.
#
# 6. 검색 근거 행 수는 main.py 출력의 '[근거 자료] N건'에서 읽습니다.
#    그 표기가 없거나 여러 번 나오면 해당 질문의 후보 행 수는 확인 불가입니다.
#    행 수만 비교하며 답변의 정확도/품질은 평가하지 않습니다.
# =============================================================================
# compare_project.py, compare_project.json, main.py는 반드시 같은 폴더에 둡니다.
PROJECT_ROOT = Path(__file__).resolve().parent
MAIN_SCRIPT_NAME = "main.py"
BASELINE_FILE_NAME = "compare_project.json"

# 현재 프로젝트 main.py는 대화형 stdin 입력입니다. 다른 CLI라면 예시를 참고해 변경하세요.
MAIN_ARGUMENTS_TEMPLATE = ()
# stdin에 질문과 종료 명령을 각각 한 줄로 전달합니다. Python 문자열의 \n은 줄바꿈입니다.
MAIN_STDIN_TEMPLATE: str | None = "{query}\n종료\n"

# compare_project.json의 queries와 질문 문자열·순서를 똑같이 맞춥니다.
TEST_QUERIES = (
    "다문화 가정",
    "노인 일자리",
    "청년 일자리",
    "지방 소멸",
    "출산 장려",
    "전통시장 활성화",
)
PYTHON_EXECUTABLE = sys.executable
BASELINE_TIMING_KEY = "전체 처리"
RUN_TIMEOUT_SECONDS = 3600
OUTPUT_DIRECTORY_NAME = "evaluation"
RUN_FOLDER_PREFIX = "answer_comparison"


def format_main_arguments(query: str) -> list[str]:
    return [part.format(query=query) for part in MAIN_ARGUMENTS_TEMPLATE]


def _load_baseline(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Baseline file not found beside compare_project.py: {path}")
    baseline = json.loads(path.read_text(encoding="utf-8"))
    records = baseline.get("queries")
    if not isinstance(records, list) or not records:
        raise ValueError("Baseline JSON must contain a non-empty 'queries' array")
    baseline_queries = [item.get("query") for item in records]
    if baseline_queries != list(TEST_QUERIES):
        raise ValueError(
            "Baseline queries do not match TEST_QUERIES in compare_project.py. "
            f"Baseline={baseline_queries!r}; TEST_QUERIES={list(TEST_QUERIES)!r}"
        )
    return records


def _reserve_run_directory(root: Path) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    base = root / f"{RUN_FOLDER_PREFIX}_{stamp}"
    result = base
    suffix = 1
    while True:
        try:
            result.mkdir(parents=True, exist_ok=False)
            return result
        except FileExistsError:
            result = Path(f"{base}_{suffix}")
            suffix += 1


def _decode_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _run_query(query: str, project_root: Path) -> dict:
    main_script = project_root / MAIN_SCRIPT_NAME
    if not main_script.is_file():
        raise FileNotFoundError(f"main.py must be beside compare_project.py: {main_script}")
    if MAIN_ARGUMENTS_TEMPLATE and MAIN_STDIN_TEMPLATE is not None:
        raise ValueError("Set MAIN_ARGUMENTS_TEMPLATE or MAIN_STDIN_TEMPLATE, not both")
    if not MAIN_ARGUMENTS_TEMPLATE and MAIN_STDIN_TEMPLATE is None:
        raise ValueError("Interactive mode requires MAIN_STDIN_TEMPLATE")

    command = [str(PYTHON_EXECUTABLE), str(main_script), *format_main_arguments(query)]
    input_text = MAIN_STDIN_TEMPLATE.format(query=query) if MAIN_STDIN_TEMPLATE is not None else ""
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    started = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=project_root,
            env=environment,
            timeout=RUN_TIMEOUT_SECONDS,
            check=False,
        )
        elapsed = time.perf_counter() - started
        return_code = completed.returncode
        output_text = completed.stdout
        stderr_text = completed.stderr
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        timed_out = True
        return_code = None
        output_text = _decode_output(exc.stdout)
        stderr_text = _decode_output(exc.stderr)

    if timed_out:
        status = "error"
        error = f"Timed out after {RUN_TIMEOUT_SECONDS} seconds"
    elif return_code != 0:
        status = "error"
        error = f"Process exited with code {return_code}"
        if stderr_text.strip():
            error += f": {stderr_text.strip()}"
    else:
        status = "success"
        error = None

    return {
        "query": query,
        "status": status,
        "error": error,
        "stderr": stderr_text,
        "return_code": return_code,
        "timed_out": timed_out,
        "output": output_text,
        "has_answer": bool(output_text.strip()),
        "answer_char_count": len(output_text.strip()),
        "answer_word_count": len(output_text.split()),
        "elapsed_seconds": round(elapsed, 4),
        "command": command,
    }


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _extract_evidence_count(output: str) -> int | None:
    """Read the target's explicit evidence-row count, not council/day metrics."""
    matches = re.findall(r"^\s*\[근거 자료\]\s*(\d+)\s*건\s*$", output, re.MULTILINE)
    return int(matches[0]) if len(matches) == 1 else None


def _row_count_verdict(delta: int | None) -> str:
    if delta is None:
        return "비교 불가"
    if delta > 0:
        return "후보가 더 많음"
    if delta < 0:
        return "후보가 더 적음"
    return "동일"


def _seconds(value: Any) -> str:
    number = _number(value)
    return f"{number:.4f}초" if number is not None else "확인 불가"


def _code_block(text: str) -> str:
    longest_backtick_run = 0
    current_run = 0
    for character in text:
        if character == "`":
            current_run += 1
            longest_backtick_run = max(longest_backtick_run, current_run)
        else:
            current_run = 0
    fence = "`" * max(3, longest_backtick_run + 1)
    body = text.rstrip("\n") or "(내용 없음)"
    return f"{fence}text\n{body}\n{fence}"


def _render_markdown_report(report: dict) -> str:
    overall = report["overall_evaluation"]
    total = overall["candidate_invocation_count"]
    candidate_count = overall["candidate_retrieved_source_row_count"]
    candidate_count_text = (
        str(candidate_count)
        if candidate_count is not None
        else f"확인 불가 (측정 {overall['candidate_retrieved_source_query_count']}/{total}개 질문)"
    )
    lines = [
        "# 전체 평가 보고서",
        "",
        f"- 실행 시각: {report['created_at']}",
        f"- 기준 파일: {Path(report['baseline_file']).name}",
        f"- 실행 메인: {Path(report['main_script']).name}",
        "",
        "## 전체 평가",
        "",
        f"- 전체 실행 상태: {'성공' if report['run_successfully'] else '일부 실패'}",
        f"- 성공한 프로세스 수: {overall['successful_process_count']}/{total}",
        f"- 성공한 답변 수: {overall['successful_answer_count']}/{total}",
        f"- 비어 있지 않은 출력 수: {overall['nonempty_output_count']}/{total}",
        f"- 전체 답변 길이: {overall['answer_char_count']:,}자 / {overall['answer_word_count']:,}단어",
        f"- 정확도: {'평가함' if overall['accuracy_evaluated'] else '평가하지 않음 (기준 답변 없음)'}",
        f"- 기준 검색 근거 행 수 합계: {overall['baseline_retrieved_source_row_count']}",
        f"- 후보 검색 근거 행 수 합계: {candidate_count_text}",
        f"- 행 수 비교 가능 질문 수: {overall['source_row_count_comparable_query_count']}/{total}",
        f"- 전체 행 수 차이: {overall['source_row_count_delta']:+d}행 ({_row_count_verdict(overall['source_row_count_delta'])})" if overall["source_row_count_delta"] is not None else "- 전체 행 수 차이: 비교 불가 (일부 질문의 후보 건수 미확인)",
    ]
    baseline_mean = overall["mean_baseline_internal_seconds"]
    candidate_mean = overall["mean_candidate_process_seconds"]
    change = overall["mean_latency_change_percent"]
    if baseline_mean is not None and candidate_mean is not None:
        change_text = f"{change:+.2f}%" if change is not None else "계산 불가"
        speed_text = {
            "faster": "후보가 더 빠름",
            "slower": "후보가 더 느림",
            "same_or_unavailable": "동일 또는 비교 불가",
        }.get(overall["speed_verdict"], overall["speed_verdict"])
        lines.append(
            f"- 평균 처리 시간: 기준선 내부 {_seconds(baseline_mean)} / "
            f"후보 프로세스 전체 {_seconds(candidate_mean)} ({change_text}, 근사 비교)"
        )
        lines.append(f"- 속도 요약: {speed_text} (근사 비교)")
    else:
        lines.append("- 평균 처리 시간: 비교 데이터 부족")
    lines.extend(["", "## 평가 기준과 한계", ""])
    lines.extend(f"- {note}" for note in report["notes"])
    lines.extend(["", "## 질문별 결과", ""])

    for index, item in enumerate(report["queries"], start=1):
        baseline_timings = item.get("baseline_timings_seconds") or {}
        baseline_timing_text = (
            ", ".join(f"{name}: {_seconds(value)}" for name, value in baseline_timings.items())
            or "기록 없음"
        )
        change_value = item.get("latency_change_percent")
        change_text = f"{change_value:+.2f}% (근사 비교)" if change_value is not None else "계산 불가"
        query_heading = item["query"].replace("`", "\\`").replace("\n", " ")
        baseline_count = item.get("baseline_retrieved_source_row_count")
        baseline_count_text = baseline_count if baseline_count is not None else "확인 불가"
        candidate_count = item.get("candidate_retrieved_source_row_count")
        candidate_count_text = candidate_count if candidate_count is not None else "확인 불가"
        row_delta = item.get("source_row_count_delta")
        lines.extend([
            f"### {index}. {query_heading}",
            "",
            f"- 상태: {'성공' if item['status'] == 'success' else '실패'}",
            f"- 기준 처리 시간: {baseline_timing_text}",
            f"- 후보 프로세스 전체 시간: {_seconds(item['timings_seconds'].get('프로세스 전체'))}",
            f"- 시간 변화: {change_text}",
            f"- 검색 근거 행 수: 기준 {baseline_count_text} / 후보 {candidate_count_text}",
            f"- 행 수 차이: {row_delta:+d}행 ({_row_count_verdict(row_delta)})" if row_delta is not None else "- 행 수 차이: 비교 불가",
            f"- 답변 길이: {item['answer_char_count']:,}자 / {item['answer_word_count']:,}단어",
            f"- 종료 코드: {item.get('return_code')}",
            "",
            "#### 답변 원문",
            "",
            _code_block(item.get("output", "")),
        ])
        if item.get("error"):
            lines.extend(["", "#### 오류", "", _code_block(item["error"])])
        if item.get("stderr", "").strip():
            lines.extend(["", "#### 표준 오류 출력", "", _code_block(item["stderr"])])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_evaluation(project_root: Path | None = None) -> dict:
    root = Path(project_root or PROJECT_ROOT).resolve()
    main_script = root / MAIN_SCRIPT_NAME
    baseline_path = root / BASELINE_FILE_NAME
    if not main_script.is_file():
        raise FileNotFoundError(f"main.py must be beside compare_project.py: {main_script}")
    baseline_queries = _load_baseline(baseline_path)

    # The script lives in <project>/model; put reports in <project>/evaluation.
    output_root = root.parent / OUTPUT_DIRECTORY_NAME
    run_dir = _reserve_run_directory(output_root)
    baseline_by_query = {item["query"]: item for item in baseline_queries}
    query_reports = []
    for query in TEST_QUERIES:
        candidate = _run_query(query, root)
        baseline = baseline_by_query[query]
        baseline_seconds = _number(baseline.get("timings_seconds", {}).get(BASELINE_TIMING_KEY))
        candidate_seconds = candidate["elapsed_seconds"]
        change_percent = (
            round((candidate_seconds - baseline_seconds) * 100 / baseline_seconds, 2)
            if baseline_seconds is not None and baseline_seconds > 0 else None
        )
        baseline_row_count = baseline.get("retrieved_source_row_count")
        candidate_row_count = _extract_evidence_count(candidate["output"])
        row_count_delta = (
            candidate_row_count - baseline_row_count
            if isinstance(baseline_row_count, int) and candidate_row_count is not None else None
        )
        query_reports.append({
            "query": query,
            "status": candidate["status"],
            "error": candidate["error"],
            "timings_seconds": {"프로세스 전체": candidate_seconds},
            "baseline_timings_seconds": baseline.get("timings_seconds", {}),
            "baseline_retrieved_source_row_count": baseline_row_count,
            "candidate_retrieved_source_row_count": candidate_row_count,
            "retrieved_source_row_count": candidate_row_count,
            "source_row_count_delta": row_count_delta,
            "source_row_count_verdict": _row_count_verdict(row_count_delta),
            "output": candidate["output"],
            "answer_char_count": candidate["answer_char_count"],
            "answer_word_count": candidate["answer_word_count"],
            "has_answer": candidate["has_answer"],
            "return_code": candidate["return_code"],
            "timed_out": candidate["timed_out"],
            "stderr": candidate["stderr"],
            "command": candidate["command"],
            "latency_change_percent": change_percent,
            "answer_accuracy": "not_evaluated",
        })

    successful_processes = sum(item["status"] == "success" for item in query_reports)
    successful_answers = sum(item["status"] == "success" and item["has_answer"] for item in query_reports)
    nonempty_outputs = sum(item["has_answer"] for item in query_reports)
    baseline_row_counts = [item.get("retrieved_source_row_count") for item in baseline_queries]
    baseline_retrieved_rows = (
        sum(baseline_row_counts)
        if all(isinstance(value, int) for value in baseline_row_counts) else None
    )
    candidate_row_counts = [item["candidate_retrieved_source_row_count"] for item in query_reports]
    candidate_retrieved_rows = (
        sum(candidate_row_counts)
        if all(value is not None for value in candidate_row_counts) else None
    )
    row_count_comparable = all(
        item["source_row_count_delta"] is not None for item in query_reports
    )
    row_count_delta = (
        candidate_retrieved_rows - baseline_retrieved_rows
        if row_count_comparable and candidate_retrieved_rows is not None and baseline_retrieved_rows is not None
        else None
    )
    baseline_times = [
        _number(item.get("timings_seconds", {}).get(BASELINE_TIMING_KEY))
        for item in baseline_queries
    ]
    candidate_times = [item["timings_seconds"]["프로세스 전체"] for item in query_reports]
    valid_baseline_times = [value for value in baseline_times if value is not None]
    mean_baseline = sum(valid_baseline_times) / len(valid_baseline_times) if valid_baseline_times else None
    mean_candidate = sum(candidate_times) / len(candidate_times) if candidate_times else None
    speed_verdict = (
        "faster" if mean_candidate is not None and mean_baseline is not None and mean_candidate < mean_baseline
        else "slower" if mean_candidate is not None and mean_baseline is not None and mean_candidate > mean_baseline
        else "same_or_unavailable"
    )
    total_chars = sum(item["answer_char_count"] for item in query_reports)
    total_words = sum(item["answer_word_count"] for item in query_reports)
    latency_change = (
        round((mean_candidate - mean_baseline) * 100 / mean_baseline, 2)
        if mean_candidate is not None and mean_baseline is not None and mean_baseline > 0 else None
    )

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "baseline_file": str(baseline_path),
        "main_script": str(main_script),
        "command_arguments_template": list(MAIN_ARGUMENTS_TEMPLATE),
        "run_successfully": successful_processes == len(TEST_QUERIES) and successful_answers == len(TEST_QUERIES),
        "overall_evaluation": {
            "accuracy_evaluated": False,
            "speed_verdict": speed_verdict,
            "mean_baseline_internal_seconds": round(mean_baseline, 4) if mean_baseline is not None else None,
            "mean_candidate_process_seconds": round(mean_candidate, 4) if mean_candidate is not None else None,
            "mean_latency_change_percent": latency_change,
            "baseline_query_count": len(baseline_queries),
            "candidate_invocation_count": len(query_reports),
            "successful_process_count": successful_processes,
            "successful_answer_count": successful_answers,
            "candidate_answer_count": successful_answers,
            "nonempty_output_count": nonempty_outputs,
            "answer_char_count": total_chars,
            "answer_word_count": total_words,
            "baseline_retrieved_source_row_count": baseline_retrieved_rows,
            "candidate_retrieved_source_row_count": candidate_retrieved_rows,
            "candidate_retrieved_source_query_count": sum(value is not None for value in candidate_row_counts),
            "source_row_count_comparable_query_count": sum(item["source_row_count_delta"] is not None for item in query_reports),
            "source_row_count_delta": row_count_delta,
            "source_row_count_verdict": _row_count_verdict(row_count_delta),
        },
        "notes": [
            "답변 정확도는 평가하지 않았습니다. 기준 JSON에 기준 답변 텍스트가 없습니다.",
            "출력은 main.py의 stdout 원문이며, 프롬프트·로그가 stdout에 포함되면 답변 글자/단어 수에도 포함됩니다.",
            "검색 근거 행 수는 stdout의 '[근거 자료] N건' 표기에서 추출해 기준선의 retrieved_source_row_count와 비교했습니다. '검색 결과 범위'는 의회·일 단위 통계이므로 행 수 비교에 사용하지 않았습니다.",
            "후보 앱이 '[근거 자료] N건' 표기를 제공하지 않거나 여러 번 출력하면 그 질문의 후보 행 수는 확인 불가로 남깁니다. 비교는 검색 근거 행 수만 대상으로 하며 답변 품질은 평가하지 않습니다.",
            "후보 시간은 프로세스 전체 시간이고 기준 시간은 내부 처리시간일 수 있어 속도 비교는 근사치입니다.",
        ],
        "queries": query_reports,
    }
    report_path = run_dir / "evaluation_report.md"
    report_path.write_text(_render_markdown_report(report), encoding="utf-8")
    return {"run_directory": run_dir, "evaluation": report, "report_path": report_path}


def main() -> int:
    try:
        result = run_evaluation()
    except Exception as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    report = result["evaluation"]
    overall = report["overall_evaluation"]
    print(f"Run saved: {result['report_path']}")
    print(f"Answers: {overall['successful_answer_count']}/{len(TEST_QUERIES)}")
    baseline_rows = overall["baseline_retrieved_source_row_count"]
    candidate_rows = overall["candidate_retrieved_source_row_count"]
    if baseline_rows is not None and candidate_rows is not None:
        row_delta = candidate_rows - baseline_rows
        english_verdict = {
            "후보가 더 많음": "candidate more",
            "후보가 더 적음": "candidate fewer",
            "동일": "same",
        }[_row_count_verdict(row_delta)]
        print(f"Evidence rows: candidate {candidate_rows} / baseline {baseline_rows}")
        print(f"Evidence row delta: {row_delta:+d} ({english_verdict})")
    else:
        available = overall["candidate_retrieved_source_query_count"]
        print(f"Evidence rows: unavailable for {len(TEST_QUERIES) - available}/{len(TEST_QUERIES)} queries")
    print(f"Speed summary (approximate): {overall['speed_verdict']}")
    print("Answer quality: not evaluated")
    return 0 if report["run_successfully"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

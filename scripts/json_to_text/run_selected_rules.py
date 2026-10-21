from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
os.chdir(PROJECT_DIR)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


import _constant as constant
from pipeline_timing import start_step, record_file, finish_step


DEFAULT_TARGETS = ["2024_경기도의회"]
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_INPUT_DIR = DATA_DIR / "org"


def load_script_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"스크립트를 불러올 수 없습니다: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_input(target: str, input_dir: str | Path = DEFAULT_INPUT_DIR) -> Path | None:
    candidate = Path(target)
    if candidate.suffix.lower() == ".json" and candidate.is_file():
        return candidate.resolve()

    stem = candidate.stem if candidate.suffix.lower() == ".json" else target
    year = stem.split("_", 1)[0]
    filename = f"{stem}.json"
    base = Path(input_dir)
    if not base.is_absolute():
        base = PROJECT_DIR / base

    if base.name.lower() in {"org", "org2"}:
        roots = [base]
    elif base.name.endswith("광역 회의록"):
        roots = [base]
    else:
        roots = [base / "org2", base / "org", base]

    checked = set()
    for root in roots:
        direct = root / filename if root.name.endswith("광역 회의록") else root / f"{year}_광역 회의록" / filename
        if direct.is_file():
            return direct
        # Also accept an input directory that already points at a year folder.
        same_dir = root / filename
        if root not in checked and same_dir.is_file():
            return same_dir
        checked.add(root)

    # Accept custom directory layouts while still matching only the requested file.
    for root in roots:
        if root.is_dir():
            matches = list(root.rglob(filename))
            if matches:
                return matches[0]

    # Permit a project-relative or absolute path even when the input is outside
    # the standard year folders.
    if candidate.suffix.lower() == ".json" and candidate.is_file():
        return candidate.resolve()
    return None


def make_sample(json_path: Path, sample_path: Path, splitter) -> tuple[str, int]:
    """Reuse step 1's extraction helpers for one JSON file only."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records = data if isinstance(data, list) else [data]
    middle = records[5:-5] if len(records) > 10 else records
    indices = list(range(len(middle)))
    random.shuffle(indices)
    samples = []

    for idx in indices:
        record = middle[idx]
        if not isinstance(record, dict) or not isinstance(record.get("MINTS_HTML"), str):
            continue
        try:
            soup = splitter.prepare_html(record["MINTS_HTML"])
            region = splitter.find_transcript_region(soup)
            if region is None:
                continue
            samples.append(splitter.shorten_long_text(region))
        except Exception as exc:
            print(f"  [레코드 건너뜀] index={idx}: {type(exc).__name__}: {exc}")
        if len(samples) >= constant.records_num:
            break

    if not samples:
        return "no_sample", len(records)

    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_path.open("w", encoding="utf-8") as f:
        for index, sample in enumerate(samples, start=1):
            f.write(f"--- [Random Sample Record {index} from {json_path.name}] ---\n")
            f.write(sample + "\n\n")
    return "success", len(records)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="선택한 회의록 JSON만 1단계 샘플 분리 후 2단계 규칙 추출합니다."
    )
    parser.add_argument(
        "targets", nargs="*",
        help="JSON 파일명(확장자 제외 가능) 또는 JSON 경로. 생략 시 기본 두 대상을 사용합니다.",
    )
    parser.add_argument(
        "--input-dir", default=str(DEFAULT_INPUT_DIR),
        help="원본 파일의 기준 폴더(예: C:\\project\\mini\\data\\org 또는 data\\org2)",
    )
    args = parser.parse_args(argv)
    targets = args.targets or DEFAULT_TARGETS

    splitter = load_script_module("1_split_sample.py", "selected_split_sample")
    rule_maker = load_script_module("2_rules_make.py", "selected_rules_make")

    run_id, step_started_at = start_step("1")
    selected_samples = []
    missing = []

    # Keep step 2 isolated to this invocation, while retaining the source sample
    # path referenced by its debug report for later review.
    selected_input_dir = DATA_DIR / "sample" / "_selected_runs" / run_id
    selected_input_dir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        json_path = resolve_input(target, args.input_dir)
        if json_path is None:
            print(f"[입력 없음] {target}: 해당 JSON 파일을 찾지 못했습니다.")
            missing.append(target)
            continue

        stem = json_path.stem
        year = stem.split("_", 1)[0]
        standard_sample = DATA_DIR / "sample" / f"{year}_광역 회의록" / f"sample_{stem}.txt"
        isolated_sample = selected_input_dir / standard_sample.name
        file_started = time.perf_counter()
        file_started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            status, record_count = make_sample(json_path, standard_sample, splitter)
            if status == "success":
                isolated_sample.write_text(standard_sample.read_text(encoding="utf-8"), encoding="utf-8")
                selected_samples.append(isolated_sample)
                print(f"[1단계 완료] {json_path.name} -> {standard_sample}")
            else:
                print(f"[1단계 확인 필요] {json_path.name}: 본문 샘플을 찾지 못했습니다.")
            record_file(run_id, "1", json_path, time.perf_counter() - file_started,
                        status, record_count, file_started_at)
        except Exception as exc:
            print(f"[1단계 오류] {json_path.name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            record_file(run_id, "1", json_path, time.perf_counter() - file_started,
                        "error", 0, file_started_at)

    finish_step(run_id, "1", step_started_at)

    if selected_samples:
        print(f"[2단계 시작] 선택된 샘플 {len(selected_samples)}개만 규칙을 추출합니다.")
        rule_maker.run(input_dir=str(selected_input_dir), output_dir=constant.rules_output_dir)
    else:
        print("[2단계 건너뜀] 규칙 추출에 사용할 샘플이 없습니다.")

    if missing:
        print(f"[완료/일부 누락] 찾지 못한 파일 {len(missing)}개: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

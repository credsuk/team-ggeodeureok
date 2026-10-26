#!/usr/bin/env python3
"""정규화 규칙 문자열이 정제 TXT에서 실제 구분자로 동작하는지 원문 노출 없이 측정한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_PATTERN = re.compile(r"^cleaned_(?P<year>20\d{2})_(?P<council>.+)\.txt$")
FIELDS = ("agenda", "speaker", "utterance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "input")
    parser.add_argument(
        "--rules",
        type=Path,
        default=PROJECT_ROOT / "data" / "canonical" / "rules" / "normalized_parsing_rules.json",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "structure_probe")
    return parser.parse_args()


def percentile(sorted_values: list[int], q: float) -> int | None:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, max(0, math.ceil(len(sorted_values) * q) - 1))
    return sorted_values[index]


def marker_descriptor(marker: str) -> dict:
    return {
        "value_length": len(marker),
        "value_sha256": hashlib.sha256(marker.encode("utf-8")).hexdigest(),
        "contains_newline": "\n" in marker or "\r" in marker,
        "contains_tab": "\t" in marker,
        "contains_colon": ":" in marker or "：" in marker,
        "contains_regex_metachar": bool(re.search(r"[\\\[\]\(\)\?\*\+\|\^\$]", marker)),
    }


def inspect_file(path: Path, key: str, rule: dict) -> dict:
    markers = {field: str(rule.get(field, "")) for field in FIELDS}
    invalid_fields = [field for field, marker in markers.items() if not marker]
    if invalid_fields:
        raise ValueError(f"빈 파싱 규칙이 있습니다: {key} / {invalid_fields}")

    line_lengths: list[int] = []
    line_count = 0
    empty_line_count = 0
    json_object_line_count = 0
    json_array_line_count = 0
    tab_line_count = 0
    pipe_line_count = 0
    unit_separator_line_count = 0
    replacement_char_count = 0
    marker_stats = {
        field: {
            **marker_descriptor(marker),
            "occurrence_count": 0,
            "containing_line_count": 0,
            "prefix_line_count": 0,
            "exact_line_count": 0,
        }
        for field, marker in markers.items()
    }
    matched_field_combinations = Counter()

    with path.open("r", encoding="utf-8-sig", errors="strict") as file:
        for line in file:
            line_count += 1
            stripped = line.strip()
            line_lengths.append(len(line.rstrip("\r\n")))
            if not stripped:
                empty_line_count += 1
            if stripped.startswith("{") and stripped.endswith("}"):
                json_object_line_count += 1
            if stripped.startswith("[") and stripped.endswith("]"):
                json_array_line_count += 1
            if "\t" in line:
                tab_line_count += 1
            if "|" in line:
                pipe_line_count += 1
            if "\x1f" in line or "\x1e" in line:
                unit_separator_line_count += 1
            replacement_char_count += line.count("\ufffd")

            matched_fields = []
            for field, marker in markers.items():
                occurrences = line.count(marker)
                if occurrences:
                    matched_fields.append(field)
                    marker_stats[field]["occurrence_count"] += occurrences
                    marker_stats[field]["containing_line_count"] += 1
                    marker_stats[field]["prefix_line_count"] += int(stripped.startswith(marker))
                    marker_stats[field]["exact_line_count"] += int(stripped == marker)
            if matched_fields:
                matched_field_combinations["+".join(matched_fields)] += 1

    line_lengths.sort()
    total_marker_occurrences = sum(value["occurrence_count"] for value in marker_stats.values())
    if all(marker_stats[field]["occurrence_count"] > 0 for field in FIELDS):
        inferred_format = "literal_rule_markers_detected"
    elif line_count and json_object_line_count / line_count >= 0.8:
        inferred_format = "jsonl_like"
    elif line_count and tab_line_count / line_count >= 0.8:
        inferred_format = "tabular_lines"
    elif total_marker_occurrences:
        inferred_format = "partial_rule_markers_detected"
    else:
        inferred_format = "rule_markers_not_literal"

    return {
        "key": key,
        "relative_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "line_count": line_count,
        "empty_line_count": empty_line_count,
        "line_length": {
            "min": line_lengths[0] if line_lengths else None,
            "p50": percentile(line_lengths, 0.50),
            "p95": percentile(line_lengths, 0.95),
            "p99": percentile(line_lengths, 0.99),
            "max": line_lengths[-1] if line_lengths else None,
            "mean": round(sum(line_lengths) / len(line_lengths), 2) if line_lengths else None,
        },
        "shape_counts": {
            "json_object_lines": json_object_line_count,
            "json_array_lines": json_array_line_count,
            "tab_lines": tab_line_count,
            "pipe_lines": pipe_line_count,
            "record_separator_lines": unit_separator_line_count,
            "replacement_chars": replacement_char_count,
        },
        "rule_marker_stats": marker_stats,
        "matched_field_combinations": dict(matched_field_combinations),
        "inferred_format": inferred_format,
        "raw_text_in_report": False,
        "rule_values_in_report": False,
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    rules_path = args.rules.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if not rules_path.exists():
        raise FileNotFoundError(rules_path)
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    if not isinstance(rules, dict):
        raise TypeError("normalized rules의 최상위는 dict여야 합니다.")
    output_dir.mkdir(parents=True, exist_ok=True)

    text_files: dict[str, Path] = {}
    for path in sorted(input_dir.rglob("cleaned_*.txt")):
        if ":Zone.Identifier" in path.name:
            continue
        match = TEXT_PATTERN.match(path.name)
        if match:
            key = f"{match.group('year')}_{match.group('council')}"
            text_files[key] = path

    missing_rules = sorted(set(text_files) - set(rules))
    missing_texts = sorted(set(rules) - set(text_files))
    if missing_rules or missing_texts:
        raise ValueError(f"TXT/규칙 불일치: missing_rules={missing_rules}, missing_texts={missing_texts}")

    file_reports = []
    for index, key in enumerate(sorted(text_files), start=1):
        print(f"[{index}/{len(text_files)}] {key}")
        report = inspect_file(text_files[key], key, rules[key])
        report["relative_path"] = str(text_files[key].relative_to(input_dir))
        file_reports.append(report)

    format_counts = Counter(report["inferred_format"] for report in file_reports)
    aggregate_marker_counts = {
        field: sum(report["rule_marker_stats"][field]["occurrence_count"] for report in file_reports)
        for field in FIELDS
    }
    summary = {
        "file_count": len(file_reports),
        "total_lines": sum(report["line_count"] for report in file_reports),
        "format_counts": dict(format_counts),
        "aggregate_rule_marker_occurrences": aggregate_marker_counts,
        "files_without_all_three_literal_markers": [
            report["key"] for report in file_reports
            if report["inferred_format"] != "literal_rule_markers_detected"
        ],
        "safe_for_literal_marker_parser": all(
            report["inferred_format"] == "literal_rule_markers_detected" for report in file_reports
        ),
        "files": file_reports,
        "raw_text_in_report": False,
        "rule_values_in_report": False,
    }
    output_path = output_dir / "structure_probe_report.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== TXT 구조 매칭 검사 완료 ===")
    print(f"파일: {len(file_reports)} | 전체 줄: {summary['total_lines']:,}")
    print(f"형식 판정: {dict(format_counts)}")
    print(f"규칙 출현 수: {aggregate_marker_counts}")
    print(f"문자열 규칙으로 안전하게 파싱 가능: {summary['safe_for_literal_marker_parser']}")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()


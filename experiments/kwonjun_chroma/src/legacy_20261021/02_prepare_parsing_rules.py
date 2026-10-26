#!/usr/bin/env python3
"""파싱 규칙 키를 정규화하고 TXT 대응 관계와 규칙 스키마를 검증한다."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_PATTERN = re.compile(r"^cleaned_(?P<year>20\d{2})_(?P<council>.+)\.txt$")
COPY_SUFFIX_PATTERN = re.compile(r"\s*\(\d+\)\s*$")
MARKER_COLUMNS = ["html_div", "html_p", "agenda_tag", "content_tag", "speaker_tag", "meeting_tag"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "input")
    parser.add_argument("--rules-dir", type=Path, default=PROJECT_ROOT / "data" / "input" / "llm_rules_outputs")
    parser.add_argument(
        "--inventory-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "inventory_text_v2" / "text_files_inventory.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "canonical" / "rules")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def normalize_rule_key(key: str) -> str:
    return COPY_SUFFIX_PATTERN.sub("", str(key).strip())


def flatten_schema(value: Any, path: str = "$", depth: int = 0) -> list[dict[str, Any]]:
    """규칙의 실제 문자열 값은 제외하고 경로·타입·길이만 기록한다."""
    if depth > 8:
        return [{"path": path, "type": "depth_limit"}]
    if isinstance(value, dict):
        rows = [{"path": path, "type": "dict", "length": len(value)}]
        for key in sorted(value, key=str):
            rows.extend(flatten_schema(value[key], f"{path}.{key}", depth + 1))
        return rows
    if isinstance(value, list):
        item_types = sorted({type(item).__name__ for item in value})
        rows = [{"path": path, "type": "list", "length": len(value), "item_types": item_types}]
        if value:
            rows.extend(flatten_schema(value[0], f"{path}[0]", depth + 1))
        return rows
    if isinstance(value, str):
        return [{
            "path": path,
            "type": "str",
            "length": len(value),
            "has_regex_metachar": bool(re.search(r"[\\\[\]\(\)\?\*\+\|\^\$]", value)),
        }]
    return [{"path": path, "type": type(value).__name__}]


def schema_signature(schema_rows: list[dict[str, Any]]) -> str:
    reduced = [{"path": row["path"], "type": row["type"]} for row in schema_rows]
    payload = json.dumps(reduced, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def load_inventory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path), "marker_totals": {}, "line_statistics": {}}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    marker_totals = {
        marker: sum(int(row.get(marker) or 0) for row in rows)
        for marker in MARKER_COLUMNS if marker in rows[0]
    }
    line_counts = [int(row.get("line_count") or 0) for row in rows]
    char_counts = [int(row.get("char_count") or 0) for row in rows]
    return {
        "available": True,
        "path": str(path),
        "row_count": len(rows),
        "marker_totals": marker_totals,
        "line_statistics": {
            "total_lines": sum(line_counts),
            "min_lines_per_file": min(line_counts) if line_counts else None,
            "max_lines_per_file": max(line_counts) if line_counts else None,
            "mean_chars_per_line": round(sum(char_counts) / max(sum(line_counts), 1), 2),
        },
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    rules_dir = args.rules_dir.expanduser().resolve()
    inventory_csv = args.inventory_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if not rules_dir.is_dir():
        raise NotADirectoryError(rules_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_text_keys: dict[str, str] = {}
    for path in sorted(input_dir.rglob("cleaned_*.txt")):
        if ":Zone.Identifier" in path.name:
            continue
        match = TEXT_PATTERN.match(path.name)
        if not match:
            continue
        key = f"{match.group('year')}_{match.group('council')}"
        if key in expected_text_keys:
            raise ValueError(f"동일 연도·의회의 TXT가 중복됩니다: {key}")
        expected_text_keys[key] = str(path.relative_to(input_dir))

    merged_rules: dict[str, Any] = {}
    source_by_key: dict[str, str] = {}
    renamed_keys: list[dict[str, str]] = []
    source_files = []
    for path in sorted(rules_dir.glob("*.json")):
        if ":Zone.Identifier" in path.name:
            continue
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise TypeError(f"규칙 JSON의 최상위는 dict여야 합니다: {path.name}")
        source_files.append({"file_name": path.name, "sha256": file_sha256(path), "rule_count": len(data)})
        for original_key, rule in data.items():
            normalized_key = normalize_rule_key(original_key)
            if normalized_key != original_key:
                renamed_keys.append({"from": original_key, "to": normalized_key})
            if normalized_key in merged_rules and merged_rules[normalized_key] != rule:
                raise ValueError(f"정규화 후 서로 다른 규칙이 같은 키를 사용합니다: {normalized_key}")
            merged_rules[normalized_key] = rule
            source_by_key[normalized_key] = path.name

    expected_set = set(expected_text_keys)
    rule_set = set(merged_rules)
    missing_rules = sorted(expected_set - rule_set)
    orphan_rules = sorted(rule_set - expected_set)

    schema_groups: dict[str, list[str]] = defaultdict(list)
    schema_examples: dict[str, list[dict[str, Any]]] = {}
    rule_index: dict[str, dict[str, Any]] = {}
    for key, rule in sorted(merged_rules.items()):
        schema_rows = flatten_schema(rule)
        signature = schema_signature(schema_rows)
        schema_groups[signature].append(key)
        schema_examples.setdefault(signature, schema_rows)
        rule_index[key] = {
            "source_file": source_by_key[key],
            "schema_signature": signature,
            "rule_value_sha256": hashlib.sha256(
                json.dumps(rule, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }

    normalized_path = output_dir / "normalized_parsing_rules.json"
    normalized_path.write_text(json.dumps(merged_rules, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "text_file_count": len(expected_text_keys),
        "normalized_rule_count": len(merged_rules),
        "source_rule_files": source_files,
        "renamed_keys": renamed_keys,
        "missing_rules": missing_rules,
        "orphan_rules": orphan_rules,
        "fallback_required": missing_rules,
        "schema_variant_count": len(schema_groups),
        "schema_groups": [
            {
                "signature": signature,
                "rule_count": len(keys),
                "rule_keys": keys,
                "schema_only": schema_examples[signature],
            }
            for signature, keys in sorted(schema_groups.items())
        ],
        "rule_index": rule_index,
        "text_structure_inventory": load_inventory(inventory_csv),
        "normalized_rules_file": str(normalized_path),
        "raw_text_in_report": False,
        "rule_string_values_in_report": False,
    }
    report_path = output_dir / "parsing_rules_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 파싱 규칙 정규화 완료 ===")
    print(f"TXT 키: {len(expected_text_keys)} | 규칙: {len(merged_rules)}")
    print(f"키 이름 수정: {renamed_keys or '없음'}")
    print(f"누락 규칙: {missing_rules or '없음'}")
    print(f"TXT 없는 규칙: {orphan_rules or '없음'}")
    print(f"규칙 스키마 유형: {len(schema_groups)}개")
    print(f"normalized rules: {normalized_path}")
    print(f"safe report: {report_path}")


if __name__ == "__main__":
    main()


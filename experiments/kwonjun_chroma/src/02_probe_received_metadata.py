#!/usr/bin/env python3
"""수정 회의록 34개와 신규 metadata JSONL 33개의 대응 관계와 스키마를 검사한다."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_PATTERN = re.compile(r"^cleaned_(20\d{2})_(.+?)\.(?:txt|jsonl|ndjson)$")
TEXT_FIELDS = {"utterance", "text", "content", "body", "raw_text", "embedding_text"}
ROLE_WORDS = {
    "의원", "의장", "부의장", "위원장", "부위원장", "시장", "도지사", "교육감",
    "부시장", "부지사", "국장", "실장", "과장", "소장", "원장", "본부장",
    "전문위원", "사무처장", "의사담당관", "관계공무원", "집행부",
}
NAME_PATTERN = re.compile(r"^[가-힣]{2,4}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "input")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "metadata_probe",
    )
    parser.add_argument("--sample-rows-per-file", type=int, default=200)
    return parser.parse_args()


def key_from_path(path: Path) -> str | None:
    match = FILE_PATTERN.match(path.name)
    return f"{match.group(1)}_{match.group(2)}" if match else None


def safe_preview(value: Any, field: str) -> Any:
    if field.lower() in TEXT_FIELDS:
        return f"<본문 {len(str(value)):,}자>" if value is not None else None
    text = "" if value is None else str(value)
    return text[:80] + ("…" if len(text) > 80 else "")


def looks_like_role(value: Any) -> bool:
    text = "" if value is None else str(value).strip()
    return text in ROLE_WORDS or any(text.endswith(role) for role in ROLE_WORDS if len(role) >= 2)


def looks_like_name(value: Any) -> bool:
    text = "" if value is None else str(value).strip()
    return bool(NAME_PATTERN.fullmatch(text)) and not looks_like_role(text)


def inspect_metadata(path: Path, limit: int) -> dict[str, Any]:
    row_count = 0
    invalid_rows = 0
    object_rows = 0
    key_counts: Counter[str] = Counter()
    null_counts: Counter[str] = Counter()
    empty_counts: Counter[str] = Counter()
    speaker_swap_suspects = 0
    type_suspects = 0
    first_record: dict[str, Any] | None = None

    with path.open("r", encoding="utf-8-sig", errors="replace") as file:
        for line in file:
            if not line.strip():
                continue
            row_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows += 1
                if row_count >= limit:
                    break
                continue
            if not isinstance(record, dict):
                if row_count >= limit:
                    break
                continue
            object_rows += 1
            if first_record is None:
                first_record = record
            key_counts.update(record.keys())
            for key, value in record.items():
                if value is None:
                    null_counts[key] += 1
                elif isinstance(value, str) and not value.strip():
                    empty_counts[key] += 1

            name = record.get("speaker_name")
            role = record.get("speaker_role")
            speaker_type = record.get("speaker_type")
            if looks_like_role(name) and looks_like_name(role):
                speaker_swap_suspects += 1
            if looks_like_name(speaker_type):
                type_suspects += 1
            if row_count >= limit:
                break

    return {
        "file": path.name,
        "sampled_rows": row_count,
        "object_rows": object_rows,
        "invalid_rows": invalid_rows,
        "keys": sorted(key_counts),
        "key_presence_in_sample": dict(key_counts),
        "null_counts_in_sample": dict(null_counts),
        "empty_counts_in_sample": dict(empty_counts),
        "speaker_name_role_swap_suspects": speaker_swap_suspects,
        "speaker_type_name_suspects": type_suspects,
        "safe_first_record": {
            key: safe_preview(value, key) for key, value in (first_record or {}).items()
        },
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    text_paths = sorted(
        path for path in input_dir.rglob("*.txt")
        if "2024_광역 회의록" in str(path.parent)
        or "2025_광역 회의록" in str(path.parent)
    )
    metadata_paths = sorted(
        path for path in (input_dir / "metadata_jsonl_python").rglob("*.jsonl")
        if ":Zone.Identifier" not in path.name
    )
    text_map = {key_from_path(path): path for path in text_paths if key_from_path(path)}
    metadata_map = {key_from_path(path): path for path in metadata_paths if key_from_path(path)}
    missing_metadata = sorted(set(text_map) - set(metadata_map))
    extra_metadata = sorted(set(metadata_map) - set(text_map))

    reports = []
    schemas: defaultdict[tuple[str, ...], list[str]] = defaultdict(list)
    field_file_counts: Counter[str] = Counter()
    for index, path in enumerate(metadata_paths, start=1):
        print(f"[{index}/{len(metadata_paths)}] {path.name}")
        report = inspect_metadata(path, args.sample_rows_per_file)
        reports.append(report)
        schema = tuple(report["keys"])
        schemas[schema].append(path.name)
        field_file_counts.update(set(report["keys"]))

    all_fields = sorted(field_file_counts)
    direct_id_fields = [
        field for field in ("chunk_id", "document_id", "source_record_no", "row_id")
        if field in all_fields
    ]
    has_utterance = any(field in all_fields for field in ("utterance", "text", "content", "body"))
    if direct_id_fields:
        recommendation = "고정 ID를 이용해 canonical과 직접 결합"
    elif has_utterance:
        recommendation = (
            "JSONL 발화 행을 새 canonical의 출발점으로 사용하고, 수정 TXT는 원문 대조·누락 검증용으로 사용"
        )
    else:
        recommendation = (
            "직접 결합 불가: 날짜·의회명만으로는 중복이 많으므로 문서/텍스트 위치 매핑 규칙이 필요"
        )

    summary = {
        "meeting_file_count": len(text_map),
        "metadata_file_count": len(metadata_map),
        "missing_metadata_for_meeting": missing_metadata,
        "metadata_without_meeting": extra_metadata,
        "schema_variant_count": len(schemas),
        "schema_variants": [
            {"fields": list(schema), "file_count": len(files), "files": files}
            for schema, files in schemas.items()
        ],
        "field_file_frequency": dict(field_file_counts),
        "all_fields": all_fields,
        "direct_id_fields": direct_id_fields,
        "has_utterance_field": has_utterance,
        "sample_quality": {
            "speaker_name_role_swap_suspects": sum(
                report["speaker_name_role_swap_suspects"] for report in reports
            ),
            "speaker_type_name_suspects": sum(
                report["speaker_type_name_suspects"] for report in reports
            ),
            "invalid_json_rows": sum(report["invalid_rows"] for report in reports),
            "sampled_rows": sum(report["sampled_rows"] for report in reports),
        },
        "recommendation": recommendation,
        "per_file": reports,
        "raw_utterance_not_in_report": True,
    }
    output_path = output_dir / "metadata_probe_report.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 신규 metadata 정렬 검사 ===")
    print(f"회의록: {len(text_map)} | metadata: {len(metadata_map)}")
    print(f"metadata 누락: {missing_metadata or '없음'}")
    print(f"회의록 없는 metadata: {extra_metadata or '없음'}")
    print(f"스키마 종류: {len(schemas)}")
    print(f"전체 필드: {all_fields}")
    print(f"고정 ID 필드: {direct_id_fields or '없음'}")
    print(
        "발화자 이름/역할 교환 의심: "
        f"{summary['sample_quality']['speaker_name_role_swap_suspects']:,}"
    )
    print(f"권장 결합 방식: {recommendation}")
    print(f"보고서: {output_path}")


if __name__ == "__main__":
    main()

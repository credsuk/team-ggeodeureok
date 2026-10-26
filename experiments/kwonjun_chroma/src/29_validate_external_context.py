#!/usr/bin/env python3
"""외부 공개지표 CSV를 참고 비교용 JSON으로 검증·변환한다.

회의록 원문은 외부로 전송하지 않는다. 외부지표는 공식 이슈 탐지·군집화·예측
입력에서 제외하며, 데이터 기준일(기본 2025-12-31)을 넘는 관측도 차단한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {
    "metric_id", "metric_name", "region", "period", "value", "unit",
    "source_name", "source_url", "data_cutoff", "usage_scope",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs" / "external" / "external_context.json",
    )
    parser.add_argument("--analysis-cutoff", default="2025-12-31")
    return parser.parse_args()


def parse_date(value: str, field: str) -> date:
    text = str(value or "").strip()
    for pattern in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            parsed = datetime.strptime(text, pattern).date()
            if pattern == "%Y-%m":
                return parsed.replace(day=1)
            if pattern == "%Y":
                return parsed.replace(month=1, day=1)
            return parsed
        except ValueError:
            continue
    raise ValueError(f"{field} 날짜 형식 오류: {value!r}")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    cutoff = parse_date(args.analysis_cutoff, "analysis_cutoff")

    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"필수 열 누락: {missing}")
        source_rows = list(reader)
    if not source_rows:
        raise ValueError("외부지표 CSV에 데이터 행이 없습니다")

    rows = []
    rejected = []
    for line_no, row in enumerate(source_rows, start=2):
        try:
            period = parse_date(row["period"], "period")
            data_cutoff = parse_date(row["data_cutoff"], "data_cutoff")
            if period > cutoff:
                raise ValueError("관측 시점이 분석 기준일 이후입니다")
            if data_cutoff > cutoff:
                raise ValueError("자료 기준일이 분석 기준일 이후입니다")
            if row["usage_scope"].strip() != "reference_only":
                raise ValueError("usage_scope는 reference_only여야 합니다")
            value = float(row["value"])
            rows.append({
                "metric_id": row["metric_id"].strip(),
                "metric_name": row["metric_name"].strip(),
                "region": row["region"].strip() or "전국",
                "period": row["period"].strip(),
                "value": value,
                "unit": row["unit"].strip(),
                "source_name": row["source_name"].strip(),
                "source_url": row["source_url"].strip(),
                "data_cutoff": row["data_cutoff"].strip(),
                "usage_scope": "reference_only",
            })
        except (TypeError, ValueError) as error:
            rejected.append({"line": line_no, "reason": str(error)})

    if not rows:
        raise RuntimeError(f"검증을 통과한 행이 없습니다: {rejected[:5]}")
    rows.sort(key=lambda row: (
        row["metric_id"], row["region"], row["period"]
    ))
    metrics = sorted({row["metric_id"] for row in rows})
    sources = sorted({row["source_name"] for row in rows})
    report = {
        "schema_version": "council_external_context_v1",
        "status": "complete",
        "analysis_cutoff": args.analysis_cutoff,
        "usage_policy": {
            "scope": "reference_only",
            "included_in_issue_detection": False,
            "included_in_forecast": False,
            "meeting_text_sent_externally": False,
        },
        "summary": {
            "accepted_rows": len(rows),
            "rejected_rows": len(rejected),
            "metric_count": len(metrics),
            "sources": sources,
        },
        "rejected": rejected,
        "rows": rows,
    }
    output = args.output.expanduser().resolve()
    atomic_json(output, report)
    print("\n=== 외부 공개지표 검증 완료 ===")
    print(f"통과: {len(rows):,} | 제외: {len(rejected):,} | 지표: {len(metrics):,}")
    print("용도: 화면 참고 비교 전용 (이슈 탐지·예측 입력 제외)")
    print(f"JSON: {output}")


if __name__ == "__main__":
    main()

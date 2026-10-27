#!/usr/bin/env python3
"""전체 청크에서 지역·연도·우선순위 균형 임베딩 평가 표본을 생성한다."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=PROJECT_ROOT / "data" / "canonical_v3" / "chunks_v1" / "chunks.parquet",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "data" / "embedding_eval" / "chunks_eval_60k.parquet",
    )
    parser.add_argument(
        "--report-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "restart_v3" / "embedding_eval_sample",
    )
    parser.add_argument("--target-rows", type=int, default=60_000)
    parser.add_argument("--high-ratio", type=float, default=0.70)
    parser.add_argument("--medium-ratio", type=float, default=0.20)
    parser.add_argument("--low-ratio", type=float, default=0.10)
    parser.add_argument("--seed", default="council-v3-eval-20261022")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if args.target_rows <= 0:
        raise ValueError("target-rows는 1 이상이어야 합니다")
    ratio_sum = args.high_ratio + args.medium_ratio + args.low_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"high/medium/low 비율의 합은 1이어야 합니다: {ratio_sum}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError("표본 파일이 이미 있습니다. 교체하려면 --overwrite를 사용하세요")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    source = str(input_path)
    total_rows = int(con.execute("SELECT count(*) FROM read_parquet(?)", [source]).fetchone()[0])
    source_keys = int(con.execute(
        "SELECT count(DISTINCT source_key) FROM read_parquet(?)", [source]
    ).fetchone()[0])
    if total_rows == 0 or source_keys == 0:
        raise RuntimeError("입력 청크가 비어 있습니다")

    quotas = {
        "high": max(1, round(args.target_rows * args.high_ratio / source_keys)),
        "medium": max(1, round(args.target_rows * args.medium_ratio / source_keys)),
        "low": max(1, round(args.target_rows * args.low_ratio / source_keys)),
    }
    con.execute("""
        CREATE TEMP TABLE ranked AS
        SELECT *, row_number() OVER (
          PARTITION BY source_key, priority
          ORDER BY hash(chunk_id || ?)
        ) AS stratum_rank
        FROM read_parquet(?)
    """, [args.seed, source])
    con.execute("""
        CREATE TEMP TABLE sample AS
        SELECT * EXCLUDE (stratum_rank)
        FROM ranked
        WHERE (priority = 'high' AND stratum_rank <= ?)
           OR (priority = 'medium' AND stratum_rank <= ?)
           OR (priority = 'low' AND stratum_rank <= ?)
    """, [quotas["high"], quotas["medium"], quotas["low"]])

    selected = int(con.execute("SELECT count(*) FROM sample").fetchone()[0])
    needed = min(max(0, args.target_rows - selected), total_rows - selected)
    if needed:
        con.execute("""
            INSERT INTO sample
            SELECT r.* EXCLUDE (stratum_rank)
            FROM ranked r
            ANTI JOIN sample s USING (chunk_id)
            ORDER BY hash(r.chunk_id || ?)
            LIMIT ?
        """, [args.seed + "-topup", needed])

    if output_path.exists():
        output_path.unlink()
    escaped_output = str(output_path).replace("'", "''")
    con.execute(f"COPY (SELECT * FROM sample ORDER BY source_key, priority, chunk_id) TO '{escaped_output}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    sample_rows = int(con.execute("SELECT count(*) FROM sample").fetchone()[0])
    unique_ids = int(con.execute("SELECT count(DISTINCT chunk_id) FROM sample").fetchone()[0])
    strata = con.execute("""
        SELECT source_key, priority, count(*) AS rows
        FROM sample GROUP BY 1, 2 ORDER BY 1, 2
    """).fetchall()
    priority_counts = {
        str(priority): int(count) for priority, count in con.execute("""
            SELECT priority, count(*) FROM sample GROUP BY 1 ORDER BY 1
        """).fetchall()
    }
    year_counts = {
        str(year): int(count) for year, count in con.execute("""
            SELECT year, count(*) FROM sample GROUP BY 1 ORDER BY 1
        """).fetchall()
    }
    council_counts = {
        str(council): int(count) for council, count in con.execute("""
            SELECT council, count(*) FROM sample GROUP BY 1 ORDER BY 1
        """).fetchall()
    }
    checks = {
        "row_count_within_target": sample_rows <= args.target_rows,
        "chunk_ids_unique": sample_rows == unique_ids,
        "all_source_keys_present": int(con.execute(
            "SELECT count(DISTINCT source_key) FROM sample"
        ).fetchone()[0]) == source_keys,
        "embedding_fields_complete": int(con.execute("""
            SELECT count(*) FROM sample
            WHERE embedding_text_raw IS NULL OR embedding_text_raw = ''
               OR embedding_text_enriched IS NULL OR embedding_text_enriched = ''
        """).fetchone()[0]) == 0,
    }
    passed = all(checks.values())
    report = {
        "passed": passed,
        "input": str(input_path),
        "output": str(output_path),
        "input_rows": total_rows,
        "target_rows": args.target_rows,
        "sample_rows": sample_rows,
        "source_key_count": source_keys,
        "seed": args.seed,
        "per_source_key_quotas": quotas,
        "priority_counts": priority_counts,
        "year_counts": year_counts,
        "council_counts": council_counts,
        "checks": checks,
    }
    report_path = report_dir / "embedding_eval_sample_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    strata_path = report_dir / "embedding_eval_sample_strata.csv"
    with strata_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["source_key", "priority", "rows"])
        writer.writerows(strata)

    print("\n=== 임베딩 평가 표본 생성 완료 ===")
    print(f"전체 청크: {total_rows:,}")
    print(f"표본: {sample_rows:,} / 목표 {args.target_rows:,}")
    print(f"source_key: {source_keys:,}")
    print(f"우선순위: {priority_counts}")
    print(f"연도: {year_counts}")
    for name, value in checks.items():
        print(f"{'PASS' if value else 'FAIL':4s} | {name}")
    print(f"최종 결과: {'PASS' if passed else 'FAIL'}")
    print(f"표본: {output_path}")
    print(f"요약: {report_path}")
    print(f"층별 분포: {strata_path}")
    if not passed:
        raise RuntimeError("임베딩 평가 표본 검증에 실패했습니다")


if __name__ == "__main__":
    main()

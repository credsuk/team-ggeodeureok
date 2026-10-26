#!/usr/bin/env python3
"""canonical 메타데이터와 부모·자식 청킹 시험 결과의 분포를 진단한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical", type=Path,
        default=PROJECT_ROOT / "data" / "canonical_v3" / "utterances.parquet",
    )
    parser.add_argument(
        "--chunks", type=Path,
        default=PROJECT_ROOT / "data" / "canonical_v3" / "chunk_test_2024_gangwon" / "chunks.parquet",
    )
    parser.add_argument("--key", default="2024_강원특별자치도의회")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs" / "restart_v3" / "chunk_test_2024_gangwon" / "chunk_audit.json",
    )
    return parser.parse_args()


def rows_to_dict(rows: list[tuple[Any, Any]]) -> dict[str, int]:
    return {str(key): int(value) for key, value in rows}


def main() -> None:
    args = parse_args()
    canonical = args.canonical.expanduser().resolve()
    chunks = args.chunks.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not canonical.is_file():
        raise FileNotFoundError(canonical)
    if not chunks.is_file():
        raise FileNotFoundError(chunks)

    con = duckdb.connect()
    canonical_path = str(canonical)
    chunks_path = str(chunks)

    total = con.execute(
        "SELECT count(*) FROM read_parquet(?) WHERE source_key = ?",
        [canonical_path, args.key],
    ).fetchone()[0]

    substantive = rows_to_dict(con.execute("""
        SELECT coalesce(CAST(is_substantive AS VARCHAR), 'NULL') AS value, count(*)
        FROM read_parquet(?)
        WHERE source_key = ?
        GROUP BY 1 ORDER BY 2 DESC
    """, [canonical_path, args.key]).fetchall())

    field_presence_row = con.execute("""
        SELECT
          count_if(nullif(trim(coalesce(policy_category, '')), '') IS NOT NULL),
          count_if(nullif(trim(coalesce(core_issue, '')), '') IS NOT NULL),
          count_if(nullif(trim(coalesce(keywords_json, '')), '') IS NOT NULL),
          count_if(nullif(trim(coalesce(summary, '')), '') IS NOT NULL),
          count_if(nullif(trim(coalesce(llm_agenda, '')), '') IS NOT NULL),
          count_if(nullif(trim(coalesce(llm_speaker, '')), '') IS NOT NULL)
        FROM read_parquet(?)
        WHERE source_key = ?
    """, [canonical_path, args.key]).fetchone()
    field_names = [
        "policy_category", "core_issue", "keywords_json",
        "summary", "llm_agenda", "llm_speaker",
    ]
    field_presence = dict(zip(field_names, map(int, field_presence_row)))

    top_categories = con.execute("""
        SELECT coalesce(nullif(trim(policy_category), ''), '<EMPTY>') AS value, count(*) AS n
        FROM read_parquet(?)
        WHERE source_key = ?
        GROUP BY 1 ORDER BY n DESC LIMIT 20
    """, [canonical_path, args.key]).fetchall()
    top_issues = con.execute("""
        SELECT coalesce(nullif(trim(core_issue), ''), '<EMPTY>') AS value, count(*) AS n
        FROM read_parquet(?)
        WHERE source_key = ?
        GROUP BY 1 ORDER BY n DESC LIMIT 20
    """, [canonical_path, args.key]).fetchall()

    chunk_total = con.execute("SELECT count(*) FROM read_parquet(?)", [chunks_path]).fetchone()[0]
    candidate = rows_to_dict(con.execute("""
        SELECT CAST(embedding_candidate AS VARCHAR), count(*)
        FROM read_parquet(?) GROUP BY 1 ORDER BY 2 DESC
    """, [chunks_path]).fetchall())
    priority = rows_to_dict(con.execute("""
        SELECT priority, count(*) FROM read_parquet(?) GROUP BY 1 ORDER BY 2 DESC
    """, [chunks_path]).fetchall())
    quality_row = con.execute("""
        SELECT
          count_if(contains(quality_flags_json, '"short_under_80"')),
          count_if(contains(quality_flags_json, '"short_under_180"')),
          count_if(contains(quality_flags_json, '"single_utterance"')),
          count_if(contains(quality_flags_json, '"mixed_substantive_context"')),
          count_if(contains(quality_flags_json, '"non_substantive"'))
        FROM read_parquet(?)
    """, [chunks_path]).fetchone()
    quality_names = [
        "short_under_80", "short_under_180", "single_utterance",
        "mixed_substantive_context", "non_substantive",
    ]
    quality_flags = dict(zip(quality_names, map(int, quality_row)))
    length_buckets = rows_to_dict(con.execute("""
        SELECT CASE
          WHEN char_length < 80 THEN '000-079'
          WHEN char_length < 180 THEN '080-179'
          WHEN char_length < 400 THEN '180-399'
          WHEN char_length < 700 THEN '400-699'
          WHEN char_length < 900 THEN '700-899'
          ELSE '900+'
        END AS bucket, count(*)
        FROM read_parquet(?) GROUP BY 1 ORDER BY 1
    """, [chunks_path]).fetchall())
    short_stats = con.execute("""
        SELECT
          count_if(char_length < 80),
          count_if(char_length < 180),
          count_if(utterance_count = 1),
          min(char_length), round(avg(char_length), 2), max(char_length)
        FROM read_parquet(?)
    """, [chunks_path]).fetchone()

    report = {
        "source_key": args.key,
        "canonical_rows": int(total),
        "is_substantive": substantive,
        "field_presence": field_presence,
        "field_presence_ratio": {
            key: round(value / total, 6) if total else 0.0
            for key, value in field_presence.items()
        },
        "top_policy_categories": [
            {"value": str(value), "count": int(count)} for value, count in top_categories
        ],
        "top_core_issues": [
            {"value": str(value), "count": int(count)} for value, count in top_issues
        ],
        "chunks": {
            "total": int(chunk_total),
            "embedding_candidate": candidate,
            "priority": priority,
            "quality_flags": quality_flags,
            "length_buckets": length_buckets,
            "under_80": int(short_stats[0]),
            "under_180": int(short_stats[1]),
            "single_utterance": int(short_stats[2]),
            "length_min": int(short_stats[3]),
            "length_mean": float(short_stats[4]),
            "length_max": int(short_stats[5]),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 청킹 및 메타데이터 진단 ===")
    print(f"대상: {args.key} | canonical: {total:,}행 | chunks: {chunk_total:,}행")
    print(f"is_substantive: {substantive}")
    print("메타데이터 채움률:")
    for field, count in field_presence.items():
        print(f"  {field:18s} {count:>8,} / {total:,} ({count / total:.2%})")
    print(f"후보: {candidate} | 우선순위: {priority}")
    print(f"품질 플래그: {quality_flags}")
    print(f"길이 구간: {length_buckets}")
    print(f"80자 미만: {short_stats[0]:,} | 180자 미만: {short_stats[1]:,} | 단일 발화: {short_stats[2]:,}")
    print("정책분야 상위값:")
    for value, count in top_categories[:10]:
        print(f"  {count:>8,} | {value}")
    print(f"보고서: {output}")


if __name__ == "__main__":
    main()

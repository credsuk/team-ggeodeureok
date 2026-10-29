#!/usr/bin/env python3
"""부모·자식 청크의 ID, 참조, 원문, 분류 규칙을 검증한다."""

from __future__ import annotations

import argparse
import csv
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
        "--parents", type=Path,
        default=PROJECT_ROOT / "data" / "canonical_v3" / "chunk_test_2024_gangwon_v2" / "parents.parquet",
    )
    parser.add_argument(
        "--chunks", type=Path,
        default=PROJECT_ROOT / "data" / "canonical_v3" / "chunk_test_2024_gangwon_v2" / "chunks.parquet",
    )
    parser.add_argument("--key", help="단일 시험 source_key. 생략하면 전체 데이터를 검증한다")
    parser.add_argument("--parent-max-chars", type=int, default=3200)
    parser.add_argument("--child-max-chars", type=int, default=950)
    parser.add_argument(
        "--report-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "restart_v3" / "chunk_test_2024_gangwon_v2",
    )
    return parser.parse_args()


def scalar(con: duckdb.DuckDBPyConnection, query: str, params: list[Any]) -> int:
    return int(con.execute(query, params).fetchone()[0])


def main() -> None:
    args = parse_args()
    canonical = args.canonical.expanduser().resolve()
    parents = args.parents.expanduser().resolve()
    chunks = args.chunks.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    for path in (canonical, parents, chunks):
        if not path.is_file():
            raise FileNotFoundError(path)

    con = duckdb.connect()
    cp, pp, chp = str(canonical), str(parents), str(chunks)
    parent_rows = scalar(con, "SELECT count(*) FROM read_parquet(?)", [pp])
    chunk_rows = scalar(con, "SELECT count(*) FROM read_parquet(?)", [chp])

    failures: dict[str, int] = {}
    failures["duplicate_parent_ids"] = scalar(con, """
        SELECT count(*) FROM (
          SELECT parent_chunk_id FROM read_parquet(?) GROUP BY 1 HAVING count(*) > 1
        )
    """, [pp])
    failures["duplicate_chunk_ids"] = scalar(con, """
        SELECT count(*) FROM (
          SELECT chunk_id FROM read_parquet(?) GROUP BY 1 HAVING count(*) > 1
        )
    """, [chp])
    failures["invalid_parent_required_fields"] = scalar(con, """
        SELECT count(*) FROM read_parquet(?)
        WHERE parent_chunk_id IS NULL OR document_id IS NULL OR source_key IS NULL
           OR council IS NULL OR meeting_date IS NULL OR text IS NULL OR text = ''
    """, [pp])
    failures["invalid_chunk_required_fields"] = scalar(con, """
        SELECT count(*) FROM read_parquet(?)
        WHERE chunk_id IS NULL OR parent_chunk_id IS NULL OR document_id IS NULL
           OR source_key IS NULL OR council IS NULL OR meeting_date IS NULL
           OR text IS NULL OR text = '' OR embedding_text_raw IS NULL
           OR embedding_text_enriched IS NULL
    """, [chp])
    failures["parent_length_mismatch"] = scalar(con, """
        SELECT count(*) FROM read_parquet(?) WHERE char_length <> length(text)
    """, [pp])
    failures["chunk_length_mismatch"] = scalar(con, """
        SELECT count(*) FROM read_parquet(?) WHERE char_length <> length(text)
    """, [chp])
    failures["parent_hash_mismatch"] = scalar(con, """
        SELECT count(*) FROM read_parquet(?) WHERE text_sha256 <> sha256(text)
    """, [pp])
    failures["chunk_hash_mismatch"] = scalar(con, """
        SELECT count(*) FROM read_parquet(?) WHERE text_sha256 <> sha256(text)
    """, [chp])
    failures["parent_over_max"] = scalar(con, """
        SELECT count(*) FROM read_parquet(?) WHERE char_length > ?
    """, [pp, args.parent_max_chars])
    failures["chunk_over_max"] = scalar(con, """
        SELECT count(*) FROM read_parquet(?) WHERE char_length > ?
    """, [chp, args.child_max_chars])
    failures["invalid_source_ranges"] = scalar(con, """
        SELECT count(*) FROM read_parquet(?)
        WHERE source_row_start < 1 OR source_row_end < source_row_start
    """, [chp])
    failures["orphan_parent_references"] = scalar(con, """
        SELECT count(*)
        FROM read_parquet(?) c
        LEFT JOIN read_parquet(?) p USING (parent_chunk_id)
        WHERE p.parent_chunk_id IS NULL
    """, [chp, pp])
    failures["parents_without_children"] = scalar(con, """
        SELECT count(*)
        FROM read_parquet(?) p
        LEFT JOIN read_parquet(?) c USING (parent_chunk_id)
        WHERE c.parent_chunk_id IS NULL
    """, [pp, chp])
    failures["parent_child_metadata_mismatch"] = scalar(con, """
        SELECT count(*)
        FROM read_parquet(?) c
        JOIN read_parquet(?) p USING (parent_chunk_id)
        WHERE c.document_id <> p.document_id
           OR c.source_key <> p.source_key
           OR c.council <> p.council
           OR c.meeting_date <> p.meeting_date
           OR coalesce(c.committee, '') <> coalesce(p.committee, '')
           OR coalesce(c.meeting_name, '') <> coalesce(p.meeting_name, '')
           OR coalesce(c.agenda_title, '') <> coalesce(p.agenda_title, '')
    """, [chp, pp])
    failures["missing_source_start_rows"] = scalar(con, """
        SELECT count(*)
        FROM read_parquet(?) c
        LEFT JOIN read_parquet(?) u
          ON c.source_key = u.source_key AND c.source_row_start = u.source_row_id
        WHERE u.utterance_id IS NULL
    """, [chp, cp])
    failures["missing_source_end_rows"] = scalar(con, """
        SELECT count(*)
        FROM read_parquet(?) c
        LEFT JOIN read_parquet(?) u
          ON c.source_key = u.source_key AND c.source_row_end = u.source_row_id
        WHERE u.utterance_id IS NULL
    """, [chp, cp])
    failures["candidate_rule_mismatch"] = scalar(con, """
        SELECT count(*) FROM read_parquet(?)
        WHERE embedding_candidate <> (substantive_true_count > 0)
    """, [chp])
    failures["priority_rule_mismatch"] = scalar(con, """
        SELECT count(*) FROM read_parquet(?)
        WHERE priority <> CASE
          WHEN substantive_true_count = 0 THEN 'low'
          WHEN substantive_ratio >= 0.5 AND char_length >= 80 THEN 'high'
          ELSE 'medium' END
    """, [chp])
    failures["invalid_json_fields"] = scalar(con, """
        SELECT count(*) FROM read_parquet(?)
        WHERE NOT json_valid(utterance_ids_json)
           OR NOT json_valid(speakers_json)
           OR NOT json_valid(policy_categories_json)
           OR NOT json_valid(core_issues_json)
           OR NOT json_valid(keywords_json)
           OR NOT json_valid(quality_flags_json)
    """, [chp])
    failures["embedding_text_missing_source"] = scalar(con, """
        SELECT count(*) FROM read_parquet(?)
        WHERE NOT contains(embedding_text_raw, text)
           OR NOT contains(embedding_text_enriched, text)
    """, [chp])
    if args.key:
        failures["unexpected_source_key"] = scalar(con, """
            SELECT count(*) FROM read_parquet(?) WHERE source_key <> ?
        """, [chp, args.key])
    else:
        failures["missing_source_keys"] = scalar(con, """
            SELECT count(*)
            FROM (SELECT DISTINCT source_key FROM read_parquet(?)) u
            LEFT JOIN (SELECT DISTINCT source_key FROM read_parquet(?)) c USING (source_key)
            WHERE c.source_key IS NULL
        """, [cp, chp])
        failures["extra_source_keys"] = scalar(con, """
            SELECT count(*)
            FROM (SELECT DISTINCT source_key FROM read_parquet(?)) c
            LEFT JOIN (SELECT DISTINCT source_key FROM read_parquet(?)) u USING (source_key)
            WHERE u.source_key IS NULL
        """, [chp, cp])

    passed = all(value == 0 for value in failures.values()) and parent_rows > 0 and chunk_rows > 0
    report = {
        "passed": passed,
        "source_key": args.key or "ALL",
        "parent_rows": parent_rows,
        "chunk_rows": chunk_rows,
        "limits": {
            "parent_max_chars": args.parent_max_chars,
            "child_max_chars": args.child_max_chars,
        },
        "failure_counts": failures,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "chunk_validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    sample_rows = con.execute("""
        WITH ranked AS (
          SELECT
            priority, chunk_id, council, meeting_date, committee, meeting_name,
            agenda_title, source_row_start, source_row_end, char_length,
            substantive_ratio, quality_flags_json, substr(text, 1, 500) AS text_preview,
            row_number() OVER (
              PARTITION BY priority ORDER BY hash(chunk_id)
            ) AS sample_rank
          FROM read_parquet(?)
        )
        SELECT * EXCLUDE (sample_rank) FROM ranked
        WHERE sample_rank <= 10
        ORDER BY priority, chunk_id
    """, [chp]).fetchall()
    columns = [item[0] for item in con.description]
    sample_path = report_dir / "chunk_validation_samples.csv"
    with sample_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(sample_rows)

    print("\n=== 부모·자식 청크 무결성 검증 ===")
    print(f"parents: {parent_rows:,} | chunks: {chunk_rows:,}")
    for name, count in failures.items():
        print(f"{'PASS' if count == 0 else 'FAIL':4s} | {name}: {count:,}")
    print(f"최종 결과: {'PASS' if passed else 'FAIL'}")
    print(f"샘플: {sample_path}")
    print(f"보고서: {report_path}")
    if not passed:
        raise RuntimeError("청크 무결성 검증에 실패했습니다. 보고서를 확인하세요.")


if __name__ == "__main__":
    main()

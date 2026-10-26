#!/usr/bin/env python3
"""원본 JSON 기반 canonical 문서·부모·검색 청크의 전체 무결성을 검증한다."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "canonical" / "raw_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "raw_v1_validation" / "validation_report.json",
    )
    parser.add_argument("--expected-strata", type=int, default=34)
    parser.add_argument("--expected-source-files", type=int, default=34)
    parser.add_argument("--max-child-chars", type=int, default=1020)
    parser.add_argument(
        "--max-replacement-document-ratio",
        type=float,
        default=0.001,
        help="원본 보존을 전제로 허용할 U+FFFD 포함 문서 비율. 기본 0.1%%",
    )
    parser.add_argument("--memory-limit", default="6GB")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def scalar(connection: duckdb.DuckDBPyConnection, query: str) -> int:
    return int(connection.execute(query).fetchone()[0])


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    paths = {
        "documents": input_dir / "documents.parquet",
        "parents": input_dir / "parents.parquet",
        "chunks": input_dir / "chunks.parquet",
        "build_report": input_dir / "build_report.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"필수 canonical 파일이 없습니다: {missing}")
    if args.expected_strata < 1 or args.expected_source_files < 1 or args.max_child_chars < 1:
        raise ValueError("expected-strata/source-files/max-child-chars는 1 이상이어야 합니다.")
    if not 0 <= args.max_replacement_document_ratio <= 1:
        raise ValueError("--max-replacement-document-ratio는 0~1 사이여야 합니다.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    connection = duckdb.connect()
    connection.execute(f"SET memory_limit={sql_string(args.memory_limit)}")
    connection.execute(f"SET threads={args.threads}")
    d = sql_string(str(paths["documents"]))
    p = sql_string(str(paths["parents"]))
    c = sql_string(str(paths["chunks"]))

    print("[1/5] 기본 행 수·ID·메타데이터 검사", flush=True)
    basic_row = connection.execute(
        f"""
        SELECT
            COUNT(*) AS total_chunks,
            COUNT(DISTINCT chunk_id) AS unique_chunk_ids,
            COUNT(*) FILTER (WHERE document_id IS NULL) AS document_id_null,
            COUNT(*) FILTER (WHERE parent_chunk_id IS NULL) AS parent_chunk_id_null,
            COUNT(*) FILTER (WHERE council IS NULL OR council = '') AS council_null,
            COUNT(*) FILTER (WHERE meeting_date IS NULL OR meeting_date = '') AS date_null,
            COUNT(*) FILTER (WHERE committee IS NULL OR committee = '') AS committee_null,
            COUNT(*) FILTER (WHERE meeting_name IS NULL OR meeting_name = '') AS meeting_name_null,
            COUNT(*) FILTER (WHERE text IS NULL OR text = '') AS text_null,
            COUNT(*) FILTER (WHERE embedding_text IS DISTINCT FROM text) AS embedding_text_mismatch,
            COUNT(*) FILTER (WHERE char_length != length(text)) AS char_length_mismatch,
            COUNT(*) FILTER (WHERE year != CAST(substr(meeting_date, 1, 4) AS SMALLINT)) AS year_mismatch,
            COUNT(*) FILTER (WHERE replacement_char_count != 0) AS replacement_char_rows,
            COUNT(*) FILTER (WHERE char_length > {args.max_child_chars}) AS oversized_chunks,
            MIN(char_length),
            ROUND(AVG(char_length), 2),
            quantile_cont(char_length, 0.50),
            quantile_cont(char_length, 0.95),
            quantile_cont(char_length, 0.99),
            MAX(char_length)
        FROM read_parquet({c})
        """
    ).fetchone()
    basic_names = [
        "total_chunks", "unique_chunk_ids", "document_id_null", "parent_chunk_id_null",
        "council_null", "date_null", "committee_null", "meeting_name_null", "text_null",
        "embedding_text_mismatch", "char_length_mismatch", "year_mismatch",
        "replacement_char_rows", "oversized_chunks", "min_chars", "avg_chars",
        "p50_chars", "p95_chars", "p99_chars", "max_chars",
    ]
    basic = dict(zip(basic_names, basic_row))

    document_count, unique_document_ids, source_file_count = connection.execute(
        f"""
        SELECT COUNT(*), COUNT(DISTINCT document_id), COUNT(DISTINCT source_file)
        FROM read_parquet({d})
        """
    ).fetchone()
    replacement_documents, replacement_characters = connection.execute(
        f"""
        SELECT
            COUNT(*) FILTER (WHERE contains(text, '�')),
            COALESCE(SUM(length(text) - length(replace(text, '�', ''))), 0)
        FROM read_parquet({d})
        """
    ).fetchone()
    replacement_document_ratio = (
        int(replacement_documents) / int(document_count) if int(document_count) else 0.0
    )
    parent_count, unique_parent_ids = connection.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT parent_chunk_id) FROM read_parquet({p})"
    ).fetchone()

    print("[2/5] 문서→부모→자식 참조 검사", flush=True)
    references = {
        "parents_without_document": scalar(
            connection,
            f"""
            SELECT COUNT(*) FROM read_parquet({p}) x
            LEFT JOIN read_parquet({d}) y USING (document_id)
            WHERE y.document_id IS NULL
            """,
        ),
        "chunks_without_document": scalar(
            connection,
            f"""
            SELECT COUNT(*) FROM read_parquet({c}) x
            LEFT JOIN read_parquet({d}) y USING (document_id)
            WHERE y.document_id IS NULL
            """,
        ),
        "chunks_without_parent": scalar(
            connection,
            f"""
            SELECT COUNT(*) FROM read_parquet({c}) x
            LEFT JOIN read_parquet({p}) y USING (parent_chunk_id)
            WHERE y.parent_chunk_id IS NULL
            """,
        ),
    }

    print("[3/5] 원문 위치·텍스트 복원 검사", flush=True)
    slices = {
        "invalid_parent_offsets": scalar(
            connection,
            f"""
            SELECT COUNT(*) FROM read_parquet({p}) x
            JOIN read_parquet({d}) y USING (document_id)
            WHERE x.char_start < 0 OR x.char_end <= x.char_start OR x.char_end > length(y.text)
            """,
        ),
        "parent_document_slice_mismatch": scalar(
            connection,
            f"""
            SELECT COUNT(*) FROM read_parquet({p}) x
            JOIN read_parquet({d}) y USING (document_id)
            WHERE x.text != substr(y.text, x.char_start + 1, x.char_end - x.char_start)
            """,
        ),
        "invalid_chunk_offsets": scalar(
            connection,
            f"""
            SELECT COUNT(*) FROM read_parquet({c}) x
            JOIN read_parquet({d}) y USING (document_id)
            WHERE x.char_start < 0 OR x.char_end <= x.char_start OR x.char_end > length(y.text)
            """,
        ),
        "chunk_document_slice_mismatch": scalar(
            connection,
            f"""
            SELECT COUNT(*) FROM read_parquet({c}) x
            JOIN read_parquet({d}) y USING (document_id)
            WHERE x.text != substr(y.text, x.char_start + 1, x.char_end - x.char_start)
            """,
        ),
        "chunk_outside_parent": scalar(
            connection,
            f"""
            SELECT COUNT(*) FROM read_parquet({c}) x
            JOIN read_parquet({p}) y USING (parent_chunk_id)
            WHERE x.document_id != y.document_id
               OR x.char_start < y.char_start
               OR x.char_end > y.char_end
            """,
        ),
    }

    print("[4/5] 지역×연도 분포·UI 잔여 검사", flush=True)
    strata_rows = connection.execute(
        f"""
        SELECT year, council, COUNT(*) AS chunks, COUNT(DISTINCT document_id) AS documents
        FROM read_parquet({c})
        GROUP BY year, council
        ORDER BY year, council
        """
    ).fetchall()
    strata = [
        {"year": int(year), "council": str(council), "chunks": int(chunks), "documents": int(documents)}
        for year, council, chunks, documents in strata_rows
    ]
    ui_residue_rows = scalar(
        connection,
        f"""
        SELECT COUNT(*) FROM read_parquet({c})
        WHERE regexp_matches(text, '(^|\\n)(접기|펼치기)(\\n|$)')
        """,
    )

    build_report = json.loads(paths["build_report"].read_text(encoding="utf-8"))
    source_failures = []
    for item in build_report.get("files", []):
        failures = {
            name: int(item.get(name, 0))
            for name in ("invalid_record_type", "invalid_date", "empty_text", "council_mismatch")
            if int(item.get(name, 0)) != 0
        }
        if failures:
            source_failures.append({"key": item.get("key"), **failures})

    checks = {
        "document_ids_unique": int(document_count) == int(unique_document_ids),
        "parent_ids_unique": int(parent_count) == int(unique_parent_ids),
        "chunk_ids_unique": int(basic["total_chunks"]) == int(basic["unique_chunk_ids"]),
        "required_chunk_fields_complete": all(
            int(basic[name]) == 0
            for name in (
                "document_id_null", "parent_chunk_id_null", "council_null", "date_null",
                "committee_null", "meeting_name_null", "text_null",
            )
        ),
        "embedding_text_exact": int(basic["embedding_text_mismatch"]) == 0,
        "lengths_exact": int(basic["char_length_mismatch"]) == 0,
        "years_consistent": int(basic["year_mismatch"]) == 0,
        "replacement_chars_within_tolerance": (
            replacement_document_ratio <= args.max_replacement_document_ratio
        ),
        "chunk_size_limit": int(basic["oversized_chunks"]) == 0,
        "references_complete": all(value == 0 for value in references.values()),
        "source_slices_exact": all(value == 0 for value in slices.values()),
        "expected_strata": len(strata) == args.expected_strata,
        "expected_source_files": int(source_file_count) == args.expected_source_files,
        "no_ui_residue": ui_residue_rows == 0,
        "source_records_valid": not source_failures,
    }
    passed = all(checks.values())
    connection.close()

    report = {
        "passed": passed,
        "input_dir": str(input_dir),
        "expected": {
            "strata": args.expected_strata,
            "source_files": args.expected_source_files,
            "max_child_chars": args.max_child_chars,
            "max_replacement_document_ratio": args.max_replacement_document_ratio,
        },
        "counts": {
            "documents": int(document_count),
            "parents": int(parent_count),
            "chunks": int(basic["total_chunks"]),
            "source_files": int(source_file_count),
            "strata": len(strata),
        },
        "basic": basic,
        "replacement_character_quality": {
            "affected_documents": int(replacement_documents),
            "replacement_characters": int(replacement_characters),
            "affected_document_ratio": round(replacement_document_ratio, 8),
            "policy": "원문 text에는 보존하고 임베딩 입력에서만 공백으로 치환",
        },
        "references": references,
        "slices": slices,
        "ui_residue_rows": ui_residue_rows,
        "source_failures": source_failures,
        "checks": checks,
        "strata": strata,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "raw_text_in_report": False,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[5/5] 검증 보고서 작성", flush=True)
    print("\n=== raw canonical 전체 검증 ===")
    print(f"문서: {int(document_count):,}")
    print(f"부모 청크: {int(parent_count):,}")
    print(f"검색 청크: {int(basic['total_chunks']):,}")
    print(f"지역×연도 층: {len(strata):,}")
    for name, value in checks.items():
        print(f"{'PASS' if value else 'FAIL':4s} | {name}")
    if int(replacement_documents):
        print(
            f"WARN | U+FFFD 포함 원본 문서 {int(replacement_documents):,}건 / "
            f"문자 {int(replacement_characters):,}개 "
            f"({replacement_document_ratio:.4%})"
        )
        print("       원문에는 보존하고 임베딩 입력에서만 공백으로 치환합니다.")
    print(f"\n최종 결과: {'PASS' if passed else 'FAIL'}")
    print(f"보고서: {report_path}")
    if not passed:
        raise RuntimeError("canonical 전체 검증에 실패했습니다. 보고서를 확인하세요.")


if __name__ == "__main__":
    main()

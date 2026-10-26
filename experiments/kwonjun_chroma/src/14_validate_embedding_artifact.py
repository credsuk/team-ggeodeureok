#!/usr/bin/env python3
"""전체 임베딩 아티팩트의 행·차원·유한값·정규화·metadata 정합성을 검사한다."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow.parquet as pq

from council_vector_runtime import EmbeddingArtifact, atomic_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = [
    "chunk_id", "parent_chunk_id", "document_id", "council", "meeting_date",
    "year", "committee", "meeting_name", "agenda_title", "priority", "text",
    "embedding_text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-dir", type=Path,
        default=PROJECT_ROOT / "data" / "embeddings" / "arctic_raw_v1",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs" / "post_embedding" / "validation.json",
    )
    parser.add_argument("--expected-dim", type=int, default=None)
    parser.add_argument("--block-rows", type=int, default=8192)
    parser.add_argument("--norm-min", type=float, default=0.97)
    parser.add_argument("--norm-max", type=float, default=1.03)
    parser.add_argument("--allow-incomplete-summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = EmbeddingArtifact(args.embedding_dir)
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    checks["vectors_are_2d"] = artifact.vectors.ndim == 2
    checks["row_counts_match"] = artifact.row_count == artifact.parquet.metadata.num_rows
    checks["dimension_matches"] = (
        args.expected_dim is None or artifact.dimension == args.expected_dim
    )
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in artifact.columns]
    checks["required_columns_present"] = not missing_columns
    details["missing_columns"] = missing_columns

    connection = duckdb.connect()
    metadata_sql = str(artifact.paths.metadata).replace("'", "''")
    stats = connection.execute(f"""
        SELECT
          count(*) AS rows,
          count(DISTINCT chunk_id) AS unique_chunk_ids,
          count(*) FILTER (
            WHERE chunk_id IS NULL OR trim(chunk_id) = ''
               OR council IS NULL OR trim(council) = ''
               OR meeting_date IS NULL
               OR committee IS NULL OR trim(committee) = ''
               OR meeting_name IS NULL OR trim(meeting_name) = ''
               OR text IS NULL OR trim(text) = ''
               OR embedding_text IS NULL OR trim(embedding_text) = ''
          ) AS incomplete_rows
        FROM read_parquet('{metadata_sql}')
    """).fetchone()
    connection.close()
    checks["chunk_ids_unique"] = stats[0] == stats[1]
    checks["required_values_complete"] = stats[2] == 0
    details.update({
        "metadata_rows": int(stats[0]),
        "unique_chunk_ids": int(stats[1]),
        "incomplete_required_rows": int(stats[2]),
    })

    finite_rows = 0
    invalid_rows = 0
    norm_outlier_rows = 0
    norm_sum = 0.0
    norm_min = float("inf")
    norm_max = float("-inf")
    for start in range(0, artifact.row_count, args.block_rows):
        end = min(start + args.block_rows, artifact.row_count)
        block = np.asarray(artifact.vectors[start:end], dtype=np.float32)
        valid = np.isfinite(block).all(axis=1)
        finite_rows += int(valid.sum())
        invalid_rows += int((~valid).sum())
        if valid.any():
            norms = np.linalg.norm(block[valid], axis=1)
            norm_sum += float(norms.sum())
            norm_min = min(norm_min, float(norms.min()))
            norm_max = max(norm_max, float(norms.max()))
            norm_outlier_rows += int(((norms < args.norm_min) | (norms > args.norm_max)).sum())

    checks["all_vectors_finite"] = invalid_rows == 0
    checks["vectors_normalized"] = norm_outlier_rows == 0
    details.update({
        "finite_rows": finite_rows,
        "invalid_rows": invalid_rows,
        "norm_outlier_rows": norm_outlier_rows,
        "norm": {
            "mean": round(norm_sum / finite_rows, 7) if finite_rows else None,
            "min": round(norm_min, 7) if finite_rows else None,
            "max": round(norm_max, 7) if finite_rows else None,
        },
        "vector_file_size_bytes": artifact.paths.vectors.stat().st_size,
    })

    summary = artifact.summary
    expected_digest = summary.get("selected_digest_sha256")
    chunk_hasher = hashlib.sha256()
    for batch in artifact.parquet.iter_batches(batch_size=16_384, columns=["chunk_id"]):
        for chunk_id in batch.column(0).to_pylist():
            chunk_hasher.update(str(chunk_id).encode("utf-8"))
            chunk_hasher.update(b"\n")
    actual_digest = chunk_hasher.hexdigest()
    checks["chunk_order_digest_matches"] = expected_digest is None or actual_digest == expected_digest
    details["expected_chunk_digest_sha256"] = expected_digest
    details["actual_chunk_digest_sha256"] = actual_digest

    summary_complete = summary.get("status") == "complete"
    summary_rows = summary.get("selected_rows", summary.get("completed_rows"))
    summary_dim = summary.get("dimension")
    checks["summary_complete"] = args.allow_incomplete_summary or summary_complete
    checks["summary_row_count_matches"] = summary_rows is None or int(summary_rows) == artifact.row_count
    checks["summary_dimension_matches"] = summary_dim is None or int(summary_dim) == artifact.dimension

    passed = all(checks.values())
    report = {
        "status": "PASS" if passed else "FAIL",
        "embedding_dir": str(artifact.paths.root),
        "vectors": str(artifact.paths.vectors),
        "metadata": str(artifact.paths.metadata),
        "row_count": artifact.row_count,
        "dimension": artifact.dimension,
        "dtype": str(artifact.vectors.dtype),
        "model_name": artifact.model_name,
        "revision": artifact.revision,
        "checks": checks,
        "details": details,
    }
    atomic_json(args.output.expanduser().resolve(), report)

    print("\n=== 전체 임베딩 검증 ===")
    print(f"행: {artifact.row_count:,} | 차원: {artifact.dimension} | dtype: {artifact.vectors.dtype}")
    for name, value in checks.items():
        print(f"{'PASS' if value else 'FAIL'} | {name}")
    print(f"NaN/Inf 행: {invalid_rows:,} | norm 범위: {details['norm']}")
    print(f"최종 결과: {report['status']}")
    print(f"보고서: {args.output.expanduser().resolve()}")
    if not passed:
        raise RuntimeError("임베딩 검증에 실패했습니다. 보고서를 확인하세요.")


if __name__ == "__main__":
    main()

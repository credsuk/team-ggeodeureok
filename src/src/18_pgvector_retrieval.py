#!/usr/bin/env python3
"""Dense와 BM25를 RRF로 결합해 정책 질의의 근거 후보를 검색한다.

기본 모드는 Arctic Dense + SQLite FTS5 BM25 + RRF다. 결합 후보를 JSON에
저장하고 다음 단계의 BGE 재순위 모델이 최종 근거를 선택한다.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from council_vector_runtime import (
    EmbeddingArtifact,
    QueryEncoder,
    atomic_json,
    json_safe,
    parent_contexts,
    rows_by_chunk_ids,
)
from bm25_runtime import BM25Index, reciprocal_rank_fusion
from search_aliases import committee_alias_values, council_alias_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="검색할 정책 또는 현안")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument(
        "--retrieval-mode", choices=["dense", "hybrid"],
        default=os.environ.get("COUNCIL_RETRIEVAL_MODE", "hybrid"),
    )
    parser.add_argument(
        "--bm25-index", type=Path,
        default=Path(os.environ.get(
            "COUNCIL_BM25_INDEX",
            PROJECT_ROOT / "data" / "bm25" / "arctic_raw_v1" / "bm25.sqlite3",
        )),
    )
    parser.add_argument("--dense-k", type=int, default=None)
    parser.add_argument("--bm25-k", type=int, default=None)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument(
        "--max-per-parent", type=int, default=1,
        help="최종 결과에 허용할 동일 부모 청크의 최대 개수",
    )
    parser.add_argument("--council")
    parser.add_argument("--year", type=int)
    parser.add_argument("--committee")
    parser.add_argument("--date-from")
    parser.add_argument("--date-to")
    parser.add_argument("--priorities", nargs="+")
    parser.add_argument("--policy-category")
    parser.add_argument(
        "--embedding-dir", type=Path,
        default=PROJECT_ROOT / "data" / "embeddings" / "arctic_raw_v1",
    )
    parser.add_argument(
        "--parent-path", type=Path,
        default=PROJECT_ROOT / "data" / "canonical_v3" / "chunks_v1" / "parents.parquet",
    )
    parser.add_argument("--dsn", default=os.environ.get("PGVECTOR_DSN"))
    parser.add_argument("--table", default="council_arctic_raw_v1")
    parser.add_argument("--ef-search", type=int, default=200)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--model-cache", type=Path,
        default=PROJECT_ROOT / "models" / "huggingface",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--no-parent", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs" / "retrieval" / "pgvector_result.json",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.query.strip():
        raise ValueError("검색 질의가 비었습니다")
    if args.top_k < 1 or args.candidate_k < args.top_k:
        raise ValueError("candidate-k는 top-k 이상이어야 합니다")
    if args.max_per_parent < 1:
        raise ValueError("max-per-parent는 1 이상이어야 합니다")
    if args.ef_search < args.candidate_k:
        raise ValueError("ef-search는 candidate-k 이상이어야 합니다")
    if args.rrf_k < 1 or args.dense_weight <= 0 or args.bm25_weight <= 0:
        raise ValueError("rrf-k는 1 이상, 결합 가중치는 0보다 커야 합니다")
    if not args.dsn:
        raise ValueError("--dsn 또는 PGVECTOR_DSN 환경변수가 필요합니다")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.table):
        raise ValueError("table 이름은 영문자·숫자·밑줄만 허용됩니다")


def build_statement(args: argparse.Namespace, table: Any) -> tuple[Any, list[Any]]:
    from psycopg import sql

    conditions = []
    parameters: list[Any] = []
    council_values = council_alias_values(args.council)
    if council_values:
        conditions.append(sql.SQL("council = ANY(%s)"))
        parameters.append(list(council_values))
    if args.year is not None:
        conditions.append(sql.SQL("year = %s"))
        parameters.append(args.year)
    committee_values = committee_alias_values(args.committee)
    if committee_values:
        conditions.append(sql.SQL("(") + sql.SQL(" OR ").join(
            sql.SQL("committee ILIKE %s") for _ in committee_values
        ) + sql.SQL(")"))
        parameters.extend(f"%{value}%" for value in committee_values)
    if args.date_from:
        conditions.append(sql.SQL("meeting_date >= %s"))
        parameters.append(args.date_from)
    if args.date_to:
        conditions.append(sql.SQL("meeting_date <= %s"))
        parameters.append(args.date_to)
    if args.priorities:
        conditions.append(sql.SQL("priority = ANY(%s)"))
        parameters.append(args.priorities)
    if args.policy_category:
        conditions.append(sql.SQL(
            "COALESCE(metadata->>'policy_categories_json', '') ILIKE %s"
        ))
        parameters.append(f"%{args.policy_category}%")

    where = (
        sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions)
        if conditions else sql.SQL("")
    )
    statement = sql.SQL("""
        SELECT
          chunk_id, parent_chunk_id, document_id, council, meeting_date, year,
          committee, meeting_name, agenda_title, priority, document, metadata,
          1 - (embedding <=> %s) AS similarity
        FROM {}{}
        ORDER BY embedding <=> %s
        LIMIT %s
    """).format(table, where)
    return statement, parameters


def row_to_candidate(row: tuple[Any, ...], rank: int) -> dict[str, Any]:
    metadata = dict(row[11] or {})
    explicit = {
        "parent_chunk_id": row[1],
        "document_id": row[2],
        "council": row[3],
        "meeting_date": row[4],
        "year": row[5],
        "committee": row[6],
        "meeting_name": row[7],
        "agenda_title": row[8],
        "priority": row[9],
    }
    for key, value in explicit.items():
        if value is not None:
            metadata[key] = json_safe(value)
    return {
        "candidate_rank": rank,
        "dense_rank": rank,
        "chunk_id": str(row[0]),
        "similarity": round(float(row[12]), 6),
        "dense_similarity": round(float(row[12]), 6),
        "retrieval_sources": ["dense"],
        "text": str(row[10] or ""),
        "metadata": {key: json_safe(value) for key, value in metadata.items()},
    }


def hydrate_bm25_candidates(
    artifact: EmbeddingArtifact,
    hits: list[Any],
) -> list[dict[str, Any]]:
    ids = [hit.chunk_id for hit in hits]
    rows = rows_by_chunk_ids(artifact.paths.metadata, ids)
    result = []
    for hit in hits:
        row = rows.get(hit.chunk_id)
        if not row:
            continue
        metadata = {
            key: json_safe(value)
            for key, value in row.items()
            if key not in {"text", "embedding_text"}
        }
        result.append({
            "candidate_rank": hit.rank,
            "bm25_rank": hit.rank,
            "chunk_id": hit.chunk_id,
            "similarity": 0.0,
            "bm25_score": round(float(hit.score), 6),
            "retrieval_sources": ["bm25"],
            "text": str(row.get("text") or ""),
            "metadata": metadata,
        })
    return result


def normalized_text_digest(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def select_diverse_results(
    candidates: list[dict[str, Any]],
    top_k: int,
    max_per_parent: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    parent_counts: dict[str, int] = {}
    skipped_exact_text = 0
    skipped_parent_limit = 0

    for row in candidates:
        digest = normalized_text_digest(row["text"])
        if digest in seen_texts:
            skipped_exact_text += 1
            continue
        parent_id = str(row["metadata"].get("parent_chunk_id") or "")
        if parent_id and parent_counts.get(parent_id, 0) >= max_per_parent:
            skipped_parent_limit += 1
            continue
        seen_texts.add(digest)
        if parent_id:
            parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
        selected.append(row)
        if len(selected) >= top_k:
            break

    return selected, {
        "skipped_exact_text_duplicates": skipped_exact_text,
        "skipped_parent_limit": skipped_parent_limit,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)

    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg import sql

    artifact = EmbeddingArtifact(args.embedding_dir)
    encoder = QueryEncoder(
        artifact,
        device=args.device,
        model_cache=args.model_cache,
        local_files_only=not args.allow_download,
    )

    encode_started = time.perf_counter()
    query_vector = encoder.encode([args.query.strip()])[0]
    embedding_ms = (time.perf_counter() - encode_started) * 1000
    if query_vector.shape != (artifact.dimension,):
        raise RuntimeError(
            f"질의 벡터 차원이 다릅니다: {query_vector.shape} != ({artifact.dimension},)"
        )
    if not np.isfinite(query_vector).all():
        raise RuntimeError("질의 벡터에 NaN 또는 Inf가 있습니다")

    table = sql.Identifier(args.table)
    statement, filter_parameters = build_statement(args, table)
    dense_k = args.dense_k or (
        args.candidate_k * 2 if args.retrieval_mode == "hybrid" else args.candidate_k
    )
    bm25_k = args.bm25_k or args.candidate_k * 2
    if args.ef_search < dense_k:
        raise ValueError("ef-search는 실제 dense-k 이상이어야 합니다")
    connection_started = time.perf_counter()
    with psycopg.connect(args.dsn, autocommit=True) as connection:
        register_vector(connection)
        connection.execute(
            "SELECT set_config('hnsw.ef_search', %s, false)",
            (str(args.ef_search),),
        )
        connection.execute(
            "SELECT set_config('hnsw.iterative_scan', 'strict_order', false)"
        )
        connection_setup_ms = (time.perf_counter() - connection_started) * 1000
        search_started = time.perf_counter()
        rows = connection.execute(
            statement,
            [query_vector, *filter_parameters, query_vector, dense_k],
        ).fetchall()
        search_ms = (time.perf_counter() - search_started) * 1000

    dense_candidates = [
        row_to_candidate(row, rank) for rank, row in enumerate(rows, start=1)
    ]
    bm25_candidates: list[dict[str, Any]] = []
    bm25_tokens: list[str] = []
    bm25_ms = 0.0
    fusion_ms = 0.0
    if args.retrieval_mode == "hybrid":
        bm25_started = time.perf_counter()
        hits, bm25_tokens = BM25Index(args.bm25_index).search(
            args.query.strip(), bm25_k,
            council=args.council, year=args.year, committee=args.committee,
            date_from=args.date_from, date_to=args.date_to,
            priorities=args.priorities, policy_category=args.policy_category,
        )
        bm25_candidates = hydrate_bm25_candidates(artifact, hits)
        bm25_ms = (time.perf_counter() - bm25_started) * 1000
        fusion_started = time.perf_counter()
        candidates = reciprocal_rank_fusion(
            dense_candidates, bm25_candidates, limit=args.candidate_k,
            rrf_k=args.rrf_k, dense_weight=args.dense_weight,
            bm25_weight=args.bm25_weight,
        )
        fusion_ms = (time.perf_counter() - fusion_started) * 1000
        for rank, row in enumerate(candidates, start=1):
            row["candidate_rank"] = rank
    else:
        candidates = dense_candidates[:args.candidate_k]
    selected, deduplication = select_diverse_results(
        candidates, args.top_k, args.max_per_parent,
    )
    contexts: dict[str, str] = {}
    parent_path = args.parent_path.expanduser().resolve()
    if not args.no_parent:
        if not parent_path.is_file():
            raise FileNotFoundError(f"부모 청크 파일이 없습니다: {parent_path}")
        contexts = parent_contexts(
            parent_path,
            [
                str(row["metadata"].get("parent_chunk_id") or "")
                for row in selected
            ],
        )

    results = []
    for final_rank, row in enumerate(selected, start=1):
        result = dict(row)
        result["rank"] = final_rank
        parent_id = str(result["metadata"].get("parent_chunk_id") or "")
        result["parent_context"] = contexts.get(parent_id)
        results.append(result)

    report = {
        "status": "complete",
        "query": args.query.strip(),
        "backend": "postgresql+pgvector",
        "retrieval_stage": (
            "hybrid_rrf_before_bge_reranking"
            if args.retrieval_mode == "hybrid"
            else "dense_baseline_before_bge_reranking"
        ),
        "model": {
            "name": artifact.model_name,
            "revision": artifact.revision,
            "dimension": artifact.dimension,
            "text_view": artifact.summary.get("text_view"),
        },
        "database": {
            "table": args.table,
            "distance": "cosine",
            "index": "hnsw",
            "ef_search": args.ef_search,
        },
        "retrieval": {
            "mode": args.retrieval_mode,
            "dense_candidate_count": len(dense_candidates),
            "bm25_candidate_count": len(bm25_candidates),
            "fused_candidate_count": len(candidates),
            "bm25_index": str(args.bm25_index.expanduser().resolve())
            if args.retrieval_mode == "hybrid" else None,
            "bm25_tokens": bm25_tokens,
            "rrf_k": args.rrf_k,
            "dense_weight": args.dense_weight,
            "bm25_weight": args.bm25_weight,
        },
        "filters": {
            "council": args.council,
            "year": args.year,
            "committee": args.committee,
            "date_from": args.date_from,
            "date_to": args.date_to,
            "priorities": args.priorities,
            "policy_category": args.policy_category,
        },
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "candidate_count": len(candidates),
        "max_per_parent": args.max_per_parent,
        "deduplication": deduplication,
        "timing_ms": {
            "query_embedding": round(embedding_ms, 3),
            "connection_setup": round(connection_setup_ms, 3),
            "database_search": round(search_ms, 3),
            "bm25_search_and_hydration": round(bm25_ms, 3),
            "rrf_fusion": round(fusion_ms, 3),
        },
        "selection": (
            f"{args.retrieval_mode} retrieval + exact-text deduplication + "
            "parent diversity; BGE reranker not applied yet"
        ),
        "results": results,
        "candidates": candidates,
    }
    output = args.output.expanduser().resolve()
    atomic_json(output, report)

    print("\n=== Dense/BM25 근거 후보 검색 완료 ===")
    print(f"질의: {args.query.strip()}")
    print(
        f"모드: {args.retrieval_mode} | Dense={len(dense_candidates)} | "
        f"BM25={len(bm25_candidates)} | 결합={len(candidates)}"
    )
    print(
        f"후보: {len(candidates)} | 최종: {len(results)} | "
        f"ef_search={args.ef_search} | 검색={search_ms:.2f}ms"
    )
    print(
        "중복 제거: "
        f"동일본문={deduplication['skipped_exact_text_duplicates']} | "
        f"동일부모={deduplication['skipped_parent_limit']}"
    )
    for row in results:
        metadata = row["metadata"]
        source_ranks = []
        if row.get("dense_rank") is not None:
            source_ranks.append(f"dense={row['dense_rank']}")
        if row.get("bm25_rank") is not None:
            source_ranks.append(f"bm25={row['bm25_rank']}")
        if row.get("rrf_rank") is not None:
            source_ranks.append(f"rrf={row['rrf_rank']}")
        print(
            f"\n[{row['rank']}] {' · '.join(source_ranks)} "
            f"| dense_similarity={row['similarity']:.4f} "
            f"| {metadata.get('council')} | {metadata.get('meeting_date')}"
        )
        print(
            f"회의록: {metadata.get('meeting_name')} "
            f"| 위원회: {metadata.get('committee')}"
        )
        print(f"안건: {metadata.get('agenda_title')}")
        print(f"원문: {row['text'][:500]}")
    print(f"\nJSON: {output}")


if __name__ == "__main__":
    main()

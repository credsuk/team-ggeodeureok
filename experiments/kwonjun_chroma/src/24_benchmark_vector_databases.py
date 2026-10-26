#!/usr/bin/env python3
"""동일 벡터·동일 질의로 Chroma와 pgvector의 지연시간 및 exact Recall@K를 비교한다."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from council_vector_runtime import EmbeddingArtifact, QueryEncoder, atomic_json, exact_topk


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=PROJECT_ROOT / "eval" / "queries.csv")
    parser.add_argument("--max-queries", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--embedding-dir", type=Path,
        default=PROJECT_ROOT / "data" / "embeddings" / "arctic_enriched_v1",
    )
    parser.add_argument("--chroma-dir", type=Path, default=PROJECT_ROOT / "data" / "chroma" / "council_final_v1")
    parser.add_argument("--collection", default="council_chunks_final_v1")
    parser.add_argument("--pg-dsn", default=os.environ.get("PGVECTOR_DSN"))
    parser.add_argument("--pg-table", default="council_chunks_final_v1")
    parser.add_argument("--pg-ef-search", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--model-cache", type=Path, default=PROJECT_ROOT / "models" / "huggingface")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--skip-chroma", action="store_true")
    parser.add_argument("--skip-pgvector", action="store_true")
    parser.add_argument("--exact-limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "benchmark" / "vector_db_benchmark.json")
    return parser.parse_args()


def load_queries(path: Path, limit: int) -> list[dict[str, str]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("queries", payload)
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    result = []
    for index, row in enumerate(rows[:limit], start=1):
        result.append({"query_id": str(row.get("query_id") or f"Q{index:03d}"), "query": str(row["query"])})
    if not result:
        raise ValueError("평가 질의가 없습니다")
    return result


def percentile(values: list[float], q: float) -> float:
    return round(float(np.percentile(np.asarray(values), q)), 3) if values else 0.0


def summarize(latencies: list[float], recalls: list[float], top1s: list[float]) -> dict[str, Any]:
    return {
        "requests": len(latencies), "latency_ms_mean": round(float(np.mean(latencies)), 3),
        "latency_ms_p50": percentile(latencies, 50), "latency_ms_p95": percentile(latencies, 95),
        "latency_ms_p99": percentile(latencies, 99), "recall_at_k": round(float(np.mean(recalls)), 6),
        "top1_accuracy": round(float(np.mean(top1s)), 6),
    }


def evaluate_backend(name: str, query_fn, vectors: np.ndarray, exact_ids: list[list[str]], top_k: int, warmup: int, repeat: int):
    latencies: list[float] = []
    recalls: list[float] = []
    top1s: list[float] = []
    details = []
    for query_index, vector in enumerate(vectors):
        for _ in range(warmup):
            query_fn(vector, top_k)
        for repeat_index in range(repeat):
            started = time.perf_counter()
            found = query_fn(vector, top_k)
            elapsed = (time.perf_counter() - started) * 1000
            truth = exact_ids[query_index]
            recall = len(set(found) & set(truth)) / top_k
            top1 = float(bool(found) and found[0] == truth[0])
            latencies.append(elapsed); recalls.append(recall); top1s.append(top1)
            details.append({"query_index": query_index, "repeat": repeat_index + 1, "latency_ms": round(elapsed, 3), "recall_at_k": recall, "top1": bool(top1)})
    return {"backend": name, "summary": summarize(latencies, recalls, top1s), "runs": details}


def main() -> None:
    args = parse_args()
    artifact = EmbeddingArtifact(args.embedding_dir)
    queries = load_queries(args.queries.expanduser().resolve(), args.max_queries)
    encoder = QueryEncoder(
        artifact, device=args.device, model_cache=args.model_cache,
        local_files_only=not args.allow_download,
    )
    encode_started = time.perf_counter()
    query_vectors = encoder.encode([row["query"] for row in queries])
    encode_ms = (time.perf_counter() - encode_started) * 1000
    indices, _, exact_ms = exact_topk(
        artifact.vectors, query_vectors, args.top_k, limit=args.exact_limit,
    )
    all_ids = artifact.chunk_ids(limit=args.exact_limit)
    exact_ids = [[all_ids[int(index)] for index in row if index >= 0] for row in indices]
    results: dict[str, Any] = {}

    if not args.skip_chroma:
        import chromadb
        collection = chromadb.PersistentClient(path=str(args.chroma_dir.expanduser().resolve())).get_collection(
            args.collection, embedding_function=None,
        )
        if collection.count() != artifact.row_count:
            raise RuntimeError(f"Chroma 행 불일치: {collection.count():,} != {artifact.row_count:,}")
        def chroma_query(vector, k):
            return collection.query(query_embeddings=[vector.tolist()], n_results=k, include=[])["ids"][0]
        results["chroma"] = evaluate_backend("chroma", chroma_query, query_vectors, exact_ids, args.top_k, args.warmup, args.repeat)

    if not args.skip_pgvector:
        if not args.pg_dsn:
            raise ValueError("pgvector 비교에는 --pg-dsn 또는 PGVECTOR_DSN이 필요합니다")
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg import sql
        connection = psycopg.connect(args.pg_dsn)
        register_vector(connection)
        connection.execute(
            "SELECT set_config('hnsw.ef_search', %s, false)",
            (str(args.pg_ef_search),),
        )
        statement = sql.SQL("SELECT chunk_id FROM {} ORDER BY embedding <=> %s LIMIT %s").format(sql.Identifier(args.pg_table))
        def pg_query(vector, k):
            return [str(row[0]) for row in connection.execute(statement, (vector, k)).fetchall()]
        try:
            results["pgvector"] = evaluate_backend("pgvector", pg_query, query_vectors, exact_ids, args.top_k, args.warmup, args.repeat)
        finally:
            connection.close()

    report = {
        "artifact": str(artifact.paths.root), "row_count": artifact.row_count,
        "dimension": artifact.dimension, "model": artifact.model_name, "top_k": args.top_k,
        "query_count": len(queries), "repeat": args.repeat, "warmup": args.warmup,
        "query_embedding_total_ms": round(encode_ms, 3), "exact_total_ms": round(exact_ms, 3),
        "exact_scope_rows": min(args.exact_limit, artifact.row_count) if args.exact_limit else artifact.row_count,
        "queries": queries, "backends": results,
    }
    atomic_json(args.output.expanduser().resolve(), report)
    print("\n=== 벡터 DB 동일 조건 비교 ===")
    print(f"질의: {len(queries)} | top-k: {args.top_k} | exact: {exact_ms:.1f}ms")
    for name, value in results.items():
        summary = value["summary"]
        print(f"{name:10s} | p50={summary['latency_ms_p50']:.2f}ms | p95={summary['latency_ms_p95']:.2f}ms | Recall@{args.top_k}={summary['recall_at_k']:.4f} | Top1={summary['top1_accuracy']:.4f}")
    print(f"결과: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()

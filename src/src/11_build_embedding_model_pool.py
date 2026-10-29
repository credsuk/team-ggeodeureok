#!/usr/bin/env python3
"""Arctic와 KURE 검색 결과의 소규모 블라인드 평가 pool을 만든다."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from council_vector_runtime import EmbeddingArtifact, QueryEncoder, exact_topk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY_IDS = [
    "Q001", "Q007", "Q012", "Q014", "Q016", "Q021",
    "Q024", "Q029", "Q033", "Q037", "Q040", "Q043",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=PROJECT_ROOT / "config/retrieval_queries_v1.json")
    parser.add_argument("--arctic-dir", type=Path, default=PROJECT_ROOT / "data/embedding_eval_vectors/arctic_raw_60k")
    parser.add_argument("--kure-dir", type=Path, default=PROJECT_ROOT / "data/embedding_eval_vectors/kure_raw_60k")
    parser.add_argument("--query-ids", nargs="+", default=DEFAULT_QUERY_IDS)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--model-cache", type=Path, default=PROJECT_ROOT / "models/huggingface")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/retrieval_eval/arctic_vs_kure_quick_v1")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def pair_id(query_id: str, chunk_id: str) -> str:
    value = hashlib.sha256(f"{query_id}\x1f{chunk_id}".encode("utf-8")).hexdigest()[:24]
    return f"pair_{value}"


def blind_order(query_id: str, chunk_id: str) -> str:
    return hashlib.sha256(f"model-blind-v1\x1f{query_id}\x1f{chunk_id}".encode("utf-8")).hexdigest()


def load_queries(path: Path, selected: list[str]) -> list[dict[str, str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))["queries"]
    by_id = {row["query_id"]: row for row in rows}
    missing = [query_id for query_id in selected if query_id not in by_id]
    if missing:
        raise KeyError(f"없는 query_id: {missing}")
    return [by_id[query_id] for query_id in selected]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    blind_path = output_dir / "blind_pool.jsonl"
    provenance_path = output_dir / "pool_provenance.parquet"
    summary_path = output_dir / "pool_summary.json"
    if any(path.exists() for path in (blind_path, provenance_path, summary_path)) and not args.overwrite:
        raise FileExistsError("출력이 이미 있습니다. 교체하려면 --overwrite를 사용하세요")

    queries = load_queries(args.queries.expanduser().resolve(), args.query_ids)
    artifacts = {
        "arctic": EmbeddingArtifact(args.arctic_dir),
        "kure": EmbeddingArtifact(args.kure_dir),
    }
    ids = {name: artifact.chunk_ids() for name, artifact in artifacts.items()}
    if ids["arctic"] != ids["kure"]:
        raise RuntimeError("Arctic/KURE chunk_id 순서가 다릅니다")
    metadata = pq.read_table(artifacts["arctic"].paths.metadata).to_pylist()
    query_texts = [row["query"] for row in queries]

    rankings: dict[str, tuple[Any, Any, float]] = {}
    for name, artifact in artifacts.items():
        encoder = QueryEncoder(
            artifact, device=args.device, model_cache=args.model_cache,
            local_files_only=args.local_files_only,
        )
        query_vectors = encoder.encode(query_texts)
        rankings[name] = exact_topk(artifact.vectors, query_vectors, args.top_k)
        print(f"{name} exact 검색 완료: {rankings[name][2]:.1f}ms", flush=True)

    provenance_rows: list[dict[str, Any]] = []
    blind_rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        pooled: dict[str, dict[str, Any]] = {}
        for name in ("arctic", "kure"):
            indices, scores, _ = rankings[name]
            for rank, (index, score) in enumerate(zip(indices[query_index], scores[query_index]), start=1):
                chunk_id = ids[name][int(index)]
                entry = pooled.setdefault(chunk_id, {
                    "metadata_index": int(index), "arctic_rank": None, "arctic_score": None,
                    "kure_rank": None, "kure_score": None,
                })
                entry[f"{name}_rank"] = rank
                entry[f"{name}_score"] = float(score)
        for chunk_id, info in pooled.items():
            row = metadata[info["metadata_index"]]
            pid = pair_id(query["query_id"], chunk_id)
            provenance_rows.append({
                "pair_id": pid, "query_id": query["query_id"], "chunk_id": chunk_id,
                "arctic_rank": info["arctic_rank"], "arctic_score": info["arctic_score"],
                "kure_rank": info["kure_rank"], "kure_score": info["kure_score"],
            })
            blind_rows.append({
                "pair_id": pid, "query_id": query["query_id"], "category": query["category"],
                "query": query["query"], "chunk_id": chunk_id, "council": row.get("council"),
                "meeting_date": row.get("meeting_date"), "committee": row.get("committee"),
                "meeting_name": row.get("meeting_name"), "agenda_title": row.get("agenda_title"),
                "text": row.get("text"), "blind_order": blind_order(query["query_id"], chunk_id),
            })

    blind_rows.sort(key=lambda row: (row["query_id"], row["blind_order"]))
    with blind_path.open("w", encoding="utf-8") as file:
        for row in blind_rows:
            row.pop("blind_order")
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    pq.write_table(pa.Table.from_pylist(provenance_rows), provenance_path, compression="zstd")
    summary = {
        "systems": ["arctic", "kure"], "text_view": "raw", "query_ids": args.query_ids,
        "query_count": len(queries), "top_k_per_system": args.top_k,
        "pool_pairs": len(blind_rows), "average_pairs_per_query": round(len(blind_rows) / len(queries), 2),
        "arctic_model": artifacts["arctic"].identity, "kure_model": artifacts["kure"].identity,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Arctic/KURE 빠른 블라인드 pool 완료 ===")
    print(f"질의: {len(queries)} | top-k: {args.top_k} | 판정 대상: {len(blind_rows)}")
    print(f"blind: {blind_path}")
    print(f"provenance: {provenance_path}")


if __name__ == "__main__":
    main()

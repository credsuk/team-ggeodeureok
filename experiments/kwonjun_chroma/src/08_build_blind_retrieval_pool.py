#!/usr/bin/env python3
"""raw/enriched 검색 결과를 합쳐 검색기 정보가 숨겨진 자동평가 pool을 만든다."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
MODEL_REVISION = "55ec6e9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=PROJECT_ROOT / "config" / "retrieval_queries_v1.json")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "embedding_eval_vectors" / "arctic_raw_60k")
    parser.add_argument("--enriched-dir", type=Path, default=PROJECT_ROOT / "data" / "embedding_eval_vectors" / "arctic_enriched_60k")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "retrieval_eval" / "arctic_raw_vs_enriched_v1")
    parser.add_argument("--model-cache", type=Path, default=PROJECT_ROOT / "models" / "huggingface")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_queries(path: Path) -> list[dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    queries = value.get("queries", [])
    ids = [item["query_id"] for item in queries]
    if not queries or len(ids) != len(set(ids)):
        raise RuntimeError("질의가 없거나 query_id가 중복되었습니다")
    return queries


def metadata_ids(path: Path) -> list[str]:
    return [str(value) for value in pq.read_table(path, columns=["chunk_id"])["chunk_id"].to_pylist()]


def exact_topk(vectors_path: Path, query_vectors: np.ndarray, top_k: int, block_rows: int = 10_000) -> tuple[np.ndarray, np.ndarray]:
    documents = np.load(vectors_path, mmap_mode="r")
    if documents.ndim != 2 or documents.shape[1] != query_vectors.shape[1]:
        raise RuntimeError(f"벡터 차원 불일치: docs={documents.shape}, queries={query_vectors.shape}")
    scores = np.full((len(query_vectors), top_k), -np.inf, dtype=np.float32)
    indices = np.full((len(query_vectors), top_k), -1, dtype=np.int64)
    for start in range(0, len(documents), block_rows):
        end = min(start + block_rows, len(documents))
        block = np.asarray(documents[start:end], dtype=np.float32)
        block_scores = query_vectors @ block.T
        local_k = min(top_k, block_scores.shape[1])
        local_positions = np.argpartition(block_scores, -local_k, axis=1)[:, -local_k:]
        local_scores = np.take_along_axis(block_scores, local_positions, axis=1)
        local_indices = local_positions.astype(np.int64) + start
        merged_scores = np.concatenate([scores, local_scores], axis=1)
        merged_indices = np.concatenate([indices, local_indices], axis=1)
        keep = np.argpartition(merged_scores, -top_k, axis=1)[:, -top_k:]
        scores = np.take_along_axis(merged_scores, keep, axis=1)
        indices = np.take_along_axis(merged_indices, keep, axis=1)
    order = np.argsort(-scores, axis=1)
    return np.take_along_axis(indices, order, axis=1), np.take_along_axis(scores, order, axis=1)


def pair_id(query_id: str, chunk_id: str) -> str:
    value = hashlib.sha256(f"{query_id}\x1f{chunk_id}".encode("utf-8")).hexdigest()[:24]
    return f"pair_{value}"


def blind_order(query_id: str, chunk_id: str) -> str:
    return hashlib.sha256(f"blind-v1\x1f{query_id}\x1f{chunk_id}".encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    queries_path = args.queries.expanduser().resolve()
    raw_dir = args.raw_dir.expanduser().resolve()
    enriched_dir = args.enriched_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    blind_path = output_dir / "blind_pool.jsonl"
    provenance_path = output_dir / "pool_provenance.parquet"
    summary_path = output_dir / "pool_summary.json"
    if any(path.exists() for path in (blind_path, provenance_path, summary_path)) and not args.overwrite:
        raise FileExistsError("출력이 이미 있습니다. 교체하려면 --overwrite를 사용하세요")

    queries = load_queries(queries_path)
    raw_meta = raw_dir / "chunk_metadata.parquet"
    enriched_meta = enriched_dir / "chunk_metadata.parquet"
    raw_vectors = raw_dir / "embeddings.f16.npy"
    enriched_vectors = enriched_dir / "embeddings.f16.npy"
    for path in (raw_meta, enriched_meta, raw_vectors, enriched_vectors):
        if not path.is_file():
            raise FileNotFoundError(path)
    raw_ids = metadata_ids(raw_meta)
    enriched_ids = metadata_ids(enriched_meta)
    if raw_ids != enriched_ids:
        raise RuntimeError("raw/enriched metadata의 chunk_id 순서가 다릅니다")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    model = SentenceTransformer(
        MODEL_NAME, revision=MODEL_REVISION,
        cache_folder=str(args.model_cache.expanduser().resolve()),
        local_files_only=args.local_files_only, device=device,
    )
    model.max_seq_length = 2048
    query_texts = [item["query"] for item in queries]
    query_vectors = model.encode(
        query_texts, prompt_name="query", batch_size=16,
        convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    raw_indices, raw_scores = exact_topk(raw_vectors, query_vectors, args.top_k)
    enriched_indices, enriched_scores = exact_topk(enriched_vectors, query_vectors, args.top_k)
    metadata = pq.read_table(raw_meta).to_pylist()

    provenance_rows: list[dict[str, Any]] = []
    blind_rows: list[dict[str, Any]] = []
    for query_pos, query in enumerate(queries):
        pooled: dict[str, dict[str, Any]] = {}
        for system, indices, scores in (
            ("raw", raw_indices[query_pos], raw_scores[query_pos]),
            ("enriched", enriched_indices[query_pos], enriched_scores[query_pos]),
        ):
            for rank, (index, score) in enumerate(zip(indices, scores), start=1):
                chunk_id = raw_ids[int(index)]
                item = pooled.setdefault(chunk_id, {
                    "raw_rank": None, "raw_score": None,
                    "enriched_rank": None, "enriched_score": None,
                    "metadata_index": int(index),
                })
                item[f"{system}_rank"] = rank
                item[f"{system}_score"] = float(score)
        for chunk_id, info in pooled.items():
            row = metadata[info["metadata_index"]]
            pid = pair_id(query["query_id"], chunk_id)
            provenance_rows.append({
                "pair_id": pid,
                "query_id": query["query_id"],
                "chunk_id": chunk_id,
                "raw_rank": info["raw_rank"],
                "raw_score": info["raw_score"],
                "enriched_rank": info["enriched_rank"],
                "enriched_score": info["enriched_score"],
            })
            blind_rows.append({
                "pair_id": pid,
                "query_id": query["query_id"],
                "category": query["category"],
                "query": query["query"],
                "chunk_id": chunk_id,
                "council": row.get("council"),
                "meeting_date": row.get("meeting_date"),
                "committee": row.get("committee"),
                "meeting_name": row.get("meeting_name"),
                "agenda_title": row.get("agenda_title"),
                "text": row.get("text"),
                "blind_order": blind_order(query["query_id"], chunk_id),
            })

    blind_rows.sort(key=lambda row: (row["query_id"], row["blind_order"]))
    with blind_path.open("w", encoding="utf-8") as file:
        for row in blind_rows:
            row.pop("blind_order", None)
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    pq.write_table(pa.Table.from_pylist(provenance_rows), provenance_path, compression="zstd")
    per_query = len(blind_rows) / len(queries)
    summary = {
        "query_count": len(queries),
        "top_k_per_system": args.top_k,
        "pool_pairs": len(blind_rows),
        "average_pairs_per_query": round(per_query, 2),
        "chunk_order_equal": True,
        "judge_visibility": "검색기·순위·점수 숨김",
        "blind_pool": str(blind_path),
        "provenance": str(provenance_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 블라인드 검색 평가 pool 생성 완료 ===")
    print(f"질의: {len(queries):,} | pair: {len(blind_rows):,} | 질의당 평균: {per_query:.1f}")
    print(f"blind: {blind_path}")
    print(f"provenance: {provenance_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Arctic 질의 임베딩으로 로컬 ChromaDB를 검색하고 FAISS 정확 검색과 비교한다."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    import chromadb
    import faiss
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError as error:
    raise SystemExit(
        "chromadb, faiss-cpu, numpy, pyarrow, torch, sentence-transformers가 필요합니다."
    ) from error


MODEL_NAME = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
MODEL_REVISION = "55ec6e9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChromaDB 정책 회의록 검색")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chroma-dir", default="data/chroma/arctic_policy_v1")
    parser.add_argument("--collection", default="council_policy_arctic_v1")
    parser.add_argument("--model-cache", default="models/huggingface")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--council")
    parser.add_argument("--committee")
    parser.add_argument("--year", type=int)
    parser.add_argument("--date-from", help="YYYYMMDD")
    parser.add_argument("--date-to", help="YYYYMMDD")
    parser.add_argument("--output", help="검색 결과 JSON 경로")
    parser.add_argument("--compare-faiss", action="store_true")
    parser.add_argument(
        "--metadata",
        default="data/index/arctic_ko_semantic_test_v2/chunk_metadata.parquet",
    )
    parser.add_argument(
        "--embeddings",
        default="data/index/arctic_ko_semantic_test_v2/embeddings.f16.npy",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA를 사용할 수 없습니다.")
    return requested


def encode_query(args: argparse.Namespace) -> tuple[np.ndarray, str | None, str]:
    device = resolve_device(args.device)
    cache = Path(args.model_cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device=device,
        cache_folder=str(cache),
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    model.max_seq_length = 2048
    model.eval()
    prompts = getattr(model, "prompts", {}) or {}
    prompt_name = "query" if "query" in prompts else None
    kwargs = {
        "batch_size": 1,
        "show_progress_bar": False,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }
    if prompt_name:
        kwargs["prompt_name"] = prompt_name
    vector = model.encode([args.query], **kwargs)
    return np.asarray(vector, dtype=np.float32), prompt_name, device


def normalize_date(value: str | None, name: str) -> int | None:
    if not value:
        return None
    cleaned = value.replace("-", "").replace(".", "")
    if len(cleaned) != 8 or not cleaned.isdigit():
        raise SystemExit(f"{name}은 YYYYMMDD 형식이어야 합니다: {value}")
    return int(cleaned)


def build_where(args: argparse.Namespace) -> dict | None:
    conditions = []
    if args.council:
        conditions.append({"council": {"$eq": args.council}})
    if args.committee:
        conditions.append({"committee": {"$eq": args.committee}})
    if args.year:
        conditions.append({"meeting_year": {"$eq": args.year}})
    date_from = normalize_date(args.date_from, "--date-from")
    date_to = normalize_date(args.date_to, "--date-to")
    if date_from is not None:
        conditions.append({"meeting_date_int": {"$gte": date_from}})
    if date_to is not None:
        conditions.append({"meeting_date_int": {"$lte": date_to}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def chroma_search(args: argparse.Namespace, query_vector: np.ndarray) -> tuple[list[dict], float]:
    chroma_dir = Path(args.chroma_dir).expanduser().resolve()
    if not chroma_dir.exists():
        raise SystemExit(f"Chroma 디렉터리가 없습니다: {chroma_dir}")
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(args.collection, embedding_function=None)
    where = build_where(args)
    started = time.perf_counter()
    result = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=args.top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    output = []
    ids = result.get("ids", [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):
        output.append({
            "rank": rank,
            "chunk_id": chunk_id,
            "similarity": 1.0 - float(distance),
            "council": (metadata or {}).get("council", ""),
            "committee": (metadata or {}).get("committee", ""),
            "meeting_date": (metadata or {}).get("meeting_date", ""),
            "meeting_name": (metadata or {}).get("meeting_name", ""),
            "agenda_title": (
                (metadata or {}).get("agenda_title_effective")
                or (metadata or {}).get("evidence_title")
                or (metadata or {}).get("agenda_title", "")
            ),
            "agenda_title_raw": (metadata or {}).get("agenda_title_raw", ""),
            "agenda_confidence": (metadata or {}).get("agenda_confidence", ""),
            "speaker_label": (metadata or {}).get("speaker_label", ""),
            "evidence_ref": (metadata or {}).get("evidence_ref", ""),
            "text": document or "",
        })
    return output, elapsed_ms


def exact_faiss_search(args: argparse.Namespace, query_vector: np.ndarray) -> tuple[list[str], float]:
    if any([args.council, args.committee, args.year, args.date_from, args.date_to]):
        raise SystemExit("--compare-faiss는 메타데이터 필터 없이 사용하세요.")
    metadata_path = Path(args.metadata).expanduser().resolve()
    embedding_path = Path(args.embeddings).expanduser().resolve()
    if not metadata_path.exists() or not embedding_path.exists():
        raise SystemExit("FAISS 비교용 metadata/embeddings 파일이 없습니다.")
    table = pq.read_table(metadata_path, columns=["embedding_row", "chunk_id"])
    rows = table.to_pylist()
    vectors = np.load(embedding_path, mmap_mode="r")
    if len(rows) != vectors.shape[0]:
        raise SystemExit("FAISS 비교 입력 행 수가 일치하지 않습니다.")
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    for start in range(0, len(rows), 4096):
        block = np.asarray(vectors[start:start + 4096], dtype=np.float32)
        faiss.normalize_L2(block)
        index.add(block)
    started = time.perf_counter()
    _, indices = index.search(query_vector, min(args.top_k, len(rows)))
    elapsed_ms = (time.perf_counter() - started) * 1000
    return [str(rows[int(index)]["chunk_id"]) for index in indices[0]], elapsed_ms


def print_results(args: argparse.Namespace, results: list[dict], elapsed_ms: float) -> None:
    print(f"\n질의: {args.query}")
    print(f"Chroma 검색: {elapsed_ms:.2f} ms | 결과 {len(results)}개")
    for row in results:
        preview = " ".join(row["text"].split())[:240]
        print(f"\n[{row['rank']}] similarity={row['similarity']:.4f} | {row['council']} | {row['meeting_date']}")
        print(f"    {row['committee']} | {row['agenda_title']}")
        print(f"    {preview}")


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k는 1 이상이어야 합니다.")
    query_vector, prompt_name, device = encode_query(args)
    results, chroma_ms = chroma_search(args, query_vector)
    print_results(args, results, chroma_ms)
    report = {
        "query": args.query,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "query_prompt": prompt_name,
        "device": device,
        "filters": build_where(args),
        "chroma_elapsed_ms": chroma_ms,
        "results": results,
    }
    if args.compare_faiss:
        exact_ids, faiss_ms = exact_faiss_search(args, query_vector)
        chroma_ids = [row["chunk_id"] for row in results]
        overlap = len(set(chroma_ids) & set(exact_ids)) / min(args.top_k, len(exact_ids))
        report["faiss_comparison"] = {
            "faiss_elapsed_ms": faiss_ms,
            f"overlap@{args.top_k}": overlap,
            "top1_match": bool(chroma_ids and exact_ids and chroma_ids[0] == exact_ids[0]),
            "faiss_ids": exact_ids,
        }
        print(f"\nFAISS 정확 검색: {faiss_ms:.2f} ms")
        print(f"Chroma/FAISS overlap@{args.top_k}: {overlap:.3f}")
        print(f"Top-1 일치: {report['faiss_comparison']['top1_match']}")
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"결과 JSON: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""검증에서 우승한 Arctic 정책 임베딩을 로컬 ChromaDB에 적재한다.

회의록 원문이나 임베딩을 외부 서비스로 전송하지 않는다. 이미 계산된
임베딩을 그대로 저장하므로 Chroma의 기본 임베딩 모델은 사용하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

try:
    import chromadb
    import numpy as np
    import pyarrow.parquet as pq
except ImportError as error:
    raise SystemExit("chromadb, numpy, pyarrow가 필요합니다.") from error


SCRIPT_VERSION = "2026-10-20-chroma-arctic-v1"
TEXT_COLUMNS = ("semantic_text", "issue_text", "chunk_text")
METADATA_COLUMNS = (
    "embedding_row", "chunk_id", "parent_chunk_id", "doc_id", "council",
    "committee", "meeting_name", "meeting_date", "agenda_title", "speaker_label",
    "evidence_ref", "char_start", "char_end", "chunk_type", "priority",
    "agenda_title_raw", "agenda_title_effective", "agenda_confidence",
    "agenda_reason", "evidence_title",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arctic 정책 임베딩 ChromaDB 구축")
    parser.add_argument(
        "--metadata",
        default="data/index/arctic_ko_semantic_test_v2/chunk_metadata.parquet",
    )
    parser.add_argument(
        "--embeddings",
        default="data/index/arctic_ko_semantic_test_v2/embeddings.f16.npy",
    )
    parser.add_argument("--chroma-dir", default="data/chroma/arctic_policy_v1")
    parser.add_argument("--collection", default="council_policy_arctic_v1")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=200)
    parser.add_argument("--max-neighbors", type=int, default=32)
    parser.add_argument("--sync-threshold", type=int, default=5000)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def safe_text(value) -> str:
    return "" if value is None else str(value).strip()


def atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def file_signature(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"입력 파일이 없습니다: {path}")
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_inputs(metadata_path: Path, embedding_path: Path):
    source = pq.ParquetFile(metadata_path)
    available = set(source.schema_arrow.names)
    missing = {"embedding_row", "chunk_id"} - available
    if missing:
        raise SystemExit("필수 메타데이터 컬럼 누락: " + ", ".join(sorted(missing)))
    text_column = next((name for name in TEXT_COLUMNS if name in available), None)
    if text_column is None:
        raise SystemExit("semantic_text, issue_text, chunk_text 중 본문 컬럼이 필요합니다.")
    columns = [name for name in METADATA_COLUMNS if name in available]
    if text_column not in columns:
        columns.append(text_column)
    rows = pq.read_table(metadata_path, columns=columns).to_pylist()
    vectors = np.load(embedding_path, mmap_mode="r")
    if vectors.ndim != 2 or vectors.shape[0] != len(rows):
        raise SystemExit(f"임베딩 shape 불일치: {vectors.shape}, metadata={len(rows):,}")

    ids = []
    documents = []
    digest = hashlib.sha256()
    seen = set()
    for expected, row in enumerate(rows):
        actual = int(row.get("embedding_row", -1))
        if actual != expected:
            raise SystemExit(f"embedding_row 순서 오류: row={expected}, value={actual}")
        chunk_id = safe_text(row.get("chunk_id"))
        text = safe_text(row.get(text_column))
        if not chunk_id or not text:
            raise SystemExit(f"빈 chunk_id 또는 본문: row={expected}")
        if chunk_id in seen:
            raise SystemExit(f"중복 chunk_id: {chunk_id}")
        seen.add(chunk_id)
        ids.append(chunk_id)
        documents.append(text)
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(hashlib.sha256(text.encode("utf-8")).digest())
        digest.update(b"\n")
    return rows, ids, documents, vectors, text_column, digest.hexdigest()


def chroma_metadata(row: dict) -> dict:
    output = {"embedding_row": int(row["embedding_row"])}
    for name in METADATA_COLUMNS:
        if name in {"embedding_row", "char_start", "char_end"}:
            continue
        value = row.get(name)
        if value is not None and safe_text(value):
            output[name] = safe_text(value)
    for name in ("char_start", "char_end"):
        value = row.get(name)
        if value is not None and safe_text(value):
            try:
                output[name] = int(value)
            except (TypeError, ValueError):
                output[name] = safe_text(value)
    date = safe_text(row.get("meeting_date")).replace("-", "").replace(".", "")
    if len(date) >= 4 and date[:4].isdigit():
        output["meeting_year"] = int(date[:4])
    if len(date) == 8 and date.isdigit():
        output["meeting_date_int"] = int(date)
    return output


def create_collection(client, args: argparse.Namespace):
    common_metadata = {
        "model_name": "dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        "model_revision": "55ec6e9",
        "normalized": True,
        "source": "local_competition_data",
    }
    configuration = {
        "hnsw": {
            "space": "cosine",
            "ef_construction": args.ef_construction,
            "ef_search": args.ef_search,
            "max_neighbors": args.max_neighbors,
            "batch_size": args.batch_size,
            "sync_threshold": max(args.sync_threshold, args.batch_size + 1),
        }
    }
    try:
        return client.create_collection(
            name=args.collection,
            metadata=common_metadata,
            configuration=configuration,
            embedding_function=None,
        )
    except TypeError:
        # Chroma 1.0 이전 버전과의 호환 설정이다.
        legacy_metadata = {
            **common_metadata,
            "hnsw:space": "cosine",
            "hnsw:construction_ef": args.ef_construction,
            "hnsw:search_ef": args.ef_search,
            "hnsw:M": args.max_neighbors,
            "hnsw:batch_size": args.batch_size,
            "hnsw:sync_threshold": max(args.sync_threshold, args.batch_size + 1),
        }
        return client.create_collection(
            name=args.collection,
            metadata=legacy_metadata,
            embedding_function=None,
        )


def verify_samples(collection, ids: list[str], vectors: np.ndarray) -> dict:
    positions = sorted({0, len(ids) // 2, len(ids) - 1})
    sample_ids = [ids[index] for index in positions]
    result = collection.get(ids=sample_ids, include=["embeddings", "metadatas", "documents"])
    returned = {chunk_id: index for index, chunk_id in enumerate(result["ids"])}
    max_abs_error = 0.0
    for position, chunk_id in zip(positions, sample_ids):
        if chunk_id not in returned:
            raise SystemExit(f"Chroma 검증 샘플 누락: {chunk_id}")
        stored = np.asarray(result["embeddings"][returned[chunk_id]], dtype=np.float32)
        original = np.asarray(vectors[position], dtype=np.float32)
        max_abs_error = max(max_abs_error, float(np.max(np.abs(stored - original))))
    return {"sample_count": len(sample_ids), "max_abs_embedding_error": max_abs_error}


def main() -> None:
    args = parse_args()
    if args.batch_size < 2:
        raise SystemExit("--batch-size는 2 이상이어야 합니다.")
    metadata_path = Path(args.metadata).expanduser().resolve()
    embedding_path = Path(args.embeddings).expanduser().resolve()
    chroma_dir = Path(args.chroma_dir).expanduser().resolve()
    metadata_signature = file_signature(metadata_path)
    embedding_signature = file_signature(embedding_path)
    print("Arctic 정책 청크 검증", flush=True)
    rows, ids, documents, vectors, text_column, digest = load_inputs(metadata_path, embedding_path)
    print(f"청크 {len(rows):,}개 | dim={vectors.shape[1]} | text={text_column}")

    if args.rebuild and chroma_dir.exists():
        print(f"기존 Chroma 디렉터리 삭제 후 재구축: {chroma_dir}")
        shutil.rmtree(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    state_path = chroma_dir / "chroma_build_state.json"
    client = chromadb.PersistentClient(path=str(chroma_dir))
    existing_names = {
        item if isinstance(item, str) else item.name
        for item in client.list_collections()
    }
    if args.collection in existing_names:
        collection = client.get_collection(args.collection, embedding_function=None)
        if collection.count() == len(rows) and state_path.exists() and not args.rebuild:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("corpus_digest_sha256") == digest:
                print(f"이미 구축 완료된 컬렉션입니다: {args.collection} ({collection.count():,}개)")
                return
        if not args.rebuild:
            raise SystemExit(
                "동일 이름의 불완전하거나 입력이 다른 컬렉션이 있습니다. "
                "확인 후 --rebuild로 다시 구축하세요."
            )
    collection = create_collection(client, args)

    started = time.time()
    for start in range(0, len(rows), args.batch_size):
        end = min(start + args.batch_size, len(rows))
        batch_vectors = np.asarray(vectors[start:end], dtype=np.float32)
        collection.add(
            ids=ids[start:end],
            embeddings=batch_vectors.tolist(),
            documents=documents[start:end],
            metadatas=[chroma_metadata(row) for row in rows[start:end]],
        )
        if end % (args.batch_size * 10) == 0 or end == len(rows):
            elapsed = max(time.time() - started, 1e-9)
            print(f"Chroma 적재 {end:,}/{len(rows):,} | {end / elapsed:.1f} chunks/s", flush=True)

    stored_count = collection.count()
    if stored_count != len(rows):
        raise SystemExit(f"적재 개수 불일치: Chroma={stored_count:,}, input={len(rows):,}")
    verification = verify_samples(collection, ids, vectors)
    state = {
        "script_version": SCRIPT_VERSION,
        "security": "회의록과 임베딩은 로컬 PersistentClient에만 저장",
        "collection": args.collection,
        "chroma_dir": str(chroma_dir),
        "metadata": metadata_signature,
        "embeddings": embedding_signature,
        "corpus_digest_sha256": digest,
        "row_count": len(rows),
        "dimension": int(vectors.shape[1]),
        "distance": "cosine",
        "model_name": "dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        "model_revision": "55ec6e9",
        "hnsw": {
            "ef_construction": args.ef_construction,
            "ef_search": args.ef_search,
            "max_neighbors": args.max_neighbors,
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "verification": verification,
    }
    atomic_write_json(state_path, state)
    print("\nChroma 구축 완료")
    print(f"컬렉션: {args.collection}")
    print(f"저장 개수: {stored_count:,}")
    print(f"검증 최대 임베딩 오차: {verification['max_abs_embedding_error']:.8f}")
    print(f"상태 파일: {state_path}")


if __name__ == "__main__":
    main()

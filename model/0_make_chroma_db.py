"""Preflight and index all configured transcript/metadata pairs into Chroma."""
import csv
import json

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

from _constant import (CHROMA_PATH, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME,
                    DATASETS, EMBEDDING_MODEL, INSERT_BATCH_SIZE)

TSV_FIELDS = ["date", "assembly", "agenda_title", "speaker_role", "speaker_name",
              "speaker_type", "utterance"]


def load_metadata(path):
    by_id = {}
    with path.open("r", encoding="utf-8-sig") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            row_id = item.get("row_id")
            if not isinstance(row_id, int) or row_id < 1 or row_id in by_id:
                raise ValueError(f"{path}:{line_no}: row_id가 유효하지 않거나 중복입니다: {row_id}")
            by_id[row_id] = item
    if not by_id:
        raise ValueError(f"메타데이터가 비어 있습니다: {path}")
    return by_id


def read_dataset(dataset):
    """Read and validate before any persistent Chroma mutation."""
    key, metadata_path, transcript_path = dataset["key"], dataset["metadata_path"], dataset["transcript_path"]
    if not key or not metadata_path.is_file() or not transcript_path.is_file():
        raise FileNotFoundError(f"dataset={key}: 입력 파일 경로를 확인하세요 ({metadata_path}, {transcript_path})")
    by_id = load_metadata(metadata_path)
    rows, found = [], set()
    with transcript_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames != TSV_FIELDS:
            raise ValueError(f"{transcript_path}: TXT 헤더 불일치. expected={TSV_FIELDS}, actual={reader.fieldnames}")
        row_id = 0
        for row in reader:
            if not row or not any(str(v or "").strip() for v in row.values()):
                continue
            row_id += 1
            if None in row or any(row.get(field) is None for field in TSV_FIELDS):
                raise ValueError(f"{transcript_path}: row_id={row_id}, 열 개수가 다릅니다")
            item = by_id.get(row_id)
            if item is not None:
                found.add(row_id)
                for field in ("date", "assembly"):
                    if item.get(field) and str(item[field]) != str(row[field]):
                        raise ValueError(f"dataset={key} row_id={row_id}: {field} 불일치")
            rows.append((row_id, row, item))
    missing = sorted(set(by_id) - found)
    if missing:
        raise ValueError(f"dataset={key}: JSONL row_id {len(missing)}건이 TXT에 없습니다. 예시={missing[:10]}")
    return by_id, rows


def scalar_metadata(item, row, dataset_key):
    merged = {**item, **{k: row[k] for k in TSV_FIELDS if k != "utterance"}, "dataset_key": dataset_key}
    return {k: v if isinstance(v, (str, int, float, bool)) else json.dumps(v, ensure_ascii=False)
            for k, v in merged.items() if v is not None}


def make_chunk_id(dataset_key, row, row_id, chunk_index):
    """Dataset key separates councils/years; chunk index keeps split IDs unique."""
    return f"{dataset_key}__row_{row_id}__chunk_{chunk_index}"


def main():
    keys = [d.get("key") for d in DATASETS]
    if not keys or any(not k for k in keys) or len(set(keys)) != len(keys):
        raise ValueError("DATASETS의 key는 비어 있지 않은 고유값이어야 합니다")
    prepared = [(d, *read_dataset(d)) for d in DATASETS]  # all preflight before DB changes
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    embedding = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embedding, metadata={"hnsw:space": "cosine"})
    for dataset, by_id, rows in prepared:
        key = dataset["key"]
        existing = collection.get(where={"dataset_key": key}, include=["metadatas"])
        old_ids = set(existing["ids"])
        docs, metas, ids, expected = [], [], [], set()
        indexed_rows = skipped = 0

        def flush():
            if ids:
                collection.upsert(ids=ids, documents=docs, metadatas=metas)
                docs.clear(); metas.clear(); ids.clear()

        for row_id, row, item in rows:
            utterance = row["utterance"].strip()
            if not utterance or item is None:
                skipped += 1
                continue
            metadata = scalar_metadata(item, row, key)
            for chunk_no, chunk in enumerate(splitter.split_text(utterance)):
                chunk_id = make_chunk_id(key, row, row_id, chunk_no)
                docs.append(chunk); metas.append({**metadata, "chunk_index": chunk_no}); ids.append(chunk_id)
                expected.add(chunk_id)
                if len(ids) >= INSERT_BATCH_SIZE:
                    flush()
            indexed_rows += 1
        flush()
        stale = old_ids - expected
        if stale:
            stale_list = list(stale)
            for start in range(0, len(stale_list), 5000):
                collection.delete(ids=stale_list[start:start + 5000])
        print(f"{key}: metadata={len(by_id):,}, indexed_rows={indexed_rows:,}, stale_deleted={len(stale):,}, unmatched_or_empty={skipped:,}")
    print(f"Chroma 저장 완료: {CHROMA_PATH} / {COLLECTION_NAME}")


if __name__ == "__main__":
    main()

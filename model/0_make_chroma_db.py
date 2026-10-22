import csv
import json
import numpy as np
import torch

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from tqdm import tqdm

from _constant import (CHROMA_PATH, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME,
                    DATASETS, EMBEDDING_MODEL, INSERT_BATCH_SIZE)

LEGACY_TSV_FIELDS = ["date", "assembly", "agenda_title", "speaker_role", "speaker_name",
                     "speaker_type", "utterance"]
TSV_FIELDS = ["MTG_DE", "RASMBLY_NM", "MTGNM", "MINTS_NM", "agenda_title",
              "speaker_role", "speaker_name", "speaker_type", "utterance"]

EMBEDDING_BATCH_SIZE = 64
CHROMA_WRITE_BATCH_SIZE = 64


class BatchedSentenceTransformerEmbeddingFunction(
    embedding_functions.SentenceTransformerEmbeddingFunction
):
    """Use a configurable encode batch size with the Chroma SentenceTransformer model."""
    def __call__(self, input):
        vectors = self._model.encode(
            list(input),
            batch_size=EMBEDDING_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return [np.asarray(vector, dtype=np.float32) for vector in vectors]


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
        if reader.fieldnames not in (TSV_FIELDS, LEGACY_TSV_FIELDS):
            raise ValueError(
                f"{transcript_path}: TXT 헤더 불일치. expected={TSV_FIELDS} 또는 {LEGACY_TSV_FIELDS}, "
                f"actual={reader.fieldnames}"
            )
        fields = reader.fieldnames
        row_id = 0
        for row in reader:
            if not row or not any(str(v or "").strip() for v in row.values()):
                continue
            row_id += 1
            if None in row or any(row.get(field) is None for field in fields):
                raise ValueError(f"{transcript_path}: row_id={row_id}, 열 개수가 다릅니다")
            item = by_id.get(row_id)
            if item is not None:
                found.add(row_id)
                for metadata_field, transcript_field in (
                    ("date", "date"), ("assembly", "assembly"),
                    ("MTG_DE", "MTG_DE"), ("RASMBLY_NM", "RASMBLY_NM"),
                    ("MTGNM", "MTGNM"), ("MINTS_NM", "MINTS_NM"),
                ):
                    if transcript_field in fields and item.get(metadata_field) and str(item[metadata_field]) != str(row[transcript_field]):
                        raise ValueError(f"dataset={key} row_id={row_id}: {transcript_field} 불일치")
            rows.append((row_id, row, item))
    missing = sorted(set(by_id) - found)
    if missing:
        raise ValueError(f"dataset={key}: JSONL row_id {len(missing)}건이 TXT에 없습니다. 예시={missing[:10]}")
    return by_id, rows


def scalar_metadata(item, row, dataset_key):
    # These JSONL fields duplicate the transcript columns under different names:
    # date/MTG_DE, assembly/RASMBLY_NM, agenda/agenda_title, speaker/speaker_role+speaker_name.
    duplicate_fields = {"date", "assembly", "agenda", "speaker"}
    merged = {
        **{k: v for k, v in item.items() if k not in duplicate_fields},
        **{k: v for k, v in row.items() if k != "utterance"},
        "dataset_key": dataset_key,
    }
    # Normalize legacy transcript/JSONL names to the preferred meeting metadata keys.
    if "MTG_DE" not in merged and item.get("date") is not None:
        merged["MTG_DE"] = item["date"]
    if "RASMBLY_NM" not in merged and item.get("assembly") is not None:
        merged["RASMBLY_NM"] = item["assembly"]
    return {k: v if isinstance(v, (str, int, float, bool)) else json.dumps(v, ensure_ascii=False)
            for k, v in merged.items() if v is not None}


def make_chunk_id(dataset_key, row, row_id, chunk_index):
    """Dataset key separates councils/years; chunk index keeps split IDs unique."""
    return f"{dataset_key}__row_{row_id}__chunk_{chunk_index}"


def main():
    start_time = time.perf_counter()
    keys = [dataset.get("key") for dataset in DATASETS]
    if not keys or any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("DATASETS의 key는 비어 있지 않은 고유값이어야 합니다")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"임베딩 장치: {torch.cuda.get_device_name(0)} (batch_size={EMBEDDING_BATCH_SIZE})")
    else:
        print("임베딩 장치: CPU (CUDA 사용 불가)")
    embedding = BatchedSentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL, device=device
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embedding, metadata={"hnsw:space": "cosine"})

    file_progress = tqdm(
        total=len(DATASETS), desc="전체 파일", unit="파일", dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )
    for file_index, dataset in enumerate(DATASETS, 1):
        file_start = time.perf_counter()
        key = dataset["key"]
        metadata_path = dataset["metadata_path"]
        transcript_path = dataset["transcript_path"]
        tqdm.write(f"[{file_index}/{len(DATASETS)}] 파일 시작: {transcript_path.name}")

        # Load and validate one pair at a time to keep memory bounded for folder inputs.
        by_id, rows = read_dataset(dataset)
        row_progress = tqdm(
            total=len(rows), desc=f"{file_index}/{len(DATASETS)} {transcript_path.stem}",
            unit="행", dynamic_ncols=True, leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        existing = collection.get(where={"dataset_key": key}, include=["metadatas"])
        old_ids = set(existing["ids"])
        docs, metas, ids, expected = [], [], [], set()
        indexed_rows = skipped = 0

        def flush():
            if ids:
                # Keep SQLite statements small to avoid "too many SQL variables".
                for offset in range(0, len(ids), CHROMA_WRITE_BATCH_SIZE):
                    end = offset + CHROMA_WRITE_BATCH_SIZE
                    collection.upsert(
                        ids=ids[offset:end],
                        documents=docs[offset:end],
                        metadatas=metas[offset:end],
                    )
                docs.clear(); metas.clear(); ids.clear()

        for row_id, row, item in rows:
            utterance = row["utterance"].strip()
            if not utterance or item is None:
                skipped += 1
                row_progress.update(1)
                continue
            metadata = scalar_metadata(item, row, key)
            for chunk_no, chunk in enumerate(splitter.split_text(utterance)):
                chunk_id = make_chunk_id(key, row, row_id, chunk_no)
                docs.append(chunk); metas.append({**metadata, "chunk_index": chunk_no}); ids.append(chunk_id)
                expected.add(chunk_id)
                if len(ids) >= INSERT_BATCH_SIZE:
                    flush()
            indexed_rows += 1
            row_progress.update(1)
            row_progress.set_postfix(chunks=len(expected), refresh=False)
        flush()
        row_progress.close()

        stale = old_ids - expected
        if stale:
            stale_list = list(stale)
            for offset in range(0, len(stale_list), 500):
                collection.delete(ids=stale_list[offset:offset + 500])

        elapsed = time.perf_counter() - file_start
        tqdm.write(
            f"파일 완료: {transcript_path.name} | metadata={len(by_id):,}, "
            f"indexed_rows={indexed_rows:,}, stale_deleted={len(stale):,}, "
            f"unmatched_or_empty={skipped:,}, 걸린 시간={elapsed:.1f}초"
        )
        file_progress.update(1)

    file_progress.close()
    total_seconds = int(round(time.perf_counter() - start_time, 0))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"전체 걸린 시간 : {hours}시 {minutes}분 {seconds}초")
    print(f"Chroma 저장 완료: {CHROMA_PATH} / {COLLECTION_NAME}")

if __name__ == "__main__":
    main()


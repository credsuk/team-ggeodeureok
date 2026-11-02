from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "data" / "metadata_jsonl_python"
DEFAULT_CHROMA_DIR = ROOT / "data" / "chroma_db"
DEFAULT_COLLECTION = "assembly_metadata"
DEFAULT_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"


def make_record_id(source_name: str, row_id: int) -> str:
    """Return a stable ID shared with the BM25 index and org_jsonl data."""
    return f"{Path(source_name).stem}:{row_id}"


def to_chroma_metadata(record: dict[str, Any], source_name: str) -> dict[str, Any]:
    """Convert JSONL fields to Chroma-supported scalar metadata values."""
    metadata: dict[str, Any] = {
        "source_file": f"{Path(source_name).stem}.txt",
        "source_jsonl": Path(source_name).name,
    }
    for key, value in record.items():
        if key == "summary" or value is None:
            continue
        if isinstance(value, bool):
            metadata[key] = value
        elif isinstance(value, int):
            metadata[key] = value
        elif isinstance(value, float):
            if math.isfinite(value):
                metadata[key] = value
        elif isinstance(value, (list, dict)):
            metadata[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            metadata[key] = str(value)
    return metadata


def prepare_record(record: dict[str, Any], source_name: str) -> tuple[str, str, dict[str, Any]]:
    row_id = record.get("row_id")
    if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id < 1:
        raise ValueError(f"{source_name}: row_id는 1 이상의 정수여야 합니다: {row_id!r}")
    summary = record.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError(f"{source_name} row_id={row_id}: 비어 있지 않은 summary가 필요합니다.")
    return make_record_id(source_name, row_id), summary.strip(), to_chroma_metadata(record, source_name)


def iter_jsonl_records(paths: Iterable[Path]):
    for path in paths:
        expected_row_id = 0
        with path.open("r", encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSON 오류: {path}:{line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"JSON 객체가 아닙니다: {path}:{line_number}")
                prepared = prepare_record(record, path.name)
                expected_row_id += 1
                if record["row_id"] != expected_row_id:
                    raise ValueError(
                        f"{path.name}: JSONL 레코드 순번 {expected_row_id}와 "
                        f"row_id={record['row_id']}가 일치하지 않습니다."
                    )
                yield prepared


def batched(items: Iterable[Any], size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_cuda_model(model_name: str):
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "필요 패키지가 없습니다. PyTorch(CUDA 빌드), sentence-transformers, chromadb를 "
            "프로젝트 가상환경에 설치하세요."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU를 사용할 수 없습니다. CPU로 자동 전환하지 않도록 중단했습니다. "
            "NVIDIA 드라이버와 PyTorch CUDA 빌드를 확인하세요."
        )
    print(f"GPU 사용: {torch.cuda.get_device_name(0)}", flush=True)
    return SentenceTransformer(model_name, device="cuda")


def ingest(input_dir: Path, chroma_dir: Path, collection_name: str, model_name: str,
           gpu_batch_size: int = 8, write_batch_size: int = 256,
           max_records: int | None = None) -> int:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("chromadb가 없습니다. 프로젝트 가상환경에 설치하세요.") from exc

    paths = sorted(input_dir.rglob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"입력 JSONL이 없습니다: {input_dir}")
    model = load_cuda_model(model_name)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "embedding_model": model_name},
    )
    existing_model = (collection.metadata or {}).get("embedding_model")
    if existing_model != model_name:
        raise RuntimeError(
            f"컬렉션 임베딩 모델 불일치: 현재={existing_model!r}, 요청={model_name!r}. "
            "다른 collection 이름을 사용하거나, 기존 DB를 별도로 백업·정리한 뒤 실행하세요."
        )

    total = 0
    records = iter_jsonl_records(paths)
    for batch in batched(records, write_batch_size):
        if max_records is not None:
            remaining = max_records - total
            if remaining <= 0:
                break
            batch = batch[:remaining]
        if not batch:
            break
        ids, documents, metadatas = zip(*batch)
        vectors = model.encode(
            list(documents),
            batch_size=gpu_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            device="cuda",
        )
        collection.upsert(
            ids=list(ids),
            documents=list(documents),
            metadatas=list(metadatas),
            embeddings=vectors.tolist(),
        )
        total += len(batch)
        if total % 10_000 < len(batch):
            print(f"적재 중: {total:,}건", flush=True)
        if max_records is not None and total >= max_records:
            break
    print(f"적재 완료: {total:,}건, collection={collection_name}, 경로={chroma_dir}")
    return total


def validate_only(input_dir: Path, max_records: int | None = None) -> int:
    paths = sorted(input_dir.rglob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"입력 JSONL이 없습니다: {input_dir}")
    count = 0
    for _ in iter_jsonl_records(paths):
        count += 1
        if max_records is not None and count >= max_records:
            break
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="JSONL 메타데이터의 summary를 CUDA 임베딩해 로컬 ChromaDB에 적재합니다."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gpu-batch-size", type=int, default=8)
    parser.add_argument("--write-batch-size", type=int, default=256)
    parser.add_argument("--max-records", type=int, default=None,help="테스트 적재용 상한. 일부 레코드만 DB에 저장합니다.")
    parser.add_argument("--dry-run", action="store_true",help="입력 JSONL만 검증하고 Chroma/GPU에는 연결하지 않습니다.")
    args = parser.parse_args()
    if args.gpu_batch_size < 1 or args.write_batch_size < 1:
        parser.error("batch size는 1 이상이어야 합니다.")
    if args.max_records is not None and args.max_records < 1:
        parser.error("--max-records는 1 이상이어야 합니다.")
    try:
        if args.dry_run:
            count = validate_only(args.input_dir, args.max_records)
            print(f"검증 완료: {count:,}건 (DB 변경 없음)")
        else:
            ingest(args.input_dir, args.chroma_dir, args.collection, args.model,
                   args.gpu_batch_size, args.write_batch_size, args.max_records)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

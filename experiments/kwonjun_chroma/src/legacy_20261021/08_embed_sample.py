#!/usr/bin/env python3
"""층화 표본을 Arctic 한국어 모델로 한 번만 임베딩해 두 DB가 공유한다.

벡터 행 순서는 embedding_ids.parquet의 row_index와 고정된다. 중간 상태를
저장하므로 동일 명령으로 재실행하면 완료 지점부터 이어서 계산한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
DEFAULT_REVISION = "55ec6e9"
EMBEDDING_SANITIZATION = "U+FFFD replacement character -> ASCII space; evidence text unchanged"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "canonical" / "embedding_sample_100k.parquet",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, help="앞에서부터 지정한 행만 smoke test")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=PROJECT_ROOT / "models" / "huggingface",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--compute-dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--read-batch-size", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda를 요청했지만 torch.cuda.is_available()이 False입니다.")
    return requested


def choose_output_dir(input_path: Path, selected_rows: int, supplied: Path | None) -> Path:
    if supplied:
        return supplied.expanduser().resolve()
    suffix = f"{selected_rows // 1000}k" if selected_rows % 1000 == 0 else str(selected_rows)
    return PROJECT_ROOT / "data" / "embeddings" / f"arctic_sample_{suffix}"


def build_id_map(corpus: Path, output: Path, row_count: int, batch_size: int) -> str:
    schema = pa.schema([("row_index", pa.int64()), ("chunk_id", pa.string())])
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        temporary.unlink()
    writer = pq.ParquetWriter(temporary, schema=schema, compression="zstd")
    digest = hashlib.sha256()
    offset = 0
    try:
        for batch in pq.ParquetFile(corpus).iter_batches(columns=["chunk_id"], batch_size=batch_size):
            if offset >= row_count:
                break
            ids = [str(value) for value in batch.column(0).to_pylist()]
            ids = ids[: row_count - offset]
            if not ids:
                break
            for chunk_id in ids:
                digest.update(chunk_id.encode("utf-8"))
                digest.update(b"\n")
            table = pa.table({
                "row_index": pa.array(range(offset, offset + len(ids)), type=pa.int64()),
                "chunk_id": pa.array(ids, type=pa.string()),
            }, schema=schema)
            writer.write_table(table)
            offset += len(ids)
    except Exception:
        writer.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        writer.close()
    if offset != row_count:
        temporary.unlink()
        raise RuntimeError(f"ID 순서표 행 수 오류: expected={row_count}, actual={offset}")
    temporary.replace(output)
    return digest.hexdigest()


def validate_resume_state(
    state: dict,
    input_sha256: str,
    row_count: int,
    dimension: int,
    args: argparse.Namespace,
) -> None:
    expected = {
        "input_sha256": input_sha256,
        "row_count": row_count,
        "shape": [row_count, dimension],
        "model_name": args.model_name,
        "revision": args.revision,
        "max_seq_length": args.max_seq_length,
        "output_dtype": "float32",
        "embedding_sanitization": EMBEDDING_SANITIZATION,
    }
    mismatches = {
        key: {"expected": value, "actual": state.get(key)}
        for key, value in expected.items() if state.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "기존 임베딩 상태가 현재 설정과 다릅니다. 별도 --output-dir 또는 "
            f"--restart를 사용하세요: {mismatches}"
        )


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit은 1 이상이어야 합니다.")
    if args.batch_size < 1 or args.read_batch_size < 1 or args.max_seq_length < 1:
        raise ValueError("batch-size/read-batch-size/max-seq-length는 1 이상이어야 합니다.")

    parquet = pq.ParquetFile(input_path)
    missing = sorted({"chunk_id", "embedding_text"} - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"임베딩 입력에 필요한 열이 없습니다: {missing}")
    total_rows = int(parquet.metadata.num_rows)
    row_count = min(total_rows, args.limit) if args.limit else total_rows
    output_dir = choose_output_dir(input_path, row_count, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = output_dir / "embeddings.f32.npy"
    ids_path = output_dir / "embedding_ids.parquet"
    state_path = output_dir / "embedding_state.json"
    manifest_path = output_dir / "embedding_manifest.json"
    input_sha256 = file_sha256(input_path)

    device = choose_device(args.device)
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None
    print(f"입력: {input_path.name} | 선택 행: {row_count:,}/{total_rows:,}")
    print(f"모델 로드: {args.model_name} | device={device} | gpu={gpu_name}")
    model = SentenceTransformer(
        args.model_name,
        revision=args.revision or None,
        cache_folder=str(args.model_cache.expanduser().resolve()),
        device=device,
        local_files_only=args.local_files_only,
    )
    model.max_seq_length = args.max_seq_length
    if device == "cuda" and args.compute_dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("현재 GPU는 bf16을 지원하지 않습니다. --compute-dtype fp16을 사용하세요.")
        model.bfloat16()
    elif device == "cuda" and args.compute_dtype == "fp16":
        model.half()
    dimension = int(model.get_sentence_embedding_dimension())

    print("chunk_id 순서표 생성")
    ids_sha256 = build_id_map(input_path, ids_path, row_count, args.read_batch_size)

    rows_completed = 0
    previous_elapsed = 0.0
    sanitized_replacement_chars = 0
    if state_path.exists() and vectors_path.exists() and not args.restart:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        validate_resume_state(state, input_sha256, row_count, dimension, args)
        rows_completed = int(state.get("rows_completed", 0))
        previous_elapsed = float(state.get("elapsed_seconds_total", 0.0))
        sanitized_replacement_chars = int(state.get("sanitized_replacement_chars", 0))
        if not 0 <= rows_completed <= row_count:
            raise RuntimeError(f"잘못된 rows_completed 값입니다: {rows_completed}")
        vectors = np.load(vectors_path, mmap_mode="r+")
        if tuple(vectors.shape) != (row_count, dimension) or vectors.dtype != np.float32:
            raise RuntimeError("기존 벡터 파일의 shape 또는 dtype이 현재 설정과 다릅니다.")
        print(f"이어하기: {rows_completed:,}/{row_count:,}")
    else:
        vectors = np.lib.format.open_memmap(
            vectors_path, mode="w+", dtype=np.float32, shape=(row_count, dimension)
        )

    started = time.perf_counter()
    global_offset = 0
    for batch in parquet.iter_batches(columns=["embedding_text"], batch_size=args.read_batch_size):
        if global_offset >= row_count:
            break
        texts = ["" if value is None else str(value) for value in batch.column(0).to_pylist()]
        texts = texts[: row_count - global_offset]
        batch_start = global_offset
        batch_end = batch_start + len(texts)
        global_offset = batch_end
        if batch_end <= rows_completed:
            continue
        if rows_completed > batch_start:
            texts = texts[rows_completed - batch_start :]
            batch_start = rows_completed
        if not texts:
            continue
        sanitized_replacement_chars += sum(text.count("\ufffd") for text in texts)
        texts = [text.replace("\ufffd", " ") for text in texts]

        encoded = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)
        if encoded.shape != (len(texts), dimension):
            raise RuntimeError(
                f"모델 출력 shape 오류: expected={(len(texts), dimension)}, actual={encoded.shape}"
            )
        vectors[batch_start:batch_end] = encoded
        vectors.flush()
        rows_completed = batch_end
        elapsed_this_run = time.perf_counter() - started
        elapsed_total = previous_elapsed + elapsed_this_run
        throughput = rows_completed / elapsed_total if elapsed_total else 0.0
        state = {
            "input_file": input_path.name,
            "input_sha256": input_sha256,
            "source_total_rows": total_rows,
            "row_count": row_count,
            "rows_completed": rows_completed,
            "shape": [row_count, dimension],
            "model_name": args.model_name,
            "revision": args.revision,
            "model_cache": str(args.model_cache.expanduser().resolve()),
            "device": device,
            "gpu_name": gpu_name,
            "compute_dtype": args.compute_dtype,
            "output_dtype": "float32",
            "max_seq_length": args.max_seq_length,
            "normalized": True,
            "document_prompt": None,
            "ids_sha256": ids_sha256,
            "elapsed_seconds_total": round(elapsed_total, 3),
            "throughput_rows_per_second": round(throughput, 3),
            "raw_data_external_transfer": False,
            "embedding_sanitization": EMBEDDING_SANITIZATION,
            "sanitized_replacement_chars": sanitized_replacement_chars,
        }
        write_json(state_path, state)
        print(
            f"임베딩: {rows_completed:,}/{row_count:,} "
            f"({rows_completed / row_count * 100:.1f}%) | {throughput:.2f} rows/s"
        )

    if rows_completed != row_count:
        raise RuntimeError(f"임베딩이 완성되지 않았습니다: {rows_completed:,}/{row_count:,}")

    sample_count = min(row_count, 10_000)
    norms = np.linalg.norm(np.asarray(vectors[:sample_count]), axis=1)
    if not np.isfinite(norms).all():
        raise RuntimeError("벡터 norm 검사에서 NaN 또는 Inf가 발견되었습니다.")
    manifest = {
        **state,
        "complete": True,
        "vectors_file": vectors_path.name,
        "vectors_size_bytes": vectors_path.stat().st_size,
        "vectors_sha256": file_sha256(vectors_path),
        "ids_file": ids_path.name,
        "ids_file_sha256": file_sha256(ids_path),
        "norm_check_rows": sample_count,
        "sample_norm_mean": round(float(norms.mean()), 8),
        "sample_norm_min": round(float(norms.min()), 8),
        "sample_norm_max": round(float(norms.max()), 8),
        "database_policy": "ChromaDB와 pgvector가 이 동일 float32 배열을 사용",
        "metadata_embedded": False,
        "embedding_input": "embedding_text only; U+FFFD is replaced with a space at inference time",
    }
    write_json(manifest_path, manifest)

    print("\n=== Arctic 공용 임베딩 완료 ===")
    print(f"shape: {(row_count, dimension)} | dtype=float32 | normalized=True")
    print(f"norm mean/min/max: {manifest['sample_norm_mean']} / {manifest['sample_norm_min']} / {manifest['sample_norm_max']}")
    print(f"vectors: {vectors_path}")
    print(f"ids: {ids_path}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""전체 검색 청크를 메모리 안전한 스트리밍 방식으로 임베딩한다."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterator

import duckdb
import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
DEFAULT_REVISION = "55ec6e9358a56d56af759bc8372e970caf8c305f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=PROJECT_ROOT / "data" / "canonical_v3" / "chunks_v1" / "chunks.parquet",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data" / "embeddings" / "arctic_raw_v1",
    )
    parser.add_argument("--text-view", choices=["raw", "enriched"], default="raw")
    parser.add_argument(
        "--priorities", nargs="+", choices=["high", "medium", "low"],
        default=["high", "medium", "low"],
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--model-cache", type=Path,
        default=PROJECT_ROOT / "models" / "huggingface",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--compute-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--read-batch-rows", type=int, default=2048)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--state-every-rows", type=int, default=2000)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def quote_sql(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare_manifest(input_path: Path, manifest_path: Path, text_view: str, priorities: list[str]) -> None:
    text_column = f"embedding_text_{text_view}"
    priority_sql = ", ".join(quote_sql(value) for value in priorities)
    source_sql = quote_sql(str(input_path))
    output_sql = quote_sql(str(manifest_path))
    con = duckdb.connect()
    con.execute(f"""
        COPY (
          SELECT
            * EXCLUDE (embedding_text_raw, embedding_text_enriched),
            {text_column} AS embedding_text
          FROM read_parquet({source_sql})
          WHERE priority IN ({priority_sql})
          ORDER BY source_key, document_id, parent_index, child_index
        ) TO {output_sql} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 10000)
    """)
    con.close()


def manifest_digest(path: Path) -> tuple[int, str]:
    parquet = pq.ParquetFile(path)
    hasher = hashlib.sha256()
    rows = 0
    for batch in parquet.iter_batches(batch_size=16_384, columns=["chunk_id"]):
        for value in batch.column(0).to_pylist():
            hasher.update(str(value).encode("utf-8"))
            hasher.update(b"\n")
            rows += 1
    return rows, hasher.hexdigest()


def iter_text_batches(path: Path, read_batch_rows: int, skip_rows: int) -> Iterator[list[str]]:
    parquet = pq.ParquetFile(path)
    seen = 0
    for batch in parquet.iter_batches(batch_size=read_batch_rows, columns=["embedding_text"]):
        values = [str(value or "").strip() for value in batch.column(0).to_pylist()]
        batch_end = seen + len(values)
        if batch_end <= skip_rows:
            seen = batch_end
            continue
        start = max(0, skip_rows - seen)
        selected = values[start:]
        if any(not value for value in selected):
            raise RuntimeError(f"빈 embedding_text 발견: manifest row {seen + start:,}")
        if selected:
            yield selected
        seen = batch_end


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 사용할 수 없습니다")
    return requested


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if device == "cpu" or name == "fp32":
        return torch.float32
    if name == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU가 bf16을 지원하지 않습니다. fp16을 사용하세요")
        return torch.bfloat16
    return torch.float16


def load_model(args: argparse.Namespace, device: str, dtype: torch.dtype) -> SentenceTransformer:
    kwargs: dict[str, Any] = {
        "cache_folder": str(args.model_cache.expanduser().resolve()),
        "local_files_only": args.local_files_only,
        "device": device,
    }
    if args.revision:
        kwargs["revision"] = args.revision
    model = SentenceTransformer(args.model_name, **kwargs)
    model.max_seq_length = args.max_seq_length
    if dtype == torch.bfloat16:
        model = model.bfloat16()
    elif dtype == torch.float16:
        model = model.half()
    else:
        model = model.float()
    model.eval()
    return model


def model_revision(model: SentenceTransformer, fallback: str | None) -> str | None:
    try:
        return str(model._first_module().auto_model.config._commit_hash or fallback)
    except (AttributeError, IndexError):
        return fallback


def norm_stats(array: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(array.astype(np.float32), axis=1)
    return {
        "mean": round(float(norms.mean()), 6),
        "min": round(float(norms.min()), 6),
        "max": round(float(norms.max()), 6),
    }


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    args.model_cache = args.model_cache.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if args.batch_size < 1 or args.read_batch_rows < args.batch_size:
        raise ValueError("batch-size는 1 이상, read-batch-rows는 batch-size 이상이어야 합니다")
    output_dir.mkdir(parents=True, exist_ok=True)
    args.model_cache.mkdir(parents=True, exist_ok=True)

    manifest = output_dir / "chunk_metadata.parquet"
    partial_vectors = output_dir / "embeddings.partial.f16.npy"
    final_vectors = output_dir / "embeddings.f16.npy"
    state_path = output_dir / "embedding_state.json"
    summary_path = output_dir / "embedding_summary.json"
    if args.restart:
        for path in (manifest, partial_vectors, final_vectors, state_path, summary_path):
            if path.exists():
                path.unlink()
    elif final_vectors.exists() and summary_path.exists():
        print(f"이미 완료된 결과입니다: {output_dir}")
        return

    if not manifest.exists():
        print(f"manifest 생성: priorities={args.priorities} | text={args.text_view}", flush=True)
        prepare_manifest(input_path, manifest, args.text_view, args.priorities)
    selected_rows, selected_digest = manifest_digest(manifest)
    if selected_rows == 0:
        raise RuntimeError("선택된 청크가 없습니다")
    print(f"manifest: {selected_rows:,}행 | {manifest}", flush=True)
    if args.prepare_only:
        print("prepare-only 완료")
        return

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.compute_dtype, device)
    started = time.perf_counter()
    print(f"모델 로드: {args.model_name} | {device} | {dtype}", flush=True)
    model = load_model(args, device, dtype)
    revision = model_revision(model, args.revision)

    if state_path.exists() and partial_vectors.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        expected = {
            "selected_rows": selected_rows,
            "selected_digest_sha256": selected_digest,
            "model_name": args.model_name,
            "revision": revision,
            "text_view": args.text_view,
            "priorities": args.priorities,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise RuntimeError(f"재개 상태 불일치: {key}")
        completed = int(state["completed_rows"])
        dimension = int(state["dimension"])
        vectors = np.lib.format.open_memmap(
            partial_vectors, mode="r+", dtype=np.float16,
            shape=(selected_rows, dimension),
        )
        print(f"재개: {completed:,}/{selected_rows:,}", flush=True)
    else:
        first_read = next(iter_text_batches(manifest, args.read_batch_rows, 0))
        first_texts = first_read[: args.batch_size]
        encoded = model.encode(
            first_texts, batch_size=args.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        dimension = int(encoded.shape[1])
        vectors = np.lib.format.open_memmap(
            partial_vectors, mode="w+", dtype=np.float16,
            shape=(selected_rows, dimension),
        )
        vectors[: len(first_texts)] = encoded.astype(np.float16)
        vectors.flush()
        completed = len(first_texts)
        state = {
            "status": "running",
            "security": "모델 파일 다운로드 외 회의록 데이터 외부 전송 없음",
            "input": str(input_path),
            "manifest": str(manifest),
            "selected_rows": selected_rows,
            "selected_digest_sha256": selected_digest,
            "priorities": args.priorities,
            "model_name": args.model_name,
            "revision": revision,
            "text_view": args.text_view,
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "compute_dtype": args.compute_dtype,
            "batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length,
            "dimension": dimension,
            "normalized": True,
            "storage_dtype": "float16",
            "completed_rows": completed,
        }
        atomic_json(state_path, state)

    last_saved = completed
    for read_batch in iter_text_batches(manifest, args.read_batch_rows, completed):
        for offset in range(0, len(read_batch), args.batch_size):
            texts = read_batch[offset: offset + args.batch_size]
            end = completed + len(texts)
            encoded = model.encode(
                texts, batch_size=args.batch_size, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            )
            vectors[completed:end] = encoded.astype(np.float16)
            completed = end
            if completed - last_saved >= args.state_every_rows or completed == selected_rows:
                vectors.flush()
                elapsed = time.perf_counter() - started
                state["completed_rows"] = completed
                state["elapsed_seconds_current_session"] = round(elapsed, 3)
                state["throughput_rows_per_second_current_session"] = round(completed / elapsed, 3)
                atomic_json(state_path, state)
                print(
                    f"진행: {completed:,}/{selected_rows:,} ({completed / selected_rows:.1%}) | "
                    f"{state['throughput_rows_per_second_current_session']:.2f}행/초",
                    flush=True,
                )
                last_saved = completed

    vectors.flush()
    sample = np.asarray(vectors[: min(10_000, selected_rows)])
    norms = norm_stats(sample)
    del vectors
    gc.collect()
    partial_vectors.replace(final_vectors)
    elapsed = time.perf_counter() - started
    state.update({
        "status": "complete",
        "completed_rows": selected_rows,
        "elapsed_seconds_current_session": round(elapsed, 3),
        "throughput_rows_per_second_current_session": round(selected_rows / elapsed, 3),
        "norms_first_10000": norms,
        "vectors": str(final_vectors),
        "vector_file_size_bytes": final_vectors.stat().st_size,
    })
    atomic_json(state_path, state)
    atomic_json(summary_path, state)

    print("\n=== 전체 임베딩 완료 ===")
    print(f"행: {selected_rows:,} | 차원: {dimension} | text={args.text_view}")
    print(f"시간: {elapsed / 3600:.2f}시간 | 처리율: {selected_rows / elapsed:.2f}행/초")
    print(f"norm: {norms}")
    print(f"vectors: {final_vectors}")
    print(f"metadata: {manifest}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""평가 청크를 로컬 임베딩 모델로 벡터화하고 재현 가능한 상태 파일을 저장한다."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PRESETS = {
    "arctic": {
        "name": "dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        "revision": "55ec6e9",
        "query_prompt": "query",
    },
    "kure": {
        "name": "nlpai-lab/KURE-v1",
        "revision": None,
        "query_prompt": None,
    },
    "bge_m3": {
        "name": "BAAI/bge-m3",
        "revision": None,
        "query_prompt": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=PROJECT_ROOT / "data" / "embedding_eval" / "chunks_eval_60k.parquet",
    )
    parser.add_argument("--model-key", choices=sorted(MODEL_PRESETS), default="arctic")
    parser.add_argument("--model-name", help="preset 대신 사용할 Hugging Face 모델명")
    parser.add_argument("--revision", help="모델 revision/commit. 미지정 시 preset 값을 사용")
    parser.add_argument("--text-view", choices=["raw", "enriched"], default="raw")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--model-cache", type=Path,
        default=PROJECT_ROOT / "models" / "huggingface",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--compute-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--state-every", type=int, default=25)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def digest_ids(ids: list[str]) -> str:
    hasher = hashlib.sha256()
    for value in ids:
        hasher.update(value.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 torch.cuda.is_available()이 False입니다")
    return requested


def torch_dtype(name: str, device: str) -> torch.dtype:
    if name == "fp32" or device == "cpu":
        return torch.float32
    if name == "bf16":
        if device == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("현재 CUDA 장치가 bf16을 지원하지 않습니다. --compute-dtype fp16을 사용하세요")
        return torch.bfloat16
    return torch.float16


def load_model(
    model_name: str,
    revision: str | None,
    cache: Path,
    local_files_only: bool,
    device: str,
    dtype: torch.dtype,
    max_seq_length: int,
) -> SentenceTransformer:
    kwargs: dict[str, Any] = {
        "cache_folder": str(cache),
        "local_files_only": local_files_only,
        "device": device,
    }
    if revision:
        kwargs["revision"] = revision
    model = SentenceTransformer(model_name, **kwargs)
    model.max_seq_length = max_seq_length
    if dtype == torch.bfloat16:
        model = model.bfloat16()
    elif dtype == torch.float16:
        model = model.half()
    else:
        model = model.float()
    model.eval()
    return model


def resolved_revision(model: SentenceTransformer, requested: str | None) -> str | None:
    try:
        commit = model._first_module().auto_model.config._commit_hash
        return str(commit) if commit else requested
    except (AttributeError, IndexError):
        return requested


def norm_stats(array: np.ndarray) -> dict[str, float]:
    if len(array) == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    norms = np.linalg.norm(array.astype(np.float32), axis=1)
    return {
        "mean": round(float(norms.mean()), 6),
        "min": round(float(norms.min()), 6),
        "max": round(float(norms.max()), 6),
    }


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if args.batch_size < 1:
        raise ValueError("batch-size는 1 이상이어야 합니다")

    preset = MODEL_PRESETS[args.model_key]
    model_name = args.model_name or preset["name"]
    revision = args.revision or preset["revision"]
    text_column = f"embedding_text_{args.text_view}"
    suffix = f"{args.model_key}_{args.text_view}"
    if args.limit:
        suffix += f"_{args.limit}"
    output_dir = (args.output_dir or PROJECT_ROOT / "data" / "embedding_eval_vectors" / suffix).expanduser().resolve()
    cache = args.model_cache.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    final_vectors = output_dir / "embeddings.f16.npy"
    partial_vectors = output_dir / "embeddings.partial.f16.npy"
    metadata_path = output_dir / "chunk_metadata.parquet"
    state_path = output_dir / "embedding_state.json"
    summary_path = output_dir / "embedding_summary.json"
    if args.restart:
        for path in (final_vectors, partial_vectors, metadata_path, state_path, summary_path):
            if path.exists():
                path.unlink()
    elif final_vectors.exists() and summary_path.exists():
        print(f"이미 완료된 결과입니다: {output_dir}")
        return

    table = pq.read_table(input_path)
    if args.limit:
        table = table.slice(0, min(args.limit, table.num_rows))
    if text_column not in table.column_names:
        raise KeyError(f"입력에 {text_column} 열이 없습니다")
    ids = [str(value) for value in table["chunk_id"].to_pylist()]
    texts = [str(value or "").strip() for value in table[text_column].to_pylist()]
    if not ids or any(not text for text in texts):
        raise RuntimeError("임베딩 대상이 비어 있거나 빈 텍스트가 있습니다")
    if len(ids) != len(set(ids)):
        raise RuntimeError("chunk_id가 중복되었습니다")
    selection_digest = digest_ids(ids)
    if not metadata_path.exists():
        pq.write_table(table, metadata_path, compression="zstd")

    device = resolve_device(args.device)
    dtype = torch_dtype(args.compute_dtype, device)
    print(f"모델 로드: {model_name} | revision={revision or 'resolved-at-runtime'}", flush=True)
    print(f"장치: {device} | dtype={dtype} | text={text_column} | rows={len(ids):,}", flush=True)
    started = time.perf_counter()
    model = load_model(model_name, revision, cache, args.local_files_only, device, dtype, args.max_seq_length)
    actual_revision = resolved_revision(model, revision)

    state: dict[str, Any]
    vectors: np.memmap
    if state_path.exists() and partial_vectors.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        expected = {
            "model_name": model_name,
            "revision": actual_revision,
            "text_column": text_column,
            "selected_digest_sha256": selection_digest,
            "selected_rows": len(ids),
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise RuntimeError(f"재개 상태 불일치: {key} | state={state.get(key)!r}, current={value!r}")
        completed = int(state["completed_rows"])
        dimension = int(state["dimension"])
        vectors = np.lib.format.open_memmap(partial_vectors, mode="r+", dtype=np.float16, shape=(len(ids), dimension))
        print(f"재개: {completed:,}/{len(ids):,}", flush=True)
    else:
        first_end = min(args.batch_size, len(texts))
        first = model.encode(
            texts[:first_end], batch_size=args.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        dimension = int(first.shape[1])
        vectors = np.lib.format.open_memmap(partial_vectors, mode="w+", dtype=np.float16, shape=(len(ids), dimension))
        vectors[:first_end] = first.astype(np.float16)
        vectors.flush()
        completed = first_end
        state = {
            "status": "running",
            "security": "모델 파일 다운로드 외 회의록 데이터 외부 전송 없음",
            "input": str(input_path),
            "selected_rows": len(ids),
            "selected_digest_sha256": selection_digest,
            "model_key": args.model_key,
            "model_name": model_name,
            "revision": actual_revision,
            "text_view": args.text_view,
            "text_column": text_column,
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

    batch_number = 0
    for start in range(completed, len(texts), args.batch_size):
        end = min(start + args.batch_size, len(texts))
        encoded = model.encode(
            texts[start:end], batch_size=args.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        vectors[start:end] = encoded.astype(np.float16)
        completed = end
        batch_number += 1
        if batch_number % args.state_every == 0 or completed == len(texts):
            vectors.flush()
            state["completed_rows"] = completed
            state["elapsed_seconds_current_session"] = round(time.perf_counter() - started, 3)
            atomic_json(state_path, state)
            print(f"진행: {completed:,}/{len(texts):,} ({completed / len(texts):.1%})", flush=True)

    vectors.flush()
    sample = np.asarray(vectors[: min(10_000, len(vectors))])
    norms = norm_stats(sample)
    del vectors
    gc.collect()
    partial_vectors.replace(final_vectors)
    elapsed = time.perf_counter() - started
    state.update({
        "status": "complete",
        "completed_rows": len(ids),
        "elapsed_seconds_current_session": round(elapsed, 3),
        "throughput_rows_per_second": round(len(ids) / elapsed, 3) if elapsed else None,
        "norms_first_10000": norms,
        "vectors": str(final_vectors),
        "metadata": str(metadata_path),
    })
    atomic_json(state_path, state)
    atomic_json(summary_path, state)

    print("\n=== 평가 표본 임베딩 완료 ===")
    print(f"모델: {model_name} | revision={actual_revision}")
    print(f"텍스트: {args.text_view} | 행: {len(ids):,} | 차원: {state['dimension']}")
    print(f"시간: {elapsed:.1f}초 | 처리율: {state['throughput_rows_per_second']:.2f}행/초")
    print(f"벡터 norm: {norms}")
    print(f"vectors: {final_vectors}")
    print(f"metadata: {metadata_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

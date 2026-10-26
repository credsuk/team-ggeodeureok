#!/usr/bin/env python3
"""pgvector 후보를 로컬 cross-encoder로 재순위해 최종 근거를 선택한다.

입력 JSON은 18_pgvector_retrieval.py가 생성한 candidates를 포함해야 한다.
모델 파일 다운로드 외에는 회의록 원문을 외부로 전송하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

from council_vector_runtime import atomic_json, parent_contexts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--model-cache", type=Path,
        default=PROJECT_ROOT / "models" / "huggingface",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-per-parent", type=int, default=1)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--parent-path", type=Path,
        default=PROJECT_ROOT / "data" / "canonical_v3" / "chunks_v1" / "parents.parquet",
    )
    parser.add_argument("--no-parent", action="store_true")
    return parser.parse_args()


def validate_payload(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    query = str(payload.get("query") or "").strip()
    candidates = payload.get("candidates")
    if not query:
        raise ValueError("입력 JSON에 query가 없습니다")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("입력 JSON에 candidates가 없습니다")
    required = {"chunk_id", "text", "metadata"}
    for index, row in enumerate(candidates, start=1):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"후보 {index}의 필수 필드가 없습니다: {missing}")
    return query, candidates


def candidate_document(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    parts = []
    committee = str(metadata.get("committee") or "").strip()
    agenda = str(metadata.get("agenda_title") or "").strip()
    if committee:
        parts.append(f"[위원회] {committee}")
    if agenda:
        parts.append(f"[안건] {agenda}")
    parts.append(f"[발언] {str(row.get('text') or '').strip()}")
    return "\n".join(parts)


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def normalized_text_digest(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def select_diverse(
    reranked: list[dict[str, Any]], top_k: int, max_per_parent: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    parent_counts: dict[str, int] = {}
    skipped_text = 0
    skipped_parent = 0
    for row in reranked:
        digest = normalized_text_digest(str(row.get("text") or ""))
        if digest in seen_texts:
            skipped_text += 1
            continue
        metadata = row.get("metadata") or {}
        parent_id = str(metadata.get("parent_chunk_id") or "")
        if parent_id and parent_counts.get(parent_id, 0) >= max_per_parent:
            skipped_parent += 1
            continue
        seen_texts.add(digest)
        if parent_id:
            parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
        selected.append(row)
        if len(selected) >= top_k:
            break
    return selected, {
        "skipped_exact_text_duplicates": skipped_text,
        "skipped_parent_limit": skipped_parent,
    }


def main() -> None:
    args = parse_args()
    if min(args.batch_size, args.max_length, args.top_k, args.max_per_parent) < 1:
        raise ValueError("batch-size, max-length, top-k, max-per-parent는 1 이상이어야 합니다")

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    query, candidates = validate_payload(payload)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 torch.cuda.is_available()이 False입니다")

    load_kwargs: dict[str, Any] = {
        "cache_dir": str(args.model_cache.expanduser().resolve()),
        "local_files_only": not args.allow_download,
    }
    if args.revision:
        load_kwargs["revision"] = args.revision

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **load_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, **load_kwargs,
    )
    if device == "cuda":
        model = model.half()
    model = model.to(device)
    model.eval()
    model_load_ms = (time.perf_counter() - load_started) * 1000

    documents = [candidate_document(row) for row in candidates]
    scores: list[float] = []
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(documents), args.batch_size):
            batch_documents = documents[start:start + args.batch_size]
            pairs = [[query, document] for document in batch_documents]
            encoded = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded, return_dict=True).logits
            batch_scores = logits.view(-1).float().cpu().tolist()
            scores.extend(float(value) for value in batch_scores)
    inference_ms = (time.perf_counter() - inference_started) * 1000

    if len(scores) != len(candidates):
        raise RuntimeError(f"재순위 점수 개수 불일치: {len(scores)} != {len(candidates)}")

    reranked = []
    for row, score in zip(candidates, scores):
        current = dict(row)
        current["dense_rank"] = int(row.get("candidate_rank") or 0)
        current["dense_similarity"] = float(row.get("similarity") or 0.0)
        current["reranker_logit"] = round(score, 6)
        current["reranker_sigmoid"] = round(sigmoid(score), 6)
        reranked.append(current)
    reranked.sort(key=lambda row: row["reranker_logit"], reverse=True)
    for rank, row in enumerate(reranked, start=1):
        row["rerank_rank"] = rank

    selected, deduplication = select_diverse(
        reranked, args.top_k, args.max_per_parent,
    )
    contexts: dict[str, str] = {}
    parent_path = args.parent_path.expanduser().resolve()
    if not args.no_parent:
        if not parent_path.is_file():
            raise FileNotFoundError(f"부모 청크 파일이 없습니다: {parent_path}")
        contexts = parent_contexts(
            parent_path,
            [
                str((row.get("metadata") or {}).get("parent_chunk_id") or "")
                for row in selected
            ],
        )

    results = []
    for rank, row in enumerate(selected, start=1):
        result = dict(row)
        result["rank"] = rank
        parent_id = str((result.get("metadata") or {}).get("parent_chunk_id") or "")
        result["parent_context"] = contexts.get(parent_id)
        results.append(result)

    commit_hash = getattr(model.config, "_commit_hash", None)
    report = {
        "status": "complete",
        "query": query,
        "source_retrieval": {
            "input": str(input_path),
            "backend": payload.get("backend"),
            "database": payload.get("database"),
            "candidate_count": len(candidates),
        },
        "reranker": {
            "model_name": args.model_name,
            "revision_requested": args.revision,
            "revision_resolved": commit_hash,
            "device": device,
            "precision": "float16" if device == "cuda" else "float32",
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "security": "모델 파일 다운로드 외 회의록 데이터 외부 전송 없음",
        },
        "selection": {
            "top_k": args.top_k,
            "max_per_parent": args.max_per_parent,
            "deduplication": deduplication,
        },
        "timing_ms": {
            "model_load": round(model_load_ms, 3),
            "reranker_inference": round(inference_ms, 3),
        },
        "results": results,
        "reranked_candidates": reranked,
    }
    output = args.output.expanduser().resolve()
    atomic_json(output, report)

    print("\n=== 로컬 재순위 완료 ===")
    print(f"질의: {query}")
    print(f"모델: {args.model_name} | 장치: {device}")
    print(
        f"후보: {len(candidates)} | 최종: {len(results)} | "
        f"추론={inference_ms:.1f}ms"
    )
    print(
        "중복 제거: "
        f"동일본문={deduplication['skipped_exact_text_duplicates']} | "
        f"동일부모={deduplication['skipped_parent_limit']}"
    )
    for row in results:
        metadata = row.get("metadata") or {}
        print(
            f"\n[{row['rank']}] dense={row['dense_rank']}→rerank={row['rerank_rank']} "
            f"| score={row['reranker_logit']:.4f}"
        )
        print(
            f"{metadata.get('council')} | {metadata.get('meeting_date')} | "
            f"{metadata.get('meeting_name')}"
        )
        print(f"안건: {metadata.get('agenda_title')}")
        print(f"원문: {str(row.get('text') or '')[:500]}")
    print(f"\nJSON: {output}")


if __name__ == "__main__":
    main()

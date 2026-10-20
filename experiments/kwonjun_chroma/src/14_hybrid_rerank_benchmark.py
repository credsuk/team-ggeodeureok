#!/usr/bin/env python3
"""Dense + BM25 + RRF + CrossEncoder 검색 성능을 비교한다.

prepare
    Arctic와 KURE의 FAISS 정확 검색, BM25 검색을 RRF로 결합하고
    BGE reranker로 재정렬한다. 기존 임베딩 평가 라벨은 유지하며 새로 검색된
    후보만 retrieval_judgments_hybrid.csv에 빈 label로 추가한다.

evaluate
    완성된 라벨을 기준으로 Arctic, KURE, Dense RRF, BM25, Hybrid RRF,
    Hybrid Reranker의 Precision/Recall/MRR/nDCG를 동일 조건에서 비교한다.

원문은 로컬에서만 처리하며 외부 API로 전송하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path

try:
    import faiss
    import numpy as np
    import pyarrow.parquet as pq
except ImportError as error:
    raise SystemExit("faiss-cpu, numpy, pyarrow가 필요합니다.") from error

try:
    import torch
except ImportError:
    torch = None

try:
    from scipy import sparse
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError as error:
    raise SystemExit("scipy와 scikit-learn이 필요합니다.") from error

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ImportError as error:
    raise SystemExit("sentence-transformers가 필요합니다.") from error


SCRIPT_VERSION = "2026-10-20-hybrid-rerank-v1"

DENSE_MODELS = {
    "arctic": {
        "name": "dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        "revision": "55ec6e9",
    },
    "kure": {
        "name": "nlpai-lab/KURE-v1",
        "revision": None,
    },
}

CONFIG_NAMES = ["arctic", "kure", "dense_rrf", "bm25", "hybrid_rrf", "hybrid_rerank"]

METADATA_COLUMNS = [
    "embedding_row", "chunk_id", "parent_chunk_id", "doc_id", "council",
    "committee", "meeting_name", "meeting_date", "agenda_title", "speaker_label",
    "semantic_text", "issue_text", "chunk_text", "evidence_ref", "char_start", "char_end",
]

JUDGMENT_FIELDS = [
    "query_id", "category", "split", "query", "display_order", "chunk_id",
    "embedding_row", "label", "annotator", "memo", "council", "committee",
    "meeting_name", "meeting_date", "agenda_title", "speaker_label", "semantic_text",
    "original_excerpt", "evidence_ref", "parent_chunk_id", "char_start", "char_end",
    "source_models", "source_ranks", "similarity_scores",
]

TOKEN_PATTERN = r"(?u)\b[가-힣A-Za-z0-9][가-힣A-Za-z0-9·]{1,}\b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense+BM25+RRF+Reranker 성능 비교")
    parser.add_argument("--mode", choices=["prepare", "evaluate"], required=True)
    parser.add_argument(
        "--metadata",
        default="data/index/arctic_ko_semantic_test_v2/chunk_metadata.parquet",
    )
    parser.add_argument("--queries", default="outputs/retrieval_eval/queries.csv")
    parser.add_argument(
        "--base-judgments",
        default="outputs/retrieval_eval/retrieval_judgments_union.csv",
        help="13번에서 사람이 라벨링한 원본 파일",
    )
    parser.add_argument(
        "--judgments",
        default="outputs/retrieval_eval/retrieval_judgments_hybrid.csv",
        help="prepare 출력이자 evaluate 입력",
    )
    parser.add_argument("--output-dir", default="outputs/hybrid_benchmark")
    parser.add_argument(
        "--arctic-embedding",
        default="data/index/arctic_ko_semantic_test_v2/embeddings.f16.npy",
    )
    parser.add_argument(
        "--kure-embedding",
        default="data/index/embedding_benchmark/kure/embeddings.f16.npy",
    )
    parser.add_argument("--model-cache", default="models/huggingface")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["auto", "bf16", "float32"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--reranker-max-length", type=int, default=1024)
    parser.add_argument("--dense-k", type=int, default=100)
    parser.add_argument("--rerank-k", type=int, default=40)
    parser.add_argument("--pool-k", type=int, default=5, help="검색 구성별 라벨링 후보 수")
    parser.add_argument("--eval-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--bm25-max-features", type=int, default=150000)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--rebuild-rankings", action="store_true")
    parser.add_argument("--overwrite-pool", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def safe_text(value) -> str:
    return "" if value is None else str(value)


def shorten(value, limit: int) -> str:
    text = re.sub(r"\s+", " ", safe_text(value)).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    if requested == "cuda" and (torch is None or not torch.cuda.is_available()):
        raise SystemExit("CUDA를 사용할 수 없습니다.")
    return requested


def resolve_dtype(requested: str, device: str):
    if torch is None:
        raise SystemExit("PyTorch가 필요합니다.")
    if requested == "float32":
        return "float32", torch.float32
    if requested == "bf16":
        if device != "cuda" or not torch.cuda.is_bf16_supported():
            raise SystemExit("현재 장치에서 BF16을 지원하지 않습니다.")
        return "bf16", torch.bfloat16
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return "bf16", torch.bfloat16
    return "float32", torch.float32


def release_memory() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def file_signature(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"입력 파일이 없습니다: {path}")
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_queries(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"질의 파일이 없습니다: {path}")
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            query_id = safe_text(row.get("query_id")).strip()
            query = safe_text(row.get("query")).strip()
            split = safe_text(row.get("split", "dev")).strip().lower()
            if query_id and query:
                if split not in {"dev", "test"}:
                    raise SystemExit(f"split 오류: {query_id}={split}")
                rows.append({
                    "query_id": query_id,
                    "category": safe_text(row.get("category")),
                    "split": split,
                    "query": query,
                })
    if not rows:
        raise SystemExit("유효한 질의가 없습니다.")
    return rows


def load_metadata(path: Path) -> tuple[list[dict], list[str], str]:
    if not path.exists():
        raise SystemExit(f"메타데이터가 없습니다: {path}")
    source = pq.ParquetFile(path)
    available = set(source.schema_arrow.names)
    missing = {"embedding_row", "chunk_id"} - available
    if missing:
        raise SystemExit("메타데이터 필수 컬럼 누락: " + ", ".join(sorted(missing)))
    text_column = next((name for name in ("semantic_text", "issue_text", "chunk_text") if name in available), None)
    if text_column is None:
        raise SystemExit("검색 본문 컬럼이 없습니다.")
    columns = [name for name in METADATA_COLUMNS if name in available]
    rows = pq.read_table(path, columns=columns).to_pylist()
    texts = []
    digest = hashlib.sha256()
    for expected, row in enumerate(rows):
        actual = int(row.get("embedding_row", -1))
        if actual != expected:
            raise SystemExit(f"embedding_row 순서 오류: {actual} != {expected}")
        text = safe_text(row.get(text_column)).strip()
        if not text:
            raise SystemExit(f"빈 검색 본문: row={expected}")
        texts.append(text)
        digest.update(safe_text(row.get("chunk_id")).encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(hashlib.sha256(text.encode("utf-8")).digest())
        digest.update(b"\n")
    return rows, texts, digest.hexdigest()


def load_embeddings(path: Path, row_count: int) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"임베딩 파일이 없습니다: {path}")
    vectors = np.load(path, mmap_mode="r")
    if vectors.ndim != 2 or vectors.shape[0] != row_count:
        raise SystemExit(f"임베딩 shape 불일치: {vectors.shape}, expected rows={row_count}")
    return vectors


def load_dense_model(alias: str, args: argparse.Namespace, device: str, dtype):
    spec = DENSE_MODELS[alias]
    cache = Path(args.model_cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    common = {
        "device": device,
        "cache_folder": str(cache),
        "local_files_only": args.local_files_only,
        "trust_remote_code": False,
    }
    if spec["revision"]:
        common["revision"] = spec["revision"]
    try:
        model = SentenceTransformer(spec["name"], model_kwargs={"torch_dtype": dtype}, **common)
    except TypeError:
        model = SentenceTransformer(spec["name"], **common)
        model.to(dtype=dtype)
    model.max_seq_length = args.max_seq_length
    model.eval()
    return model


def encode_queries(model, query_texts: list[str], batch_size: int) -> tuple[np.ndarray, str | None]:
    prompts = getattr(model, "prompts", {}) or {}
    prompt_name = "query" if "query" in prompts else None
    kwargs = {
        "batch_size": batch_size,
        "show_progress_bar": False,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }
    if prompt_name:
        kwargs["prompt_name"] = prompt_name
    vectors = model.encode(query_texts, **kwargs)
    return np.asarray(vectors, dtype=np.float32), prompt_name


def exact_dense_rankings(
    alias: str,
    embedding_path: Path,
    queries: list[dict],
    row_count: int,
    args: argparse.Namespace,
    device: str,
    dtype,
) -> dict[str, list[dict]]:
    embeddings = load_embeddings(embedding_path, row_count)
    model = load_dense_model(alias, args, device, dtype)
    query_vectors, prompt_name = encode_queries(model, [row["query"] for row in queries], args.batch_size)
    if query_vectors.shape[1] != embeddings.shape[1]:
        raise SystemExit(f"{alias} 질의/문서 차원 불일치")
    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    for start in range(0, row_count, 4096):
        block = np.asarray(embeddings[start:start + 4096], dtype=np.float32)
        faiss.normalize_L2(block)
        index.add(block)
    scores, indices = index.search(query_vectors, min(args.dense_k, row_count))
    output = {}
    for query, found_rows, found_scores in zip(queries, indices, scores):
        output[query["query_id"]] = [
            {"row": int(row), "score": float(score)}
            for row, score in zip(found_rows, found_scores)
            if int(row) >= 0
        ]
    print(f"{alias} 정확 검색 완료 | query_prompt={prompt_name}")
    del index, model
    release_memory()
    return output


def build_bm25_matrix(texts: list[str], args: argparse.Namespace):
    print("BM25 어휘·문서행렬 생성", flush=True)
    vectorizer = CountVectorizer(
        token_pattern=TOKEN_PATTERN,
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=args.bm25_max_features,
        dtype=np.float32,
    )
    counts = vectorizer.fit_transform(texts).tocsr().astype(np.float32)
    row_count = counts.shape[0]
    doc_len = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    avgdl = float(doc_len.mean()) if len(doc_len) else 1.0
    df = np.asarray((counts > 0).sum(axis=0)).ravel().astype(np.float32)
    idf = np.log1p((row_count - df + 0.5) / (df + 0.5)).astype(np.float32)
    row_ids = np.repeat(np.arange(row_count, dtype=np.int32), np.diff(counts.indptr))
    tf = counts.data
    norm = args.bm25_k1 * (1.0 - args.bm25_b + args.bm25_b * doc_len[row_ids] / max(avgdl, 1e-9))
    counts.data = idf[counts.indices] * (tf * (args.bm25_k1 + 1.0)) / (tf + norm)
    print(f"BM25 행렬: {counts.shape[0]:,} x {counts.shape[1]:,} | nnz={counts.nnz:,}")
    return vectorizer, counts


def top_rows_from_scores(scores: np.ndarray, top_k: int) -> list[dict]:
    top_k = min(top_k, len(scores))
    if top_k == 0:
        return []
    candidate = np.argpartition(scores, -top_k)[-top_k:]
    candidate = candidate[np.argsort(scores[candidate])[::-1]]
    return [{"row": int(row), "score": float(scores[row])} for row in candidate if scores[row] > 0]


def bm25_rankings(texts: list[str], queries: list[dict], args: argparse.Namespace) -> dict[str, list[dict]]:
    vectorizer, matrix = build_bm25_matrix(texts, args)
    query_counts = vectorizer.transform([row["query"] for row in queries]).tocsr()
    query_counts.data[:] = 1.0
    score_matrix = (query_counts @ matrix.T).tocsr()
    output = {}
    for query_index, query in enumerate(queries):
        row = score_matrix.getrow(query_index)
        dense_scores = np.zeros(matrix.shape[0], dtype=np.float32)
        dense_scores[row.indices] = row.data
        output[query["query_id"]] = top_rows_from_scores(dense_scores, args.dense_k)
    del matrix, score_matrix
    gc.collect()
    return output


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    top_k: int,
    rrf_k: int,
) -> list[dict]:
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[int(item["row"])] += 1.0 / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]
    return [{"row": row, "score": float(score)} for row, score in ordered]


def load_reranker(args: argparse.Namespace, device: str, dtype):
    cache = Path(args.model_cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    common = {
        "device": device,
        "max_length": args.reranker_max_length,
        "local_files_only": args.local_files_only,
        "trust_remote_code": False,
    }
    model_args = {"torch_dtype": dtype, "cache_dir": str(cache)}
    tokenizer_args = {"cache_dir": str(cache)}
    try:
        model = CrossEncoder(
            args.reranker_model,
            automodel_args=model_args,
            tokenizer_args=tokenizer_args,
            **common,
        )
    except TypeError:
        # sentence-transformers 버전에 따라 추가 인자 이름이 다를 수 있다.
        # 이 경우 Hugging Face 기본 캐시를 사용하되 추론 동작은 동일하다.
        model = CrossEncoder(args.reranker_model, **common)
    return model


def rerank_all(
    hybrid: dict[str, list[dict]],
    queries: list[dict],
    texts: list[str],
    args: argparse.Namespace,
    device: str,
    dtype,
) -> dict[str, list[dict]]:
    print(f"Reranker 로딩: {args.reranker_model}", flush=True)
    model = load_reranker(args, device, dtype)
    output = {}
    query_map = {row["query_id"]: row["query"] for row in queries}
    for index, (query_id, candidates) in enumerate(hybrid.items(), start=1):
        selected = candidates[: args.rerank_k]
        pairs = [(query_map[query_id], texts[int(item["row"])]) for item in selected]
        current_batch = args.batch_size
        while True:
            try:
                scores = model.predict(
                    pairs,
                    batch_size=current_batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                break
            except RuntimeError as error:
                if device != "cuda" or "out of memory" not in str(error).lower() or current_batch == 1:
                    raise
                current_batch = max(1, current_batch // 2)
                torch.cuda.empty_cache()
                print(f"Reranker GPU 메모리 부족: batch={current_batch}")
        scores = np.asarray(scores)
        if scores.ndim > 1:
            scores = scores[:, -1]
        reranked = sorted(
            (
                {"row": int(item["row"]), "score": float(score)}
                for item, score in zip(selected, scores)
            ),
            key=lambda item: (-item["score"], item["row"]),
        )
        output[query_id] = reranked
        if index % 10 == 0 or index == len(hybrid):
            print(f"Reranker {index}/{len(hybrid)} 질의 완료", flush=True)
    del model
    release_memory()
    return output


def expected_cache_state(
    args: argparse.Namespace,
    metadata_digest: str,
    queries_path: Path,
    arctic_path: Path,
    kure_path: Path,
) -> dict:
    return {
        "script_version": SCRIPT_VERSION,
        "metadata_digest": metadata_digest,
        "queries_sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
        "arctic": file_signature(arctic_path),
        "kure": file_signature(kure_path),
        "reranker_model": args.reranker_model,
        "dense_k": args.dense_k,
        "rerank_k": args.rerank_k,
        "rrf_k": args.rrf_k,
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "bm25_max_features": args.bm25_max_features,
    }


def build_or_load_rankings(
    metadata: list[dict],
    texts: list[str],
    metadata_digest: str,
    queries: list[dict],
    queries_path: Path,
    args: argparse.Namespace,
) -> dict:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rankings_path = output_dir / "hybrid_rankings.json"
    state_path = output_dir / "hybrid_rankings_state.json"
    arctic_path = Path(args.arctic_embedding).expanduser().resolve()
    kure_path = Path(args.kure_embedding).expanduser().resolve()
    expected = expected_cache_state(args, metadata_digest, queries_path, arctic_path, kure_path)
    if rankings_path.exists() and state_path.exists() and not args.rebuild_rankings:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state == expected:
            print(f"기존 하이브리드 검색 캐시 재사용: {rankings_path}")
            return json.loads(rankings_path.read_text(encoding="utf-8"))
        raise SystemExit("기존 하이브리드 캐시 설정이 다릅니다. --rebuild-rankings로 다시 생성하세요.")

    device = resolve_device(args.device)
    precision_name, dtype = resolve_dtype(args.precision, device)
    print(f"장치: {device} | precision={precision_name}")
    arctic = exact_dense_rankings("arctic", arctic_path, queries, len(texts), args, device, dtype)
    kure = exact_dense_rankings("kure", kure_path, queries, len(texts), args, device, dtype)
    bm25 = bm25_rankings(texts, queries, args)

    dense_rrf = {}
    hybrid_rrf = {}
    for query in queries:
        query_id = query["query_id"]
        dense_rrf[query_id] = reciprocal_rank_fusion(
            [arctic[query_id], kure[query_id]], args.dense_k, args.rrf_k
        )
        hybrid_rrf[query_id] = reciprocal_rank_fusion(
            [arctic[query_id], kure[query_id], bm25[query_id]], args.dense_k, args.rrf_k
        )
    hybrid_rerank = rerank_all(hybrid_rrf, queries, texts, args, device, dtype)
    rankings = {
        "arctic": arctic,
        "kure": kure,
        "dense_rrf": dense_rrf,
        "bm25": bm25,
        "hybrid_rrf": hybrid_rrf,
        "hybrid_rerank": hybrid_rerank,
    }
    atomic_write_json(rankings_path, rankings)
    atomic_write_json(state_path, expected)
    return rankings


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"CSV 파일이 없습니다: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def prepare_hybrid_judgments(
    base_path: Path,
    output_path: Path,
    queries: list[dict],
    metadata: list[dict],
    rankings: dict,
    pool_k: int,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise SystemExit(
            f"하이브리드 라벨 파일이 이미 있습니다: {output_path}\n"
            "기존 결과를 유지하거나 --overwrite-pool을 사용하세요."
        )
    base_rows = read_csv_rows(base_path)
    # --overwrite-pool은 후보 풀을 다시 계산한다는 뜻이지, 사람이 입력한 기존
    # 하이브리드 라벨을 지운다는 뜻이 아니다. 기존 출력이 있으면 우선 보존한다.
    seed_rows = list(base_rows)
    if output_path.exists():
        hybrid_rows = read_csv_rows(output_path)
        by_key = {(row["query_id"], row["chunk_id"]): dict(row) for row in seed_rows}
        for row in hybrid_rows:
            by_key[(row["query_id"], row["chunk_id"])] = dict(row)
        seed_rows = list(by_key.values())
    existing = {(row["query_id"], row["chunk_id"]): dict(row) for row in seed_rows}
    query_lookup = {row["query_id"]: row for row in queries}

    # 평가할 모든 새 구성의 top-k를 pooling해야 특정 구성에 불리한
    # 미판정 문서(평가 시 0으로 간주)가 생기지 않는다.
    sources_to_add = ["dense_rrf", "bm25", "hybrid_rrf", "hybrid_rerank"]
    for source in sources_to_add:
        for query_id, items in rankings[source].items():
            query = query_lookup[query_id]
            for rank, item in enumerate(items[:pool_k], start=1):
                row_index = int(item["row"])
                meta = metadata[row_index]
                chunk_id = safe_text(meta.get("chunk_id"))
                key = (query_id, chunk_id)
                if key in existing:
                    previous_sources = [value for value in safe_text(existing[key].get("source_models")).split("|") if value]
                    if source not in previous_sources:
                        existing[key]["source_models"] = "|".join(previous_sources + [source])
                    continue
                existing[key] = {
                    "query_id": query_id,
                    "category": query["category"],
                    "split": query["split"],
                    "query": query["query"],
                    "display_order": "",
                    "chunk_id": chunk_id,
                    "embedding_row": row_index,
                    "label": "",
                    "annotator": "",
                    "memo": "",
                    "council": safe_text(meta.get("council")),
                    "committee": safe_text(meta.get("committee")),
                    "meeting_name": safe_text(meta.get("meeting_name")),
                    "meeting_date": safe_text(meta.get("meeting_date")),
                    "agenda_title": shorten(meta.get("agenda_title"), 300),
                    "speaker_label": shorten(meta.get("speaker_label"), 120),
                    "semantic_text": shorten(meta.get("semantic_text") or meta.get("issue_text") or meta.get("chunk_text"), 1200),
                    "original_excerpt": shorten(meta.get("chunk_text"), 1200),
                    "evidence_ref": safe_text(meta.get("evidence_ref")),
                    "parent_chunk_id": safe_text(meta.get("parent_chunk_id")),
                    "char_start": safe_text(meta.get("char_start")),
                    "char_end": safe_text(meta.get("char_end")),
                    "source_models": source,
                    "source_ranks": str(rank),
                    "similarity_scores": f"{float(item['score']):.6f}",
                }

    grouped = defaultdict(list)
    for row in existing.values():
        grouped[row["query_id"]].append(row)
    output = []
    for query in queries:
        rows = grouped[query["query_id"]]
        rows.sort(
            key=lambda row: hashlib.sha1(
                f"hybrid|{row['query_id']}|{row['chunk_id']}".encode()
            ).hexdigest()
        )
        for order, row in enumerate(rows, start=1):
            row["display_order"] = order
            output.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=JUDGMENT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)
    new_count = sum(not safe_text(row.get("label")).strip() for row in output)
    print(f"하이브리드 라벨 파일: {output_path}")
    print(f"전체 {len(output):,}행 | 새로 검수할 빈 label {new_count:,}행")


def load_qrels(path: Path, allow_incomplete: bool) -> dict[str, dict[str, int]]:
    rows = read_csv_rows(path)
    qrels = defaultdict(dict)
    blanks = []
    for line_number, row in enumerate(rows, start=2):
        label_text = safe_text(row.get("label")).strip()
        if not label_text:
            blanks.append(line_number)
            if not allow_incomplete:
                continue
            label = 0
        else:
            try:
                label = int(label_text)
            except ValueError:
                raise SystemExit(f"label 오류: line={line_number}, value={label_text}")
            if label not in {0, 1, 2}:
                raise SystemExit(f"label은 0/1/2만 가능합니다: line={line_number}")
        qrels[row["query_id"]][row["chunk_id"]] = label
    if blanks and not allow_incomplete:
        raise SystemExit(
            f"빈 label이 {len(blanks):,}개 있습니다. 모두 입력한 뒤 evaluate 하세요. "
            "진단용은 --allow-incomplete를 사용할 수 있습니다."
        )
    return dict(qrels)


def dcg(labels: list[int]) -> float:
    return sum((2 ** label - 1) / math.log2(rank + 2) for rank, label in enumerate(labels))


def query_metrics(ranked_ids: list[str], qrels: dict[str, int], k: int) -> dict:
    grades = [int(qrels.get(chunk_id, 0)) for chunk_id in ranked_ids[:k]]
    relevant_total = sum(label > 0 for label in qrels.values())
    found = sum(label > 0 for label in grades)
    first = next((rank for rank, label in enumerate(grades, start=1) if label > 0), None)
    ideal = sorted(qrels.values(), reverse=True)[:k]
    ideal_score = dcg(ideal)
    return {
        "precision": found / k,
        "recall": found / relevant_total if relevant_total else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "ndcg": dcg(grades) / ideal_score if ideal_score else 0.0,
        "relevant_total": relevant_total,
        "relevant_found": found,
    }


def evaluate(
    rankings: dict,
    queries: list[dict],
    metadata: list[dict],
    qrels: dict,
    eval_k: int,
    output_dir: Path,
) -> None:
    per_query = []
    for config in CONFIG_NAMES:
        for query in queries:
            query_id = query["query_id"]
            ranked_ids = [safe_text(metadata[int(item["row"])].get("chunk_id")) for item in rankings[config][query_id]]
            metrics = query_metrics(ranked_ids, qrels.get(query_id, {}), eval_k)
            per_query.append({
                "config": config,
                **query,
                f"precision@{eval_k}": metrics["precision"],
                f"recall@{eval_k}": metrics["recall"],
                f"mrr@{eval_k}": metrics["mrr"],
                f"ndcg@{eval_k}": metrics["ndcg"],
                "relevant_total": metrics["relevant_total"],
                "relevant_found": metrics["relevant_found"],
            })

    metric_names = [f"precision@{eval_k}", f"recall@{eval_k}", f"mrr@{eval_k}", f"ndcg@{eval_k}"]
    summary = []
    for config in CONFIG_NAMES:
        for split in ("dev", "test", "all"):
            selected = [
                row for row in per_query
                if row["config"] == config and (split == "all" or row["split"] == split)
            ]
            usable = [row for row in selected if row["relevant_total"] > 0]
            item = {
                "config": config,
                "split": split,
                "query_count": len(selected),
                "queries_with_relevant": len(usable),
            }
            for metric in metric_names:
                item[metric] = sum(row[metric] for row in usable) / len(usable) if usable else 0.0
            summary.append(item)

    dev = [row for row in summary if row["split"] == "dev"]
    winner = max(
        dev,
        key=lambda row: (row[f"ndcg@{eval_k}"], row[f"recall@{eval_k}"], row[f"mrr@{eval_k}"]),
    )["config"]
    output_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = output_dir / "hybrid_benchmark_per_query.csv"
    summary_csv_path = output_dir / "hybrid_benchmark_summary.csv"
    summary_json_path = output_dir / "hybrid_benchmark_summary.json"
    with per_query_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_query[0]))
        writer.writeheader()
        writer.writerows(per_query)
    with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    atomic_write_json(summary_json_path, {
        "script_version": SCRIPT_VERSION,
        "selection_rule": f"dev nDCG@{eval_k}, 동률이면 Recall@{eval_k}, MRR@{eval_k}",
        "recommended_config": winner,
        "results": summary,
        "security": "모델 다운로드 외 회의록 데이터 외부 전송 없음",
    })
    print("\n=== 하이브리드 검색 비교(dev) ===")
    for row in sorted(dev, key=lambda item: item[f"ndcg@{eval_k}"], reverse=True):
        print(
            f"{row['config']:16s} | nDCG={row[f'ndcg@{eval_k}']:.4f} | "
            f"Recall={row[f'recall@{eval_k}']:.4f} | MRR={row[f'mrr@{eval_k}']:.4f}"
        )
    print(f"추천 구성: {winner}")
    print(f"요약: {summary_json_path}")


def validate_args(args: argparse.Namespace) -> None:
    positive = [
        args.batch_size, args.max_seq_length, args.reranker_max_length, args.dense_k,
        args.rerank_k, args.pool_k, args.eval_k, args.rrf_k, args.bm25_max_features,
    ]
    if any(value < 1 for value in positive):
        raise SystemExit("크기·K·배치 관련 값은 1 이상이어야 합니다.")
    if args.pool_k < args.eval_k:
        raise SystemExit("--pool-k는 --eval-k 이상이어야 합니다.")
    if args.rerank_k < args.pool_k:
        raise SystemExit("--rerank-k는 --pool-k 이상이어야 합니다.")
    if args.dense_k < args.rerank_k:
        raise SystemExit("--dense-k는 --rerank-k 이상이어야 합니다.")
    if args.bm25_k1 <= 0 or not 0 <= args.bm25_b <= 1:
        raise SystemExit("BM25는 k1>0, 0<=b<=1이어야 합니다.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    metadata_path = Path(args.metadata).expanduser().resolve()
    queries_path = Path(args.queries).expanduser().resolve()
    base_judgments = Path(args.base_judgments).expanduser().resolve()
    judgments = Path(args.judgments).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    queries = load_queries(queries_path)
    print("메타데이터 검증", flush=True)
    metadata, texts, metadata_digest = load_metadata(metadata_path)
    print(f"정책 청크 {len(texts):,}개 | 질의 {len(queries):,}개")
    rankings = build_or_load_rankings(
        metadata, texts, metadata_digest, queries, queries_path, args
    )

    if args.mode == "prepare":
        prepare_hybrid_judgments(
            base_judgments, judgments, queries, metadata, rankings,
            args.pool_k, args.overwrite_pool,
        )
        return

    qrels = load_qrels(judgments, args.allow_incomplete)
    evaluate(rankings, queries, metadata, qrels, args.eval_k, output_dir)


if __name__ == "__main__":
    main()

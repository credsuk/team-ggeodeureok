#!/usr/bin/env python3
"""정책 의미 청크에서 임베딩 모델을 공정하게 비교한다.

prepare
    Arctic-ko, KURE-v1, BGE-M3로 동일한 25,340개 정책 의미 청크를 검색하고
    모델별 Top-K의 합집합을 블라인드 라벨링 CSV로 만든다. 기존 Arctic 벡터는
    기본적으로 재사용하며, 나머지 모델 임베딩은 중단 후 재개할 수 있다.

evaluate
    사람이 label 열에 입력한 0/1/2 관련도를 이용하여 모델별 Precision, Recall,
    MRR, nDCG를 dev/test로 나누어 계산한다.

회의록 원문은 로컬에서만 처리하며 외부 LLM/API로 전송하지 않는다.
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
    from sentence_transformers import SentenceTransformer
except ImportError as error:
    raise SystemExit("sentence-transformers가 필요합니다.") from error


SCRIPT_VERSION = "2026-10-20-embedding-benchmark-v1"

MODEL_SPECS = {
    "arctic": {
        "name": "dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        "revision": "55ec6e9",
        "description": "현재 기준 모델",
    },
    "kure": {
        "name": "nlpai-lab/KURE-v1",
        "revision": None,
        "description": "한국어 질의-문서 검색 특화 후보",
    },
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "revision": None,
        "description": "다국어·장문·하이브리드 확장 후보",
    },
}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arctic/KURE/BGE-M3 임베딩 검색 성능 비교")
    parser.add_argument("--mode", choices=["prepare", "evaluate"], required=True)
    parser.add_argument(
        "--metadata",
        default="data/index/arctic_ko_semantic_test_v2/chunk_metadata.parquet",
    )
    parser.add_argument("--queries", default="outputs/retrieval_eval/queries.csv")
    parser.add_argument(
        "--judgments",
        default="outputs/retrieval_eval/retrieval_judgments_union.csv",
        help="prepare 출력이자 evaluate 입력",
    )
    parser.add_argument("--output-dir", default="outputs/retrieval_eval")
    parser.add_argument("--cache-dir", default="data/index/embedding_benchmark")
    parser.add_argument(
        "--reuse-arctic-embedding",
        default="data/index/arctic_ko_semantic_test_v2/embeddings.f16.npy",
    )
    parser.add_argument(
        "--models", default="arctic,kure,bge-m3",
        help="쉼표 구분 모델 별칭: arctic,kure,bge-m3",
    )
    parser.add_argument("--model-cache", default="models/huggingface")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["auto", "bf16", "float32"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--pool-k", type=int, default=8, help="모델별 라벨링 후보 수")
    parser.add_argument("--eval-k", type=int, default=5)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite-pool", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="빈 label을 0으로 간주. 중간 진단 외 최종 점수에는 권장하지 않음",
    )
    return parser.parse_args()


def atomic_write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def safe_text(value) -> str:
    return "" if value is None else str(value)


def shorten(value, limit: int) -> str:
    text = re.sub(r"\s+", " ", safe_text(value)).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    if requested == "cuda" and (torch is None or not torch.cuda.is_available()):
        raise SystemExit("CUDA를 사용할 수 없습니다. PyTorch CUDA 설치와 nvidia-smi를 확인하세요.")
    return requested


def resolve_precision(requested: str, device: str):
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


def parse_models(value: str) -> list[str]:
    aliases = []
    for alias in value.split(","):
        alias = alias.strip().lower()
        if not alias:
            continue
        if alias not in MODEL_SPECS:
            raise SystemExit(f"지원하지 않는 모델 별칭: {alias}")
        if alias not in aliases:
            aliases.append(alias)
    if not aliases:
        raise SystemExit("--models에 모델을 하나 이상 지정하세요.")
    return aliases


def load_queries(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"질의 파일이 없습니다: {path}\n먼저 12_make_retrieval_evalset.py를 실행하세요.")
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            query_id = safe_text(row.get("query_id")).strip()
            query = safe_text(row.get("query")).strip()
            split = safe_text(row.get("split", "dev")).strip().lower()
            if query_id and query:
                if split not in {"dev", "test"}:
                    raise SystemExit(f"질의 split 오류: {query_id}={split}")
                rows.append({
                    "query_id": query_id,
                    "query": query,
                    "category": safe_text(row.get("category")),
                    "split": split,
                })
    if not rows:
        raise SystemExit("유효한 평가 질의가 없습니다.")
    if len({row["query_id"] for row in rows}) != len(rows):
        raise SystemExit("query_id가 중복되었습니다.")
    return rows


def load_metadata(path: Path) -> tuple[list[dict], list[str], str]:
    if not path.exists():
        raise SystemExit(f"메타데이터가 없습니다: {path}")
    source = pq.ParquetFile(path)
    available = set(source.schema_arrow.names)
    required = {"embedding_row", "chunk_id"}
    missing = required - available
    if missing:
        raise SystemExit("메타데이터 필수 컬럼 누락: " + ", ".join(sorted(missing)))
    text_column = next((name for name in ("semantic_text", "issue_text", "chunk_text") if name in available), None)
    if text_column is None:
        raise SystemExit("semantic_text/issue_text/chunk_text 중 하나가 필요합니다.")
    columns = [name for name in METADATA_COLUMNS if name in available]
    rows = pq.read_table(path, columns=columns).to_pylist()
    digest = hashlib.sha256()
    texts = []
    for expected_row, row in enumerate(rows):
        actual_row = int(row.get("embedding_row", -1))
        if actual_row != expected_row:
            raise SystemExit(f"embedding_row 순서 오류: expected={expected_row}, actual={actual_row}")
        text = safe_text(row.get(text_column)).strip()
        if not text:
            raise SystemExit(f"빈 임베딩 본문: embedding_row={expected_row}")
        texts.append(text)
        digest.update(safe_text(row.get("chunk_id")).encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(hashlib.sha256(text.encode("utf-8")).digest())
        digest.update(b"\n")
    return rows, texts, digest.hexdigest()


def load_model(alias: str, args: argparse.Namespace, device: str, torch_dtype):
    spec = MODEL_SPECS[alias]
    model_cache = Path(args.model_cache).expanduser().resolve()
    model_cache.mkdir(parents=True, exist_ok=True)
    common = {
        "device": device,
        "cache_folder": str(model_cache),
        "local_files_only": args.local_files_only,
        "trust_remote_code": False,
    }
    if spec["revision"]:
        common["revision"] = spec["revision"]
    print(f"모델 로딩: {alias} | {spec['name']} | {device}/{torch_dtype}", flush=True)
    try:
        model = SentenceTransformer(spec["name"], model_kwargs={"torch_dtype": torch_dtype}, **common)
    except TypeError:
        model = SentenceTransformer(spec["name"], **common)
        model.to(dtype=torch_dtype)
    model.max_seq_length = args.max_seq_length
    model.eval()
    return model


def is_oom(error: RuntimeError) -> bool:
    return "out of memory" in str(error).lower()


def encode_batch(model, texts: list[str], batch_size: int, device: str, is_query: bool):
    current = max(1, batch_size)
    prompts = getattr(model, "prompts", {}) or {}
    prompt_name = None
    if is_query and "query" in prompts:
        prompt_name = "query"
    elif not is_query:
        for candidate in ("document", "passage", "corpus"):
            if candidate in prompts:
                prompt_name = candidate
                break
    while True:
        kwargs = {
            "batch_size": current,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
        if prompt_name:
            kwargs["prompt_name"] = prompt_name
        try:
            vectors = model.encode(texts, **kwargs)
            return np.asarray(vectors, dtype=np.float32), current, prompt_name
        except RuntimeError as error:
            if device != "cuda" or not is_oom(error) or current == 1:
                raise
            current = max(1, current // 2)
            torch.cuda.empty_cache()
            print(f"GPU 메모리 부족: batch={current}로 재시도", flush=True)


def cache_paths(cache_dir: Path, alias: str) -> tuple[Path, Path]:
    model_dir = cache_dir / alias
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "embeddings.f16.npy", model_dir / "embedding_state.json"


def validate_reused_embedding(path: Path, row_count: int) -> np.ndarray:
    if not path.exists():
        raise SystemExit(
            f"기존 Arctic 임베딩이 없습니다: {path}\n"
            "경로가 다르면 --reuse-arctic-embedding으로 지정하세요."
        )
    vectors = np.load(path, mmap_mode="r")
    if vectors.ndim != 2 or vectors.shape[0] != row_count:
        raise SystemExit(f"기존 Arctic 벡터 행 수 불일치: {vectors.shape} / expected={row_count:,}")
    return vectors


def encode_or_reuse_corpus(
    alias: str,
    texts: list[str],
    metadata_digest: str,
    args: argparse.Namespace,
    device: str,
    torch_dtype,
):
    if alias == "arctic" and args.reuse_arctic_embedding:
        path = Path(args.reuse_arctic_embedding).expanduser().resolve()
        vectors = validate_reused_embedding(path, len(texts))
        print(f"기존 Arctic 벡터 재사용: {path} | {vectors.shape}", flush=True)
        return vectors, None, {"reused": True, "path": str(path)}

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    embedding_path, state_path = cache_paths(cache_dir, alias)
    spec = MODEL_SPECS[alias]
    expected = {
        "script_version": SCRIPT_VERSION,
        "model_alias": alias,
        "model_name": spec["name"],
        "model_revision": spec["revision"],
        "metadata_digest": metadata_digest,
        "row_count": len(texts),
        "max_seq_length": args.max_seq_length,
        "output_dtype": "float16",
    }
    state = None
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        differences = [key for key, value in expected.items() if state.get(key) != value]
        if differences:
            raise SystemExit(f"{alias} 기존 캐시 옵션 불일치: {', '.join(differences)}")
        if state.get("phase") == "complete" and embedding_path.exists():
            vectors = np.load(embedding_path, mmap_mode="r")
            if vectors.shape[0] != len(texts):
                raise SystemExit(f"{alias} 캐시 행 수 불일치: {vectors.shape}")
            print(f"{alias} 완성 캐시 재사용: {embedding_path}", flush=True)
            return vectors, None, state
    elif embedding_path.exists():
        raise SystemExit(f"{alias} 벡터는 있지만 상태 파일이 없습니다: {embedding_path}")

    model = load_model(alias, args, device, torch_dtype)
    processed = int((state or {}).get("processed_count", 0))
    session_start_count = processed
    effective_batch = int((state or {}).get("effective_batch_size", args.batch_size))
    dimension = (state or {}).get("dimension")
    embeddings = None
    if processed:
        embeddings = np.load(embedding_path, mmap_mode="r+")
        dimension = int(embeddings.shape[1])

    started = time.time()
    for start in range(processed, len(texts), 256):
        end = min(start + 256, len(texts))
        vectors, used_batch, document_prompt = encode_batch(
            model, texts[start:end], effective_batch, device, is_query=False
        )
        effective_batch = min(effective_batch, used_batch)
        if embeddings is None:
            dimension = int(vectors.shape[1])
            embeddings = np.lib.format.open_memmap(
                embedding_path, mode="w+", dtype=np.float16, shape=(len(texts), dimension)
            )
        if vectors.shape != (end - start, dimension):
            raise RuntimeError(f"{alias} 예상하지 못한 벡터 shape: {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise RuntimeError(f"{alias} 벡터에 NaN/inf가 있습니다.")
        embeddings[start:end] = vectors.astype(np.float16)
        embeddings.flush()
        processed = end
        state = {
            **expected,
            "phase": "embedding",
            "processed_count": processed,
            "dimension": dimension,
            "effective_batch_size": effective_batch,
            "document_prompt": document_prompt,
            "device": device,
            "updated_at_unix": time.time(),
        }
        atomic_write_json(state_path, state)
        elapsed = time.time() - started
        session_done = processed - session_start_count
        rate = session_done / max(elapsed, 1e-9)
        eta = (len(texts) - processed) / max(rate, 1e-9)
        print(
            f"{alias} 임베딩 {processed:,}/{len(texts):,} "
            f"({processed / len(texts) * 100:.1f}%) | {rate:.1f}/s | ETA {format_duration(eta)}",
            flush=True,
        )

    state.update({"phase": "complete", "completed_at_unix": time.time()})
    atomic_write_json(state_path, state)
    return embeddings, model, state


def make_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    row_count, dimension = embeddings.shape
    index = faiss.IndexFlatIP(int(dimension))
    for start in range(0, row_count, 4096):
        vectors = np.asarray(embeddings[start:start + 4096], dtype=np.float32)
        faiss.normalize_L2(vectors)
        index.add(vectors)
    return index


def release_device_memory() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def retrieve_all_models(
    aliases: list[str],
    queries: list[dict],
    texts: list[str],
    metadata_digest: str,
    args: argparse.Namespace,
) -> dict[str, dict[str, list[tuple[int, float]]]]:
    device = resolve_device(args.device)
    precision_name, torch_dtype = resolve_precision(args.precision, device)
    print(f"검색 비교 장치: {device} | precision={precision_name}")
    query_texts = [row["query"] for row in queries]
    rankings: dict[str, dict[str, list[tuple[int, float]]]] = {}

    for alias in aliases:
        embeddings, model, _ = encode_or_reuse_corpus(
            alias, texts, metadata_digest, args, device, torch_dtype
        )
        if model is None:
            model = load_model(alias, args, device, torch_dtype)
        query_vectors, _, query_prompt = encode_batch(
            model, query_texts, min(16, max(1, args.batch_size)), device, is_query=True
        )
        if query_vectors.shape[1] != embeddings.shape[1]:
            raise SystemExit(
                f"{alias} 질의/문서 차원 불일치: {query_vectors.shape[1]} != {embeddings.shape[1]}"
            )
        index = make_faiss_index(embeddings)
        search_k = min(max(args.pool_k, args.eval_k), index.ntotal)
        scores, rows = index.search(query_vectors, search_k)
        rankings[alias] = {}
        for query, found_rows, found_scores in zip(queries, rows, scores):
            rankings[alias][query["query_id"]] = [
                (int(row), float(score))
                for row, score in zip(found_rows, found_scores)
                if int(row) >= 0
            ]
        print(f"{alias} 검색 완료 | query_prompt={query_prompt} | top={search_k}")
        del index
        del model
        release_device_memory()
    return rankings


def load_existing_labels(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    labels = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            key = (safe_text(row.get("query_id")), safe_text(row.get("chunk_id")))
            if all(key):
                labels[key] = row
    return labels


def prepare_judgments(
    queries: list[dict],
    metadata: list[dict],
    rankings: dict[str, dict[str, list[tuple[int, float]]]],
    path: Path,
    pool_k: int,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(
            f"라벨링 파일이 이미 있습니다: {path}\n"
            "기존 라벨 보호를 위해 중단했습니다. 후보를 다시 만들려면 백업 후 --overwrite-pool을 사용하세요."
        )
    old = load_existing_labels(path)
    output = []
    aliases = list(rankings)
    for query in queries:
        pooled: dict[int, dict] = {}
        for alias in aliases:
            for rank, (row_index, score) in enumerate(
                rankings[alias][query["query_id"]][:pool_k], start=1
            ):
                item = pooled.setdefault(row_index, {"models": [], "ranks": [], "scores": []})
                item["models"].append(alias)
                item["ranks"].append(str(rank))
                item["scores"].append(f"{score:.6f}")

        query_rows = []
        for row_index, source in pooled.items():
            meta = metadata[row_index]
            chunk_id = safe_text(meta.get("chunk_id"))
            previous = old.get((query["query_id"], chunk_id), {})
            random_key = hashlib.sha1(f"{query['query_id']}|{chunk_id}".encode()).hexdigest()
            query_rows.append((random_key, {
                "query_id": query["query_id"],
                "category": query["category"],
                "split": query["split"],
                "query": query["query"],
                "chunk_id": chunk_id,
                "embedding_row": row_index,
                "label": safe_text(previous.get("label")),
                "annotator": safe_text(previous.get("annotator")),
                "memo": safe_text(previous.get("memo")),
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
                "source_models": "|".join(source["models"]),
                "source_ranks": "|".join(source["ranks"]),
                "similarity_scores": "|".join(source["scores"]),
            }))
        query_rows.sort(key=lambda pair: pair[0])
        for display_order, (_, row) in enumerate(query_rows, start=1):
            row["display_order"] = display_order
            output.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=JUDGMENT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)
    print(f"라벨링 후보 생성: {path} | {len(output):,}행")
    print("label 열에 0/1/2를 모두 입력한 뒤 --mode evaluate를 실행하세요.")


def load_qrels(path: Path, allow_incomplete: bool) -> tuple[dict, dict]:
    if not path.exists():
        raise SystemExit(f"라벨링 파일이 없습니다: {path}\n먼저 --mode prepare를 실행하세요.")
    qrels = defaultdict(dict)
    row_lookup = {}
    blanks = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for line_number, row in enumerate(csv.DictReader(file), start=2):
            query_id = safe_text(row.get("query_id"))
            chunk_id = safe_text(row.get("chunk_id"))
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
                    raise SystemExit(f"label은 0/1/2만 가능합니다: line={line_number}, value={label_text}")
                if label not in {0, 1, 2}:
                    raise SystemExit(f"label은 0/1/2만 가능합니다: line={line_number}, value={label}")
            qrels[query_id][chunk_id] = label
            row_lookup[int(row["embedding_row"])] = chunk_id
    if blanks and not allow_incomplete:
        preview = ", ".join(map(str, blanks[:10]))
        raise SystemExit(
            f"아직 label이 비어 있는 행이 {len(blanks):,}개입니다(line {preview}...). "
            "모두 검수한 뒤 실행하세요. 중간 진단만 하려면 --allow-incomplete를 사용하세요."
        )
    return dict(qrels), row_lookup


def dcg(labels: list[int]) -> float:
    return sum((2 ** label - 1) / math.log2(rank + 2) for rank, label in enumerate(labels))


def metrics_for_query(ranked_chunk_ids: list[str], qrels: dict[str, int], k: int) -> dict:
    ranked = ranked_chunk_ids[:k]
    grades = [int(qrels.get(chunk_id, 0)) for chunk_id in ranked]
    relevant_total = sum(label > 0 for label in qrels.values())
    relevant_found = sum(label > 0 for label in grades)
    first_rank = next((rank for rank, label in enumerate(grades, start=1) if label > 0), None)
    ideal = sorted(qrels.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    return {
        "precision": relevant_found / k,
        "recall": relevant_found / relevant_total if relevant_total else 0.0,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "ndcg": dcg(grades) / ideal_dcg if ideal_dcg else 0.0,
        "relevant_total": relevant_total,
        "relevant_found": relevant_found,
    }


def evaluate(
    aliases: list[str],
    queries: list[dict],
    metadata: list[dict],
    rankings: dict[str, dict[str, list[tuple[int, float]]]],
    qrels: dict,
    eval_k: int,
    output_dir: Path,
) -> None:
    per_query = []
    for alias in aliases:
        for query in queries:
            query_id = query["query_id"]
            judged = qrels.get(query_id, {})
            ranked_ids = [safe_text(metadata[row].get("chunk_id")) for row, _ in rankings[alias][query_id]]
            metrics = metrics_for_query(ranked_ids, judged, eval_k)
            per_query.append({
                "model": alias,
                **query,
                f"precision@{eval_k}": metrics["precision"],
                f"recall@{eval_k}": metrics["recall"],
                f"mrr@{eval_k}": metrics["mrr"],
                f"ndcg@{eval_k}": metrics["ndcg"],
                "relevant_total": metrics["relevant_total"],
                "relevant_found": metrics["relevant_found"],
            })

    metric_names = [f"precision@{eval_k}", f"recall@{eval_k}", f"mrr@{eval_k}", f"ndcg@{eval_k}"]
    summary_rows = []
    for alias in aliases:
        for split in ("dev", "test", "all"):
            selected = [
                row for row in per_query
                if row["model"] == alias and (split == "all" or row["split"] == split)
            ]
            with_answer = [row for row in selected if row["relevant_total"] > 0]
            summary = {
                "model": alias,
                "split": split,
                "query_count": len(selected),
                "queries_with_relevant": len(with_answer),
            }
            for metric in metric_names:
                summary[metric] = sum(row[metric] for row in with_answer) / len(with_answer) if with_answer else 0.0
            summary_rows.append(summary)

    dev_rows = [row for row in summary_rows if row["split"] == "dev"]
    winner = max(
        dev_rows,
        key=lambda row: (row[f"ndcg@{eval_k}"], row[f"recall@{eval_k}"], row[f"mrr@{eval_k}"]),
    )["model"]

    output_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = output_dir / "embedding_benchmark_per_query.csv"
    summary_csv_path = output_dir / "embedding_benchmark_summary.csv"
    summary_json_path = output_dir / "embedding_benchmark_summary.json"
    with per_query_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_query[0]))
        writer.writeheader()
        writer.writerows(per_query)
    with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    payload = {
        "script_version": SCRIPT_VERSION,
        "security": "모델 다운로드 외 회의록 데이터 외부 전송 없음",
        "selection_rule": f"dev nDCG@{eval_k}, 동률이면 Recall@{eval_k}, MRR@{eval_k}",
        "recommended_model": winner,
        "models": {alias: MODEL_SPECS[alias] for alias in aliases},
        "results": summary_rows,
        "warning": "test 점수는 최종 설정 확정 후 한 번만 해석할 것",
    }
    atomic_write_json(summary_json_path, payload)

    print("\n=== 임베딩 비교 결과(dev) ===")
    for row in sorted(dev_rows, key=lambda item: item[f"ndcg@{eval_k}"], reverse=True):
        print(
            f"{row['model']:8s} | nDCG={row[f'ndcg@{eval_k}']:.4f} | "
            f"Recall={row[f'recall@{eval_k}']:.4f} | MRR={row[f'mrr@{eval_k}']:.4f}"
        )
    print(f"추천 모델: {winner}")
    print(f"요약: {summary_json_path}")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.pool_k < 1 or args.eval_k < 1:
        raise SystemExit("batch-size, pool-k, eval-k는 1 이상이어야 합니다.")
    if args.pool_k < args.eval_k:
        raise SystemExit("공정한 평가를 위해 --pool-k는 --eval-k 이상이어야 합니다.")
    aliases = parse_models(args.models)
    metadata_path = Path(args.metadata).expanduser().resolve()
    queries_path = Path(args.queries).expanduser().resolve()
    judgments_path = Path(args.judgments).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    queries = load_queries(queries_path)
    print("정책 의미 청크 메타데이터 검증", flush=True)
    metadata, texts, metadata_digest = load_metadata(metadata_path)
    print(f"청크 {len(texts):,}개 | 질의 {len(queries):,}개 | 모델 {', '.join(aliases)}")

    rankings = retrieve_all_models(aliases, queries, texts, metadata_digest, args)
    if args.mode == "prepare":
        prepare_judgments(
            queries, metadata, rankings, judgments_path, args.pool_k, args.overwrite_pool
        )
        summary = {
            "script_version": SCRIPT_VERSION,
            "mode": "prepare",
            "metadata": str(metadata_path),
            "metadata_digest": metadata_digest,
            "chunk_count": len(texts),
            "query_count": len(queries),
            "models": aliases,
            "pool_k_per_model": args.pool_k,
            "judgments": str(judgments_path),
        }
        atomic_write_json(output_dir / "benchmark_prepare_summary.json", summary)
        return

    qrels, _ = load_qrels(judgments_path, args.allow_incomplete)
    evaluate(aliases, queries, metadata, rankings, qrels, args.eval_k, output_dir)


if __name__ == "__main__":
    main()

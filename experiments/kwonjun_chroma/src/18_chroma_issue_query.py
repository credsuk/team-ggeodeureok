#!/usr/bin/env python3
"""Chroma 청크 검색을 검수된 이슈·전파 타임라인·예측과 연결한다."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

try:
    import chromadb
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError as error:
    raise SystemExit(
        "chromadb, numpy, pyarrow, torch, sentence-transformers가 필요합니다."
    ) from error


SCRIPT_VERSION = "2026-10-20-chroma-issue-query-v1"
MODEL_NAME = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
MODEL_REVISION = "55ec6e9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chroma 기반 이슈·전파·예측 통합 검색")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k-issues", type=int, default=3)
    parser.add_argument("--candidate-chunks", type=int, default=50)
    parser.add_argument("--evidence-per-issue", type=int, default=5)
    parser.add_argument("--min-cluster-similarity", type=float, default=0.55)
    parser.add_argument("--chroma-dir", default="data/chroma/arctic_policy_v1")
    parser.add_argument("--collection", default="council_policy_arctic_v1")
    parser.add_argument(
        "--assignments",
        default=(
            "data/index/arctic_ko_semantic_test_v2/clustering_k160/"
            "cluster_assignments.parquet"
        ),
    )
    parser.add_argument("--timelines", default="outputs/issue_timeline_final/issue_timelines.json")
    parser.add_argument("--forecasts", default="outputs/issue_forecast_final/issue_forecasts.json")
    parser.add_argument("--model-cache", default="models/huggingface")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--council")
    parser.add_argument("--committee")
    parser.add_argument("--year", type=int)
    parser.add_argument("--date-from")
    parser.add_argument("--date-to")
    parser.add_argument("--output-dir", default="outputs/chroma_issue_search")
    return parser.parse_args()


def clean(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def safe_int(value, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def string_list(value) -> list[str]:
    if isinstance(value, list):
        return [clean(item) for item in value if clean(item)]
    cleaned = clean(value)
    return [part.strip() for part in cleaned.split("|") if part.strip()]


def slug(value: str) -> str:
    result = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", value).strip("_")
    return result[:60] or "query"


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA를 사용할 수 없습니다.")
    return requested


def normalize_date(value: str | None, name: str) -> int | None:
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    if len(digits) != 8:
        raise SystemExit(f"{name}은 YYYYMMDD 형식이어야 합니다: {value}")
    return int(digits)


def build_where(args: argparse.Namespace) -> dict | None:
    conditions = []
    if args.council:
        conditions.append({"council": {"$eq": args.council}})
    if args.committee:
        conditions.append({"committee": {"$eq": args.committee}})
    if args.year:
        conditions.append({"meeting_year": {"$eq": args.year}})
    date_from = normalize_date(args.date_from, "--date-from")
    date_to = normalize_date(args.date_to, "--date-to")
    if date_from is not None:
        conditions.append({"meeting_date_int": {"$gte": date_from}})
    if date_to is not None:
        conditions.append({"meeting_date_int": {"$lte": date_to}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def encode_query(args: argparse.Namespace) -> tuple[np.ndarray, str, str | None]:
    device = resolve_device(args.device)
    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        cache_folder=str(Path(args.model_cache).expanduser().resolve()),
        device=device,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    model.max_seq_length = 2048
    model.eval()
    prompt_name = "query" if "query" in (getattr(model, "prompts", {}) or {}) else None
    kwargs = {
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }
    if prompt_name:
        kwargs["prompt_name"] = prompt_name
    vector = np.asarray(model.encode([args.query], **kwargs), dtype=np.float32)
    return vector, device, prompt_name


def load_assignment_lookup(path: Path) -> tuple[dict[str, dict], dict[int, dict]]:
    if not path.exists():
        raise SystemExit(f"군집 배정 파일이 없습니다: {path}")
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    required = {"cluster_id", "cluster_similarity"}
    if not required <= available:
        raise SystemExit("군집 배정 파일에 cluster_id, cluster_similarity가 필요합니다.")
    columns = [name for name in ("chunk_id", "embedding_row", "cluster_id", "cluster_similarity") if name in available]
    by_chunk = {}
    by_row = {}
    for batch in parquet.iter_batches(batch_size=8192, columns=columns):
        for row in batch.to_pylist():
            cluster_id = safe_int(row.get("cluster_id"))
            similarity = safe_float(row.get("cluster_similarity"), -1.0)
            item = {"cluster_id": cluster_id, "cluster_similarity": similarity}
            chunk_id = clean(row.get("chunk_id"))
            embedding_row = safe_int(row.get("embedding_row"))
            if chunk_id:
                previous = by_chunk.get(chunk_id)
                if previous is None or similarity > previous["cluster_similarity"]:
                    by_chunk[chunk_id] = item
            if embedding_row >= 0:
                previous = by_row.get(embedding_row)
                if previous is None or similarity > previous["cluster_similarity"]:
                    by_row[embedding_row] = item
    if not by_chunk and not by_row:
        raise SystemExit("군집 배정에서 chunk_id 또는 embedding_row를 읽지 못했습니다.")
    return by_chunk, by_row


def load_json_list(path: Path, required: bool = True) -> list[dict]:
    if not path.exists():
        if required:
            raise SystemExit(f"입력 JSON이 없습니다: {path}")
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise SystemExit(f"JSON 배열이어야 합니다: {path}")
    return value


def search_chunks(args: argparse.Namespace, vector: np.ndarray) -> tuple[list[dict], float]:
    client = chromadb.PersistentClient(path=str(Path(args.chroma_dir).expanduser().resolve()))
    collection = client.get_collection(args.collection, embedding_function=None)
    started = time.perf_counter()
    result = collection.query(
        query_embeddings=vector.tolist(),
        n_results=args.candidate_chunks,
        where=build_where(args),
        include=["documents", "metadatas", "distances"],
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    rows = []
    for rank, (chunk_id, document, metadata, distance) in enumerate(zip(
        result["ids"][0],
        (result.get("documents") or [[]])[0],
        (result.get("metadatas") or [[]])[0],
        (result.get("distances") or [[]])[0],
    ), start=1):
        metadata = metadata or {}
        rows.append({
            "retrieval_rank": rank,
            "chunk_id": chunk_id,
            "embedding_row": safe_int(metadata.get("embedding_row")),
            "retrieval_similarity": 1.0 - float(distance),
            "council": clean(metadata.get("council")),
            "committee": clean(metadata.get("committee")),
            "meeting_date": clean(metadata.get("meeting_date")),
            "meeting_name": clean(metadata.get("meeting_name")),
            "agenda_title": clean(
                metadata.get("agenda_title_effective")
                or metadata.get("evidence_title")
                or metadata.get("agenda_title")
            ),
            "agenda_title_raw": clean(metadata.get("agenda_title_raw")),
            "agenda_confidence": clean(metadata.get("agenda_confidence")),
            "speaker_label": clean(metadata.get("speaker_label")),
            "evidence_ref": clean(metadata.get("evidence_ref")),
            "text": clean(document),
        })
    return rows, elapsed_ms


def aggregate_issues(
    chunks: list[dict],
    by_chunk: dict[str, dict],
    by_row: dict[int, dict],
    timelines: list[dict],
    forecasts: list[dict],
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    timeline_by_cluster = {safe_int(row.get("cluster_id")): row for row in timelines}
    forecast_by_issue = {clean(row.get("issue_id")): row for row in forecasts}
    grouped = defaultdict(list)
    unmatched = 0
    rejected_similarity = 0
    rejected_unreviewed = 0
    for chunk in chunks:
        assignment = by_chunk.get(chunk["chunk_id"]) or by_row.get(chunk["embedding_row"])
        if not assignment:
            unmatched += 1
            continue
        cluster_similarity = assignment["cluster_similarity"]
        if cluster_similarity < args.min_cluster_similarity:
            rejected_similarity += 1
            continue
        cluster_id = assignment["cluster_id"]
        if cluster_id not in timeline_by_cluster:
            rejected_unreviewed += 1
            continue
        rank_discount = 1.0 / math.log2(chunk["retrieval_rank"] + 1.0)
        contribution = max(0.0, chunk["retrieval_similarity"]) * max(0.0, cluster_similarity) * rank_discount
        grouped[cluster_id].append({
            **chunk,
            "cluster_similarity": cluster_similarity,
            "score_contribution": contribution,
        })

    scored = []
    for cluster_id, hits in grouped.items():
        hits.sort(key=lambda row: (-row["score_contribution"], row["retrieval_rank"]))
        best = hits[0]["score_contribution"]
        support_bonus = min(0.15, 0.03 * math.log1p(len(hits)))
        top_support = sum(row["score_contribution"] for row in hits[:5]) / min(5, len(hits))
        score = 0.65 * best + 0.35 * top_support + support_bonus
        scored.append((score, cluster_id, hits))
    scored.sort(key=lambda item: (item[0], len(item[2])), reverse=True)

    results = []
    for rank, (score, cluster_id, hits) in enumerate(scored[:args.top_k_issues], start=1):
        timeline = timeline_by_cluster[cluster_id]
        issue_id = clean(timeline.get("issue_id"))
        results.append({
            "rank": rank,
            "issue_search_score": round(score, 6),
            "matched_chunk_count": len(hits),
            "query_evidence": [
                {**row, "score_contribution": round(row["score_contribution"], 6)}
                for row in hits[:args.evidence_per_issue]
            ],
            **timeline,
            "forecast": forecast_by_issue.get(issue_id),
        })
    diagnostics = {
        "retrieved_chunk_count": len(chunks),
        "matched_reviewed_chunk_count": sum(len(hits) for _, _, hits in scored),
        "unmatched_assignment_count": unmatched,
        "below_cluster_similarity_count": rejected_similarity,
        "unreviewed_cluster_count": rejected_unreviewed,
        "matched_issue_count": len(scored),
    }
    return results, diagnostics


def print_result(args: argparse.Namespace, results: list[dict], diagnostics: dict, elapsed_ms: float) -> None:
    print(f"\n질의: {args.query}")
    print(f"Chroma 후보 검색: {elapsed_ms:.2f} ms | {diagnostics['retrieved_chunk_count']}개")
    print(
        f"검수 이슈 연결: {diagnostics['matched_reviewed_chunk_count']}개 청크 / "
        f"{diagnostics['matched_issue_count']}개 이슈"
    )
    for row in results:
        route = " → ".join(
            f"{event.get('council')}({event.get('first_date')})"
            for event in row.get("propagation", [])[:8]
        )
        print(f"\n[{row['rank']}] {row.get('issue_name')} ({row.get('issue_id')})")
        print(f"  점수: {row['issue_search_score']:.4f} | 검색 근거 {row['matched_chunk_count']}개")
        print(
            f"  최초 관측: {row.get('first_observed_date')} / "
            f"{', '.join(string_list(row.get('origin_councils')))}"
        )
        print(f"  전파 순서: {route or '후속 의회 없음'}")
        if row.get("query_evidence"):
            evidence = row["query_evidence"][0]
            print(
                f"  대표 근거: {evidence['council']} {evidence['meeting_date']} | "
                f"{evidence['agenda_title']}"
            )
            print(f"  원문: {evidence['text'][:300]}")
        forecast = row.get("forecast")
        if forecast:
            print(
                f"  예측: {forecast.get('predicted_direction')} | "
                f"다음 달 {forecast.get('predicted_mention_count')}건 | "
                f"확산확률 {safe_float(forecast.get('spread_probability')):.3f}"
            )


def main() -> None:
    args = parse_args()
    if min(args.top_k_issues, args.candidate_chunks, args.evidence_per_issue) < 1:
        raise SystemExit("top-k와 후보·근거 개수는 1 이상이어야 합니다.")
    if not -1.0 <= args.min_cluster_similarity <= 1.0:
        raise SystemExit("--min-cluster-similarity는 -1 이상 1 이하여야 합니다.")
    vector, device, prompt_name = encode_query(args)
    chunks, elapsed_ms = search_chunks(args, vector)
    by_chunk, by_row = load_assignment_lookup(Path(args.assignments).expanduser().resolve())
    timelines = load_json_list(Path(args.timelines).expanduser().resolve())
    forecasts = load_json_list(Path(args.forecasts).expanduser().resolve(), required=False)
    results, diagnostics = aggregate_issues(
        chunks, by_chunk, by_row, timelines, forecasts, args
    )
    print_result(args, results, diagnostics, elapsed_ms)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{slug(args.query)}.json"
    payload = {
        "script_version": SCRIPT_VERSION,
        "security": "로컬 Chroma·군집·타임라인·예측 파일만 사용",
        "query": args.query,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "query_prompt": prompt_name,
        "device": device,
        "filters": build_where(args),
        "chroma_elapsed_ms": round(elapsed_ms, 3),
        "diagnostics": diagnostics,
        "results": results,
        "interpretation_note": (
            "전파는 제공 데이터 내 최초 관측 순서이며 실제 정책 인과관계를 의미하지 않습니다. "
            "예측은 제공 데이터 기준일까지의 월별 관측만 사용합니다."
        ),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n통합 결과 JSON: {output_path}")


if __name__ == "__main__":
    main()

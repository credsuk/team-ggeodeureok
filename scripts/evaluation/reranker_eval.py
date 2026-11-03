from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import statistics
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_BASE = "BAAI/bge-reranker-v2-m3"
MODEL_KOREAN = "dragonkue/bge-reranker-v2-m3-ko"
SETTINGS = ("BAAI only", "Korean only", "Equal-weight ensemble")
REVIEW_FIELDS = [
    "query_id", "query_type", "question", "reranker_query", "candidate_id", "date", "assembly",
    "meeting_title", "agenda", "candidate_text", "relevance", "review_notes",
]


def build_reranker_query(issue: str, keywords: tuple[str, ...]) -> str:
    return " ".join([issue.strip(), *(item.strip() for item in keywords if item.strip())]).strip()


def blind_candidate_order(query_id: str, evidence: list[Any]) -> list[Any]:
    """Return a repeatable order that does not expose the hybrid retrieval rank."""
    return sorted(
        evidence,
        key=lambda item: hashlib.sha256(f"{query_id}\0{item.id}".encode("utf-8")).hexdigest(),
    )


def cached_revision(model_id: str, cache_root: Path | None = None) -> str:
    if cache_root is None:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
        except ImportError:
            return ""
        cache_root = Path(HF_HUB_CACHE)
    ref_path = cache_root / f"models--{model_id.replace('/', '--')}" / "refs" / "main"
    return ref_path.read_text(encoding="utf-8").strip() if ref_path.is_file() else ""


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float | None:
    if k < 1:
        raise ValueError("k must be at least 1")
    grades = list(relevance.values())
    if not any(grade > 0 for grade in grades):
        return None

    def dcg(values: list[int]) -> float:
        return sum((2**grade - 1) / math.log2(position + 2) for position, grade in enumerate(values[:k]))

    actual = [relevance.get(candidate_id, 0) for candidate_id in ranked_ids]
    ideal = sorted(grades, reverse=True)
    ideal_score = dcg(ideal)
    if ideal_score == 0:
        return None
    return dcg(actual) / ideal_score


def mrr_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float | None:
    if k < 1:
        raise ValueError("k must be at least 1")
    if not any(grade > 0 for grade in relevance.values()):
        return None
    for position, candidate_id in enumerate(ranked_ids[:k], start=1):
        if relevance.get(candidate_id, 0) > 0:
            return 1.0 / position
    return 0.0


def _split_values(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split("|") if part.strip())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")


def prepare_review(queries_path: Path, output_path: Path, *, candidate_limit: int = 300, pool_size: int = 30) -> int:
    if candidate_limit < 1 or pool_size < 1:
        raise ValueError("candidate_limit and pool_size must be positive")
    query_rows = _read_csv(queries_path)
    if not query_rows:
        raise ValueError(f"질문 CSV가 비어 있습니다: {queries_path}")
    ids = [row.get("query_id", "").strip() for row in query_rows]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("query_id는 비어 있지 않고 중복되지 않아야 합니다.")

    from src.core.config import Settings
    from src.core.models import QueryIntent
    from src.search.retrieval import load_retriever
    import chromadb

    settings = replace(Settings.from_env(), candidate_limit=candidate_limit)
    if not settings.allow_model_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    retriever = load_retriever(settings, client=client)
    output_rows: list[dict[str, Any]] = []

    for query_row in query_rows:
        query_id = query_row["query_id"].strip()
        intent = QueryIntent(
            issue=query_row.get("issue", "").strip() or query_row.get("question", "").strip(),
            keywords=_split_values(query_row.get("keywords", "")),
            assemblies=_split_values(query_row.get("assemblies", "")),
            date_start=query_row.get("date_start", "").strip() or None,
            date_end=query_row.get("date_end", "").strip() or None,
            request_type="issue_flow",
        )
        reranker_query = build_reranker_query(intent.issue, intent.keywords)
        evidence = blind_candidate_order(query_id, retriever.retrieve(intent)[:pool_size])
        for item in evidence:
            output_rows.append({
                "query_id": query_id,
                "query_type": query_row.get("query_type", "issue").strip(),
                "question": query_row.get("question", "").strip(),
                "reranker_query": reranker_query,
                "candidate_id": item.id,
                "date": item.date,
                "assembly": item.assembly,
                "meeting_title": item.meeting_title,
                "agenda": item.agenda,
                "candidate_text": item.text[:4000],
                "relevance": "",
                "review_notes": "",
            })
        print(f"{query_id}: retrieved={len(evidence)}")

    _write_csv(output_path, REVIEW_FIELDS, output_rows)
    metadata_path = output_path.with_name(f"{output_path.stem}.meta.json")
    _write_json(metadata_path, {
        "query_count": len(query_rows),
        "query_ids": ids,
        "data_version": settings.data_version,
        "chroma_collection": settings.collection_name,
        "chroma_record_count": int(retriever.collection.count()),
        "embedding_model": settings.embedding_model,
        "retrieval": {
            "dense_limit": settings.dense_limit,
            "lexical_limit": settings.lexical_limit,
            "candidate_limit": candidate_limit,
            "review_pool_size": pool_size,
            "dense_rrf_weight": settings.dense_rrf_weight,
            "bm25_rrf_weight": settings.lexical_rrf_weight,
        },
        "comparison_settings": [
            {"name": "BAAI only", "models": [MODEL_BASE]},
            {"name": "Korean only", "models": [MODEL_KOREAN]},
            {"name": "Equal-weight ensemble", "models": [MODEL_BASE, MODEL_KOREAN], "weights": [1.0, 1.0]},
        ],
        "model_revisions": {
            MODEL_BASE: cached_revision(MODEL_BASE),
            MODEL_KOREAN: cached_revision(MODEL_KOREAN),
        },
        "scope_note": "Reranker comparison is conditional on the same first-stage hybrid retrieval candidate pool; it does not measure first-stage recall.",
    })
    return len(output_rows)


def _load_review_groups(review_path: Path) -> dict[str, list[dict[str, str]]]:
    rows = _read_csv(review_path)
    if not rows:
        raise ValueError(f"검토 CSV가 비어 있습니다: {review_path}")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for line_number, row in enumerate(rows, start=2):
        query_id = row.get("query_id", "").strip()
        candidate_id = row.get("candidate_id", "").strip()
        if not query_id or not candidate_id or not row.get("candidate_text", "").strip():
            raise ValueError(f"{line_number}행: query_id, candidate_id, candidate_text가 필요합니다.")
        raw_label = row.get("relevance", "").strip()
        if raw_label not in {"0", "1", "2"}:
            raise ValueError(f"{line_number}행 relevance는 0, 1, 2 중 하나여야 합니다. 현재값={raw_label!r}")
        groups[query_id].append(row)
    for query_id, items in groups.items():
        candidate_ids = [row["candidate_id"].strip() for row in items]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"{query_id}: candidate_id가 중복되었습니다.")
    return groups


def evaluate_review(
    review_path: Path,
    output_dir: Path,
    *,
    k: int = 10,
    device: str | None = None,
    rankers: dict[str, Any] | None = None,
) -> dict[str, Path]:
    if k < 1:
        raise ValueError("k must be at least 1")
    groups = _load_review_groups(review_path)

    from src.core.models import Evidence

    if rankers is None:
        from src.core.config import Settings
        from src.search.reranking import CrossEncoderReranker

        settings = Settings.from_env()
        model_device = device or settings.reranker_device
        if not settings.allow_model_download:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("평가에 sentence-transformers가 필요합니다.") from exc

        # Load each checkpoint once and share it across the three comparisons.
        base_model = CrossEncoder(MODEL_BASE, device=model_device)
        korean_model = CrossEncoder(MODEL_KOREAN, device=model_device)
        rankers = {
            "BAAI only": CrossEncoderReranker(MODEL_BASE, device=model_device, model=base_model),
            "Korean only": CrossEncoderReranker(MODEL_KOREAN, device=model_device, model=korean_model),
            "Equal-weight ensemble": CrossEncoderReranker(
                [MODEL_BASE, MODEL_KOREAN], device=model_device,
                model=[base_model, korean_model], model_weights=[1.0, 1.0],
            ),
        }
    elif set(rankers) != set(SETTINGS):
        raise ValueError(f"rankers는 다음 세 설정을 모두 제공해야 합니다: {SETTINGS}")

    ranking_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    no_positive_queries = 0
    no_evidence_top1: dict[str, list[int]] = {setting: [] for setting in SETTINGS}

    for query_id, rows in groups.items():
        first = rows[0]
        evidence = [Evidence(
            id=row["candidate_id"].strip(),
            text=row["candidate_text"],
            date=row.get("date", ""),
            assembly=row.get("assembly", ""),
            meeting_title=row.get("meeting_title", ""),
            agenda=row.get("agenda", ""),
        ) for row in rows]
        relevance = {row["candidate_id"].strip(): int(row["relevance"]) for row in rows}
        has_positive = any(value > 0 for value in relevance.values())
        if not has_positive:
            no_positive_queries += 1
        query_text = first.get("reranker_query", "").strip() or first.get("question", "").strip()

        for setting in SETTINGS:
            ranked = rankers[setting].rerank(query_text, evidence, limit=len(evidence))
            ranked_ids = [item.id for item in ranked]
            ndcg = ndcg_at_k(ranked_ids, relevance, k)
            mrr = mrr_at_k(ranked_ids, relevance, k)
            first_grade = relevance.get(ranked_ids[0], 0) if ranked_ids else 0
            if first.get("query_type", "").strip() == "no_evidence_control":
                no_evidence_top1[setting].append(first_grade)
            per_query_rows.append({
                "query_id": query_id,
                "query_type": first.get("query_type", ""),
                "setting": setting,
                f"NDCG@{k}": "" if ndcg is None else ndcg,
                f"MRR@{k}": "" if mrr is None else mrr,
                "top1_relevance": first_grade,
                "has_positive_label": has_positive,
            })
            for position, item in enumerate(ranked, start=1):
                ranking_rows.append({
                    "query_id": query_id,
                    "setting": setting,
                    "rank": position,
                    "candidate_id": item.id,
                    "relevance": relevance[item.id],
                    "ranker_score": item.score,
                })

    model_revisions = {
        MODEL_BASE: cached_revision(MODEL_BASE),
        MODEL_KOREAN: cached_revision(MODEL_KOREAN),
    }
    summary_rows: list[dict[str, Any]] = []
    for setting in SETTINGS:
        matching = [row for row in per_query_rows if row["setting"] == setting and row["has_positive_label"]]
        ndcg_values = [float(row[f"NDCG@{k}"]) for row in matching if row[f"NDCG@{k}"] != ""]
        mrr_values = [float(row[f"MRR@{k}"]) for row in matching if row[f"MRR@{k}"] != ""]
        control = no_evidence_top1[setting]
        summary_rows.append({
            "setting": setting,
            "model_base": MODEL_BASE,
            "model_base_revision": model_revisions[MODEL_BASE],
            "model_korean": MODEL_KOREAN,
            "model_korean_revision": model_revisions[MODEL_KOREAN],
            "ensemble_weights": "1.0,1.0",
            "reviewed_candidate_rows": sum(len(items) for items in groups.values()),
            "mean_review_pool_size": statistics.mean(len(items) for items in groups.values()),
            "judged_queries_with_relevant_candidates": len(matching),
            "queries_with_no_positive_label_in_pool": no_positive_queries,
            f"macro_NDCG@{k}": statistics.mean(ndcg_values) if ndcg_values else "",
            f"macro_MRR@{k}": statistics.mean(mrr_values) if mrr_values else "",
            "no_evidence_control_count": len(control),
            "no_evidence_control_mean_top1_relevance": statistics.mean(control) if control else "",
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "summary.csv",
        "per_query": output_dir / "per_query.csv",
        "rankings": output_dir / "rankings.csv",
    }
    _write_csv(paths["summary"], list(summary_rows[0]), summary_rows)
    _write_csv(paths["per_query"], list(per_query_rows[0]), per_query_rows)
    _write_csv(paths["rankings"], list(ranking_rows[0]), ranking_rows)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="BAAI-only / Korean-only / ensemble reranker evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="create a blinded human-review candidate CSV")
    prepare.add_argument("--queries", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--candidate-limit", type=int, default=300)
    prepare.add_argument("--pool-size", type=int, default=30)
    evaluate = subparsers.add_parser("evaluate", help="score a completed relevance-labeled review CSV")
    evaluate.add_argument("--review", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--k", type=int, default=10)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            count = prepare_review(args.queries, args.output, candidate_limit=args.candidate_limit, pool_size=args.pool_size)
            print(f"review candidates saved: {count:,} rows -> {args.output}")
            print(f"metadata: {args.output.with_name(f'{args.output.stem}.meta.json')}")
        else:
            paths = evaluate_review(args.review, args.output_dir, k=args.k, device=args.device)
            for key, path in paths.items():
                print(f"{key}: {path}")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"오류: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

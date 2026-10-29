#!/usr/bin/env python3
"""블라인드 관련성 라벨로 Arctic와 KURE의 빠른 검색 성능을 비교한다."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, default=PROJECT_ROOT / "outputs/retrieval_eval/arctic_vs_kure_quick_v1/pool_provenance.parquet")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "outputs/retrieval_eval/arctic_vs_kure_quick_v1/auto_judgements/relevance_labels.jsonl")
    parser.add_argument("--arctic-summary", type=Path, default=PROJECT_ROOT / "data/embedding_eval_vectors/arctic_raw_60k/embedding_summary.json")
    parser.add_argument("--kure-summary", type=Path, default=PROJECT_ROOT / "data/embedding_eval_vectors/kure_raw_60k/embedding_summary.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/retrieval_eval/arctic_vs_kure_quick_v1/comparison")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--practical-margin", type=float, default=0.02)
    return parser.parse_args()


def read_labels(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    return {row["pair_id"]: row for row in rows}


def dcg(grades: list[int], k: int = 10) -> float:
    return sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades[:k]))


def metrics(grades: list[int], pool_grades: list[int]) -> dict[str, float]:
    ideal = dcg(sorted(pool_grades, reverse=True))
    first = next((index + 1 for index, grade in enumerate(grades) if grade > 0), None)
    denominator = min(10, len(grades))
    return {
        "ndcg_at_10": dcg(grades) / ideal if ideal else 0.0,
        "precision_at_10": sum(grade > 0 for grade in grades[:10]) / denominator if denominator else 0.0,
        "direct_precision_at_10": sum(grade == 2 for grade in grades[:10]) / denominator if denominator else 0.0,
        "mrr": 1.0 / first if first else 0.0,
    }


def main() -> None:
    args = parse_args()
    paths = [args.provenance, args.labels, args.arctic_summary, args.kure_summary]
    for path in paths:
        if not path.expanduser().resolve().is_file():
            raise FileNotFoundError(path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = read_labels(args.labels.expanduser().resolve())
    rows = pq.read_table(args.provenance.expanduser().resolve()).to_pylist()
    missing = [row["pair_id"] for row in rows if row["pair_id"] not in labels]
    if missing:
        raise RuntimeError(f"라벨이 없는 pair: {len(missing):,}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["query_id"]].append(row)

    names = ["arctic", "kure"]
    metric_names = ["ndcg_at_10", "precision_at_10", "direct_precision_at_10", "mrr"]
    values = {name: {metric: [] for metric in metric_names} for name in names}
    per_query = []
    for query_id in sorted(grouped):
        query_rows = grouped[query_id]
        pool_grades = [int(labels[row["pair_id"]]["grade"]) for row in query_rows]
        item: dict[str, Any] = {"query_id": query_id, "pool_size": len(query_rows)}
        for name in names:
            ranked = sorted(
                (row for row in query_rows if row[f"{name}_rank"] is not None),
                key=lambda row: int(row[f"{name}_rank"]),
            )
            grades = [int(labels[row["pair_id"]]["grade"]) for row in ranked]
            result = metrics(grades, pool_grades)
            for metric, value in result.items():
                values[name][metric].append(value)
                item[f"{name}_{metric}"] = round(value, 6)
        per_query.append(item)

    aggregate = {
        name: {metric: round(float(np.mean(metric_values)), 6) for metric, metric_values in result.items()}
        for name, result in values.items()
    }
    arctic = np.asarray(values["arctic"]["ndcg_at_10"], dtype=np.float64)
    kure = np.asarray(values["kure"]["ndcg_at_10"], dtype=np.float64)
    differences = kure - arctic
    rng = np.random.default_rng(args.seed)
    bootstrap = np.empty(args.bootstrap, dtype=np.float64)
    for index in range(args.bootstrap):
        sample = rng.integers(0, len(differences), size=len(differences))
        bootstrap[index] = differences[sample].mean()
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    difference = float(differences.mean())
    clear = bool(lower > 0 or upper < 0)

    summaries = {
        "arctic": json.loads(args.arctic_summary.read_text(encoding="utf-8")),
        "kure": json.loads(args.kure_summary.read_text(encoding="utf-8")),
    }
    throughput = {
        name: float(summary.get("throughput_rows_per_second") or 0.0)
        for name, summary in summaries.items()
    }
    if clear and abs(difference) >= args.practical_margin:
        winner = "kure" if difference > 0 else "arctic"
        reason = "nDCG@10 차이가 통계적으로 명확하고 실질 차이 기준을 통과"
    else:
        winner = max(throughput, key=throughput.get)
        reason = "검색 정확도 실질 동률로 문서 임베딩 처리율이 높은 모델 선택"

    report = {
        "evaluation": "quick_screening_v1", "query_count": len(grouped),
        "label_count": len(labels), "aggregate_metrics": aggregate,
        "kure_minus_arctic_ndcg_at_10": round(difference, 6),
        "ndcg_difference_95pct_ci": [round(float(lower), 6), round(float(upper), 6)],
        "statistically_clear_95pct": clear, "practical_margin": args.practical_margin,
        "throughput_rows_per_second": throughput, "recommended_model": winner,
        "selection_reason": reason,
        "model_identity": {name: {key: value for key, value in summary.items() if key in {"model_name", "revision", "dimension", "text_view"}} for name, summary in summaries.items()},
    }
    summary_path = output_dir / "embedding_model_comparison.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output_dir / "embedding_model_per_query.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_query[0]))
        writer.writeheader(); writer.writerows(per_query)

    print("\n=== Arctic/KURE 빠른 성능 비교 완료 ===")
    for name in names:
        result = aggregate[name]
        print(f"{name:7s} | nDCG@10={result['ndcg_at_10']:.4f} | P@10={result['precision_at_10']:.4f} | DirectP@10={result['direct_precision_at_10']:.4f} | MRR={result['mrr']:.4f} | embed={throughput[name]:.2f}행/초")
    print(f"KURE-Arctic nDCG@10: {difference:.4f} | 95% CI [{lower:.4f}, {upper:.4f}]")
    print(f"추천 모델: {winner} | {reason}")
    print(f"summary: {summary_path}")
    print(f"per-query: {csv_path}")


if __name__ == "__main__":
    main()

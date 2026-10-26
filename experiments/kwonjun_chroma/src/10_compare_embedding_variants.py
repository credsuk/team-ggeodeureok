#!/usr/bin/env python3
"""자동 관련성 라벨로 raw/enriched 검색 성능을 공정하게 비교한다."""

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
    parser.add_argument(
        "--provenance", type=Path,
        default=PROJECT_ROOT / "outputs" / "retrieval_eval" / "arctic_raw_vs_enriched_v1" / "pool_provenance.parquet",
    )
    parser.add_argument(
        "--labels", type=Path,
        default=PROJECT_ROOT / "outputs" / "retrieval_eval" / "arctic_raw_vs_enriched_v1" / "auto_judgements" / "relevance_labels.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "retrieval_eval" / "arctic_raw_vs_enriched_v1" / "comparison",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261022)
    return parser.parse_args()


def read_labels(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            result[row["pair_id"]] = row
    return result


def dcg(grades: list[int], k: int) -> float:
    return sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades[:k]))


def metrics(grades: list[int], all_pool_grades: list[int], k: int = 10) -> dict[str, float]:
    ideal = sorted(all_pool_grades, reverse=True)
    denominator = dcg(ideal, k)
    relevant_total = sum(grade > 0 for grade in all_pool_grades)
    relevant_at_20 = sum(grade > 0 for grade in grades[:20])
    first_relevant = next((index + 1 for index, grade in enumerate(grades) if grade > 0), None)
    return {
        "ndcg_at_10": dcg(grades, k) / denominator if denominator else 0.0,
        "precision_at_10": sum(grade > 0 for grade in grades[:10]) / min(10, len(grades)) if grades else 0.0,
        "direct_precision_at_10": sum(grade == 2 for grade in grades[:10]) / min(10, len(grades)) if grades else 0.0,
        "pooled_recall_at_20": relevant_at_20 / relevant_total if relevant_total else 0.0,
        "mrr": 1.0 / first_relevant if first_relevant else 0.0,
    }


def main() -> None:
    args = parse_args()
    provenance_path = args.provenance.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (provenance_path, labels_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = read_labels(labels_path)
    provenance = pq.read_table(provenance_path).to_pylist()
    missing = [row["pair_id"] for row in provenance if row["pair_id"] not in labels]
    if missing:
        raise RuntimeError(f"라벨이 없는 pair가 있습니다: {len(missing):,}")

    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in provenance:
        by_query[row["query_id"]].append(row)
    per_query_rows = []
    metric_names = ["ndcg_at_10", "precision_at_10", "direct_precision_at_10", "pooled_recall_at_20", "mrr"]
    system_values: dict[str, dict[str, list[float]]] = {
        system: {name: [] for name in metric_names} for system in ("raw", "enriched")
    }
    for query_id in sorted(by_query):
        rows = by_query[query_id]
        pool_grades = [int(labels[row["pair_id"]]["grade"]) for row in rows]
        query_result: dict[str, Any] = {"query_id": query_id, "pool_size": len(rows)}
        for system in ("raw", "enriched"):
            ranked = sorted(
                (row for row in rows if row[f"{system}_rank"] is not None),
                key=lambda row: int(row[f"{system}_rank"]),
            )
            grades = [int(labels[row["pair_id"]]["grade"]) for row in ranked]
            values = metrics(grades, pool_grades)
            for name, value in values.items():
                system_values[system][name].append(value)
                query_result[f"{system}_{name}"] = round(value, 6)
        query_result["ndcg10_winner"] = (
            "enriched" if query_result["enriched_ndcg_at_10"] > query_result["raw_ndcg_at_10"]
            else "raw" if query_result["raw_ndcg_at_10"] > query_result["enriched_ndcg_at_10"]
            else "tie"
        )
        per_query_rows.append(query_result)

    aggregate = {
        system: {
            name: round(float(np.mean(values)), 6)
            for name, values in metrics_by_name.items()
        }
        for system, metrics_by_name in system_values.items()
    }
    raw_ndcg = np.asarray(system_values["raw"]["ndcg_at_10"], dtype=np.float64)
    enriched_ndcg = np.asarray(system_values["enriched"]["ndcg_at_10"], dtype=np.float64)
    differences = enriched_ndcg - raw_ndcg
    rng = np.random.default_rng(args.seed)
    bootstrap_means = np.empty(args.bootstrap, dtype=np.float64)
    for index in range(args.bootstrap):
        sample = rng.integers(0, len(differences), size=len(differences))
        bootstrap_means[index] = differences[sample].mean()
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    diff_mean = float(differences.mean())
    # np.quantile 반환값으로 비교하면 numpy.bool_가 생성된다.
    # 표준 json 모듈은 numpy.bool_를 직렬화하지 못하므로 Python bool로 변환한다.
    statistically_clear = bool(lower > 0 or upper < 0)
    if statistically_clear:
        winner = "enriched" if diff_mean > 0 else "raw"
    else:
        winner = "tie"

    win_counts = {
        "raw": sum(row["ndcg10_winner"] == "raw" for row in per_query_rows),
        "enriched": sum(row["ndcg10_winner"] == "enriched" for row in per_query_rows),
        "tie": sum(row["ndcg10_winner"] == "tie" for row in per_query_rows),
    }
    summary = {
        "query_count": len(by_query),
        "label_count": len(labels),
        "aggregate_metrics": aggregate,
        "ndcg10_enriched_minus_raw": round(diff_mean, 6),
        "ndcg10_difference_95pct_ci": [round(float(lower), 6), round(float(upper), 6)],
        "statistically_clear_95pct": statistically_clear,
        "per_query_wins": win_counts,
        "recommended_text_view": winner,
        "decision_rule": "nDCG@10 차이의 query bootstrap 95% CI가 0을 넘지 않을 때만 승자 확정",
        "label_source": "local_llm_double_judge_with_auto_adjudication",
    }
    summary_path = output_dir / "embedding_variant_comparison.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output_dir / "embedding_variant_per_query.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_query_rows[0]))
        writer.writeheader()
        writer.writerows(per_query_rows)

    print("\n=== raw/enriched 검색 성능 비교 완료 ===")
    for system in ("raw", "enriched"):
        values = aggregate[system]
        print(
            f"{system:8s} | nDCG@10={values['ndcg_at_10']:.4f} | "
            f"P@10={values['precision_at_10']:.4f} | "
            f"Recall@20={values['pooled_recall_at_20']:.4f} | MRR={values['mrr']:.4f}"
        )
    print(f"enriched-raw nDCG@10: {diff_mean:.4f} | 95% CI [{lower:.4f}, {upper:.4f}]")
    print(f"질의별 승수: {win_counts}")
    print(f"추천: {winner}")
    print(f"summary: {summary_path}")
    print(f"per-query: {csv_path}")


if __name__ == "__main__":
    main()

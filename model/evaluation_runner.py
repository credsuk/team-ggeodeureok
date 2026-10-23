"""Score ranked source IDs from a human-reviewed Korean retrieval CSV."""
import argparse
import csv
import json
from pathlib import Path

try:
    from .retrieval_evaluation import ranking_metrics
except ImportError:
    from retrieval_evaluation import ranking_metrics


def evaluate_csv(input_path, output_path, k=10):
    input_path, output_path = Path(input_path), Path(output_path)
    rows = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"query_id", "query", "relevant_source_ids", "ranked_source_ids"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns: {sorted(required)}")
        for row in reader:
            relevant = {item.strip() for item in row["relevant_source_ids"].split("|") if item.strip()}
            ranked = [item.strip() for item in row["ranked_source_ids"].split("|") if item.strip()]
            if row["query"].strip() and relevant:
                rows.append({"query_id": row["query_id"], "query": row["query"],
                             "relevant_count": len(relevant), "metrics": ranking_metrics(ranked, relevant, k)})
    if not rows:
        raise ValueError("no labeled queries: add query and relevant source IDs before evaluation")
    metric_names = ["precision_at_k", "recall_at_k", "reciprocal_rank", "hit_at_k"]
    aggregate = {name: sum(row["metrics"][name] for row in rows) / len(rows) for name in metric_names}
    report = {"query_count": len(rows), "k": k, "metrics": aggregate, "queries": rows}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Human-labeled CSV")
    parser.add_argument("--output", type=Path, default=Path("evaluation-results.json"))
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be >= 1")
    report = evaluate_csv(args.input, args.output, args.k)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

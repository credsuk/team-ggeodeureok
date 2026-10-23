import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_runner import evaluate_csv


class EvaluationRunnerTests(unittest.TestCase):
    def test_evaluates_rankings_for_labeled_queries(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "labels.csv"
            output = Path(temp) / "scores.json"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["query_id", "query", "relevant_source_ids", "ranked_source_ids"])
                writer.writeheader()
                writer.writerow({"query_id":"q1", "query":"청년 주거", "relevant_source_ids":"r1|r3", "ranked_source_ids":"r1|x|r3"})
            report = evaluate_csv(source, output, k=2)
            self.assertEqual(report["query_count"], 1)
            self.assertEqual(report["metrics"]["hit_at_k"], 1.0)
            self.assertAlmostEqual(report["metrics"]["recall_at_k"], 0.5)
            self.assertTrue(output.is_file())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["query_count"], 1)

    def test_fails_when_required_labels_are_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "labels.csv"
            source.write_text("query_id,query,relevant_source_ids,ranked_source_ids\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no labeled queries"):
                evaluate_csv(source, Path(temp) / "out.json")


if __name__ == "__main__":
    unittest.main()

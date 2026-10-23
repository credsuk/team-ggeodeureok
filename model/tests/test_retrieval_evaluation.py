import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_evaluation import ranking_metrics


class RankingMetricsTests(unittest.TestCase):
    def test_calculates_precision_recall_and_reciprocal_rank(self):
        result = ranking_metrics(["a", "x", "b", "y"], {"b", "a", "c"}, k=2)
        self.assertEqual(result["precision_at_k"], 0.5)
        self.assertAlmostEqual(result["recall_at_k"], 1 / 3)
        self.assertEqual(result["reciprocal_rank"], 1.0)
        self.assertEqual(result["hit_at_k"], 1)

    def test_reports_precision_at_k_with_k_denominator_even_for_short_lists(self):
        result = ranking_metrics(["relevant"], {"relevant"}, k=10)
        self.assertEqual(result["precision_at_k"], 0.1)
        self.assertEqual(result["recall_at_k"], 1.0)

    def test_returns_zero_for_no_relevant_labels(self):
        result = ranking_metrics(["a", "b"], set(), k=2)
        self.assertEqual(result["precision_at_k"], 0.0)
        self.assertEqual(result["recall_at_k"], 0.0)
        self.assertEqual(result["reciprocal_rank"], 0.0)

    def test_reciprocal_rank_tracks_first_relevant_rank(self):
        result = ranking_metrics(["x", "a", "b"], {"a", "b"}, k=3)
        self.assertEqual(result["reciprocal_rank"], 0.5)


if __name__ == "__main__":
    unittest.main()

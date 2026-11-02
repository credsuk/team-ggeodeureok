import unittest

from src.models import Evidence
from src.reranking import CrossEncoderReranker


class FakeCrossEncoder:
    def predict(self, pairs, batch_size, show_progress_bar):
        self.pairs = pairs
        return [0.1, 0.9]


class RerankerTests(unittest.TestCase):
    def test_rescores_full_query_text_pairs_and_returns_highest_score_first(self):
        model = FakeCrossEncoder()
        reranker = CrossEncoderReranker("unused", device="cpu", model=model)
        evidence = [Evidence(id="a", text="first source"), Evidence(id="b", text="second source")]
        result = reranker.rerank("issue query", evidence, limit=1)
        self.assertEqual([item.id for item in result], ["b"])
        self.assertEqual(model.pairs[0], ["issue query", "first source"])
        self.assertEqual(result[0].score, 0.9)


if __name__ == "__main__":
    unittest.main()

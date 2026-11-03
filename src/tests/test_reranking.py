import unittest

from src.core.models import Evidence
from src.search.reranking import CrossEncoderReranker


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

    def test_multiple_rerankers_fuse_ranks_with_configured_weights(self):
        class FakeModel:
            def __init__(self, scores):
                self.scores = scores

            def predict(self, pairs, batch_size, show_progress_bar):
                return self.scores

        evidence = [Evidence(id="a", text="first source"), Evidence(id="b", text="second source")]
        reranker = CrossEncoderReranker(
            ["model-a", "model-b"], device="cpu",
            model=[FakeModel([0.9, 0.1]), FakeModel([0.1, 0.9])],
            model_weights=[3.0, 1.0], rrf_k=60,
        )
        result = reranker.rerank("issue query", evidence, limit=2)
        self.assertEqual([item.id for item in result], ["a", "b"])
        self.assertAlmostEqual(result[0].score, 3 / 61 + 1 / 62)

    def test_zero_weight_reranker_is_not_invoked(self):
        class InactiveModel:
            called = False

            def predict(self, pairs, batch_size, show_progress_bar):
                self.called = True
                raise AssertionError("0 가중치 모델은 호출되면 안 됩니다")

        class ActiveModel:
            def predict(self, pairs, batch_size, show_progress_bar):
                return [0.2, 0.8]

        inactive = InactiveModel()
        reranker = CrossEncoderReranker(
            ["inactive", "active"], device="cpu",
            model=[inactive, ActiveModel()], model_weights=[0.0, 1.0],
        )
        result = reranker.rerank(
            "issue query", [Evidence(id="a", text="first"), Evidence(id="b", text="second")], limit=2
        )
        self.assertFalse(inactive.called)
        self.assertEqual([item.id for item in result], ["b", "a"])

    def test_model_ensemble_implementation_lives_in_model_with_legacy_imports(self):
        import inspect
        from pathlib import Path
        from model.ensemble.cross_encoder import CrossEncoderReranker as ModelCrossEncoderReranker
        from model.ensemble.rank_fusion import reciprocal_rank_fusion as model_reciprocal_rank_fusion
        from src.search.ranking import reciprocal_rank_fusion as legacy_reciprocal_rank_fusion
        from src.search.reranking import CrossEncoderReranker as legacy_CrossEncoderReranker

        project_root = Path(__file__).resolve().parents[2]
        for implementation in (ModelCrossEncoderReranker, model_reciprocal_rank_fusion):
            source_path = Path(inspect.getsourcefile(implementation) or inspect.getfile(implementation)).resolve()
            self.assertTrue(source_path.is_relative_to(project_root / "model" / "ensemble"))
        self.assertIs(legacy_CrossEncoderReranker, ModelCrossEncoderReranker)
        self.assertIs(legacy_reciprocal_rank_fusion, model_reciprocal_rank_fusion)


if __name__ == "__main__":
    unittest.main()

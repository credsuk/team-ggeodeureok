import unittest

from src.core.models import Evidence, QueryIntent
from src.search.ranking import reciprocal_rank_fusion, validate_citations


class IntentTests(unittest.TestCase):
    def test_cache_key_ignores_keyword_order_and_whitespace(self):
        first = QueryIntent(issue="  school meals ", keywords=("budget", "meal"))
        second = QueryIntent(issue="school meals", keywords=("meal", "budget"))
        self.assertEqual(first.cache_key, second.cache_key)


class RankingTests(unittest.TestCase):
    def test_rrf_merges_same_source_and_keeps_best_fused_order(self):
        dense = [{"id": "a"}, {"id": "b"}]
        lexical = [{"id": "b"}, {"id": "c"}]
        result = reciprocal_rank_fusion(dense, lexical, k=0)
        self.assertEqual([item.id for item in result], ["b", "a", "c"])
        self.assertEqual(result[0].rrf_score, 1.5)

    def test_weighted_rrf_can_prioritize_dense_results(self):
        dense = [{"id": "dense-first"}, {"id": "shared"}]
        lexical = [{"id": "lexical-first"}, {"id": "shared"}]
        result = reciprocal_rank_fusion(
            dense, lexical, k=60, dense_weight=3.0, lexical_weight=1.0
        )
        self.assertLess(
            [item.id for item in result].index("dense-first"),
            [item.id for item in result].index("lexical-first"),
        )

    def test_weighted_rrf_allows_disabling_one_source_but_not_both(self):
        dense = [{"id": "dense"}]
        lexical = [{"id": "lexical"}]
        self.assertEqual(
            [item.id for item in reciprocal_rank_fusion(
                dense, lexical, k=60, dense_weight=1.0, lexical_weight=0.0
            )],
            ["dense"],
        )
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion(dense, lexical, dense_weight=0.0, lexical_weight=0.0)

    def test_citations_keep_only_ids_in_retrieved_evidence(self):
        evidence = [Evidence(id="doc:7", text="meeting excerpt", date="20240502", assembly="Seoul")]
        self.assertEqual(validate_citations(["doc:7", "made-up"], evidence), ["doc:7"])
        self.assertEqual(validate_citations(["made-up"], evidence), ["doc:7"])
        self.assertEqual(validate_citations(["made-up"], evidence, fallback_to_first=False), [])
        self.assertEqual(validate_citations(["made-up"], []), [])


if __name__ == "__main__":
    unittest.main()

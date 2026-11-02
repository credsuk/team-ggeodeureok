import unittest

from src.models import Evidence, QueryIntent
from src.ranking import reciprocal_rank_fusion, validate_citations


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

    def test_citations_keep_only_ids_in_retrieved_evidence(self):
        evidence = [Evidence(id="doc:7", text="meeting excerpt", date="20240502", assembly="Seoul")]
        self.assertEqual(validate_citations(["doc:7", "made-up"], evidence), ["doc:7"])
        self.assertEqual(validate_citations(["made-up"], evidence), ["doc:7"])
        self.assertEqual(validate_citations(["made-up"], evidence, fallback_to_first=False), [])
        self.assertEqual(validate_citations(["made-up"], []), [])


if __name__ == "__main__":
    unittest.main()

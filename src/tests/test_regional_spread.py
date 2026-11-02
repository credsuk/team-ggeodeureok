import unittest

from src.models import Evidence, QueryIntent
from src.regional_spread import build_regional_spread, select_spread_evidence


class RegionalSpreadTests(unittest.TestCase):
    def test_lists_other_councils_only_from_summary_backed_evidence(self):
        intent = QueryIntent(issue="학교 급식 예산", assemblies=("서울특별시의회",))
        evidence = [
            Evidence(id="a:1", text="", date="20240110", assembly="서울특별시의회"),
            Evidence(id="b:2", text="", date="20240220", assembly="부산광역시의회"),
            Evidence(id="c:3", text="", date="20240315", assembly="대구광역시의회"),
        ]

        result = build_regional_spread(
            intent, evidence, {"a:1", "b:2"}, other_council_search_performed=True
        )

        self.assertEqual(result["status"], "observed_related_discussions")
        self.assertEqual(result["basis_record_count"], 1)
        self.assertEqual(result["origin_councils"][0]["assembly"], "서울특별시의회")
        self.assertEqual(result["origin_record_count"], 1)
        self.assertEqual([row["assembly"] for row in result["observed_councils"]], ["부산광역시의회"])
        self.assertIn("기준 의회 서울특별시의회", result["outlook"])
        self.assertIn("입증하지 않습니다", result["outlook"])

    def test_explicit_single_council_scope_reports_spread_as_unassessable(self):
        intent = QueryIntent(issue="학교 급식 예산", assemblies=("서울특별시의회",))
        evidence = [Evidence(id="a:1", text="", date="20240110", assembly="서울특별시의회")]

        result = build_regional_spread(intent, evidence, {"a:1"})

        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertTrue(result["scope_limited"])
        self.assertIn("판단할 수 없습니다", result["outlook"])
        self.assertNotIn("확산되지 않았습니다", result["outlook"])

    def test_single_council_without_explicit_filter_does_not_claim_corpus_wide_absence(self):
        intent = QueryIntent(issue="학교 급식 예산")
        evidence = [Evidence(id="a:1", text="", date="20240110", assembly="서울특별시의회")]

        result = build_regional_spread(intent, evidence, {"a:1"})

        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertFalse(result["scope_limited"])
        self.assertIn("전체 corpus", result["outlook"])


    def test_completed_cross_council_search_with_no_summaries_does_not_claim_absence(self):
        intent = QueryIntent(issue="학교 급식 예산", assemblies=("서울특별시의회",))
        evidence = [Evidence(id="a:1", text="원문", date="20240110", assembly="서울특별시의회")]

        result = build_regional_spread(
            intent, evidence, {"a:1"}, other_council_search_performed=True
        )

        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertFalse(result["scope_limited"])
        self.assertIn("판단할 수 없", result["outlook"])
        self.assertIn("전체 corpus", result["outlook"])

    def test_spread_selection_prefers_one_record_per_other_council(self):
        evidence = [
            Evidence(id="seoul:1", text="r", assembly="서울특별시의회"),
            Evidence(id="busan:1", text="r", assembly="부산광역시의회"),
            Evidence(id="busan:2", text="r", assembly="부산광역시의회"),
            Evidence(id="daegu:1", text="r", assembly="대구광역시의회"),
        ]

        selected = select_spread_evidence(evidence, {"서울특별시의회"}, limit=2)

        self.assertEqual([item.id for item in selected], ["busan:1", "daegu:1"])


if __name__ == "__main__":
    unittest.main()

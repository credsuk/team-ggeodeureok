import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis_scope import apply_analysis_period, in_analysis_scope, normalize_council


class AnalysisPeriodTests(unittest.TestCase):
    def test_applies_project_period_when_query_has_no_year(self):
        params = apply_analysis_period({"topic": "예산", "start_year": None, "end_year": None})
        self.assertEqual((params["start_year"], params["end_year"]), ("2024", "2025"))

    def test_intersects_requested_years_with_project_period(self):
        params = apply_analysis_period({"start_year": "2023", "end_year": "2024"})
        self.assertEqual((params["start_year"], params["end_year"]), ("2024", "2024"))

    def test_out_of_scope_requested_period_remains_empty(self):
        params = apply_analysis_period({"start_year": "2026", "end_year": "2026"})
        self.assertGreater(params["start_year"], params["end_year"])

    def test_does_not_modify_the_callers_params(self):
        original = {"start_year": None, "end_year": None}
        result = apply_analysis_period(original)
        self.assertEqual(original, {"start_year": None, "end_year": None})
        self.assertEqual((result["start_year"], result["end_year"]), ("2024", "2025"))


class AnalysisScopeTests(unittest.TestCase):
    def test_accepts_valid_2024_2025_metropolitan_council(self):
        self.assertTrue(in_analysis_scope({"MTG_DE": "2025-02-16", "RASMBLY_NM": "제주특별자치도의회"}, {}))

    def test_rejects_date_outside_project_period(self):
        self.assertFalse(in_analysis_scope({"MTG_DE": "2023-12-31", "RASMBLY_NM": "경기도의회"}, {}))

    def test_rejects_misleading_substring_council_name(self):
        self.assertFalse(in_analysis_scope({"date": "20250101", "assembly": "경기도의회 자료"}, {}))

    def test_rejects_non_metropolitan_or_unknown_council(self):
        self.assertFalse(in_analysis_scope({"date": "20250101", "assembly": "수원시의회"}, {}))
        self.assertFalse(in_analysis_scope({"date": "20250101"}, {}))

    def test_council_name_aliases_normalize(self):
        self.assertEqual(normalize_council("강원도의회"), normalize_council("강원특별자치도의회"))

    def test_honors_explicit_council_filter(self):
        params = {"councils": ["제주특별자치도의회"]}
        self.assertTrue(in_analysis_scope({"date": "20250101", "assembly": "제주특별자치도의회"}, params))
        self.assertFalse(in_analysis_scope({"date": "20250101", "assembly": "충청남도의회"}, params))


if __name__ == "__main__":
    unittest.main()

import unittest

from src.core.intent import IntentExtractor
from src.core.models import QueryIntent


class FakeLLM:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def chat_json(self, system, user, schema=None):
        self.calls.append((system, user, schema))
        return self.output


class IntentExtractorTests(unittest.TestCase):
    def test_extracts_structured_issue_and_normalizes_dates(self):
        llm = FakeLLM({
            "issue": "학교 급식 예산",
            "keywords": ["급식", "예산"],
            "assemblies": ["서울특별시의회"],
            "date_start": "2024-01-01",
            "date_end": "2025-12-31",
            "request_type": "issue_flow",
            "as_of_date": "2026-10-30",
        })
        result = IntentExtractor(llm).extract("서울 급식 예산은?")
        self.assertEqual(result, QueryIntent(
            issue="학교 급식 예산",
            keywords=("급식", "예산"),
            assemblies=("서울특별시의회",),
            date_start="20240101",
            date_end="20251231",
            request_type="issue_flow",
            as_of_date="20261030",
        ))
        self.assertIsNotNone(llm.calls[0][2])

    def test_year_only_scope_expands_to_full_calendar_year(self):
        llm = FakeLLM({
            "issue": "교통 정책", "keywords": [], "assemblies": [],
            "date_start": "2024", "date_end": "2024", "request_type": "issue_flow", "as_of_date": "2025-01-01",
        })
        result = IntentExtractor(llm).extract("2024년 교통 정책")
        self.assertEqual(result.date_start, "20240101")
        self.assertEqual(result.date_end, "20241231")

    def test_council_short_names_normalize_to_metadata_labels(self):
        llm = FakeLLM({
            "issue": "교통 정책", "keywords": [], "assemblies": ["서울", "강원도의회", "알수없는 의회"],
            "date_start": "", "date_end": "", "request_type": "issue_flow", "as_of_date": "2025-01-01",
        })
        result = IntentExtractor(llm).extract("서울과 강원 교통 정책")
        self.assertEqual(result.assemblies, ("서울특별시의회", "강원특별자치도의회"))


if __name__ == "__main__":
    unittest.main()

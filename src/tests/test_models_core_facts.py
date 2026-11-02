import unittest

from src.models import AnalysisResult, Evidence, QueryIntent


class AnalysisResultCoreFactsTests(unittest.TestCase):
    def test_core_facts_survive_cache_serialization(self):
        result = AnalysisResult(
            intent=QueryIntent(issue="급식 예산"), answer="요약", first_seen=None, flow=[],
            forecast="전망", confidence="보통", forecast_as_of="20261101",
            core_facts=["예산 증액이 논의됐다."],
            likely_discussion_topics=[
                {"topic": "다음 연도 예산 편성", "source_ids": ["doc:1"]},
                {"topic": "지원 대상 조정", "source_ids": ["doc:2"]},
            ],
            forecast_citations=["doc:forecast-1"],
        )

        restored = AnalysisResult.from_dict(result.to_dict())

        self.assertEqual(restored.core_facts, ["예산 증액이 논의됐다."])
        self.assertEqual(
            restored.likely_discussion_topics,
            [
                {"topic": "다음 연도 예산 편성", "source_ids": ["doc:1"]},
                {"topic": "지원 대상 조정", "source_ids": ["doc:2"]},
            ],
        )
        self.assertEqual(restored.forecast_citations, ["doc:forecast-1"])

    def test_regional_spread_survives_cache_serialization(self):
        spread = {
            "status": "observed_related_discussions",
            "scope_limited": False,
            "basis_record_count": 2,
            "observed_councils": [{"assembly": "부산광역시의회", "record_count": 1, "date_range": {"start": "20240220", "end": "20240220"}, "source_ids": ["busan:2"]}],
            "outlook": "관련 논의가 확인됐습니다.",
            "causality_established": False,
            "probability_estimated": False,
        }
        result = AnalysisResult(
            intent=QueryIntent(issue="급식 예산"), answer="요약", first_seen=None, flow=[],
            forecast="전망", confidence="낮음", forecast_as_of="20261101",
            regional_spread=spread,
            regional_spread_evidence=[Evidence(id="busan:2", text="급식 예산 심의", date="20240220", assembly="부산광역시의회")],
        )

        restored = AnalysisResult.from_dict(result.to_dict())

        self.assertEqual(restored.regional_spread, spread)
        self.assertEqual(restored.regional_spread_evidence[0].id, "busan:2")

    def test_old_cached_result_without_core_facts_remains_readable(self):
        legacy = {
            "intent": {"issue": "급식 예산"}, "answer": "저장된 요약", "first_seen": None,
            "flow": [], "forecast": "전망", "confidence": "낮음", "forecast_as_of": "20261101",
        }

        restored = AnalysisResult.from_dict(legacy)

        self.assertEqual(restored.core_facts, [])
        self.assertEqual(restored.regional_spread, {})
        self.assertEqual(restored.regional_spread_evidence, [])


if __name__ == "__main__":
    unittest.main()

import json
import unittest

from src.models import Evidence, QueryIntent
from src.pipeline import AnalysisWriter, FORECAST_SCHEMA, SUMMARY_SCHEMA


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.schemas = []
        self.users = []

    def chat_json(self, system, user, schema):
        self.schemas.append(schema)
        self.users.append(user)
        return self.response


class AnalysisWriterTests(unittest.TestCase):
    def test_forecast_call_receives_summaries_metadata_and_stats_not_source_text(self):
        class RecordingLLM:
            def __init__(self, responses):
                self.responses = responses
                self.calls = []
            def chat_json(self, system, user, schema):
                self.calls.append((system, user, schema))
                return self.responses[len(self.calls) - 1]

        raw_marker = "FULL_SOURCE_ONLY_MARKER_한정"
        llm = RecordingLLM([
            {
                "answer": "두 회의에서 학교 급식 예산을 논의했다.",
                "core_facts": ["예산 검토가 논의됐다."],
                "evidence_summaries": [
                    {"source_id": "doc:1", "summary": "급식 예산 증액안을 검토했다."},
                    {"source_id": "doc:2", "summary": "다음 연도 급식 예산을 다시 논의했다."},
                ],
                "citations": ["doc:1", "doc:2"],
            },
            {
                "forecast": "예산 검토가 다음 회기에도 이어질 수 있습니다.",
                "likely_discussion_topics": [
                    {"topic": "다음 연도 예산 편성", "source_ids": ["doc:1"]},
                    {"topic": "지원 대상 조정", "source_ids": ["doc:2"]},
                    {"topic": "근거 없는 일정 전망", "source_ids": ["doc:3"]},
                ],
                "forecast_citations": ["doc:1", "doc:2"],
                "confidence": "보통",
            },
        ])
        evidence = [
            Evidence(
                id="doc:1", text=f"{raw_marker} 급식 예산 증액을 검토했다.",
                date="20240101", assembly="서울특별시의회", meeting_title="본회의",
                minutes_title="제1회 회의록", agenda="학교 급식 예산",
            ),
            Evidence(
                id="doc:2", text="다음 연도 급식 예산 재검토", date="20250102",
                assembly="서울특별시의회", meeting_title="예산결산특별위원회",
                minutes_title="제2회 회의록", agenda="학교 급식 예산",
            ),
            Evidence(
                id="doc:3", text=f"{raw_marker} 원문 발언은 예측 단계로 전달되면 안 됩니다.",
                date="20250104", assembly="서울특별시의회", meeting_title="본회의",
                minutes_title="제3회 회의록", agenda="학교 급식 예산",
            ),
        ]

        result = AnalysisWriter(llm).generate(
            "학교 급식 예산 전망", QueryIntent(issue="학교 급식 예산"), evidence
        )

        self.assertEqual(len(llm.calls), 2)
        forecast_user = llm.calls[1][1]
        forecast_payload = json.loads(forecast_user)
        self.assertNotIn(raw_marker, forecast_user)
        self.assertNotIn('"excerpt"', forecast_user)
        self.assertIn("신뢰도 필드가 불확실성을 표시하므로", llm.calls[1][0])
        self.assertIn("경고를 반복하지 마세요", llm.calls[1][0])
        self.assertTrue(any(item["summary_kind"] == "metadata_only" for item in forecast_payload["evidence_summaries"]))
        self.assertEqual(forecast_payload["evidence_summaries"][0]["summary"], "급식 예산 증액안을 검토했다.")
        self.assertEqual(forecast_payload["evidence_summaries"][0]["assembly"], "서울특별시의회")
        self.assertEqual(forecast_payload["evidence_summaries"][0]["date"], "20240101")
        self.assertEqual(forecast_payload["statistics"]["candidate_record_count"], 3)
        self.assertEqual(forecast_payload["statistics"]["summarized_record_count"], 2)
        self.assertEqual(forecast_payload["statistics"]["distinct_meeting_days"], 3)
        self.assertEqual(forecast_payload["statistics"]["date_range"], {"start": "20240101", "end": "20250104"})
        self.assertEqual(forecast_payload["statistics"]["records_by_year"], {"2024": 1, "2025": 2})
        self.assertEqual(forecast_payload["statistics"]["records_by_assembly"], [{"assembly": "서울특별시의회", "records": 3}])
        self.assertEqual(forecast_payload["statistics"]["keyword_signal_records"]["budget_finance"], 2)
        self.assertEqual(forecast_payload["topic_summary"], "두 회의에서 학교 급식 예산을 논의했다.")
        self.assertEqual(forecast_payload["core_facts"], ["예산 검토가 논의됐다."])
        self.assertIn("likely_discussion_topics", llm.calls[1][2]["properties"])
        self.assertIn("source_id를 1~3개 연결하세요", llm.calls[1][0])
        self.assertEqual(
            result["likely_discussion_topics"],
            [
                {"topic": "다음 연도 예산 편성", "source_ids": ["doc:1"]},
                {"topic": "지원 대상 조정", "source_ids": ["doc:2"]},
            ],
        )
        self.assertEqual(result["forecast_citations"], ["doc:1", "doc:2"])
        self.assertEqual(result["forecast"], "예산 검토가 다음 회기에도 이어질 수 있습니다.")

    def test_forecast_payload_caps_summary_fields(self):
        class RecordingLLM:
            def __init__(self):
                self.calls = []
            def chat_json(self, system, user, schema):
                self.calls.append((system, user, schema))
                if len(self.calls) == 1:
                    return {
                        "answer": "A" * 1500,
                        "core_facts": ["B" * 500],
                        "evidence_summaries": [{"source_id": "doc:1", "summary": "C" * 500}],
                        "citations": ["doc:1"],
                    }
                return {"forecast": "조건부 전망", "forecast_citations": ["doc:1"], "confidence": "낮음"}

        llm = RecordingLLM()
        evidence = [Evidence(id="doc:1", text="원문", date="20250101", assembly="서울특별시의회")]
        AnalysisWriter(llm).generate("질문", QueryIntent(issue="안건"), evidence)
        payload = json.loads(llm.calls[1][1])
        self.assertEqual(len(payload["topic_summary"]), 1200)
        self.assertEqual(len(payload["core_facts"][0]), 320)
        self.assertEqual(len(payload["evidence_summaries"][0]["summary"]), 320)

    def test_forecast_receives_regional_spread_evidence_and_returns_assessment(self):
        class RecordingLLM:
            def __init__(self):
                self.calls = []
            def chat_json(self, system, user, schema):
                self.calls.append((system, user, schema))
                if len(self.calls) == 1:
                    return {
                        "answer": "학교 급식 예산 논의가 확인됐습니다.",
                        "core_facts": ["두 의회에서 관련 안건을 논의했습니다."],
                        "evidence_summaries": [
                            {"source_id": "seoul:1", "summary": "급식 예산 확대를 검토했습니다."},
                            {"source_id": "busan:2", "summary": "급식 예산 지원안을 심의했습니다."},
                        ],
                        "citations": ["seoul:1", "busan:2"],
                    }
                return {"forecast": "추가 논의 가능성", "forecast_citations": ["seoul:1", "busan:2"], "confidence": "낮음"}

        llm = RecordingLLM()
        raw_spread_marker = "SPREAD_RAW_ONLY_MARKER_다른의회"
        evidence = [
            Evidence(id="seoul:1", text="급식 예산 확대 검토", date="20240110", assembly="서울특별시의회"),
        ]
        spread_evidence = [
            Evidence(id="busan:2", text=f"{raw_spread_marker} 급식 예산 지원 심의", date="20240220", assembly="부산광역시의회"),
        ]
        result = AnalysisWriter(llm).generate(
            "다른 시도 확산 가능성",
            QueryIntent(issue="학교 급식 예산", assemblies=("서울특별시의회",)),
            evidence,
            spread_evidence,
        )
        payload = json.loads(llm.calls[1][1])

        self.assertIn(raw_spread_marker, llm.calls[0][1])
        self.assertNotIn(raw_spread_marker, llm.calls[1][1])
        self.assertEqual(payload["regional_spread"]["status"], "observed_related_discussions")
        self.assertEqual(payload["regional_spread"]["basis_record_count"], 1)
        self.assertEqual(payload["regional_spread"]["origin_councils"][0]["assembly"], "서울특별시의회")
        self.assertEqual(payload["regional_spread"]["observed_councils"][0]["assembly"], "부산광역시의회")
        self.assertIn("입증하지 않습니다", payload["regional_spread"]["outlook"])
        self.assertEqual(result["regional_spread"], payload["regional_spread"])

    def test_writer_returns_structured_core_facts(self):
        llm = FakeLLM({
            "answer": "요약", "core_facts": ["급식 예산을 증액하기로 논의했다."],
            "forecast": "추가 심의 가능성", "forecast_citations": ["doc:1"],
            "confidence": "보통", "citations": ["doc:1"],
        })
        writer = AnalysisWriter(llm)
        evidence = [Evidence(id="doc:1", text="급식 예산 증액", date="20250101", assembly="서울특별시의회")]

        result = writer.generate("급식 예산은?", QueryIntent(issue="급식 예산"), evidence)

        self.assertIn("evidence_summaries", SUMMARY_SCHEMA["required"])
        self.assertIn("forecast_citations", FORECAST_SCHEMA["required"])
        self.assertEqual(llm.schemas, [SUMMARY_SCHEMA, FORECAST_SCHEMA])
        self.assertEqual(result["core_facts"], ["급식 예산을 증액하기로 논의했다."])
        self.assertEqual(result["forecast_citations"], ["doc:1"])
        self.assertEqual(result["answer"], "요약")

    def test_empty_model_facts_fall_back_to_grounded_evidence_instead_of_heading(self):
        llm = FakeLLM({
            "answer": "핵심 사실:", "core_facts": [], "forecast": "", "confidence": "낮음", "citations": [],
        })
        writer = AnalysisWriter(llm)
        evidence = [Evidence(
            id="doc:1", text="학교 급식 예산을 늘려야 한다고 발언했다.",
            date="20250101", assembly="서울특별시의회",
        )]

        result = writer.generate("급식 예산은?", QueryIntent(issue="급식 예산"), evidence)

        self.assertEqual(
            result["core_facts"],
            ["20250101 · 서울특별시의회: 학교 급식 예산을 늘려야 한다고 발언했다."],
        )

    def test_empty_forecast_uses_concise_grounded_text_and_separate_low_confidence(self):
        llm = FakeLLM({
            "answer": "요약", "core_facts": ["급식 예산 논의"], "forecast": "",
            "forecast_citations": [], "confidence": "높음", "citations": [],
        })
        writer = AnalysisWriter(llm)
        evidence = [
            Evidence(id="doc:1", text="급식 예산 논의", date="20240101", assembly="서울특별시의회"),
            Evidence(id="doc:2", text="예산 재검토가 논의됨", date="20240301", assembly="서울특별시의회"),
        ]

        result = writer.generate("급식 예산 전망", QueryIntent(issue="급식 예산"), evidence)

        self.assertTrue(result["forecast"].strip())
        self.assertIn("20240101", result["forecast"])
        self.assertIn("20240301", result["forecast"])
        self.assertIn("후속 안건 논의", result["forecast"])
        self.assertNotIn("신뢰도", result["forecast"])
        self.assertNotIn("단정", result["forecast"])
        self.assertEqual(result["forecast_citations"], ["doc:1", "doc:2"])
        self.assertEqual(result["confidence"], "낮음")

    def test_unreferenced_recommendation_is_replaced_by_evidence_based_forecast(self):
        llm = FakeLLM({
            "answer": "요약", "core_facts": ["급식 예산을 논의했다."],
            "forecast": "1. 예산을 확대해야 합니다.", "forecast_citations": [],
            "confidence": "높음", "citations": ["doc:1"],
        })
        evidence = [
            Evidence(id="doc:1", text="급식 예산을 검토함", date="20240101", assembly="서울특별시의회"),
            Evidence(id="doc:2", text="급식 예산 후속 논의", date="20240301", assembly="서울특별시의회"),
        ]

        result = AnalysisWriter(llm).generate(
            "급식 예산 전망", QueryIntent(issue="급식 예산"), evidence
        )

        self.assertNotIn("1. 예산을 확대해야 합니다", result["forecast"])
        self.assertIn("20240101", result["forecast"])
        self.assertIn("20240301", result["forecast"])
        self.assertEqual(result["forecast_citations"], ["doc:1", "doc:2"])
        self.assertEqual(result["confidence"], "낮음")

    def test_empty_forecast_without_evidence_does_not_claim_a_direction(self):
        llm = FakeLLM({
            "answer": "", "core_facts": [], "forecast": "", "confidence": "높음", "citations": [],
        })

        result = AnalysisWriter(llm).generate(
            "급식 예산 전망", QueryIntent(issue="급식 예산"), [],
        )

        self.assertIn("검색된 근거가 없어", result["forecast"])
        self.assertEqual(result["confidence"], "낮음")


if __name__ == "__main__":
    unittest.main()

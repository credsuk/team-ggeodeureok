import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_engine import LLMEngine


class SummaryCitationFallbackTests(unittest.TestCase):
    def setUp(self):
        self.evidence = [{
            "source_id": "dataset-2025:17",
            "date": "20250210",
            "council": "서울특별시의회",
            "excerpt": "청년 주거 지원을 확대하고 신청 절차를 간소화해야 한다는 논의가 있었습니다.",
        }]

    def run_with_response(self, response, evidence=None):
        engine = LLMEngine.__new__(LLMEngine)
        object.__setattr__(engine, "_json_call", lambda *_args: response)
        return engine.synthesize_analysis(
            {"query": "청년 주거 지원", "topic": "청년 주거"},
            self.evidence if evidence is None else evidence,
            {},
        )

    def test_missing_summary_citation_uses_a_verbatim_cited_evidence_excerpt(self):
        result = self.run_with_response({
            "summary": "모델이 생성했지만 근거 ID가 없는 요약",
            "summary_evidence_ids": [],
            "forecast": "자료 부족",
            "forecast_evidence_ids": [],
            "forecast_status": "insufficient_evidence",
        })

        self.assertIn(self.evidence[0]["excerpt"], result["summary"])
        self.assertEqual(result["summary_evidence_ids"], ["dataset-2025:17"])
        self.assertNotIn("요약 문장에 연결되는 근거 ID가 없어", result["summary"])

    def test_invalid_summary_citation_is_repaired_with_an_allowed_source_id(self):
        result = self.run_with_response({
            "summary": "모델이 생성한 요약",
            "summary_evidence_ids": ["invented:999"],
            "forecast": "자료 부족",
            "forecast_evidence_ids": [],
            "forecast_status": "insufficient_evidence",
        })

        self.assertEqual(result["summary_evidence_ids"], ["dataset-2025:17"])
        self.assertIn(self.evidence[0]["excerpt"], result["summary"])

    def test_fallback_quote_preserves_the_source_text_exactly(self):
        excerpt = "이 사업이 인접 지역으로 확산될 수 있다는 우려가 제기됐습니다."
        evidence = [{**self.evidence[0], "excerpt": excerpt}]
        result = self.run_with_response({
            "summary": "요약", "summary_evidence_ids": [],
            "forecast": "자료 부족", "forecast_evidence_ids": [],
            "forecast_status": "insufficient_evidence",
        }, evidence=evidence)

        self.assertIn(excerpt, result["summary"])

    def test_valid_summary_and_citation_are_preserved(self):
        result = self.run_with_response({
            "summary": "청년 주거 지원 확대가 논의되었습니다.",
            "summary_evidence_ids": ["dataset-2025:17"],
            "forecast": "자료 부족",
            "forecast_evidence_ids": [],
            "forecast_status": "insufficient_evidence",
        })

        self.assertEqual(result["summary"], "청년 주거 지원 확대가 논의되었습니다.")
        self.assertEqual(result["summary_evidence_ids"], ["dataset-2025:17"])

    def test_does_not_invent_a_source_id_when_evidence_has_no_usable_id(self):
        result = self.run_with_response({
            "summary": "근거 ID 없는 요약",
            "summary_evidence_ids": [],
            "forecast": "자료 부족",
            "forecast_evidence_ids": [],
            "forecast_status": "insufficient_evidence",
        }, evidence=[{"source_id": "", "excerpt": "출처 식별자가 없는 발언"}])

        self.assertEqual(result["summary_evidence_ids"], [])
        self.assertNotIn("요약 문장에 연결되는 근거 ID가 없어", result["summary"])
        self.assertIn("유효한 출처 ID", result["summary"])

    def test_empty_summary_without_an_excerpt_does_not_keep_model_citation_ids(self):
        result = self.run_with_response({
            "summary": "",
            "summary_evidence_ids": ["dataset-2025:17"],
            "forecast": "자료 부족",
            "forecast_evidence_ids": [],
            "forecast_status": "insufficient_evidence",
        }, evidence=[{**self.evidence[0], "excerpt": ""}])

        self.assertEqual(result["summary_evidence_ids"], [])
        self.assertIn("발언 발췌", result["summary"])

    def test_malformed_source_id_is_skipped_without_crashing(self):
        result = self.run_with_response({
            "summary": "요약", "summary_evidence_ids": [],
            "forecast": "자료 부족", "forecast_evidence_ids": [],
            "forecast_status": "insufficient_evidence",
        }, evidence=[{"source_id": [], "excerpt": "식별자 형식이 잘못된 발언"}])

        self.assertEqual(result["summary_evidence_ids"], [])
        self.assertIn("유효한 출처 ID", result["summary"])


if __name__ == "__main__":
    unittest.main()

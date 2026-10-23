"""Offline stdlib regressions executing the real analyzer without integrations.

Only integration imports are stripped from its AST; no config (.env), models,
vector stores, or network clients are imported or instantiated.
"""
import ast
import json
from pathlib import Path
import types
import unittest
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().parents[1] / "src" / "trend_analyzer.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
tree.body = [
    node for node in tree.body
    if not (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src."))
    and not (isinstance(node, ast.Import) and any(alias.name == "requests" for alias in node.names))
]
module = types.ModuleType("offline_trend_analyzer")
exec(compile(tree, str(SOURCE), "exec"), module.__dict__)


def hit(date="2024-01-01", assembly="가의회", original=None, prefix=""):
    if original is None:
        original = "지역소멸 문제를 해결하기 위한 정책과 지원 방안을 충분히 검토하고 논의해야 합니다."
    content = prefix + "[원문 발언]\n" + original
    return (types.SimpleNamespace(page_content=content, metadata={
        "date": date, "assembly": assembly, "agenda_title": "정책 검토",
        "mints_nm": "본회의 회의록", "summary": "메타데이터 요약",
    }), 0.1)


def analyzer(hits, response=""):
    obj = module.PolicyTrendAnalyzer.__new__(module.PolicyTrendAnalyzer)
    obj.indexer = types.SimpleNamespace(query=Mock(return_value=hits))
    obj.cutoff_date = None
    obj.db_type = "chroma"
    obj.collection_name = "offline"
    obj.embedding_model_name = "offline"
    obj.llm_model = "offline"
    obj._call_ollama = Mock(return_value=response)
    return obj


class TrendRegressions(unittest.TestCase):
    def test_only_original_keyword_hits_can_establish_first_observed(self):
        unrelated = hit("2020-01-01", original="도로 정비 및 교통 시설 개선을 위한 예산과 공사 일정에 대해 충분한 검토가 필요합니다.", prefix="지역소멸 요약과 키워드\n")
        obj = analyzer([unrelated])
        self.assertEqual(obj.analyze_issue("지역소멸")["status"], "not_found")
        obj._call_ollama.assert_not_called()
        obj = analyzer([unrelated, hit("2024-01-01", original="지역 소멸 문제를 해결하기 위한 정책과 지원 방안을 충분히 검토하고 논의해야 합니다.")])
        result = obj.analyze_issue("지역소멸")
        self.assertEqual(result["first_observed"]["iso_date"], "2024-01-01")
        self.assertEqual(result["metadata"]["analyzed_chunks"], 1)
        self.assertEqual(analyzer([hit(original="AI 정책의 지원 방안에 대해 충분히 검토하고 관련 예산과 조례를 논의해야 합니다.")]).analyze_issue("a i")["status"], "success")
        obj = analyzer([hit()])
        self.assertEqual(obj.analyze_issue(" \t")["status"], "not_found")
        obj._call_ollama.assert_not_called()


    def test_flow_uses_distinct_councils_and_caps_supplied_evidence(self):
        hits = [hit("2024-01-01"), hit("2024-01-02"), hit("2024-01-03"),
                hit("2024-02-01", "나의회"), hit("2024-02-02", "나의회"),
                hit("2024-03-01", "다의회"), hit("2024-04-01", "라의회")]
        for cap in (0, 1, 2, 10):
            with self.subTest(cap=cap):
                obj = analyzer(hits)
                result = obj.analyze_issue("지역소멸", max_flow_items=cap)
                expected = ["가의회"] + ["나의회", "다의회", "라의회"][:cap]
                self.assertEqual([ev["assembly"] for ev in result["evidences"]], expected)
                self.assertEqual(len(result["cross_council_trend"]), len(expected) - 1)
                prompt = obj._call_ollama.call_args.args[0]
                self.assertNotIn("2024년 1월 2일", prompt)
                self.assertIn("지역소멸", prompt)
                self.assertNotIn("메타데이터 요약", prompt)
                for council in ("나의회", "다의회", "라의회"):
                    self.assertEqual(council in prompt, council in expected)

    def test_first_observed_explicitly_labels_retrieval_candidate_scope(self):
        result = analyzer([hit("2024-02-01"), hit("2024-01-01")]).analyze_issue("지역소멸")
        self.assertEqual(result["first_observed"]["iso_date"], "2024-01-01")
        self.assertEqual(result["metadata"]["first_observed_scope"], "retrieval_candidates")
        self.assertIn("검색 후보 내 최초 관측", result["first_observed"]["display"])
        self.assertEqual(result["submission_table_row"]["등장시점"], result["first_observed"]["display"])

    def test_context_window_budget_includes_ellipses_and_compound_keyword(self):
        extract = module.extract_context_window
        for text, keyword in (("앞" * 300 + "지역소멸" + "뒤" * 300, "지역소멸"),
                              ("앞" * 300 + "지역 소멸" + "뒤" * 300, "지역소멸"),
                              ("앞" * 300 + "AI" + "뒤" * 300, "a i"),
                              ("가" * 500, "없는말")):
            for limit in (0, 1, 3, 6, 10, 160, 180):
                with self.subTest(keyword=keyword, limit=limit):
                    span = extract("메타 정보 [원문 발언] " + text, keyword, limit)
                    self.assertLessEqual(len(span), limit)
                    if limit >= 160 and keyword != "없는말":
                        self.assertIn("".join(keyword.split()).casefold(), "".join(span.split()).casefold())
        self.assertEqual(extract("[원문 발언] 짧은 원문", "원문"), "짧은 원문")

    def test_unavailable_or_invalid_prediction_is_explicit_not_fabricated(self):
        for response in ("", "not json", "[]", "null", '"text"', "42",
                         '{"future_prediction": []}', '{"future_prediction": 12}',
                         '{"future_prediction": "  "}', '{"future_prediction": {"claim": "x"}}'):
            with self.subTest(response=response):
                result = analyzer([hit()], response).analyze_issue("지역소멸")
                self.assertEqual(result["status"], "success")  # CLI schema preserved
                self.assertEqual(result["metadata"]["prediction_status"], "unavailable")
                self.assertIsInstance(result["future_prediction"], str)
                self.assertIn("예측 불가", result["future_prediction"])
                self.assertNotIn("지속 확산", result["future_prediction"])
                self.assertEqual(result["submission_table_row"]["향후 예측"], result["future_prediction"])
        result = analyzer([hit()], json.dumps({"future_prediction": "추가 근거가 필요합니다."})).analyze_issue("지역소멸")
        self.assertEqual(result["metadata"]["prediction_status"], "available")
        self.assertEqual(result["future_prediction"], "추가 근거가 필요합니다.")


if __name__ == "__main__":
    unittest.main()

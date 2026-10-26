import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_module_names = ("llm_engine", "retriever", "reranker")
_previous_modules = {name: sys.modules.get(name) for name in _module_names}
for module_name, class_name in (("llm_engine", "LLMEngine"), ("retriever", "HybridRetriever"), ("reranker", "DocumentReranker")):
    module = types.ModuleType(module_name)
    setattr(module, class_name, type(class_name, (), {}))
    sys.modules[module_name] = module
import main
for module_name, previous_module in _previous_modules.items():
    if previous_module is None:
        sys.modules.pop(module_name, None)
    else:
        sys.modules[module_name] = previous_module


class HistoryLoggingTests(unittest.TestCase):
    def test_history_persists_input_extracted_params_output_and_timings(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test_history.json"
            params = {
                "topic": "청년 주거", "query": "청년 주거 정책", "include_terms": ["청년", "주거"],
                "start_year": "2024", "end_year": "2025", "councils": ["제주특별자치도의회"],
                "intent": "현황",
            }
            output = {"results": [{"issueKeyword": "청년 주거", "summary": "요약", "evidence": []}]}
            timings = {"질문 분석 LLM": 0.25, "검색": 1.5, "전체 처리": 2.0}
            with patch("main.os.path.dirname", return_value=temp):
                main._append_history("청년 주거 현황 분석", output, timings, params=params)
            entry = json.loads(path.read_text(encoding="utf-8"))[0]
            self.assertEqual(entry["input"], "청년 주거 현황 분석")
            self.assertEqual(entry["extracted_params"], params)
            self.assertEqual(entry["output"], output)
            self.assertEqual(entry["timings_seconds"], timings)
            self.assertNotIn("sourceId", json.dumps(entry, ensure_ascii=False))

    def test_history_appends_records_and_marks_failed_run(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test_history.json"
            with patch("main.os.path.dirname", return_value=temp):
                main._append_history("first", {"ok": True}, {"전체 처리": 0.1}, params={"topic": "one"})
                main._append_history("second", None, {"전체 처리": 0.2}, params={"topic": "two"}, error="model unavailable")
            history = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual([entry["input"] for entry in history], ["first", "second"])
            self.assertIsNone(history[1]["output"])
            self.assertEqual(history[1]["error"], "model unavailable")
            self.assertEqual(history[1]["status"], "error")

    def test_history_excludes_unavailable_source_id_fields_from_nested_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test_history.json"
            output = {"results": [{"summary": "요약", "evidence": [{
                "sourceId": "missing", "chunkIds": ["chunk-1"], "excerpt": "발언 근거"
            }]}]}
            with patch("main.os.path.dirname", return_value=temp):
                main._append_history("질문", output, {"전체 처리": 0.3}, params={"topic": "주제"})
            entry = json.loads(path.read_text(encoding="utf-8"))[0]
            serialized = json.dumps(entry["output"], ensure_ascii=False)
            self.assertNotIn("sourceId", serialized)
            self.assertNotIn("chunkIds", serialized)
            self.assertIn("발언 근거", serialized)

    def test_run_stores_extracted_params_and_total_timing_on_failure(self):
        pipeline = main.LocalRAGPipeline.__new__(main.LocalRAGPipeline)
        class QueryLLM:
            def extract_search_params(self, _):
                return {"topic": "교통", "query": "교통 정책", "include_terms": [], "start_year": None, "end_year": None, "councils": []}
        class FailingRetriever:
            def search(self, *_args, **_kwargs):
                raise RuntimeError("test retrieval failure")
        pipeline.llm = QueryLLM()
        pipeline.retriever = FailingRetriever()
        pipeline.reranker = object()
        with self.assertRaisesRegex(RuntimeError, "test retrieval failure"):
            pipeline.run("교통 정책 찾아줘")
        self.assertEqual(pipeline.last_params["topic"], "교통")
        self.assertIn("전체 처리", pipeline.last_timings)
        self.assertIn("test retrieval failure", pipeline.last_error)

    def test_first_observation_uses_ranked_candidates_without_a_second_retriever_scan(self):
        pipeline = main.LocalRAGPipeline.__new__(main.LocalRAGPipeline)

        class QueryLLM:
            def __init__(self):
                self.evidence = None

            def extract_search_params(self, _query):
                return {
                    "topic": "정책 변화", "query": "정책 변화", "include_terms": [],
                    "start_year": None, "end_year": None, "councils": [], "intent": "현황",
                }

            def synthesize_analysis(self, _params, evidence, _metrics):
                self.evidence = evidence
                return {
                    "summary": "요약", "summary_evidence_ids": [], "forecast": "",
                    "forecast_evidence_ids": [], "forecast_status": "insufficient_evidence",
                }

        class CandidateRetriever:
            def __init__(self):
                self.rows = [
                    {"id": "late", "text": "후속 발언", "metadata": {
                        "dataset_key": "d", "row_id": 2, "date": "20250201",
                        "assembly": "서울특별시의회", "agenda_title": "안건", "MINTS_NM": "회의록",
                    }, "chunk_ids": ["late-chunk"], "vector_similarity": 0.96},
                    {"id": "early", "text": "초기 발언", "metadata": {
                        "dataset_key": "d", "row_id": 1, "date": "20240102",
                        "assembly": "서울특별시의회", "agenda_title": "안건", "MINTS_NM": "회의록",
                    }, "chunk_ids": ["early-chunk"], "vector_similarity": 0.70},
                ]
                self.search_calls = 0

            def search(self, *_args, **_kwargs):
                self.search_calls += 1
                return self.rows

            def expand_rows(self, candidates):
                return candidates

        class CandidateReranker:
            def rank_and_filter(self, _query, candidates, **_kwargs):
                return [dict(row, rerank_score=0.95 if row["id"] == "early" else 0.6) for row in candidates]

        retriever = CandidateRetriever()
        llm = QueryLLM()
        pipeline.llm = llm
        pipeline.retriever = retriever
        pipeline.reranker = CandidateReranker()

        output = pipeline.run("정책 변화")
        result = output["results"][0]

        self.assertEqual(retriever.search_calls, 1)
        self.assertEqual([row["date"] for row in result["firstObserved"]], ["20240102"])
        self.assertEqual([row["date"] for row in result["subsequentDiscussions"]], ["20250201"])
        self.assertNotIn("Vector DB 주제어 검색 및 최초 관측", pipeline.last_timings)
        self.assertEqual([row["source_id"] for row in llm.evidence], ["d:2", "d:1"])
        self.assertEqual([row["vector_similarity"] for row in llm.evidence], [0.96, 0.70])
        self.assertEqual([row["reranker_score"] for row in llm.evidence], [0.6, 0.95])

    def test_expands_at_most_50_top_reranked_candidates(self):
        pipeline = main.LocalRAGPipeline.__new__(main.LocalRAGPipeline)

        class QueryLLM:
            def extract_search_params(self, _query):
                return {
                    "topic": "주택", "query": "주택", "include_terms": [],
                    "start_year": None, "end_year": None, "councils": [], "intent": "현황",
                }

            def synthesize_analysis(self, _params, _evidence, _metrics):
                return {
                    "summary": "요약", "summary_evidence_ids": [], "forecast": "전망",
                    "forecast_evidence_ids": [], "forecast_status": "insufficient_evidence",
                }

        class CandidateRetriever:
            def __init__(self):
                self.rows = [
                    {"id": f"d:{i}", "text": f"발언 {i}", "vector_similarity": 0.7,
                     "metadata": {"dataset_key": "d", "row_id": i, "date": "20240101",
                                  "assembly": f"의회{i}", "agenda_title": "안건", "MINTS_NM": "회의록"},
                     "chunk_ids": []}
                    for i in range(1, 76)
                ]
                self.expanded_count = None

            def search(self, *_args, **_kwargs):
                return self.rows

            def expand_rows(self, candidates):
                self.expanded_count = len(candidates)
                return candidates

        class CandidateReranker:
            def __init__(self):
                self.requested_top_n = None

            def rank_and_filter(self, _query, candidates, **kwargs):
                self.requested_top_n = kwargs.get("top_n")
                return [dict(row, rerank_score=float(100 - i)) for i, row in enumerate(candidates)]

        retriever = CandidateRetriever()
        reranker = CandidateReranker()
        pipeline.llm = QueryLLM()
        pipeline.retriever = retriever
        pipeline.reranker = reranker

        pipeline.run("주택 정책")

        self.assertEqual(reranker.requested_top_n, 50)
        self.assertEqual(retriever.expanded_count, 50)


if __name__ == "__main__":
    unittest.main()

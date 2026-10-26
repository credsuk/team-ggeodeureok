import sys
import unittest
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))

from evaluation.build_main_baseline import extract_expected_rows, make_stdin, merge_query_results


class MainBaselineHelpersTests(unittest.TestCase):
    def test_stdin_runs_all_six_queries_then_exits(self):
        self.assertEqual(
            make_stdin(["신혼부부", "노인 일자리"]),
            "신혼부부\n노인 일자리\n종료\n",
        )

    def test_incremental_update_replaces_only_requested_query_and_keeps_benchmark_order(self):
        existing = {"queries": [
            {"query": "신혼부부", "status": "success", "retrieved_source_row_count": 0},
            {"query": "노인 일자리", "status": "success", "retrieved_source_row_count": 13},
        ]}
        refreshed = [{"query": "다문화 가정", "status": "success", "retrieved_source_row_count": 8}]

        merged = merge_query_results(
            existing, refreshed, ["다문화 가정", "노인 일자리"]
        )

        self.assertEqual([item["query"] for item in merged], ["다문화 가정", "노인 일자리"])
        self.assertEqual([item["retrieved_source_row_count"] for item in merged], [8, 13])

    def test_extracts_source_ids_and_original_excerpts_from_main_history(self):
        history = [{
            "input": "신혼부부",
            "extracted_params": {"topic": "신혼부부"},
            "timings_seconds": {"전체 처리": 2.5},
            "status": "success",
            "output": {"results": [{"evidence": [{
                "datasetKey": "2024_gicho", "originalRowId": 17,
                "date": "2024-01-02", "council": "샘플의회",
                "agendaTitle": "주거 지원", "excerpt": "청년·신혼부부 임대 지원을 확대한다.",
            }]}]},
        }]
        rows = extract_expected_rows(history)
        self.assertEqual(rows["신혼부부"][0]["source_id"], "2024_gicho:17")
        self.assertEqual(rows["신혼부부"][0]["excerpt"], "청년·신혼부부 임대 지원을 확대한다.")

    def test_skips_entries_without_real_source_identity(self):
        rows = extract_expected_rows([{
            "input": "지방 소멸", "output": {"results": [{"evidence": [
                {"datasetKey": "", "originalRowId": None, "excerpt": "출처 없는 내용"},
            ]}]},
        }])
        self.assertEqual(rows, {"지방 소멸": []})


if __name__ == "__main__":
    unittest.main()

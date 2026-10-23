import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_engine import LLMEngine
from retrieval_utils import (
    expand_topic_terms,
    first_day_and_limited_followups,
    group_candidates_by_date,
    merge_overlapping_chunks,
)


class QueryExtractionTests(unittest.TestCase):
    def test_expands_known_aliases_without_calling_a_live_llm(self):
        engine = LLMEngine()
        engine._json_call = lambda *_: {
            "topic": "지역 소멸", "query": "지역 소멸 논의", "include_terms": ["지역 소멸"],
            "start_year": None, "end_year": None, "councils": [], "intent": "현황",
        }
        result = engine.extract_search_params("지역 소멸 논의 현황")
        self.assertEqual(result["include_terms"], ["지역 소멸", "지방소멸", "지방소멸대응기금"])


class ExpandTopicTermsTests(unittest.TestCase):
    def test_expands_known_equivalent_terms_for_matching_topic(self):
        self.assertEqual(
            expand_topic_terms("지역 소멸", "현황", ["지역 소멸"]),
            ["지역 소멸", "지방소멸", "지방소멸대응기금"],
        )

    def test_does_not_expand_unrelated_topics(self):
        self.assertEqual(expand_topic_terms("청년 주거", "현황", ["청년"]), ["청년"])


class MergeOverlappingChunksTests(unittest.TestCase):
    def test_removes_boundary_overlap_without_dropping_repeated_text(self):
        chunks = [(0, "앞 문장 반복"), (1, "반복 그리고 뒤 문장")]
        self.assertEqual(merge_overlapping_chunks(chunks, overlap=3), "앞 문장 반복 그리고 뒤 문장")

    def test_accepts_production_chunk_tuples_with_ids(self):
        chunks = [(0, "chunk-0", "앞 문장 반복"), (1, "chunk-1", "반복 그리고 뒤 문장")]
        self.assertEqual(merge_overlapping_chunks(chunks, overlap=3), "앞 문장 반복 그리고 뒤 문장")

    def test_handles_chunks_without_overlap(self):
        chunks = [(0, "첫 번째"), (1, "두 번째")]
        self.assertEqual(merge_overlapping_chunks(chunks, overlap=3), "첫 번째두 번째")


class DateGroupTests(unittest.TestCase):
    def test_groups_chunks_by_date_and_deduplicates_source_rows(self):
        candidates = [
            {"id": "b-chunk", "metadata": {"date": "2025-01-02", "dataset_key": "b", "row_id": 1}},
            {"id": "a-chunk-1", "metadata": {"date": "20241230", "dataset_key": "a", "row_id": 4}},
            {"id": "a-chunk-2", "metadata": {"date": "2024-12-30", "dataset_key": "a", "row_id": 4}},
        ]
        groups = group_candidates_by_date(candidates)
        self.assertEqual([date for date, _ in groups], ["20241230", "20250102"])
        self.assertEqual(len(groups[0][1]), 1)

    def test_scan_checks_full_first_day_then_limits_followups(self):
        candidates = [
            {"id": "later-c", "metadata": {"date": "20250103", "row_id": 3}},
            {"id": "early-a", "metadata": {"date": "20241230", "row_id": 1}},
            {"id": "early-b", "metadata": {"date": "20241230", "row_id": 2}},
            {"id": "later-a", "metadata": {"date": "20250102", "row_id": 4}},
            {"id": "later-b", "metadata": {"date": "20250102", "row_id": 5}},
        ]
        checked = []

        def all_relevant(day):
            checked.extend(row["id"] for row in day)
            return day

        date, first, later = first_day_and_limited_followups(candidates, all_relevant, followup_limit=1)
        self.assertEqual(date, "20241230")
        self.assertEqual({row["id"] for row in first}, {"early-a", "early-b"})
        self.assertEqual([row["id"] for row in later], ["later-a", "later-b"])
        self.assertNotIn("later-c", checked)

    def test_groups_chunks_by_date_and_deduplicates_source_rows(self):
        candidates = [
            {"id": "b-chunk", "metadata": {"date": "2025-01-02", "dataset_key": "b", "row_id": 1}},
            {"id": "a-chunk-1", "metadata": {"date": "20241230", "dataset_key": "a", "row_id": 4}},
            {"id": "a-chunk-2", "metadata": {"date": "2024-12-30", "dataset_key": "a", "row_id": 4}},
        ]
        groups = group_candidates_by_date(candidates)
        self.assertEqual([date for date, _ in groups], ["20241230", "20250102"])
        self.assertEqual(len(groups[0][1]), 1)


if __name__ == "__main__":
    unittest.main()

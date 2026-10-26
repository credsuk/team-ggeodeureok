import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_engine import LLMEngine
from retrieval_utils import (
    expand_topic_terms,
    first_observation_from_candidates,
    merge_overlapping_chunks,
)


class CandidateTimelineTests(unittest.TestCase):
    def test_finds_earliest_date_and_caps_later_rows_without_database_access(self):
        rows = [
            {"id": "later-b", "metadata": {"date": "2025-01-03"}},
            {"id": "first-a", "metadata": {"date": "20241230"}},
            {"id": "later-a", "metadata": {"date": "2025-01-02"}},
            {"id": "first-b", "metadata": {"date": "2024-12-30"}},
        ]

        first_date, first_rows, later_rows = first_observation_from_candidates(rows, followup_limit=1)

        self.assertEqual(first_date, "20241230")
        self.assertEqual({row["id"] for row in first_rows}, {"first-a", "first-b"})
        self.assertEqual([row["id"] for row in later_rows], ["later-a"])

    def test_returns_empty_timeline_when_candidates_have_no_valid_dates(self):
        self.assertEqual(
            first_observation_from_candidates([{"id": "undated", "metadata": {"date": ""}}]),
            (None, [], []),
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


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source_reference import reference_metadata


class SourceReferenceTests(unittest.TestCase):
    def test_preserves_meeting_minutes_title_and_agenda_separately(self):
        result = reference_metadata({
            "dataset_key": "2025_gwangyeok__cleaned_jeju",
            "row_id": 42,
            "MINTS_NM": "제주특별자치도의회 제432회 회의록",
            "MTGNM": "보건복지안전위원회",
            "agenda_title": "청년 주거 지원 논의",
            "MTG_DE": "20250612",
            "RASMBLY_NM": "제주특별자치도의회",
        })
        self.assertEqual(result["meetingTitle"], "제주특별자치도의회 제432회 회의록")
        self.assertEqual(result["agendaTitle"], "청년 주거 지원 논의")
        self.assertEqual(result["datasetKey"], "2025_gwangyeok__cleaned_jeju")
        self.assertEqual(result["originalRowId"], 42)
        self.assertEqual(result["sourceId"], "2025_gwangyeok__cleaned_jeju:42")

    def test_uses_available_fallbacks_without_fabricating_titles(self):
        result = reference_metadata({"MTGNM": "본회의", "MTG_DE": "20250101"})
        self.assertEqual(result["meetingTitle"], "")
        self.assertEqual(result["agendaTitle"], "본회의")
        self.assertEqual(result["datasetKey"], "")
        self.assertIsNone(result["originalRowId"])
        self.assertEqual(result["sourceId"], "")


if __name__ == "__main__":
    unittest.main()

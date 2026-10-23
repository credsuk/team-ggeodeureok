import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_scope import in_analysis_scope


class StrictDateTests(unittest.TestCase):
    def test_rejects_impossible_calendar_date(self):
        self.assertFalse(in_analysis_scope({"MTG_DE": "20251399", "RASMBLY_NM": "경기도의회"}, {}))

    def test_accepts_compact_and_iso_dates(self):
        self.assertTrue(in_analysis_scope({"MTG_DE": "20240229", "RASMBLY_NM": "경기도의회"}, {}))
        self.assertTrue(in_analysis_scope({"date": "2025-12-31", "assembly": "경기도의회"}, {}))


if __name__ == "__main__":
    unittest.main()

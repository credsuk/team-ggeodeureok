import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coverage_report import inspect_input_coverage

HEADER = "MTG_DE\tRASMBLY_NM\tMTGNM\tMINTS_NM\tagenda_title\tspeaker_role\tspeaker_name\tspeaker_type\tutterance\n"


class CoverageReportTests(unittest.TestCase):
    def _roots(self, root):
        metadata, transcripts = root / "metadata", root / "transcripts"
        (metadata / "2025").mkdir(parents=True)
        (transcripts / "2025").mkdir(parents=True)
        return metadata, transcripts

    def test_reports_missing_councils_and_years_instead_of_assuming_scope_is_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            metadata, transcripts = self._roots(Path(temp))
            (metadata / "2025" / "jeju.jsonl").write_text(
                '{"row_id":1,"MTG_DE":"20250102","RASMBLY_NM":"제주특별자치도의회"}\n', encoding="utf-8"
            )
            (transcripts / "2025" / "jeju.txt").write_text(
                HEADER + "20250102\t제주특별자치도의회\t본회의\t제주 회의록\t안건\t의원\t홍길동\t의원\t발언\n",
                encoding="utf-8",
            )
            report = inspect_input_coverage(metadata, transcripts)
            self.assertEqual(report["paired_datasets"], 1)
            self.assertEqual(report["source_rows"], 1)
            self.assertEqual(report["years_present"], ["2025"])
            self.assertEqual(report["councils_present"], ["제주특별자치도의회"])
            self.assertEqual(len(report["missing_years"]), 1)
            self.assertGreater(len(report["missing_councils"]), 0)

    def test_reports_missing_council_year_combinations_even_when_unions_are_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata, transcripts = root / "metadata", root / "transcripts"
            metadata.mkdir(); transcripts.mkdir()
            councils = [
                "서울특별시의회", "부산광역시의회", "대구광역시의회", "인천광역시의회",
                "광주광역시의회", "대전광역시의회", "울산광역시의회", "세종특별자치시의회",
                "경기도의회", "강원특별자치도의회", "충청북도의회", "충청남도의회",
                "전북특별자치도의회", "전라남도의회", "경상북도의회", "경상남도의회",
                "제주특별자치도의회",
            ]
            for index, council in enumerate(councils):
                year = "2024" if index % 2 == 0 else "2025"
                (metadata / f"{index}.jsonl").write_text(
                    json.dumps({"row_id": 1, "MTG_DE": f"{year}0102", "RASMBLY_NM": council}, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                (transcripts / f"{index}.txt").write_text(
                    HEADER + f"{year}0102\t{council}\t본회의\t회의록\t안건\t의원\t홍길동\t의원\t발언\n",
                    encoding="utf-8",
                )
            report = inspect_input_coverage(metadata, transcripts)
            self.assertFalse(report["scope_complete"])
            self.assertEqual(len(report["missing_council_year_combinations"]), len(councils))

    def test_rejects_blank_transcript_scope_fields_even_if_metadata_has_values(self):
        with tempfile.TemporaryDirectory() as temp:
            metadata, transcripts = self._roots(Path(temp))
            (metadata / "2025" / "jeju.jsonl").write_text(
                '{"row_id":1,"MTG_DE":"20250102","RASMBLY_NM":"제주특별자치도의회"}\n', encoding="utf-8"
            )
            (transcripts / "2025" / "jeju.txt").write_text(
                HEADER + "\t\t본회의\t회의록\t안건\t의원\t홍길동\t의원\t발언\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "blank.*MTG_DE|blank.*RASMBLY_NM"):
                inspect_input_coverage(metadata, transcripts)

    def test_rejects_orphan_metadata_and_transcript_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata, transcripts = root / "metadata", root / "transcripts"
            metadata.mkdir(); transcripts.mkdir()
            (metadata / "orphan.jsonl").write_text('{"row_id":1}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pair"):
                inspect_input_coverage(metadata, transcripts)

    def test_counts_empty_transcript_as_zero_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            metadata, transcripts = self._roots(Path(temp))
            (metadata / "2025" / "empty.jsonl").write_text('{"row_id":1}\n', encoding="utf-8")
            (transcripts / "2025" / "empty.txt").write_text(HEADER, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata rows lack"):
                inspect_input_coverage(metadata, transcripts)

    def test_blank_lines_do_not_shift_transcript_row_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            metadata, transcripts = self._roots(Path(temp))
            (metadata / "2025" / "jeju.jsonl").write_text(
                '{"row_id":1,"MTG_DE":"20250102","RASMBLY_NM":"제주특별자치도의회"}\n', encoding="utf-8"
            )
            row = "20250102\t제주특별자치도의회\t본회의\t제주 회의록\t안건\t의원\t홍길동\t의원\t발언\n"
            (transcripts / "2025" / "jeju.txt").write_text(HEADER + "\t\t\t\t\t\t\t\t\n" + row, encoding="utf-8")
            report = inspect_input_coverage(metadata, transcripts)
            self.assertEqual(report["source_rows"], 1)

    def test_rejects_unexpected_transcript_header(self):
        with tempfile.TemporaryDirectory() as temp:
            metadata, transcripts = self._roots(Path(temp))
            (metadata / "2025" / "jeju.jsonl").write_text('{"row_id":1}\n', encoding="utf-8")
            (transcripts / "2025" / "jeju.txt").write_text("wrong\theader\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "header"):
                inspect_input_coverage(metadata, transcripts)


if __name__ == "__main__":
    unittest.main()

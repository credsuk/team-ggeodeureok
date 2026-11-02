import csv
import tempfile
import unittest
from pathlib import Path

from src.source_index import SourceTextReader, build_source_index


class NestedSourceTextPathTests(unittest.TestCase):
    def test_reader_finds_indexed_txt_in_nested_year_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "to_text"
            nested_dir = source_dir / "2025_meetings"
            nested_dir.mkdir(parents=True)
            source_file = nested_dir / "cleaned_2025_세종특별자치시의회.txt"
            with source_file.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["MTG_DE", "RASMBLY_NM", "MTGNM", "MINTS_NM", "agenda_title", "utterance"],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow({
                    "MTG_DE": "2025-01-01", "RASMBLY_NM": "세종특별자치시의회",
                    "MTGNM": "본회의", "MINTS_NM": "정례회",
                    "agenda_title": "예산", "utterance": "중첩 폴더의 원문 발언",
                })

            db_path = root / "source_metadata.sqlite3"
            build_source_index(source_dir, db_path)

            text = SourceTextReader(db_path, source_dir).read(
                ["cleaned_2025_세종특별자치시의회:1"]
            )

            self.assertEqual(text["cleaned_2025_세종특별자치시의회:1"], "중첩 폴더의 원문 발언")


if __name__ == "__main__":
    unittest.main()

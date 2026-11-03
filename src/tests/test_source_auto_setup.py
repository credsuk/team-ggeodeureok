import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data.source_index import ensure_source_index, lookup_source_metadata, sqlite_connection


class SourceIndexAutoSetupTests(unittest.TestCase):
    def _write_source(self, source_dir: Path) -> None:
        source_dir.mkdir(parents=True)
        with (source_dir / "minutes.txt").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["MTG_DE", "RASMBLY_NM", "MTGNM", "MINTS_NM", "agenda_title", "utterance"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerow({
                "MTG_DE": "2024-01-01", "RASMBLY_NM": "서울특별시의회",
                "MTGNM": "본회의", "MINTS_NM": "임시회",
                "agenda_title": "예산", "utterance": "예산 발언",
            })

    def test_missing_index_is_built_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "txt"
            self._write_source(source_dir)

            created = ensure_source_index(source_dir, root / "source.sqlite3")

            self.assertTrue(created)
            self.assertEqual(len(lookup_source_metadata(root / "source.sqlite3", ["minutes:1"])), 1)

    def test_existing_valid_index_is_reused_without_rebuilding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "txt"
            self._write_source(source_dir)
            db_path = root / "source.sqlite3"
            with sqlite_connection(db_path) as conn:
                conn.execute("""CREATE TABLE source_metadata (
                    doc_id TEXT PRIMARY KEY, source_file TEXT, row_id INTEGER, byte_offset INTEGER,
                    date TEXT, assembly TEXT, meeting_title TEXT, minutes_title TEXT, agenda TEXT
                )""")
                conn.execute("INSERT INTO source_metadata VALUES ('minutes:1', 'minutes', 1, 0, '20240101', '서울특별시의회', '본회의', '임시회', '예산')")

            with patch("src.data.source_index.build_source_index", side_effect=AssertionError("must not rebuild")):
                created = ensure_source_index(source_dir, db_path)

            self.assertFalse(created)
            with sqlite_connection(db_path) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM source_metadata").fetchone()[0], 1)

    def test_existing_invalid_index_fails_without_rebuilding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "txt"
            self._write_source(source_dir)
            db_path = root / "source.sqlite3"
            db_path.write_bytes(b"not sqlite")

            with patch("src.data.source_index.build_source_index", side_effect=AssertionError("must not rebuild")):
                with self.assertRaisesRegex(RuntimeError, "자동 재생성은 하지 않았습니다"):
                    ensure_source_index(source_dir, db_path)


if __name__ == "__main__":
    unittest.main()

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.source_index import main, sqlite_connection


class SourceIndexCliTests(unittest.TestCase):
    def test_cli_reuses_existing_valid_index_without_rebuilding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "txt"
            input_dir.mkdir()
            db_path = root / "source.sqlite3"
            with sqlite_connection(db_path) as conn:
                conn.execute("""CREATE TABLE source_metadata (
                    doc_id TEXT PRIMARY KEY, source_file TEXT, row_id INTEGER, byte_offset INTEGER,
                    date TEXT, assembly TEXT, meeting_title TEXT, minutes_title TEXT, agenda TEXT
                )""")
                conn.execute("INSERT INTO source_metadata VALUES ('minutes:1', 'minutes', 1, 0, '20240101', '서울특별시의회', '본회의', '임시회', '예산')")

            with patch.object(sys, "argv", ["source_index.py", "--input-dir", str(input_dir), "--db", str(db_path)]):
                with patch("src.source_index.build_source_index", side_effect=AssertionError("must not rebuild")):
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        self.assertEqual(main(), 0)

            self.assertIn("재사용", output.getvalue())


if __name__ == "__main__":
    unittest.main()

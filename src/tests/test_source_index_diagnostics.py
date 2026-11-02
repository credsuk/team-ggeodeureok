import contextlib
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.source_index import build_source_index
import src.source_index as source_index


class SourceIndexLockDiagnosticTests(unittest.TestCase):
    def test_sqlite_connection_closes_even_when_index_work_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.sqlite3"
            with source_index.sqlite_connection(db) as conn:
                conn.execute("CREATE TABLE sample (value INTEGER)")
                conn.execute("INSERT INTO sample VALUES (7)")
            with self.assertRaises(sqlite3.ProgrammingError):
                conn.execute("SELECT 1")

    def test_cleanup_lock_does_not_hide_original_index_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "txt"
            source_dir.mkdir()
            # Deliberately malformed schema to create a known primary failure.
            (source_dir / "bad.txt").write_text("unexpected_column\nvalue\n", encoding="utf-8")
            db_path = root / "source.sqlite3"
            stderr = io.StringIO()
            original_unlink = Path.unlink

            def deny_temp_cleanup(path, *args, **kwargs):
                if path.suffix == ".tmp":
                    raise PermissionError(32, "simulated sharing violation", str(path))
                return original_unlink(path, *args, **kwargs)

            with patch.object(Path, "unlink", deny_temp_cleanup):
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaisesRegex(ValueError, "필수 TXT 열"):
                        build_source_index(source_dir, db_path)
            self.assertIn("source index cleanup failed", stderr.getvalue())
            self.assertIn("stage=", stderr.getvalue())
            self.assertTrue(list(root.glob(".*.tmp")))


if __name__ == "__main__":
    unittest.main()

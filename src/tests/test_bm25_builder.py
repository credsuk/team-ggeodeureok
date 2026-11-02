import contextlib
import importlib.util
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = ROOT / "scripts" / "make_chroma_db" / "2_build_bm25_index.py"
SPEC = importlib.util.spec_from_file_location("bm25_builder", BUILDER_PATH)
bm25_builder = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(bm25_builder)


class SimulatedWinError32(PermissionError):
    @property
    def winerror(self):
        return 32


class BM25BuilderDiagnosticTests(unittest.TestCase):
    def test_managed_sqlite_connection_is_closed_after_committing(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.sqlite3"
            with bm25_builder.sqlite_connection(db_path) as conn:
                conn.execute("CREATE TABLE sample (value INTEGER)")
                conn.execute("INSERT INTO sample VALUES (1)")
            with self.assertRaises(sqlite3.ProgrammingError):
                conn.execute("SELECT 1")
            with bm25_builder.sqlite_connection(db_path) as check:
                self.assertEqual(check.execute("SELECT value FROM sample").fetchone()[0], 1)

    def test_lock_diagnostic_reports_stage_paths_process_and_file_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp_path = Path(tmp) / ".assembly_bm25.sqlite3.test.tmp"
            target_path = Path(tmp) / "assembly_bm25.sqlite3"
            temp_path.write_bytes(b"partial sqlite file")
            message = bm25_builder.format_lock_diagnostic(
                stage="atomic_replace",
                temp_path=temp_path,
                target_path=target_path,
                error=PermissionError(13, "access denied"),
                process_id=4321,
            )
            self.assertIn("stage=atomic_replace", message)
            self.assertIn(str(temp_path), message)
            self.assertIn(str(target_path), message)
            self.assertIn("pid=4321", message)
            self.assertIn("temp_exists=True", message)
            self.assertIn("temp_size=19", message)
            self.assertIn("winerror=", message)
    def test_replace_failure_logs_stage_and_keeps_temp_state_before_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "fixture.jsonl").write_text(
                json.dumps({"source_file": "fixture.txt", "row_id": 1, "utterance": "테스트 발언"}) + "\n",
                encoding="utf-8",
            )
            target = root / "assembly_bm25.sqlite3"
            stderr = io.StringIO()
            with patch.object(bm25_builder.os, "replace", side_effect=SimulatedWinError32(13, "sharing violation")):
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SimulatedWinError32):
                        bm25_builder.build_index(input_dir, target)
            diagnostic = stderr.getvalue()
            self.assertIn("stage=atomic replace", diagnostic)
            self.assertIn("winerror=32", diagnostic)
            self.assertIn(str(target), diagnostic)
            self.assertIn("temp_exists=True", diagnostic)
            self.assertIn("inserted=1", diagnostic)
            self.assertEqual(list(root.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()

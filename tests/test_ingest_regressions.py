"""Dependency-free regression tests; real ingestion code, fake DB/model boundaries."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]


class Progress:
    def __init__(self, **kwargs):
        self.n = kwargs.get("initial", 0)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def update(self, n):
        self.n += n
    def set_postfix(self, *args):
        pass
    @staticmethod
    def write(*args):
        pass


def load_ingest():
    tree = ast.parse((ROOT / "run_ingest.py").read_text(encoding="utf-8"))
    # Keep stdlib imports and all real functions; skip heavyweight integration imports.
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 or isinstance(node, (ast.Import, ast.ImportFrom))
                 and not (isinstance(node, ast.ImportFrom)
                          and ((node.module or "").startswith("src.") or node.module == "tqdm"))]
    ns = dict(globals(), BATCH_SIZE=500, CHUNK_SIZE=800, CHUNK_OVERLAP=100,
              MIN_CHAR_LEN=40, COLLECTION_NAME="test", CHROMA_DIR=ROOT / "chroma_db")
    exec(compile(tree, str(ROOT / "run_ingest.py"), "exec"), ns)
    return ns


class IngestRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.ns = load_ingest()
        self.ns["CHECKPOINT_FILE"] = self.base / "checkpoint.json"
        self.jsonl = self.base / "sample.jsonl"
        self.txt = self.base / "sample.txt"
        self.jsonl.write_text("{}\n" * 4)
        self.txt.write_text("header\n" + "text\n" * 4)
        self.records = [SimpleNamespace(
            row_id=i, file_stem="sample", utterance=f"speech-{i}", date="2024-01-01",
            year=2024, assembly="council", mints_nm="minutes", mtg_nm="meeting",
            agenda_title="agenda", speaker_name="name", speaker_role="role",
            speaker_type="type", policy_category="policy", core_issue="issue",
            keywords=[], summary="summary", is_substantive=True) for i in range(1, 5)]
        self.indexer = Mock()
        self.indexer.count.return_value = 0
        self.written_ids = []
        self.indexer.add_batch.side_effect = lambda ids, docs, metas: self.written_ids.extend(ids)
        self.loader = Mock()
        self.loader.discover_matched_pairs.return_value = [(self.jsonl, self.txt)]
        self.starts = []
        def stream(**kwargs):
            self.starts.append(kwargs["start_row"])
            return iter([r for r in self.records if r.row_id > kwargs["start_row"]
                         and (not kwargs["max_rows"] or r.row_id <= kwargs["max_rows"])])
        self.loader.stream_hybrid_records.side_effect = stream
        self.chunker = Mock()
        self.chunker.split_text.side_effect = lambda text: [text]
        self.ns.update(tqdm=Progress, HybridCouncilLoader=Mock(return_value=self.loader),
                       ChromaIndexer=Mock(return_value=self.indexer),
                       PGVectorIndexer=Mock(return_value=self.indexer),
                       CouncilMinuteChunker=Mock(return_value=self.chunker),
                       resolve_embedding_model=lambda name: name or "test-model",
                       get_pg_connection_string=lambda: "postgresql://synthetic/db")

    def run_ingest(self, **kwargs):
        options = dict(data_dir=self.base, change_text_dir=self.base,
                       augment_text=False, batch_size=10)
        options.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            self.ns["run_hybrid_ingest"](**options)

    def test_repeated_sample_does_not_mark_file_complete(self):
        self.run_ingest(sample_lines=2)
        self.run_ingest(sample_lines=2)
        checkpoint = json.loads(self.ns["CHECKPOINT_FILE"].read_text())
        self.assertNotIn("sample", checkpoint["completed_files"])
        self.run_ingest()
        self.assertEqual(self.starts, [0, 2])
        self.assertEqual(self.written_ids, [f"sample_r{i}_c0" for i in range(1, 5)])

    def test_interrupt_does_not_commit_unfinished_record(self):
        def split(text):
            if text == "speech-2":
                raise KeyboardInterrupt
            return [text]
        self.chunker.split_text.side_effect = split
        with self.assertRaises(SystemExit):
            self.run_ingest()
        self.assertEqual(self.written_ids, [])
        self.chunker.split_text.side_effect = lambda text: [text]
        self.run_ingest()
        self.assertEqual(self.starts, [0, 0])
        self.assertEqual(self.written_ids, [f"sample_r{i}_c0" for i in range(1, 5)])


    def test_changed_database_rejects_old_checkpoint(self):
        self.run_ingest(db_type="pgvector", pg_conn="postgresql://synthetic/first")
        self.indexer.add_batch.reset_mock()
        with self.assertRaisesRegex(ValueError, "no-resume|reset"):
            self.run_ingest(db_type="pgvector", pg_conn="postgresql://synthetic/second")
        self.indexer.add_batch.assert_not_called()

    def test_interrupt_after_commit_resumes_from_committed_row(self):
        def split(text):
            if text == "speech-3":
                raise KeyboardInterrupt
            return [text]
        self.chunker.split_text.side_effect = split
        with self.assertRaises(SystemExit):
            self.run_ingest(batch_size=2)
        checkpoint = json.loads(self.ns["CHECKPOINT_FILE"].read_text())
        self.assertEqual(checkpoint["file_progress"]["sample"]["last_row_id"], 2)
        self.chunker.split_text.side_effect = lambda text: [text]
        self.run_ingest(batch_size=2)
        self.assertEqual(self.starts, [0, 2])
        self.assertEqual(self.written_ids, [f"sample_r{i}_c0" for i in range(1, 5)])

    def test_interrupt_during_chunk_generation_discards_partial_record(self):
        self.chunker.split_text.side_effect = lambda text: [text + "-a", text + "-b"]
        def augment(record, chunk):
            if chunk == "speech-2-b":
                raise KeyboardInterrupt
            return chunk
        self.ns["create_augmented_text"] = augment
        with self.assertRaises(SystemExit):
            self.run_ingest(augment_text=True)
        self.assertEqual(self.written_ids, [])

    def test_legacy_checkpoint_requires_explicit_restart(self):
        self.ns["CHECKPOINT_FILE"].write_text(json.dumps({
            "db_type": "chroma", "collection": "test", "completed_files": ["sample"]}))
        with self.assertRaisesRegex(ValueError, "no-resume|reset"):
            self.run_ingest()
        self.run_ingest(no_resume=True)
        self.assertEqual(self.starts, [0])

    def test_changed_input_directory_rejects_old_checkpoint(self):
        self.run_ingest()
        with self.assertRaisesRegex(ValueError, "no-resume|reset"):
            self.run_ingest(data_dir=self.base / "other")

    def test_changed_embedding_settings_reject_old_checkpoint(self):
        self.run_ingest()
        with self.assertRaisesRegex(ValueError, "no-resume|reset"):
            self.run_ingest(model_name="other-model")

    def test_checkpoint_does_not_store_connection_credentials(self):
        self.run_ingest(db_type="pgvector", pg_conn="postgresql://user:SYNTHETIC_ONLY@host/db")
        checkpoint = json.loads(self.ns["CHECKPOINT_FILE"].read_text())
        self.assertIn("context", checkpoint)
        self.assertNotIn("SYNTHETIC_ONLY", json.dumps(checkpoint))


if __name__ == "__main__":
    unittest.main()

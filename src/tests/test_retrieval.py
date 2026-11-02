import tempfile
from pathlib import Path
import unittest

from src.models import Evidence, QueryIntent
from src.retrieval import BM25Repository, HybridRetriever, build_chroma_filter, make_fts_query, metadata_matches_scope
from src.source_index import SourceTextReader, build_source_index, sqlite_connection
from src.clustering import cluster_evidence


class RetrievalTests(unittest.TestCase):
    def test_builds_chroma_filter_from_only_explicit_scope(self):
        intent = QueryIntent(
            issue="급식 예산",
            assemblies=("서울특별시의회",),
            date_start="20240101",
            date_end="20251231",
        )
        self.assertEqual(build_chroma_filter(intent), {"assembly": {"$in": ["서울특별시의회"]}})
        self.assertFalse(metadata_matches_scope({"assembly": "서울특별시의회", "date": "20231231"}, intent))
        self.assertTrue(metadata_matches_scope({"assembly": "서울특별시의회", "date": "20240601"}, intent))
        self.assertIsNone(build_chroma_filter(QueryIntent(issue="급식", date_start="20240101")))
        self.assertEqual(make_fts_query("급식 예산"), '"급식" OR "예산"')

    def test_groups_hits_using_metadata_core_issue(self):
        hits = [
            Evidence(id="a:1", text="one", date="20240101", assembly="서울", metadata={"core_issue": "학교 급식"}),
            Evidence(id="b:2", text="two", date="20240201", assembly="부산", metadata={"core_issue": "학교 급식"}),
            Evidence(id="c:3", text="three", date="20240301", assembly="대전", metadata={"core_issue": "버스 노선"}),
        ]
        clusters = cluster_evidence(hits)
        self.assertEqual([item["count"] for item in clusters], [2, 1])
        self.assertEqual(clusters[0]["issue"], "학교 급식")


class BM25RepositoryTests(unittest.TestCase):
    def test_joins_real_fts_hits_to_source_metadata_and_orders_by_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            bm25_path = Path(tmp) / "bm25.sqlite3"
            source_path = Path(tmp) / "source.sqlite3"
            with sqlite_connection(bm25_path) as conn:
                conn.execute("CREATE VIRTUAL TABLE bm25_docs USING fts5(chroma_id UNINDEXED, source_file UNINDEXED, row_id UNINDEXED, text, tokenize='unicode61')")
                conn.executemany("INSERT INTO bm25_docs VALUES (?, ?, ?, ?)", [
                    ("late:2", "late.txt", "2", "학교 급식 예산 논의"),
                    ("early:1", "early.txt", "1", "학교 급식 예산 논의"),
                ])
            with sqlite_connection(source_path) as conn:
                conn.execute("CREATE TABLE source_metadata (doc_id TEXT, source_file TEXT, row_id INTEGER, byte_offset INTEGER, date TEXT, assembly TEXT, meeting_title TEXT, minutes_title TEXT, agenda TEXT)")
                conn.executemany("INSERT INTO source_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [
                    ("late:2", "late", 2, 0, "20250201", "서울", "본회의", "회기", "급식"),
                    ("early:1", "early", 1, 0, "20240101", "서울", "본회의", "회기", "급식"),
                ])
            repo = BM25Repository(bm25_path, source_path)
            result = repo.search("급식 예산", QueryIntent(issue="급식", assemblies=("서울",)), 5, chronological=True)
            self.assertEqual([item["id"] for item in result], ["early:1", "late:2"])
            self.assertEqual(result[0]["meeting_title"], "본회의")


class HybridRetrieverTests(unittest.TestCase):
    def test_selected_evidence_text_is_loaded_from_original_txt_not_chroma_summary(self):
        import csv
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "to_text"
            source_dir.mkdir()
            source_file = source_dir / "cleaned_2024_서울특별시의회.txt"
            with source_file.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["MTG_DE", "RASMBLY_NM", "MTGNM", "MINTS_NM", "agenda_title", "utterance"], delimiter="\t")
                writer.writeheader()
                writer.writerow({"MTG_DE": "2024-02-01", "RASMBLY_NM": "서울특별시의회", "MTGNM": "본회의", "MINTS_NM": "임시회", "agenda_title": "급식", "utterance": "급식 예산 원본 TXT 발언"})
            metadata_db = root / "source.sqlite3"
            build_source_index(source_dir, metadata_db)
            bm25_db = root / "bm25.sqlite3"
            with sqlite_connection(bm25_db) as conn:
                conn.execute("CREATE VIRTUAL TABLE bm25_docs USING fts5(chroma_id UNINDEXED, source_file UNINDEXED, row_id UNINDEXED, text, tokenize='unicode61')")
                conn.execute("INSERT INTO bm25_docs VALUES (?, ?, ?, ?)", ("cleaned_2024_서울특별시의회:1", source_file.name, "1", "급식 예산 indexed copy"))
            class FakeCollection:
                def count(self): return 1
                def query(self, **kwargs): return {"ids": [["cleaned_2024_서울특별시의회:1"]], "distances": [[0.1]]}
                def get(self, **kwargs): return {"ids": ["cleaned_2024_서울특별시의회:1"], "metadatas": [{"date": "20240201", "assembly": "서울특별시의회", "core_issue": "학교 급식"}]}
            settings = SimpleNamespace(dense_limit=5, lexical_limit=5, candidate_limit=5, source_metadata_db=metadata_db)
            retriever = HybridRetriever(
                FakeCollection(), object(), BM25Repository(bm25_db, metadata_db),
                SourceTextReader(metadata_db, source_dir), settings,
            )
            found = retriever.retrieve(QueryIntent(issue="급식", keywords=("예산",)), [0.1, 0.2])
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].text, "급식 예산 원본 TXT 발언")
            self.assertNotIn("indexed copy", found[0].text)


if __name__ == "__main__":
    unittest.main()

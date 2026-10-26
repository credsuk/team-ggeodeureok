import importlib.util
import sys
import types
import unittest
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))


def load_retriever_module():
    """Load the production retriever with dependency-only modules stubbed."""
    saved = {name: sys.modules.get(name) for name in (
        "chromadb", "chromadb.utils", "chromadb.utils.embedding_functions"
    )}
    chromadb = types.ModuleType("chromadb")
    setattr(chromadb, "PersistentClient", object)
    utils = types.ModuleType("chromadb.utils")
    embedding_functions = types.ModuleType("chromadb.utils.embedding_functions")
    setattr(utils, "embedding_functions", embedding_functions)
    setattr(chromadb, "utils", utils)
    sys.modules["chromadb"] = chromadb
    sys.modules["chromadb.utils"] = utils
    sys.modules["chromadb.utils.embedding_functions"] = embedding_functions
    try:
        spec = importlib.util.spec_from_file_location(
            "_mini_retriever_test_target", MODEL_DIR / "retriever.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError("could not load the project retriever module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


retriever_module = load_retriever_module()


class FakeCollection:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def count(self):
        return 10

    def query(self, **kwargs):
        self.calls.append(kwargs)
        query_texts = kwargs["query_texts"]
        grouped = [self.responses[text] for text in query_texts]
        return {
            "ids": [[item[0] for item in group] for group in grouped],
            "documents": [[item[1] for item in group] for group in grouped],
            "metadatas": [[item[2] for item in group] for group in grouped],
            "distances": [[item[3] for item in group] for group in grouped],
        }


class DenseQueryBatchTests(unittest.TestCase):
    def test_batches_queries_in_one_chroma_call_and_preserves_rrf_scores(self):
        council = "서울특별시의회"
        meta_a = {"date": "20240115", "assembly": council}
        meta_b = {"date": "20250115", "assembly": council}
        responses = {
            "질문": [("a", "문서 A", meta_a, 0.1), ("b", "문서 B", meta_b, 0.2)],
            "주제": [("a", "문서 A", meta_a, 0.3)],
            "별칭": [("b", "문서 B", meta_b, 0.4)],
        }
        collection = FakeCollection(responses)
        retriever = retriever_module.HybridRetriever.__new__(retriever_module.HybridRetriever)
        retriever.collection = collection
        retriever.distance_metric = "cosine"

        candidates = retriever._dense_candidates(
            ["질문", "주제", "별칭"], limit=2, params={}
        )

        self.assertEqual(len(collection.calls), 1)
        self.assertEqual(collection.calls[0]["query_texts"], ["질문", "주제", "별칭"])
        self.assertAlmostEqual(candidates["a"]["rrf"], 1 / 61 + 1 / 61)
        self.assertEqual(candidates["b"]["rrf"], 1 / 62 + 1 / 61)
        self.assertEqual(candidates["a"]["distance"], 0.1)
        self.assertAlmostEqual(candidates["a"]["vector_similarity"], 0.9)
        self.assertAlmostEqual(candidates["b"]["vector_similarity"], 0.8)


class SimilarityThresholdTests(unittest.TestCase):
    def test_search_keeps_only_dense_candidates_at_or_above_point_six(self):
        meta = {"date": "20240115", "assembly": "서울특별시의회"}
        responses = {
            "질문": [
                ("boundary", "경계 문서", meta, 0.4),
                ("below", "미달 문서", meta, 0.4001),
                ("multi-query", "다중 질의 문서", meta, 0.45),
            ],
            "주제": [("multi-query", "다중 질의 문서", meta, 0.39)],
        }
        retriever = retriever_module.HybridRetriever.__new__(retriever_module.HybridRetriever)
        retriever.collection = FakeCollection(responses)
        retriever.distance_metric = "cosine"
        lexical_only = {
            "id": "lexical-only", "text": "정확 일치", "metadata": meta,
            "rrf": 1.0, "retrieval_sources": ["bm25"],
        }
        boundary_lexical = {
            "id": "boundary", "text": "경계 문서", "metadata": meta,
            "rrf": 0.5, "retrieval_sources": ["bm25"],
        }
        retriever._lexical_candidates = lambda _terms, _params: {
            "lexical-only": lexical_only, "boundary": boundary_lexical,
        }

        results = retriever.search(
            {"query": "질문", "topic": "주제", "include_terms": []}, top_k=100
        )

        by_id = {row["id"]: row for row in results}
        self.assertEqual(set(by_id), {"boundary", "multi-query"})
        self.assertAlmostEqual(by_id["boundary"]["vector_similarity"], 0.6)
        self.assertAlmostEqual(by_id["multi-query"]["vector_similarity"], 0.61)
        self.assertIn("bm25", by_id["boundary"]["retrieval_sources"])


class ExpandRowsVectorSimilarityTests(unittest.TestCase):
    def test_keeps_the_highest_chunk_similarity_for_the_reconstructed_source_row(self):
        class RowCollection:
            def get(self, **_kwargs):
                return {
                    "ids": ["chunk-0", "chunk-1"],
                    "documents": ["앞 문장", "뒷 문장"],
                    "metadatas": [
                        {"dataset_key": "d", "row_id": 1, "chunk_index": 0, "date": "20240101"},
                        {"dataset_key": "d", "row_id": 1, "chunk_index": 1, "date": "20240101"},
                    ],
                }

        retriever = retriever_module.HybridRetriever.__new__(retriever_module.HybridRetriever)
        retriever.collection = RowCollection()
        expanded = retriever.expand_rows([
            {"id": "chunk-0", "metadata": {"dataset_key": "d", "row_id": 1}, "vector_similarity": 0.72},
            {"id": "chunk-1", "metadata": {"dataset_key": "d", "row_id": 1}, "vector_similarity": 0.91},
        ])

        self.assertEqual(len(expanded), 1)
        self.assertAlmostEqual(expanded[0]["vector_similarity"], 0.91)

    def test_does_not_interpret_l2_distance_as_cosine_similarity(self):
        collection = FakeCollection({
            "질문": [("a", "문서 A", {"date": "20240115", "assembly": "서울특별시의회"}, 0.25)],
        })
        retriever = retriever_module.HybridRetriever.__new__(retriever_module.HybridRetriever)
        retriever.collection = collection
        retriever.distance_metric = "l2"

        candidates = retriever._dense_candidates(["질문"], limit=1, params={})

        self.assertEqual(candidates["a"]["distance"], 0.25)
        self.assertIsNone(candidates["a"]["vector_similarity"])


if __name__ == "__main__":
    unittest.main()

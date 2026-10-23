"""Offline regression tests; run with stdlib unittest, without ML dependencies."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def load_pipeline():
    """Execute the real module with temporary, fail-closed import stubs."""
    package = ModuleType("src")
    package.__path__ = []
    config = ModuleType("src.config")
    vars(config).update(
        COLLECTION_NAME="test",
        DEFAULT_DB_TYPE="chroma",
        EMBEDDING_MODEL_NAME="test",
        get_pg_connection_string=Mock(side_effect=AssertionError("No config reads")),
        resolve_embedding_model=Mock(side_effect=AssertionError("No model loading")),
    )
    indexer = ModuleType("src.indexer")
    vars(indexer)["ChromaIndexer"] = Mock(side_effect=AssertionError("No database initialization"))
    pg_indexer = ModuleType("src.pg_indexer")
    vars(pg_indexer)["PGVectorIndexer"] = Mock(side_effect=AssertionError("No database initialization"))
    requests = ModuleType("requests")
    vars(requests)["post"] = Mock(side_effect=AssertionError("No network requests"))
    stubs = {
        "src": package,
        "src.config": config,
        "src.indexer": indexer,
        "src.pg_indexer": pg_indexer,
        "requests": requests,
    }
    path = Path(__file__).resolve().parents[1] / "src" / "pipeline.py"
    spec = importlib.util.spec_from_file_location("_pipeline_regressions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class PipelineRegressions(unittest.TestCase):
    def setUp(self):
        pipeline_class = load_pipeline().CouncilRAGPipeline
        self.pipeline = pipeline_class.__new__(pipeline_class)
        self.pipeline.indexer = SimpleNamespace(query=Mock(return_value=[]))
        self.pipeline._call_ollama = Mock(return_value="")

    def test_empty_filtered_search_does_not_retry_without_filters(self):
        cases = [
            ({"assembly": "Council"}, {"assembly": "Council"}),
            (
                {"assembly": "Council", "year": 2024, "speaker_name": "Kim",
                 "policy_category": "교육/문화"},
                {"$and": [{"assembly": "Council"}, {"year": 2024},
                          {"speaker_name": "Kim"}, {"policy_category": "교육/문화"}]},
            ),
        ]
        for filters, expected_where in cases:
            with self.subTest(filters=filters):
                self.pipeline.indexer.query.reset_mock()
                self.assertEqual(self.pipeline.search_db("education", filters, top_k=7), [])
                self.pipeline.indexer.query.assert_called_once_with(
                    query_text="education", n_results=7, where=expected_where
                )

    def test_none_reranker_truncates_in_original_order_without_llm(self):
        self.pipeline.reranker_type = "none"
        docs = [SimpleNamespace(page_content=f"document {i}", metadata={}) for i in range(3)]
        candidates = list(zip(docs, [0.8, 0.2, 0.1]))

        result = self.pipeline.rerank_documents("education", candidates, top_n=2)

        self.pipeline._call_ollama.assert_not_called()
        self.assertEqual(result, [
            {"rank": 1, "rerank_score": 0.2, "initial_distance": 0.8, "document": docs[0]},
            {"rank": 2, "rerank_score": 0.8, "initial_distance": 0.2, "document": docs[1]},
        ])
        self.assertEqual(candidates, list(zip(docs, [0.8, 0.2, 0.1])))

    def test_nonpositive_limit_returns_empty_without_any_reranker(self):
        candidates = [(SimpleNamespace(page_content="document", metadata={}), 0.2),
                      (SimpleNamespace(page_content="other", metadata={}), 0.3)]
        for reranker_type in ("none", "llm", "cross-encoder"):
            for limit in (0, -1):
                with self.subTest(reranker_type=reranker_type, top_n=limit):
                    self.pipeline.reranker_type = reranker_type
                    self.pipeline._cross_encoder = Mock()
                    self.pipeline._cross_encoder.predict.return_value = [0.8, 0.7]
                    self.pipeline._call_ollama.reset_mock()

                    result = self.pipeline.rerank_documents("education", candidates, top_n=limit)

                    self.pipeline._call_ollama.assert_not_called()
                    self.pipeline._cross_encoder.predict.assert_not_called()
                    self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()

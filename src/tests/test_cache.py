import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.cache import AnalysisCache
from src.models import AnalysisResult, QueryIntent


class CacheTests(unittest.TestCase):
    def test_sqlite_context_closes_connection_after_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = AnalysisCache(Path(tmp) / "cache.sqlite3")
            with cache._connect() as connection:
                connection.execute("SELECT 1")

            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")

    def test_exact_intent_cache_round_trips_result_and_version_gates_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = AnalysisCache(Path(tmp) / "cache.sqlite3")
            intent = QueryIntent(issue="school meals", keywords=("budget",))
            result = AnalysisResult(
                intent=intent,
                answer="근거 내에서 확인된 결과",
                first_seen=None,
                flow=[],
                forecast="증가 가능성",
                confidence="낮음",
                forecast_as_of="2025-12-31",
                citations=[],
            )
            cache.put(
                intent, result, data_version="corpus-v1", prompt_version="p1",
                model_id="local-model", question="School-meals budget?",
            )
            found = cache.get(intent, data_version="corpus-v1", prompt_version="p1", model_id="local-model")
            self.assertIsNotNone(found)
            self.assertTrue(found.cache_hit)
            self.assertEqual(found.answer, result.answer)
            normalized_question = cache.get_by_question(
                "school meals budget!", as_of_date="2025-12-31",
                data_version="corpus-v1", prompt_version="p1", model_id="local-model",
            )
            self.assertIsNotNone(normalized_question)
            self.assertTrue(normalized_question.cache_hit)
            self.assertIsNone(cache.get_by_question(
                "school meals budget", as_of_date="2025-12-30",
                data_version="corpus-v1", prompt_version="p1", model_id="local-model",
            ))
            self.assertIsNone(cache.get_by_question(
                "school meals budget", as_of_date="2025-12-31",
                data_version="corpus-v1", prompt_version="p2", model_id="local-model",
            ))
            self.assertIsNone(cache.get(intent, data_version="corpus-v2", prompt_version="p1", model_id="local-model"))
            self.assertIsNone(cache.get(intent, data_version="corpus-v1", prompt_version="p2", model_id="local-model"))

    def test_clear_removes_all_sqlite_analysis_results_and_returns_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = AnalysisCache(Path(tmp) / "cache.sqlite3")
            intent = QueryIntent(issue="급식 예산")
            result = AnalysisResult(
                intent=intent, answer="답변", first_seen=None, flow=[], forecast="전망",
                confidence="낮음", forecast_as_of="20261031",
            )
            cache.put(
                intent, result, data_version="v1", prompt_version="p1", model_id="m1",
                question="급식 예산",
            )
            self.assertEqual(cache.count(), 1)
            self.assertEqual(
                cache.get_by_question(
                    "급식 예산", as_of_date="20261031", data_version="v1",
                    prompt_version="p1", model_id="m1",
                ).answer,
                "답변",
            )

            self.assertEqual(cache.clear(), 1)
            self.assertEqual(cache.count(), 0)
            self.assertIsNone(
                cache.get_by_question(
                    "급식 예산", as_of_date="20261031", data_version="v1",
                    prompt_version="p1", model_id="m1",
                )
            )

    def test_semantic_cache_clear_deletes_ids_in_bounded_batches(self):
        from src.cache import SemanticQuestionCache

        class PagedCollection:
            def __init__(self):
                self.ids = [f"q{i}" for i in range(5)]
                self.batch_sizes = []
            def get(self, limit, include):
                self.batch_sizes.append(limit)
                return {"ids": self.ids[:limit]}
            def delete(self, ids):
                self.ids = [item for item in self.ids if item not in ids]

        collection = PagedCollection()
        cache = SemanticQuestionCache(
            collection, similarity_threshold=0.94, data_version="v1",
            prompt_version="p1", model_id="m1",
        )

        self.assertEqual(cache.clear(batch_size=2), 5)
        self.assertEqual(collection.ids, [])
        self.assertEqual(collection.batch_sizes, [2, 2, 2, 2])

    def test_semantic_question_cache_requires_close_distance_and_matching_scope(self):
        from src.cache import SemanticQuestionCache
        class FakeCollection:
            def __init__(self, distance, issue="학교 급식"):
                self.distance = distance
                self.issue = issue
                self.query_args = None
            def query(self, **kwargs):
                self.query_args = kwargs
                return {"ids": [["q1"]], "distances": [[self.distance]], "metadatas": [[{"cache_key": "saved-key", "issue": self.issue}]]}
            def add(self, **kwargs):
                self.add_args = kwargs
        intent = QueryIntent(issue="학교 급식", assemblies=("서울특별시의회",))
        collection = FakeCollection(0.03)
        index = SemanticQuestionCache(collection, similarity_threshold=0.94, data_version="v1", prompt_version="p1", model_id="m1")
        self.assertEqual(index.find([0.1, 0.2], intent), "saved-key")
        self.assertIn("where", collection.query_args)
        self.assertEqual(next(item["data_version"] for item in collection.query_args["where"]["$and"] if "data_version" in item), "v1")
        distant = SemanticQuestionCache(FakeCollection(0.12), similarity_threshold=0.94, data_version="v1", prompt_version="p1", model_id="m1")
        self.assertIsNone(distant.find([0.1, 0.2], intent))
        unrelated = SemanticQuestionCache(FakeCollection(0.01, "버스 노선"), similarity_threshold=0.94, data_version="v1", prompt_version="p1", model_id="m1")
        self.assertIsNone(unrelated.find([0.1, 0.2], intent))


if __name__ == "__main__":
    unittest.main()

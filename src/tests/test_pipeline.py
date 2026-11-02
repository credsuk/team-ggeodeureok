from datetime import date
import tempfile
import unittest
from pathlib import Path

from src.cache import AnalysisCache
from src.models import AnalysisResult, Evidence, QueryIntent
from src.pipeline import AnalysisPipeline
from src.regional_spread import build_regional_spread


class FakeIntentExtractor:
    def __init__(self, intent):
        self.intent = intent
        self.calls = 0
    def extract(self, question, history=None):
        self.calls += 1
        return self.intent


class FakeSemanticCache:
    def __init__(self, key=None):
        self.key = key
        self.remembered = []
    def find(self, embedding, intent):
        return self.key
    def remember(self, embedding, query_text, cache_key, intent):
        self.remembered.append(cache_key)


class FakeRetriever:
    def __init__(self, evidence):
        self.evidence = evidence
        self.retrieval_calls = 0
        self.intents = []
    def embed_query(self, text):
        return [0.1, 0.2]
    def retrieve(self, intent, query_embedding=None):
        self.retrieval_calls += 1
        self.intents.append(intent)
        if intent.assemblies:
            return [item for item in self.evidence if item.assembly in intent.assemblies]
        return self.evidence


class FakeReranker:
    def __init__(self):
        self.calls = 0
    def rerank(self, query, evidence, limit):
        self.calls += 1
        return evidence[:limit]


class FakeWriter:
    def __init__(self):
        self.calls = 0
        self.spread_evidence = []
    def generate(self, question, intent, evidence, spread_evidence=None):
        self.calls += 1
        self.spread_evidence = list(spread_evidence or [])
        full_evidence = [*evidence, *self.spread_evidence]
        return {
            "answer": "회의록 근거 요약",
            "core_facts": ["급식 예산 논의가 있었다."],
            "likely_discussion_topics": ([
                {"topic": "추가 예산 심의", "source_ids": [evidence[0].id, "invented:9"]}
            ] if evidence else []),
            "forecast": "증가 가능성",
            "forecast_citations": [item.id for item in full_evidence[:4]],
            "confidence": "높음",
            "citations": [evidence[0].id],
            "regional_spread": build_regional_spread(intent, full_evidence, {item.id for item in full_evidence}),
        }


class PipelineTests(unittest.TestCase):
    def test_cache_hit_reuses_answer_and_skips_retrieval_rerank_and_forecast(self):
        with tempfile.TemporaryDirectory() as tmp:
            intent = QueryIntent(issue="학교 급식", keywords=("예산",), as_of_date=date.today().strftime("%Y%m%d"))
            evidence = [Evidence(id="doc:1", text="급식 예산 논의", date="20240201", assembly="서울특별시의회")]
            cache = AnalysisCache(Path(tmp) / "cache.sqlite3")
            semantic = FakeSemanticCache()
            retriever = FakeRetriever(evidence)
            reranker = FakeReranker()
            writer = FakeWriter()
            intent_extractor = FakeIntentExtractor(intent)
            pipeline = AnalysisPipeline(
                intent_extractor=intent_extractor, cache=cache,
                semantic_cache=semantic, retriever=retriever, reranker=reranker,
                writer=writer, data_version="v1", prompt_version="p1", model_id="local",
                rerank_limit=20,
            )
            first = pipeline.ask("서울 급식 예산 흐름")
            second = pipeline.ask("서울 급식 예산 흐름", history=[{"role": "user", "content": "서울 급식 예산 흐름"}])
            self.assertEqual(first.first_seen.id, "doc:1")
            self.assertEqual(first.core_facts, ["급식 예산 논의가 있었다."])
            self.assertEqual(
                first.likely_discussion_topics,
                [{"topic": "추가 예산 심의", "source_ids": ["doc:1"]}],
            )
            self.assertTrue(first.forecast)
            self.assertEqual(first.forecast_citations, ["doc:1"])
            self.assertEqual(first.confidence, "낮음")  # 한 날짜·한 근거면 높은 신뢰도를 허용하지 않는다.
            self.assertEqual(second.forecast, first.forecast)
            self.assertEqual(second.likely_discussion_topics, first.likely_discussion_topics)
            self.assertTrue(second.cache_hit)
            self.assertEqual(intent_extractor.calls, 1)
            self.assertEqual(retriever.retrieval_calls, 1)
            self.assertEqual(reranker.calls, 1)
            self.assertEqual(writer.calls, 1)
            self.assertEqual(len(semantic.remembered), 1)

    def test_forecast_confidence_is_bounded_by_cited_date_coverage(self):
        evidence = [
            Evidence(id=f"doc:{index}", text="근거", date=date)
            for index, date in enumerate(("20240101", "20240201", "20240301", "20240401"), start=1)
        ]
        self.assertEqual(
            AnalysisPipeline._forecast_confidence("높음", ["doc:1"], evidence), "낮음"
        )
        self.assertEqual(
            AnalysisPipeline._forecast_confidence("높음", ["doc:1", "doc:2"], evidence), "보통"
        )
        self.assertEqual(
            AnalysisPipeline._forecast_confidence(
                "높음", ["doc:1", "doc:2", "doc:3", "doc:4"], evidence
            ),
            "높음",
        )

    def test_explicit_council_query_searches_other_councils_for_spread(self):
        with tempfile.TemporaryDirectory() as tmp:
            intent = QueryIntent(issue="학교 급식 예산", assemblies=("서울특별시의회",), as_of_date="20261102")
            evidence = [
                Evidence(id="seoul:1", text="급식 예산 검토", date="20240101", assembly="서울특별시의회"),
                Evidence(id="busan:2", text="급식 예산 논의", date="20240201", assembly="부산광역시의회"),
                Evidence(id="daegu:3", text="급식 예산 심의", date="20240301", assembly="대구광역시의회"),
            ]
            retriever = FakeRetriever(evidence)
            writer = FakeWriter()
            pipeline = AnalysisPipeline(
                intent_extractor=FakeIntentExtractor(intent), cache=AnalysisCache(Path(tmp) / "cache.sqlite3"),
                semantic_cache=None, retriever=retriever, reranker=FakeReranker(), writer=writer,
                data_version="v1", prompt_version="p1", model_id="local",
            )

            result = pipeline.ask("서울의 급식 예산이 다른 시도로 확산될 가능성은?")

            self.assertEqual(retriever.retrieval_calls, 2)
            self.assertEqual(retriever.intents[1].assemblies, ())
            self.assertEqual({item.id for item in writer.spread_evidence}, {"busan:2", "daegu:3"})
            self.assertEqual(result.regional_spread["status"], "observed_related_discussions")
            self.assertEqual(
                {item.id for item in result.regional_spread_evidence}, {"busan:2", "daegu:3"}
            )

    def test_pipeline_persists_only_retrieved_council_ids_in_regional_spread(self):
        with tempfile.TemporaryDirectory() as tmp:
            intent = QueryIntent(issue="학교 급식 예산", as_of_date="20261102")
            evidence = [
                Evidence(id="doc:1", text="급식 예산", date="20240101", assembly="서울특별시의회"),
                Evidence(id="doc:2", text="급식 예산", date="20240201", assembly="부산광역시의회"),
            ]
            class RegionalWriter(FakeWriter):
                def generate(self, question, intent, evidence, spread_evidence=None):
                    result = super().generate(question, intent, evidence, spread_evidence)
                    result["regional_spread"] = {
                        "observed_councils": [
                            {"assembly": "서울특별시의회", "source_ids": ["doc:1", "invented:9"]},
                            {"assembly": "부산광역시의회", "source_ids": ["doc:2"]},
                        ],
                    }
                    return result

            pipeline = AnalysisPipeline(
                intent_extractor=FakeIntentExtractor(intent), cache=AnalysisCache(Path(tmp) / "cache.sqlite3"),
                semantic_cache=None, retriever=FakeRetriever(evidence), reranker=FakeReranker(),
                writer=RegionalWriter(), data_version="v1", prompt_version="p1", model_id="local",
            )

            result = pipeline.ask("급식 예산 타 시도 확산?")

            self.assertEqual(result.regional_spread["status"], "observed_related_discussions")
            self.assertEqual(result.regional_spread["basis_record_count"], 2)
            source_ids = {source_id for row in result.regional_spread["observed_councils"] for source_id in row["source_ids"]}
            self.assertEqual(source_ids, {"doc:1", "doc:2"})

    def test_semantically_similar_cached_question_skips_full_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_intent = QueryIntent(issue="학교 급식", keywords=("예산",), as_of_date="20261030")
            new_intent = QueryIntent(issue="학교 급식 예산", keywords=("지원",), as_of_date="20261030")
            saved = AnalysisResult(
                intent=original_intent, answer="저장된 분석", first_seen=None, flow=[],
                forecast="변화 가능", confidence="낮음", forecast_as_of="20261030",
            )
            cache = AnalysisCache(Path(tmp) / "cache.sqlite3")
            cache.put(original_intent, saved, data_version="v1", prompt_version="p1", model_id="local")
            retriever = FakeRetriever([Evidence(id="doc:1", text="unused")])
            reranker = FakeReranker()
            writer = FakeWriter()
            pipeline = AnalysisPipeline(
                intent_extractor=FakeIntentExtractor(new_intent), cache=cache,
                semantic_cache=FakeSemanticCache(original_intent.cache_key), retriever=retriever,
                reranker=reranker, writer=writer, data_version="v1", prompt_version="p1",
                model_id="local", rerank_limit=20,
            )
            result = pipeline.ask("학교 급식 예산 지원은?")
            self.assertTrue(result.cache_hit)
            self.assertEqual(result.answer, "저장된 분석")
            self.assertEqual(retriever.retrieval_calls, 0)
            self.assertEqual(reranker.calls, 0)
            self.assertEqual(writer.calls, 0)


if __name__ == "__main__":
    unittest.main()

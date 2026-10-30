import unittest

from api.services import RadarApiService


class FakePostgresRepository:
    def __init__(self):
        self.search_calls = []
        self.monthly = [
            {"month": "2025-01-01", "document_count": 1, "council_count": 1, "councils": ["서울특별시의회"]},
            {"month": "2025-02-01", "document_count": 2, "council_count": 1, "councils": ["서울특별시의회"]},
            {"month": "2025-03-01", "document_count": 3, "council_count": 2, "councils": ["서울특별시의회", "부산광역시의회"]},
        ]

    def search(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        return {
            "query": query,
            "backend": "postgresql_pgvector_full_text",
            "retrieval_mode": "hybrid_rrf",
            "result_count": 1,
            "results": [{
                "chunk_id": "chunk-1", "document_id": "doc-1", "text": "회의록 근거",
                "metadata": {"council": "서울특별시의회", "meeting_date": "2025-03-15"},
                "retrieval_sources": ["dense", "text"],
            }],
            "search_ms": 3.2,
            "query_embedding_ms": 1.1,
        }

    def monthly_counts(self, query, filters=None):
        return self.monthly

    def council_first_observations(self, query, filters=None):
        return [
            {"council": "서울특별시의회", "first_observation_date": "2025-01-15", "document_count": 2},
            {"council": "부산광역시의회", "first_observation_date": "2025-03-15", "document_count": 1},
        ]

    def get_evidence(self, chunk_id):
        return {"chunk_id": chunk_id, "text": "DB 원문", "metadata": {"council": "서울특별시의회"}}

    def get_document(self, document_id, limit=100):
        return [{"chunk_id": "chunk-1", "document_id": document_id, "text": "DB 원문", "metadata": {}}]

    def validate(self):
        return {"table": "council_arctic_raw_v1", "vector_dimensions": 1024}

    def embed_query(self, query):
        return [0.0] * 1024


ISSUES = [
    {"issue_key": "birth_rate", "issue_name": "저출생·출산율 대응 정책",
     "query": "출산율 저하 대응 정책", "aliases": "저출생 정책|출산 지원", "category": "보건/복지"},
    {"issue_key": "small_business", "issue_name": "소상공인 경영 안정 지원",
     "query": "소상공인 경영 안정 지원 정책", "aliases": "자영업자|골목상권", "category": "경제/산업"},
]


class RadarServiceTests(unittest.TestCase):
    def setUp(self):
        self.repository = FakePostgresRepository()
        self.service = RadarApiService(repository=self.repository, issue_definitions=ISSUES)

    def test_preflight_checks_database_and_encoder_dimension(self):
        result = self.service.validate_database()
        self.assertEqual(result["table"], "council_arctic_raw_v1")

    def test_fixed_issue_list_comes_from_config_without_web_bundle(self):
        items = self.service.list_issues()
        self.assertEqual([item["issue_key"] for item in items], ["birth_rate", "small_business"])
        self.assertEqual(self.repository.search_calls, [])

    def test_fixed_issue_detail_searches_postgres_and_includes_database_forecast(self):
        payload = self.service.get_issue("birth_rate")
        self.assertEqual(self.repository.search_calls[0][0], "출산율 저하 대응 정책")
        self.assertEqual(payload["issue"]["issue_id"], "birth_rate")
        self.assertEqual(payload["evidence"][0]["chunk_id"], "chunk-1")
        self.assertEqual(payload["forecast"]["method"], "trailing_3_month_mean")
        self.assertEqual(payload["forecast"]["predicted_document_count"], 2)
        self.assertEqual(payload["forecast"]["basis"], "postgresql_full_text_matches")
        self.assertEqual(payload["propagation"][0]["first_observation_date"], "2025-01-15")

    def test_fixed_radar_match_uses_postgres_results_not_precomputed_json(self):
        result = self.service.radar_search("저출생 정책")
        self.assertEqual(result["match_type"], "fixed_issue")
        self.assertEqual(result["matched_issue_id"], "birth_rate")
        self.assertEqual(result["evidence"][0]["chunk_id"], "chunk-1")

    def test_unmatched_radar_query_runs_hybrid_search_against_postgres(self):
        result = self.service.radar_search("해양 미세플라스틱 오염")
        self.assertEqual(result["match_type"], "dynamic")
        self.assertEqual(result["query"], "해양 미세플라스틱 오염")
        self.assertEqual(result["retrieval"]["mode"], "hybrid_rrf")
        self.assertEqual(self.repository.search_calls[-1][0], "해양 미세플라스틱 오염")

    def test_document_and_evidence_reads_use_postgres_repository(self):
        doc = self.service.document("doc-1")
        evidence = self.service.evidence("chunk-1")
        self.assertEqual(doc["items"][0]["text"], "DB 원문")
        self.assertEqual(evidence["text"], "DB 원문")


if __name__ == "__main__":
    unittest.main()

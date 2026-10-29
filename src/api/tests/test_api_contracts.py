import unittest

from fastapi.testclient import TestClient

from api.dependencies import get_api_service
from api.main import app


class FakeService:
    def list_issues(self):
        return [{"issue_key": "birth_rate", "issue_name": "저출생"}]

    def get_issue(self, issue_key):
        if issue_key != "birth_rate":
            return None
        return {"issue": {"issue_id": issue_key}, "external_followup": None}

    def _attach_fixed_followup(self, payload):
        return {**payload, "external_followup_status": "not_precomputed"}

    def radar_search(self, query):
        return {"query": query, "match_type": "fixed_issue", "issue": {"issue_id": "birth_rate"}}

    def document_search(self, query, **kwargs):
        return {"query": query, "result_count": 1, "results": [{"chunk_id": "c1"}]}

    def document(self, document_id, limit=100):
        return {"document_id": document_id, "result_count": 1, "items": []}

    def evidence(self, chunk_id):
        return {"chunk_id": chunk_id, "text": "근거 발언"}

    def chat(self, request):
        return {"answer": "근거 기반 답변", "citations": [], "conversation_id": request.conversation_id}


class ApiContractTests(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[get_api_service] = lambda: FakeService()
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_issue_catalog_returns_items(self):
        response = self.client.get("/api/v1/issues")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"][0]["issue_key"], "birth_rate")

    def test_issue_detail_returns_payload(self):
        response = self.client.get("/api/v1/issues/birth_rate")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["issue"]["issue_id"], "birth_rate")

    def test_unknown_issue_returns_404(self):
        response = self.client.get("/api/v1/issues/not-a-known-issue")
        self.assertEqual(response.status_code, 404)

    def test_radar_returns_match_type(self):
        response = self.client.post("/api/v1/radar/search", json={"query": "저출생"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["match_type"], "fixed_issue")

    def test_blank_radar_query_is_rejected(self):
        response = self.client.post("/api/v1/radar/search", json={"query": "   "})
        self.assertEqual(response.status_code, 422)

    def test_document_search_accepts_query(self):
        response = self.client.get("/api/v1/documents/search", params={"q": "저출생"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result_count"], 1)

    def test_evidence_endpoint_returns_chunk(self):
        response = self.client.get("/api/v1/evidence/chunk-1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["chunk_id"], "chunk-1")

    def test_document_detail_returns_document(self):
        response = self.client.get("/api/v1/documents/document-1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["document_id"], "document-1")

    def test_chat_accepts_question(self):
        response = self.client.post("/api/v1/chat", json={"message": "최초 관측은?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "근거 기반 답변")

    def test_chat_status_is_available_without_ollama(self):
        response = self.client.get("/api/v1/chat/status")
        self.assertEqual(response.status_code, 200)
        self.assertIn("reachable", response.json())
        self.assertIn("model_available", response.json())


if __name__ == "__main__":
    unittest.main()

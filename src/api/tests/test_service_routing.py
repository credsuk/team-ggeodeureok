import json
import tempfile
import unittest
from pathlib import Path

from api.services import RadarApiService


class RadarServiceTests(unittest.TestCase):
    def test_curated_query_reuses_precomputed_issue_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload_path = root / "birth_rate_issue.json"
            payload = {
                "schema_version": "council_issue_web_v2",
                "issue": {"issue_id": "birth_rate", "issue_name": "저출생", "aliases": ["출산율"]},
                "summary": {"council_count": 3},
                "external_followup": {"mode": "backtest", "outlook": {"top_candidates": []}},
            }
            payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            (root / "issue_catalog.json").write_text(json.dumps({
                "schema_version": "council_issue_catalog_v1",
                "issues": [{"issue_key": "birth_rate", "issue_id": "birth_rate",
                            "issue_name": "저출생", "payload": str(payload_path)}],
            }, ensure_ascii=False), encoding="utf-8")

            service = RadarApiService()
            service.web_dir = root
            result = service.radar_search("저출생 정책")

            self.assertEqual(result["match_type"], "fixed_issue")
            self.assertEqual(result["matched_issue_id"], "birth_rate")
            self.assertEqual(result["external_followup_status"], "available")
            self.assertEqual(result["summary"]["council_count"], 3)

    def test_new_query_uses_cached_dynamic_result_without_running_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "issue_catalog.json").write_text(json.dumps({
                "schema_version": "council_issue_catalog_v1", "issues": []
            }), encoding="utf-8")
            service = RadarApiService()
            service.web_dir = root
            service.dynamic_dir = root / "dynamic"
            query = "새로운 농촌 교통 주제"
            output = service._dynamic_result_path(query)
            output.parent.mkdir(parents=True)
            output.write_text(json.dumps({
                "schema_version": "council_dynamic_issue_v1",
                "status": "provisional_until_review",
                "issue": {"issue_name": query},
                "source_files": {"timeline": "private/path"},
            }), encoding="utf-8")

            result = service.radar_search(query)

            self.assertEqual(result["match_type"], "dynamic")
            self.assertEqual(result["issue"]["issue_name"], query)
            self.assertNotIn("source_files", result)
            self.assertIn("external_followup_status", result)


if __name__ == "__main__":
    unittest.main()

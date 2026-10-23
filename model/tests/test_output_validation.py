import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from output_validation import validate_output_evidence


class OutputEvidenceTests(unittest.TestCase):
    def test_rejects_source_id_that_disagrees_with_row_provenance(self):
        output = {"results": [{
            "summaryEvidenceIds": ["other:9"], "forecast": {"evidenceIds": []},
            "evidence": [{"sourceId": "other:9", "datasetKey": "dataset", "originalRowId": 1}],
            "firstObserved": [], "subsequentDiscussions": [],
        }]}
        with self.assertRaisesRegex(ValueError, "does not match source-row provenance"):
            validate_output_evidence(output)

    def test_accepts_only_existing_source_ids_for_llm_claims(self):
        output = {
            "results": [{
                "summaryEvidenceIds": ["dataset:1"],
                "forecast": {"evidenceIds": ["dataset:2"]},
                "evidence": [
                    {"sourceId": "dataset:1", "datasetKey": "dataset", "originalRowId": 1},
                    {"sourceId": "dataset:2", "datasetKey": "dataset", "originalRowId": 2},
                ],
                "firstObserved": [{"sourceId": "dataset:1", "datasetKey": "dataset", "originalRowId": 1}],
                "subsequentDiscussions": [{"sourceId": "dataset:2", "datasetKey": "dataset", "originalRowId": 2}],
            }]
        }
        self.assertTrue(validate_output_evidence(output))

    def test_rejects_llm_citation_not_present_in_retrieved_evidence(self):
        output = {"results": [{
            "summaryEvidenceIds": ["invented"], "forecast": {"evidenceIds": []},
            "evidence": [{"sourceId": "dataset:1", "datasetKey": "dataset", "originalRowId": 1}], "firstObserved": [], "subsequentDiscussions": [],
        }]}
        with self.assertRaisesRegex(ValueError, "not present"):
            validate_output_evidence(output)

    def test_rejects_malformed_source_row_provenance(self):
        output = {"results": [{
            "summaryEvidenceIds": [], "forecast": {"evidenceIds": []},
            "evidence": [{"sourceId": "dataset:1", "datasetKey": "dataset", "originalRowId": 1}],
            "firstObserved": [{"sourceId": "dataset:1", "datasetKey": "dataset", "originalRowId": None}],
            "subsequentDiscussions": [],
        }]}
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_output_evidence(output)


if __name__ == "__main__":
    unittest.main()

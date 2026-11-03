import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.config import load_env_file


class ConfigTests(unittest.TestCase):
    def test_settings_paths_remain_anchored_to_project_root_after_package_move(self):
        from src.core.config import Settings
        project_root = Path(__file__).resolve().parents[2]
        settings = Settings.from_env()
        self.assertEqual(settings.root, project_root)
        self.assertEqual(settings.data_dir, project_root / "data")

    def test_settings_load_search_and_reranker_weights_from_environment(self):
        values = {
            "DENSE_RRF_WEIGHT": "2.5",
            "BM25_RRF_WEIGHT": "0",
            "RERANKER_MODELS": "model-a, model-b",
            "RERANKER_WEIGHTS": "3, 1",
        }
        with patch.dict(os.environ, values, clear=False):
            from src.core.config import Settings
            settings = Settings.from_env()
        self.assertEqual(settings.dense_rrf_weight, 2.5)
        self.assertEqual(settings.lexical_rrf_weight, 0.0)
        self.assertEqual(settings.reranker_models, ("model-a", "model-b"))
        self.assertEqual(settings.reranker_weights, (3.0, 1.0))
        self.assertIn("model-b", settings.cache_identity)

    def test_cache_identity_changes_when_retrieval_weights_change(self):
        from src.core.config import Settings
        with patch.dict(os.environ, {"DENSE_RRF_WEIGHT": "1", "BM25_RRF_WEIGHT": "1"}, clear=False):
            baseline = Settings.from_env().cache_identity
        with patch.dict(os.environ, {"DENSE_RRF_WEIGHT": "2", "BM25_RRF_WEIGHT": "1"}, clear=False):
            changed = Settings.from_env().cache_identity
        self.assertNotEqual(baseline, changed)

    def test_settings_reject_invalid_search_weights(self):
        from src.core.config import Settings
        with patch.dict(os.environ, {"DENSE_RRF_WEIGHT": "0", "BM25_RRF_WEIGHT": "0"}, clear=False):
            with self.assertRaises(ValueError):
                Settings.from_env()
        with patch.dict(os.environ, {"DENSE_RRF_WEIGHT": "-1", "BM25_RRF_WEIGHT": "1"}, clear=False):
            with self.assertRaises(ValueError):
                Settings.from_env()

    def test_env_file_fills_missing_values_without_overriding_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("APP_TEST_FROM_FILE=loaded\nAPP_TEST_FROM_SHELL=file\n# comment\n", encoding="utf-8")
            with patch.dict(os.environ, {"APP_TEST_FROM_SHELL": "shell"}, clear=False):
                load_env_file(env_file)
                self.assertEqual(os.environ["APP_TEST_FROM_FILE"], "loaded")
                self.assertEqual(os.environ["APP_TEST_FROM_SHELL"], "shell")


if __name__ == "__main__":
    unittest.main()

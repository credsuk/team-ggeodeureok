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

    def test_settings_uses_example_when_env_file_is_missing(self):
        from src.core import config
        from src.core.config import Settings

        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            src_dir = project_root / "src"
            src_dir.mkdir()
            (src_dir / ".env.example").write_text(
                "OLLAMA_MODEL=example-model\nDATA_VERSION=example-data\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True), patch.object(config, "ROOT", project_root):
                settings = Settings.from_env()

        self.assertEqual(settings.ollama_model, "example-model")
        self.assertEqual(settings.data_version, "example-data")

    def test_model_download_is_enabled_by_default_when_unset(self):
        from src.core import config
        from src.core.config import Settings

        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            (project_root / "src").mkdir()
            with patch.dict(os.environ, {}, clear=True), patch.object(config, "ROOT", project_root):
                settings = Settings.from_env()

        self.assertTrue(settings.allow_model_download)

    def test_model_download_can_be_disabled_explicitly(self):
        from src.core import config
        from src.core.config import Settings

        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            (project_root / "src").mkdir()
            with patch.dict(os.environ, {"ALLOW_MODEL_DOWNLOAD": "0"}, clear=True), patch.object(config, "ROOT", project_root):
                settings = Settings.from_env()

        self.assertFalse(settings.allow_model_download)

    def test_settings_prefers_env_file_when_both_env_files_exist(self):
        from src.core import config
        from src.core.config import Settings

        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            src_dir = project_root / "src"
            src_dir.mkdir()
            (src_dir / ".env").write_text("OLLAMA_MODEL=local-model\n", encoding="utf-8")
            (src_dir / ".env.example").write_text("OLLAMA_MODEL=example-model\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True), patch.object(config, "ROOT", project_root):
                settings = Settings.from_env()

        self.assertEqual(settings.ollama_model, "local-model")

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

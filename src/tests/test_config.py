import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import load_env_file


class ConfigTests(unittest.TestCase):
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

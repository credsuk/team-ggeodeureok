import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main as launcher


class LauncherTests(unittest.TestCase):
    def test_env_file_loads_values_without_overwriting_shell_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text('MODEL="exaone3.5:7.8b"\nEXTRA=value # comment\n', encoding="utf-8")
            environment = {"MODEL": "shell-model"}

            loaded = launcher.load_env_file(env_file, environment)

            self.assertEqual(loaded, 1)
            self.assertEqual(environment["MODEL"], "shell-model")
            self.assertEqual(environment["EXTRA"], "value")

    def test_ollama_start_is_skipped_when_service_is_already_reachable(self):
        with patch.object(launcher, "ollama_is_reachable", return_value=True), \
             patch.object(launcher.subprocess, "Popen") as popen:
            ready = launcher.ensure_ollama({"OLLAMA_URL": "http://127.0.0.1:11434"})
        self.assertTrue(ready)
        popen.assert_not_called()

    def test_ollama_start_is_skipped_for_non_local_url(self):
        with patch.object(launcher.subprocess, "Popen") as popen:
            ready = launcher.ensure_ollama({"OLLAMA_URL": "http://ollama-host:11434"})
        self.assertFalse(ready)
        popen.assert_not_called()

    def test_ollama_start_reports_missing_executable(self):
        with patch.object(launcher, "ollama_is_reachable", return_value=False), \
             patch.object(launcher.shutil, "which", return_value=None):
            ready = launcher.ensure_ollama({"OLLAMA_URL": "http://127.0.0.1:11434"})
        self.assertFalse(ready)

    def test_ollama_is_started_and_bind_address_matches_configured_url(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(launcher, "SRC_ROOT", Path(directory)),
                patch.object(launcher, "ollama_is_reachable", side_effect=[False, True]),
                patch.object(launcher.shutil, "which", return_value="/usr/bin/ollama"),
                patch.object(launcher.subprocess, "Popen") as popen,
            ):
                ready = launcher.ensure_ollama({"OLLAMA_URL": "http://127.0.0.1:11435"}, wait_seconds=1)
            self.assertTrue(ready)
            args, kwargs = popen.call_args
            self.assertEqual(args[0], ["/usr/bin/ollama", "serve"])
            self.assertEqual(kwargs["env"]["OLLAMA_HOST"], "127.0.0.1:11435")

    def test_main_aborts_before_uvicorn_if_ollama_start_fails(self):
        import io
        import uvicorn
        from contextlib import redirect_stdout

        output = io.StringIO()
        with (
            patch.object(launcher.os, "chdir"),
            patch.object(launcher, "load_env_file", return_value=0),
            patch.object(launcher, "ensure_ollama", return_value=False),
            patch.dict(launcher.os.environ, {"AUTO_START_OLLAMA": "1"}),
            patch.object(uvicorn, "run") as run_server,
            redirect_stdout(output),
            self.assertRaises(SystemExit) as raised,
        ):
            launcher.main()

        self.assertEqual(raised.exception.code, 1)
        run_server.assert_not_called()
        self.assertIn("서버를 시작하지 않았습니다", output.getvalue())

    def test_main_aborts_before_uvicorn_if_preflight_fails(self):
        import io
        import uvicorn
        from contextlib import redirect_stdout

        output = io.StringIO()
        with (
            patch.object(launcher.os, "chdir"),
            patch.object(launcher, "load_env_file", return_value=0),
            patch.object(launcher, "ensure_ollama", return_value=True),
            patch.object(launcher, "validate_api_prerequisites", side_effect=RuntimeError("DB 연결 실패")),
            patch.dict(launcher.os.environ, {"AUTO_START_OLLAMA": "1"}),
            patch.object(uvicorn, "run") as run_server,
            redirect_stdout(output),
            self.assertRaises(SystemExit) as raised,
        ):
            launcher.main()

        self.assertEqual(raised.exception.code, 1)
        run_server.assert_not_called()
        self.assertIn("DB 연결 실패", output.getvalue())
        self.assertIn("서버를 시작하지 않았습니다", output.getvalue())

    def test_main_invokes_uvicorn_with_configured_host_and_port(self):
        import uvicorn
        with (
            patch.object(launcher.os, "chdir"),
            patch.object(launcher, "load_env_file", return_value=0),
            patch.object(launcher, "ensure_ollama", return_value=True),
            patch.object(launcher, "validate_api_prerequisites"),
            patch.dict(launcher.os.environ, {"HOST": "127.0.0.1", "PORT": "8011", "AUTO_START_OLLAMA": "1"}),
            patch.object(uvicorn, "run") as run_server,
        ):
            launcher.main()
        run_server.assert_called_once_with(
            "api.main:app", host="127.0.0.1", port=8011, log_level="info"
        )


if __name__ == "__main__":
    unittest.main()

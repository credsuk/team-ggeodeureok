import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.app import run_entrypoint


class AppEntrypointTests(unittest.TestCase):
    def test_direct_python_entrypoint_starts_a_local_streamlit_app(self):
        app_instance = SimpleNamespace(run=Mock())
        st = SimpleNamespace(
            runtime=SimpleNamespace(exists=Mock(return_value=False)),
            App=Mock(return_value=app_instance),
        )
        render_ui = Mock()
        script = Path("src/app.py")

        run_entrypoint(st, script, render_ui)

        st.App.assert_called_once_with(str(script.resolve()))
        app_instance.run.assert_called_once_with()
        render_ui.assert_not_called()

    def test_streamlit_runner_executes_ui_without_starting_nested_server(self):
        st = SimpleNamespace(
            runtime=SimpleNamespace(exists=Mock(return_value=True)),
            App=Mock(),
        )
        render_ui = Mock()

        run_entrypoint(st, Path("src/app.py"), render_ui)

        render_ui.assert_called_once_with()
        st.App.assert_not_called()


if __name__ == "__main__":
    unittest.main()

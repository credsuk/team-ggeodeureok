import tomllib
import unittest
from pathlib import Path


from src.app import APP_STYLE


class StreamlitUiConfigTests(unittest.TestCase):
    def test_hides_default_toolbar_menu_without_hiding_sidebar_reopen_control(self):
        project_root = Path(__file__).resolve().parents[2]
        config = tomllib.loads((project_root / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
        self.assertEqual(config["client"]["toolbarMode"], "minimal")
        app_source = (project_root / "src" / "app.py").read_text(encoding="utf-8")
        self.assertIn("menu_items={}", app_source)
        self.assertNotIn('[data-testid="stToolbar"] { display: none !important; }', APP_STYLE)


if __name__ == "__main__":
    unittest.main()

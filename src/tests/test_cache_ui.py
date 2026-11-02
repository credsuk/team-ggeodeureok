from contextlib import nullcontext
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from src import app
from src.app import clear_pipeline_caches


class FakeMainPage:
    def __init__(self):
        self.buttons = []
        self.titles = []
        self.column_layouts = []
    def columns(self, spec, **kwargs):
        self.column_layouts.append((spec, kwargs))
        return nullcontext(), nullcontext()
    def title(self, label):
        self.titles.append(label)
    def button(self, label, **kwargs):
        self.buttons.append((label, kwargs))
        return False


class CacheButtonServiceTests(unittest.TestCase):
    def test_cache_reset_button_is_small_and_next_to_page_title(self):
        renderer = getattr(app, "render_page_header", None)
        self.assertTrue(callable(renderer), "page header needs a title and adjacent cache action")
        st = FakeMainPage()

        renderer(st, disabled=False)

        self.assertEqual(st.titles, ["지방의회 회의록 분석"])
        self.assertEqual(st.column_layouts, [([6, 1], {"vertical_alignment": "center"})])
        self.assertEqual(st.buttons[0][0], "캐시 초기화")
        self.assertFalse(st.buttons[0][1]["disabled"])
        self.assertEqual(st.buttons[0][1]["width"], "content")
        self.assertIn("저장된 답변", st.buttons[0][1]["help"])

    def test_clear_pipeline_caches_clears_exact_and_semantic_stores(self):
        exact = SimpleNamespace(clear=Mock(return_value=3))
        semantic = SimpleNamespace(clear=Mock(return_value=5))
        pipeline = SimpleNamespace(cache=exact, semantic_cache=semantic)

        self.assertEqual(clear_pipeline_caches(pipeline), (3, 5))
        exact.clear.assert_called_once_with()
        semantic.clear.assert_called_once_with()

    def test_clear_pipeline_caches_handles_disabled_semantic_cache(self):
        exact = SimpleNamespace(clear=Mock(return_value=2))
        pipeline = SimpleNamespace(cache=exact, semantic_cache=None)

        self.assertEqual(clear_pipeline_caches(pipeline), (2, 0))
        exact.clear.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

import unittest

from src.snippets import extract_relevant_excerpt


class RelevantExcerptTests(unittest.TestCase):
    def test_excerpt_centers_on_query_terms_instead_of_showing_only_the_start(self):
        text = "처음 발언입니다. " + ("관련 없는 일반 발언입니다. " * 10) + "학교 급식 예산을 증액해야 합니다. " + ("뒤쪽 일반 발언입니다. " * 10)

        excerpt = extract_relevant_excerpt(text, ["학교 급식", "예산"], max_chars=100)

        self.assertIn("급식 예산을 증액해야 합니다", excerpt)
        self.assertTrue(excerpt.startswith("…"))
        self.assertLess(len(excerpt), len(text))

    def test_excerpt_uses_start_of_text_when_no_query_term_matches(self):
        text = "첫 문장입니다. " + ("다음 발언입니다. " * 20)

        excerpt = extract_relevant_excerpt(text, ["학교 급식"], max_chars=40)

        self.assertTrue(excerpt.startswith("첫 문장입니다."))
        self.assertTrue(excerpt.endswith("…"))
        self.assertLess(len(excerpt), len(text))


if __name__ == "__main__":
    unittest.main()

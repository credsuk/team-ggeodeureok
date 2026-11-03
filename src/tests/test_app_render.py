import contextlib
import json
import unittest

from src import app
from src.app import render_result
from src.core.models import AnalysisResult, Evidence, QueryIntent


class FakeStreamlit:
    def __init__(self):
        self.markdowns = []
        self.top_level_markdowns = []
        self.expander_markdowns = []
        self.texts = []
        self.subheaders = []
        self.expanders = []
        self.warnings = []
        self.deck_charts = []
        self.dataframes = []
        self.captions = []
        self.expander_options = {}
        self.expander_depth = 0
        self.max_expander_depth = 0
        self.events = []
    def markdown(self, value):
        value = str(value)
        self.events.append(("markdown", value))
        self.markdowns.append(value)
        if self.expander_depth:
            self.expander_markdowns.append(value)
        else:
            self.top_level_markdowns.append(value)
    def text(self, value):
        self.texts.append(str(value))
    def subheader(self, value):
        self.subheaders.append(str(value))
    def caption(self, value):
        value = str(value)
        self.events.append(("caption", value))
        self.captions.append(value)
    def info(self, value):
        pass
    def warning(self, value):
        self.warnings.append(str(value))
    def expander(self, label, expanded=False):
        self.expanders.append(label)
        self.events.append(("expander", label))
        self.expander_options[label] = expanded
        @contextlib.contextmanager
        def tracked():
            self.expander_depth += 1
            self.max_expander_depth = max(self.max_expander_depth, self.expander_depth)
            try:
                yield
            finally:
                self.expander_depth -= 1
        return tracked()
    def container(self, **kwargs):
        return contextlib.nullcontext()
    def map(self, *args, **kwargs):
        pass
    def pydeck_chart(self, deck, **kwargs):
        self.deck_charts.append(deck)
    def dataframe(self, data, *args, **kwargs):
        self.dataframes.append(data)


class ResultRenderingTests(unittest.TestCase):
    def test_structured_core_facts_render_even_when_answer_is_only_a_heading(self):
        st = FakeStreamlit()
        result = AnalysisResult(
            intent=QueryIntent(issue="급식 예산"), answer="핵심 사실:", first_seen=None,
            flow=[Evidence(id="doc:1", text="급식 예산", date="20250101", assembly="서울특별시의회")],
            forecast="심의가 이어질 수 있습니다.", confidence="높음", forecast_as_of="20261101",
            core_facts=["급식 예산 증액 필요성이 논의됐습니다."],
            likely_discussion_topics=[
                {"topic": "다음 연도 예산 편성", "source_ids": ["doc:1"]},
                {"topic": "급식 지원 대상 조정", "source_ids": ["doc:1"]},
            ],
        )

        render_result(st, result)

        self.assertIn("급식 예산 증액 필요성이 논의됐습니다.", " ".join(st.markdowns))
        self.assertIn("향후 논의 가능 주제", st.subheaders)
        self.assertIn("다음 연도 예산 편성", " ".join(st.markdowns))
        self.assertIn("급식 지원 대상 조정", " ".join(st.markdowns))
        self.assertIn("`doc:1`", " ".join(st.markdowns))
        self.assertIn("핵심 사실", st.subheaders)
        self.assertLess(
            st.subheaders.index("향후 전망 · 기준일 20261101"),
            st.subheaders.index("핵심 사실"),
        )
        confidence_index = next(i for i, event in enumerate(st.events) if event == ("caption", "신뢰도: 높음"))
        forecast_index = next(i for i, event in enumerate(st.events) if event[0] == "markdown" and "심의가 이어질 수 있습니다." in event[1])
        self.assertLess(confidence_index, forecast_index)
        self.assertIn("심의가 이어질 수 있습니다.", " ".join(st.top_level_markdowns))
        self.assertNotIn("심의가 이어질 수 있습니다.", " ".join(st.expander_markdowns))
        details_index = next(i for i, event in enumerate(st.events) if event == ("expander", "의회별 위치와 논의 순서"))
        self.assertLess(forecast_index, details_index)

    def test_empty_forecast_is_visibly_flagged(self):
        st = FakeStreamlit()
        result = AnalysisResult(
            intent=QueryIntent(issue="급식 예산"), answer="", first_seen=None, flow=[],
            forecast="", confidence="낮음", forecast_as_of="20261101",
            core_facts=["예산 논의가 있었습니다."],
        )

        render_result(st, result)

        self.assertTrue(any("전망이 없습니다" in message for message in st.warnings))
        self.assertEqual(st.captions.count("신뢰도: 낮음"), 1)
        self.assertFalse(any("정성 평가" in caption or "미래 결과를 확인한 값" in caption for caption in st.captions))
        self.assertFalse(any("예측 단계에는 원문 발언" in caption for caption in st.captions))

    def test_forecast_basis_is_expandable_and_uses_cited_evidence(self):
        st = FakeStreamlit()
        evidence = Evidence(
            id="doc:forecast-1", text="급식 예산을 다시 검토하기로 논의했습니다.",
            date="20250301", assembly="서울특별시의회", agenda="급식 예산",
        )
        result = AnalysisResult(
            intent=QueryIntent(issue="급식 예산"), answer="",
            first_seen=None, flow=[evidence], forecast="후속 예산 심의가 이어질 수 있습니다.",
            confidence="낮음", forecast_as_of="20261101",
            forecast_citations=["doc:forecast-1"],
        )

        render_result(st, result)

        self.assertIn("전망 근거 보기", st.expanders)
        self.assertTrue(any("출처 ID: `doc:forecast-1`" in caption for caption in st.captions))
        self.assertLessEqual(st.max_expander_depth, 1)

    def test_regional_spread_section_shows_council_basis_and_caveat(self):
        st = FakeStreamlit()
        flow = [
            Evidence(id="seoul:1", text="급식 예산 검토", date="20240110", assembly="서울특별시의회"),
        ]
        spread_evidence = [
            Evidence(id="busan:2", text="급식 지원 심의", date="20240220", assembly="부산광역시의회"),
        ]
        spread = {
            "status": "observed_related_discussions",
            "scope_limited": False,
            "basis_record_count": 1,
            "origin_record_count": 1,
            "origin_councils": [
                {"assembly": "서울특별시의회", "record_count": 1, "date_range": {"start": "20240110", "end": "20240110"}, "source_ids": ["seoul:1"]},
            ],
            "observed_councils": [
                {"assembly": "부산광역시의회", "record_count": 1, "date_range": {"start": "20240220", "end": "20240220"}, "source_ids": ["busan:2"]},
            ],
            "outlook": "여러 의회에서 관련 논의가 관측됐으나 전파 관계를 입증하지 않습니다.",
            "causality_established": False,
            "probability_estimated": False,
        }
        result = AnalysisResult(
            intent=QueryIntent(issue="학교 급식 예산", assemblies=("서울특별시의회",)), answer="요약", first_seen=None,
            flow=flow, forecast="조건부 전망", confidence="낮음", forecast_as_of="20261102",
            regional_spread=spread,
            regional_spread_evidence=spread_evidence,
        )

        render_result(st, result)

        self.assertIn("다른 시·도 확산 가능성", st.subheaders)
        self.assertIn(spread["outlook"], st.markdowns)
        self.assertTrue(any("- 기준 의회: 서울특별시의회" in item for item in st.markdowns))
        self.assertTrue(any("- 비교 의회: 부산광역시의회" in item for item in st.markdowns))
        self.assertTrue(any("출처 ID: `busan:2`" in caption for caption in st.captions))
        self.assertIn("급식 지원 심의", " ".join(st.texts))
        self.assertTrue(any("확산 확률" in caption for caption in st.captions))

    def test_first_observation_shows_relevant_excerpt_and_expands_full_text(self):
        st = FakeStreamlit()
        full_text = "첫머리 발언. " + ("일반 안건 설명입니다. " * 12) + "학교 급식 예산을 증액해야 합니다. " + ("마무리 발언입니다. " * 12)
        evidence = Evidence(
            id="doc:1", text=full_text, date="20250101", assembly="서울특별시의회",
            meeting_title="본회의", agenda="학교 급식 예산",
        )
        result = AnalysisResult(
            intent=QueryIntent(issue="학교 급식", keywords=("예산",)), answer="요약",
            first_seen=evidence, flow=[], forecast="전망", confidence="보통",
            forecast_as_of="20261101", core_facts=["예산이 논의됐습니다."],
        )

        render_result(st, result)

        self.assertIn("학교 급식 예산을 증액해야 합니다", " ".join(st.texts))
        self.assertIn("원문 전체 보기", st.expanders)
        self.assertIn(full_text, st.texts)

    def test_timeline_clusters_and_citations_start_collapsed_without_nested_expanders(self):
        st = FakeStreamlit()
        evidence = Evidence(
            id="doc:1", text="급식 예산 심의", date="20250101", assembly="서울특별시의회",
        )
        result = AnalysisResult(
            intent=QueryIntent(issue="급식 예산"), answer="요약", first_seen=evidence,
            flow=[evidence], forecast="후속 심의 여부를 지켜봐야 합니다.",
            confidence="보통", forecast_as_of="20261101", core_facts=["예산 논의가 있었습니다."],
            clusters=[{"issue": "급식 예산", "count": 1}], citations=["doc:1"],
        )

        render_result(st, result)

        self.assertFalse(st.expander_options["의회별 위치와 논의 순서"])
        self.assertFalse(st.expander_options["쟁점별 근거 묶음"])
        self.assertFalse(st.expander_options["출처와 인용"])
        self.assertLessEqual(st.max_expander_depth, 1)
    def test_chat_history_groups_each_query_and_full_result_in_one_collapsed_expander(self):
        renderer = getattr(app, "render_chat_history", None)
        self.assertTrue(callable(renderer), "chat history needs a question-group renderer")
        st = FakeStreamlit()
        evidence = Evidence(
            id="doc:1", text="학교 급식 예산 논의", date="20250101",
            assembly="서울특별시의회", meeting_title="본회의",
        )
        result = AnalysisResult(
            intent=QueryIntent(issue="학교 급식 예산"), answer="회의록 근거 요약",
            first_seen=evidence, flow=[evidence], forecast="추가 논의가 확인됩니다.",
            confidence="낮음", forecast_as_of="20261102", citations=["doc:1"],
            forecast_citations=["doc:1"], core_facts=["예산 증액이 논의됐습니다."],
            clusters=[{"issue": "학교 급식 예산", "count": 1}],
        )
        messages = [
            {"role": "user", "content": "학교 급식 예산 흐름 알려줘"},
            {"role": "assistant", "content": result.answer, "result": result.to_dict()},
        ]

        renderer(st, messages)

        self.assertGreater(len(st.expanders), 1)
        self.assertIn("학교 급식 예산", st.expanders[0])
        self.assertFalse(st.expander_options[st.expanders[0]])
        self.assertEqual(st.max_expander_depth, 2)
        self.assertIn("의회별 위치와 논의 순서", st.expanders)
        self.assertFalse(st.expander_options["의회별 위치와 논의 순서"])
        self.assertFalse(st.expander_options["출처와 인용"])
        self.assertIn("회의록 근거 요약", st.expander_markdowns)
        self.assertIn("추가 논의가 확인됩니다.", st.expander_markdowns)
        self.assertIn("질문: 학교 급식 예산 흐름 알려줘", st.captions)

    def test_map_markers_show_numeric_discussion_order(self):
        st = FakeStreamlit()
        flow = [
            Evidence(id="a:1", text="first", date="20240101", assembly="부산광역시의회", meeting_title="회의1"),
            Evidence(id="a:2", text="same council again", date="20240201", assembly="부산광역시의회", meeting_title="회의2"),
            Evidence(id="b:3", text="next", date="20240301", assembly="서울특별시의회", meeting_title="회의3"),
        ]
        result = AnalysisResult(
            intent=QueryIntent(issue="논의 흐름"), answer="요약", first_seen=flow[0],
            flow=flow, forecast="후속 논의 확인이 필요합니다.", confidence="보통",
            forecast_as_of="20261101", core_facts=["두 의회에서 논의됐습니다."],
        )

        render_result(st, result)

        self.assertEqual(len(st.deck_charts), 1)
        deck = json.loads(st.deck_charts[0].to_json())
        layers = deck["layers"]
        circle_layer = next(layer for layer in layers if layer["@@type"] == "ScatterplotLayer")
        text_layer = next(layer for layer in layers if layer["@@type"] == "TextLayer")
        self.assertEqual(len(layers), 2)
        self.assertEqual(len(circle_layer["data"]), 2)
        self.assertEqual([point["order_label"] for point in text_layer["data"]], ["1", "2"])
        self.assertEqual(
            [(row["발언 순서"], row["지도 번호"]) for row in st.dataframes[0]],
            [(1, "1"), (2, ""), (3, "2")],
        )
        self.assertEqual(circle_layer["getRadius"], 18)
        self.assertEqual(circle_layer["radiusUnits"], "pixels")
        self.assertEqual(text_layer["getSize"], 27)
        self.assertEqual(text_layer["sizeUnits"], "pixels")
        self.assertFalse(circle_layer["pickable"])
        self.assertFalse(text_layer["pickable"])
        self.assertNotIn("tooltip", deck)


if __name__ == "__main__":
    unittest.main()

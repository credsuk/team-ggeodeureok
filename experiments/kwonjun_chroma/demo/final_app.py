#!/usr/bin/env python3
"""꺼드럭's 의정이슈레이더 대표 정책 이슈 TOP 10·검색·Agent 통합 화면."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import time
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import pydeck as pdk
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.local_issue_agent import (  # noqa: E402
    LocalCouncilAgent,
    probe_ollama,
    route_fixed_issue,
)
from agent.followup_tools import FollowupTools  # noqa: E402
from agent.query_pipeline import INTENTS, QuestionPipeline  # noqa: E402
from council_vector_runtime import CouncilSearchService, SearchFilters  # noqa: E402


WEB_DIR = Path(
    os.environ.get("COUNCIL_FINAL_WEB_DIR", PROJECT_ROOT / "outputs" / "web_final")
).expanduser().resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_catalog(path_text: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    payload = load_json(Path(path_text))
    if payload.get("schema_version") != "council_issue_catalog_v1":
        raise ValueError("지원하지 않는 이슈 카탈로그입니다")
    return payload


@st.cache_data(show_spinner=False)
def load_payload(path_text: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    payload = load_json(Path(path_text))
    if payload.get("schema_version") != "council_issue_web_v2":
        raise ValueError("지원하지 않는 최종 이슈 JSON입니다")
    return payload


@st.cache_resource(show_spinner="Arctic·BM25 하이브리드 검색기를 연결하는 중입니다…")
def load_search_service() -> CouncilSearchService:
    return CouncilSearchService(
        backend="pgvector",
        embedding_dir=Path(os.environ.get(
            "COUNCIL_EMBEDDING_DIR",
            PROJECT_ROOT / "data" / "embeddings" / "arctic_raw_v1",
        )).expanduser().resolve(),
        parent_path=Path(os.environ.get(
            "COUNCIL_PARENT_PATH",
            PROJECT_ROOT / "data" / "canonical_v3" / "chunks_v1" / "parents.parquet",
        )).expanduser().resolve(),
        pg_dsn=os.environ.get("PGVECTOR_DSN"),
        pg_table=os.environ.get("COUNCIL_PG_TABLE", "council_arctic_raw_v1"),
        pg_ef_search=int(os.environ.get("COUNCIL_PG_EF_SEARCH", "200")),
        retrieval_mode=os.environ.get("COUNCIL_RETRIEVAL_MODE", "hybrid"),
        bm25_index=Path(os.environ.get(
            "COUNCIL_BM25_INDEX",
            PROJECT_ROOT / "data" / "bm25" / "arctic_raw_v1" / "bm25.sqlite3",
        )).expanduser().resolve(),
        rrf_k=int(os.environ.get("COUNCIL_RRF_K", "60")),
        dense_weight=float(os.environ.get("COUNCIL_DENSE_WEIGHT", "1.0")),
        bm25_weight=float(os.environ.get("COUNCIL_BM25_WEIGHT", "1.0")),
        device=os.environ.get("COUNCIL_DEVICE", "auto"),
        model_cache=Path(os.environ.get(
            "COUNCIL_MODEL_CACHE", PROJECT_ROOT / "models" / "huggingface"
        )).expanduser().resolve(),
        local_files_only=os.environ.get("COUNCIL_ALLOW_DOWNLOAD", "0") != "1",
    )


@st.cache_resource(show_spinner=False)
def load_followup_tools(
    signature: str, _payloads: list[dict[str, Any]], _service: Any,
) -> FollowupTools:
    """동일한 결과 묶음에서는 외부 조회 캐시와 도구 인스턴스를 재사용한다."""
    del signature
    return FollowupTools(issue_payloads=_payloads, search_service=_service)


@st.cache_data(ttl=5, show_spinner=False)
def load_ollama_status(ollama_url: str, model: str) -> dict[str, Any]:
    return probe_ollama(ollama_url, model, timeout_seconds=2.5)


def direction_label(value: str) -> str:
    return {
        "growth": "논의 증가", "persistence": "논의 지속",
        "attenuation": "관심 약화",
    }.get(value, value or "판단 보류")


def stage_label(value: str) -> str:
    return {
        "first_observation": "최초 관측", "early_followup_30d": "30일 내 후속",
        "expansion_90d": "90일 내 확장", "later_observation": "후기 관측",
    }.get(value, value)


def forecast_interpretation(
    issue_name: str,
    single_forecast: dict[str, Any],
    multi_forecast: dict[str, Any],
) -> tuple[str, str, float | None]:
    """예측 수치를 발표·데모에서 읽기 쉬운 조건부 문장으로 바꾼다."""
    if multi_forecast:
        probability = float(
            multi_forecast.get("discussion_active_probability") or 0
        )
        predicted = multi_forecast.get("predicted_mention_count")
        target = str(multi_forecast.get("target_month") or "다음 달")
        if probability >= 0.70:
            outlook = (
                f"‘{issue_name}’ 관련 사업 집행 상황, 예산 반영 여부와 정책 효과를 "
                "점검하는 후속 논의가 계속될 가능성이 높습니다."
            )
        elif probability >= 0.45:
            outlook = (
                f"‘{issue_name}’ 논의가 전체 지역으로 빠르게 확산되기보다는 기존 관측 의회를 중심으로 "
                "후속 질의와 집행 점검 논의가 이어질 가능성이 있습니다."
            )
        else:
            outlook = (
                f"‘{issue_name}’의 새로운 지역 확산보다는 기존 안건의 정리와 사업 점검 중심으로 "
                "논의 강도가 낮아질 가능성이 있습니다."
            )
        headline = (
            f"{target}에도 논의가 이어질 확률 {probability:.1%}"
            + (f" · 예상 언급 {int(predicted)}건" if predicted is not None else "")
        )
        return headline, outlook, probability

    if single_forecast.get("status") == "provisional_baseline":
        direction = direction_label(str(single_forecast.get("direction") or ""))
        predicted = single_forecast.get("predicted_mention_count")
        target = str(single_forecast.get("target_month") or "다음 달")
        direction_text = {
            "논의 증가": "유사 정책 제안과 후속 점검 논의가 늘어날 가능성이 있습니다.",
            "논의 지속": "기존 관측 의회를 중심으로 관련 논의가 이어질 가능성이 있습니다.",
            "관심 약화": "새로운 확산보다는 기존 논의의 정리·점검이 중심이 될 가능성이 있습니다.",
        }.get(direction, "현재 근거만으로 논의 방향을 단정하기 어렵습니다.")
        headline = f"{target} 조건부 전망: {direction}"
        if predicted is not None:
            headline += f" · 예상 언급 {int(predicted)}건"
        return headline, direction_text, None

    return (
        "향후 논의 방향을 계산할 근거가 부족합니다",
        str(single_forecast.get("reason") or "최소 관측 기간 또는 근거 수가 부족합니다."),
        None,
    )


def build_forecast_marker(
    payload: dict[str, Any], source_nodes: list[dict[str, Any]], visible_count: int,
) -> dict[str, Any] | None:
    """마지막 관측 의회에 수치 예측의 기준점을 표시한다.

    현재 모델은 '다음 도시'를 예측하지 않으므로 해당 노드를
    신규 도시 예측으로 해석하지 않도록 명시적 설명을 포함한다.
    """
    if not source_nodes or visible_count < len(source_nodes):
        return None
    latest = source_nodes[-1]
    issue = payload.get("issue") or {}
    single = payload.get("forecast") or {}
    multi = payload.get("multi_issue_forecast") or {}
    headline, outlook, probability = forecast_interpretation(
        str(issue.get("issue_name") or "해당 이슈"), single, multi,
    )
    evaluation = multi.get("required_evaluation") or {}
    rmse = float(evaluation.get("mention_count_rmse") or 0.0)
    return {
        "marker_type": "forecast_anchor",
        "position": [latest["longitude"], latest["latitude"]],
        "council": latest.get("council"),
        "order": "NEXT",
        "label": "NEXT",
        "first_observation_date": multi.get("target_month") or single.get("target_month") or "다음 달",
        "mention_count": multi.get("predicted_mention_count") or single.get("predicted_mention_count") or 0,
        "stage_label": "다음 달 조건부 전망 · 클릭하여 평가 지표 보기",
        "forecast_headline": headline,
        "forecast_outlook": outlook,
        "discussion_active_probability": probability,
        "predicted_mention_count": multi.get("predicted_mention_count") or single.get("predicted_mention_count"),
        "target_month": multi.get("target_month") or single.get("target_month"),
        "discussion_active_f1": float(evaluation.get("discussion_active_f1") or 0.0),
        "mention_count_mae": float(evaluation.get("mention_count_mae") or 0.0),
        "mention_count_mse": rmse ** 2,
        "mention_count_rmse": rmse,
        "directional_accuracy": float(evaluation.get("directional_accuracy") or 0.0),
        "note": "다음 도시 예측이 아니라 마지막 관측 의회에 예측 정보를 연결한 기준 마커",
    }


def issue_map(payload: dict[str, Any], visible_count: int | None = None) -> pdk.Deck:
    """관측 순서에 따라 노드와 연결선을 점진적으로 보여준다."""
    source_nodes = sorted(
        list((payload.get("map") or {}).get("nodes") or []),
        key=lambda row: int(row.get("order") or 0),
    )
    if visible_count is None:
        visible_count = len(source_nodes)
    visible_count = max(1, min(int(visible_count), len(source_nodes)))

    background_nodes = []
    visible_nodes = []
    for index, row in enumerate(source_nodes):
        node = dict(row)
        node["position"] = [row["longitude"], row["latitude"]]
        node["label"] = str(row.get("order") or index + 1)
        node["stage_label"] = stage_label(str(row.get("stage") or ""))
        node["radius"] = min(
            45000,
            15000 + max(int(row.get("mention_count") or 1), 1) ** 0.5 * 2500,
        )
        background_nodes.append(node)
        if index >= visible_count:
            continue
        if index == visible_count - 1:
            node["fill_color"] = [222, 65, 55, 248]
            node["line_color"] = [255, 238, 218, 255]
            node["radius"] *= 1.28
        elif index == 0:
            node["fill_color"] = [229, 150, 34, 242]
            node["line_color"] = [255, 246, 214, 255]
        else:
            node["fill_color"] = [44, 137, 164, 225]
            node["line_color"] = [244, 251, 244, 255]
        visible_nodes.append(node)

    edges = list((payload.get("map") or {}).get("edges") or [])
    visible_edges = edges[:max(visible_count - 1, 0)]
    latest_node = visible_nodes[-1] if visible_nodes else None
    forecast_node = build_forecast_marker(payload, source_nodes, visible_count)
    background = pdk.Layer(
        "ScatterplotLayer", data=background_nodes, get_position="position",
        get_radius=9000, get_fill_color=[125, 132, 119, 55],
        get_line_color=[125, 132, 119, 95], line_width_min_pixels=1,
        stroked=True, pickable=False, id="background-nodes",
    )
    arcs = pdk.Layer(
        "ArcLayer", data=visible_edges,
        get_source_position="source", get_target_position="target",
        get_source_color=[105, 112, 79, 175],
        get_target_color=[222, 65, 55, 235],
        get_width=4, great_circle=True, pickable=True, id="timeline-arcs",
    )
    points = pdk.Layer(
        "ScatterplotLayer", data=visible_nodes, get_position="position",
        get_radius="radius", get_fill_color="fill_color",
        get_line_color="line_color", line_width_min_pixels=3,
        stroked=True, pickable=True, id="observed-nodes",
    )
    latest_ring = pdk.Layer(
        "ScatterplotLayer", data=[latest_node] if latest_node else [],
        get_position="position", get_radius="radius * 1.42",
        filled=False, stroked=True, get_line_color=[222, 65, 55, 225],
        line_width_min_pixels=5, pickable=False, id="latest-ring",
    )
    labels = pdk.Layer(
        "TextLayer", data=visible_nodes, get_position="position", get_text="label",
        get_color=[255, 255, 255, 255], get_size=17,
        get_alignment_baseline="center", get_text_anchor="middle",
        id="timeline-labels",
    )
    forecast_marker = pdk.Layer(
        "ScatterplotLayer", data=[forecast_node] if forecast_node else [],
        get_position="position", get_radius=61000,
        get_fill_color=[224, 58, 73, 80], get_line_color=[224, 58, 73, 255],
        line_width_min_pixels=6, stroked=True, filled=True,
        pickable=True, auto_highlight=True, id="forecast-marker",
    )
    forecast_label = pdk.Layer(
        "TextLayer", data=[forecast_node] if forecast_node else [],
        get_position="position", get_text="label", get_color=[151, 38, 48, 255],
        get_size=18, get_pixel_offset=[0, -43], get_text_anchor="middle",
        get_alignment_baseline="center", id="forecast-label",
    )
    center = (payload.get("map") or {}).get("center") or {
        "latitude": 36.2, "longitude": 127.8,
    }
    return pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=pdk.ViewState(
            latitude=center["latitude"], longitude=center["longitude"],
            zoom=(payload.get("map") or {}).get("zoom", 6.0), pitch=22,
        ),
        layers=[background, arcs, latest_ring, points, labels, forecast_marker, forecast_label],
        tooltip={"html": (
            "<b>{order}. {council}</b><br/>{first_observation_date}<br/>"
            "근거/예상 {mention_count}건<br/>{stage_label}"
        ), "style": {"backgroundColor": "#434936", "color": "#fffaf0"}},
    )


def render_forecast_detail(payload: dict[str, Any]) -> None:
    """예측 전망, 입력 신호, 시간순 검증 지표를 한 패널에 표시한다."""
    issue = payload.get("issue") or {}
    single = payload.get("forecast") or {}
    multi = payload.get("multi_issue_forecast") or {}
    evaluation = multi.get("required_evaluation") or {}
    rolling = multi.get("rolling_backtest") or {}
    baseline = evaluation.get("last_month_persistence_comparison") or {}
    title, outlook, probability = forecast_interpretation(
        str(issue.get("issue_name") or "해당 이슈"), single, multi,
    )
    monthly = list(payload.get("monthly_series") or [])
    latest = monthly[-1] if monthly else {}
    predicted_count = multi.get("predicted_mention_count")
    if predicted_count is None:
        predicted_count = single.get("predicted_mention_count")
    evidence_signals = [
        f"최근 관측월 언급 {int(latest.get('mention_count') or 0)}건",
        f"3개월 평균 {float(latest.get('mention_mean_3m') or 0):.1f}건",
        f"활성 의회 {int(latest.get('active_council_count') or 0)}개",
        f"누적 관측 의회 {int(latest.get('cumulative_council_count') or 0)}개",
    ]
    st.markdown(
        "<div class='forecast-spotlight'>"
        "<div class='forecast-kicker'>CLICKED NEXT MARKER · 향후 논의 전망</div>"
        f"<div class='forecast-title'>{escape(title)}</div>"
        f"<div class='forecast-copy'>{escape(outlook)}</div>"
        f"<div class='forecast-note'>근거 신호: {escape(' · '.join(evidence_signals))}</div>"
        "<div class='forecast-note'>주의: NEXT 마커는 다음 도시를 예측한 것이 아니라 "
        "마지막 관측 의회에 다음 달 수치 예측을 연결한 기준 마커입니다.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    if probability is not None:
        probability_columns = st.columns(2)
        probability_columns[0].metric("다음 달 논의 지속 확률", f"{float(probability):.1%}")
        probability_columns[1].metric(
            "다음 달 예상 언급",
            f"{int(predicted_count)}건" if predicted_count is not None else "근거 부족",
        )

    f1 = float(evaluation.get("discussion_active_f1") or 0.0)
    mae = float(evaluation.get("mention_count_mae") or 0.0)
    rmse = float(evaluation.get("mention_count_rmse") or 0.0)
    direction = float(evaluation.get("directional_accuracy") or 0.0)
    first_metrics = st.columns(3)
    first_metrics[0].metric("논의 지속 F1", f"{f1:.3f}")
    first_metrics[1].metric("언급량 MAE", f"{mae:.3f}")
    first_metrics[2].metric("언급량 MSE", f"{rmse ** 2:.3f}")
    second_metrics = st.columns(3)
    second_metrics[0].metric("언급량 RMSE", f"{rmse:.3f}")
    second_metrics[1].metric("방향성 정확도", f"{direction:.1%}")
    second_metrics[2].metric("Rolling folds", int(rolling.get("fold_count") or 0))
    st.caption(
        "단순 전월 유지 기준선 대비 · "
        f"F1 Δ {float(baseline.get('classification_f1_delta') or 0):+.3f} · "
        f"MAE 개선 {float(baseline.get('count_mae_improvement_percent') or 0):+.1f}% · "
        f"RMSE 개선 {float(baseline.get('count_rmse_improvement_percent') or 0):+.1f}%"
    )


def selected_layer_objects(event: Any, layer_id: str) -> list[dict[str, Any]]:
    if event is None:
        return []
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    if not selection:
        return []
    objects = getattr(selection, "objects", None)
    if objects is None and isinstance(selection, dict):
        objects = selection.get("objects")
    return list((objects or {}).get(layer_id) or [])


def render_timeline_panel(
    payload: dict[str, Any], visible_count: int, interactive: bool = True,
) -> None:
    """타임라인 단계의 지도와 소형 관측 표를 같이 표시한다."""
    left, right = st.columns([3.35, 1.15], gap="medium")
    selected_forecast: list[dict[str, Any]] = []
    issue_identity = str((payload.get("issue") or {}).get("issue_id") or "dynamic")
    with left:
        chart = issue_map(payload, visible_count=visible_count)
        if interactive:
            event = st.pydeck_chart(
                chart, use_container_width=True, height=610,
                on_select="rerun", selection_mode="single-object",
                key=f"timeline_map_{issue_identity}",
            )
            selected_forecast = selected_layer_objects(event, "forecast-marker")
        else:
            st.pydeck_chart(chart, use_container_width=True, height=610)
    with right:
        st.markdown("#### 관측 순서")
        st.caption(f"{visible_count}단계까지 시간순 근거")
        table = pd.DataFrame(payload.get("propagation") or [])
        if table.empty:
            st.info("표시할 관측 순서가 없습니다.")
            return
        table = table.sort_values("observation_order").head(visible_count).copy()
        table["순서"] = table["observation_order"].map(lambda value: f"{int(value):02d}")
        table["의회"] = table["council"].map(
            lambda value: str(value).replace("특별자치시의회", "").replace("특별자치도의회", "").replace("광역시의회", "").replace("특별시의회", "").replace("도의회", "")
        )
        table["관측일"] = table["first_observation_date"].astype(str)
        table["시차"] = table["lag_days_from_first"].map(lambda value: f"+{int(value)}일")
        st.dataframe(
            table[["순서", "의회", "관측일", "시차"]],
            hide_index=True,
            use_container_width=True,
            height=520,
        )
        total_nodes = len((payload.get("map") or {}).get("nodes") or [])
        panel_key = f"forecast_panel_{issue_identity}"
        if interactive and visible_count >= total_nodes:
            st.caption("지도의 선홍색 NEXT 마커를 클릭하세요.")
            if st.button(
                "🎯 NEXT 전망 보기", key=f"forecast_fallback_{issue_identity}",
                use_container_width=True,
            ):
                st.session_state[panel_key] = True
    if selected_forecast:
        st.session_state[f"forecast_panel_{issue_identity}"] = True
    if interactive and st.session_state.get(f"forecast_panel_{issue_identity}"):
        render_forecast_detail(payload)


def render_evidence(rows: list[dict[str, Any]], expanded: int = 2) -> None:
    for index, row in enumerate(rows, start=1):
        metadata = row.get("metadata") or row
        title = (
            f"{index}. {metadata.get('council', '미상')} · "
            f"{metadata.get('meeting_date', '날짜 미상')} · "
            f"{metadata.get('meeting_name', '회의록명 미상')}"
        )
        with st.expander(title, expanded=index <= expanded):
            st.markdown(
                f"<div class='evidence-meta'><b>위원회</b> "
                f"{escape(str(metadata.get('committee') or '미상'))} · "
                f"<b>안건</b> {escape(str(metadata.get('agenda_title') or '미상'))}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='evidence-card'>{escape(str(row.get('text') or ''))}</div>",
                unsafe_allow_html=True,
            )
            if row.get("parent_context"):
                with st.popover("앞뒤 문맥"):
                    st.write(row["parent_context"])


FOLLOWUP_STAGE_LABELS = {
    "논의_후_광역조례화": "논의 → 광역 조례화",
    "광역조례_선행_후_논의": "광역 조례 선행 → 논의",
    "논의_진행_조례_미확인": "논의 중 · 조례 미확인",
    "회의록_미관측_광역조례_존재": "회의록 미관측 · 광역 조례 있음",
    "기초조례_선행_광역논의_미관측": "기초 조례 선행 · 광역 논의 없음",
}
NEXT_STEP_LABELS = {
    "광역_조례화": "광역 조례화",
    "광역의회_논의_진입": "광역의회 논의 진입",
    "관망": "관망",
}


def render_external_followup(followup: dict[str, Any] | None) -> None:
    """외부 공공데이터로 보강한 후속 흐름을 참고 정보로 표시한다."""
    st.subheader("타 의회 후속 흐름 · 외부 공공데이터 참고")
    if not followup:
        st.info(
            "외부 공공데이터 결과가 아직 결합되지 않았습니다. API 키 또는 사전 캐시를 준비한 뒤 "
            "`src/32a_external_followup_flow.py`와 `src/34_build_final_web_bundle.py`를 실행하세요."
        )
        return

    view = followup.get("outlook") or {}
    counts = followup.get("external_counts") or {}
    mode = str(followup.get("mode") or "backtest")
    as_of = followup.get("as_of") or followup.get("analysis_cutoff")
    st.caption(
        f"{'조회일 기준 참고' if mode == 'live' else '대회 기준일 백테스트'} · "
        f"외부 데이터 {as_of} · 회의록 "
        f"{followup.get('minutes_cutoff') or followup.get('analysis_cutoff')}까지 · "
        f"검색어 {', '.join(followup.get('keywords') or []) or '없음'} · "
        "규칙 기반 참고 점수(확률·인과관계 아님) · 예측 모델 입력 제외 · 회의록 원문 외부 전송 없음"
    )
    if mode == "live":
        st.warning("LIVE 결과는 대회 공식 예측과 분리된 사후 탐색 정보입니다.")
    if not followup.get("data_sources"):
        st.warning("API 인증값이 없어 외부 자료 없이 회의록 관측만 반영했습니다.")
    for error in followup.get("errors") or []:
        st.warning(
            f"외부 조회 오류 ({error.get('source')} · {error.get('keyword')}): "
            f"{error.get('error')}"
        )

    top = view.get("top_candidates") or []
    cols = st.columns(4)
    cols[0].metric("광역 조례화 의회", f"{len(view.get('already_institutionalized') or [])}곳")
    cols[1].metric("후속 1순위", top[0]["council"] if top else "근거 부족")
    lag = view.get("median_lag_days_minutes_to_ordinance")
    cols[2].metric("논의→조례화 중앙", f"{int(lag)}일" if lag is not None else "근거 부족")
    bills = counts.get("national_bills_last_12m", view.get("national_bills_in_window"))
    cols[3].metric("국회 의안(12개월)", f"{int(bills or 0)}건")

    after = view.get("followups_after_minutes_cutoff") or []
    if after:
        st.success(f"회의록 기준일 이후 실제 광역 조례화: {', '.join(after)}")
    for row in (followup.get("regional_need") or {}).get("indicators") or []:
        st.caption(
            f"지역 정책 수요 지표: {row.get('name')} ({row.get('period')}) — "
            f"{row.get('attribution')}"
        )

    flow_col, outlook_col = st.columns([1.3, 1], gap="large")
    with flow_col:
        st.markdown("**의회별 후속 단계**")
        flows = pd.DataFrame([
            row for row in followup.get("followup_flow") or []
            if row.get("stage") != "미관측"
        ])
        if not flows.empty:
            flows["단계"] = flows["stage"].map(
                lambda value: FOLLOWUP_STAGE_LABELS.get(value, value)
            )
            st.dataframe(
                flows[[
                    "council", "단계", "minutes_first_observation",
                    "first_regional_ordinance", "lag_days_minutes_to_ordinance",
                    "basic_ordinance_count",
                ]].rename(columns={
                    "council": "의회", "minutes_first_observation": "회의록 첫 논의",
                    "first_regional_ordinance": "광역 조례 공포",
                    "lag_days_minutes_to_ordinance": "소요일",
                    "basic_ordinance_count": "기초 조례",
                }),
                hide_index=True, use_container_width=True,
            )
        else:
            st.info("회의록·조례 모두에서 관측된 의회가 없습니다.")
    with outlook_col:
        st.markdown("**후속 조례화·논의 진입 후보**")
        if top:
            st.dataframe(
                pd.DataFrame([{
                    "순위": row["rank"], "의회": row["council"],
                    "예상 다음 단계": NEXT_STEP_LABELS.get(
                        row["expected_next_step"], row["expected_next_step"]
                    ),
                    "점수": row["followup_score"],
                } for row in top]),
                hide_index=True, use_container_width=True,
                column_config={
                    "점수": st.column_config.ProgressColumn(
                        min_value=0.0, max_value=1.0, format="%.2f"
                    )
                },
            )
        check = followup.get("post_cutoff_check") or {}
        if check.get("status") == "complete":
            st.caption(
                f"사후 점검(~{check.get('checked_until')}): 상위 {check.get('top_k')} 중 "
                f"{', '.join(check.get('top_k_hits') or []) or '적중 없음'} · "
                f"precision@k {check.get('precision_at_k')}"
            )
    with st.expander("점수 규칙 · 출처 · 한계"):
        st.json({"scoring_rule": view.get("scoring_rule"), "external_counts": counts})
        for source in followup.get("data_sources") or []:
            st.caption(source.get("attribution"))
        for line in followup.get("limitations") or []:
            st.caption(f"- {line}")


def dynamic_result_path(query: str) -> Path:
    normalized = " ".join(query.split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return PROJECT_ROOT / "outputs" / "dynamic_issues" / f"query_{digest}" / "05_dynamic_result.json"


def run_dynamic_analysis(query: str) -> tuple[dict[str, Any], str]:
    normalized = " ".join(query.split())
    command = [
        sys.executable, "-u", "src/33_analyze_free_query.py",
        "--query", normalized,
        "--device", os.environ.get("COUNCIL_DEVICE", "auto"),
        "--reuse",
    ]
    if os.environ.get("COUNCIL_ALLOW_DOWNLOAD", "0") == "1":
        command.append("--allow-download")
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, check=True, capture_output=True,
        text=True, timeout=1800,
    )
    path = dynamic_result_path(normalized)
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_json(path), completed.stdout


st.set_page_config(
    page_title="의정이슈레이더 | 꺼드럭's", page_icon="📡", layout="wide",
    initial_sidebar_state="expanded",
)
if "font_size_mode" not in st.session_state:
    st.session_state["font_size_mode"] = "기본"

font_root_px = {
    "작게": 14,
    "기본": 16,
    "크게": 18,
}.get(str(st.session_state.get("font_size_mode")), 16)

st.markdown("""
<style>
:root{--ivory:#f7f1e6;--paper:#fffaf1;--olive:#4b523b;--khaki:#747b59;
--coral:#eb604f;--ink:#30352b;--line:#ddd4c3}
html{font-size:__FONT_ROOT_PX__px}
.stApp{background:linear-gradient(180deg,#f5efe3 0,#faf7ef 34%,#f7f1e6 100%);color:var(--ink)}
.block-container{max-width:1580px;padding-top:1rem;padding-bottom:4rem}
[data-testid="stSidebar"]{background:#eae4d6;border-right:1px solid #d5cebc}
.brand-team{font-size:.73rem;font-weight:900;letter-spacing:.15em;color:#9b5047}
.brand-title{font-size:1.55rem;font-weight:900;letter-spacing:-.06em;color:#414733}
.hero{position:relative;overflow:hidden;border-radius:26px;padding:1.65rem 2rem;margin:.2rem 0 1rem;background:radial-gradient(circle at 86% 0,rgba(242,103,87,.36),transparent 24%),linear-gradient(125deg,#424936,#6d7453);color:#fff9ed;box-shadow:0 20px 46px rgba(65,72,51,.16)}
.hero-content{position:relative;z-index:2;max-width:78%}
.hero h1{color:#fff9ed;margin:.1rem 0 .35rem;font-size:2.35rem;letter-spacing:-.065em}.hero p{margin:0;color:#eee9d9}
.radar-art{position:absolute;right:2rem;top:50%;width:128px;height:128px;transform:translateY(-50%);border:1px solid rgba(255,245,220,.42);border-radius:50%;opacity:.88;background:repeating-radial-gradient(circle,transparent 0 19px,rgba(255,245,220,.22) 20px 21px),conic-gradient(from 25deg,rgba(239,91,76,.72),transparent 18%,transparent 100%)}
.radar-art:before,.radar-art:after{content:"";position:absolute;background:rgba(255,245,220,.32)}
.radar-art:before{left:50%;top:0;width:1px;height:100%}.radar-art:after{left:0;top:50%;width:100%;height:1px}
.radar-dot{position:absolute;left:71%;top:28%;width:10px;height:10px;background:#ffdfcf;border:2px solid #df4c42;border-radius:50%;box-shadow:0 0 0 7px rgba(237,87,75,.22),0 0 18px rgba(255,221,205,.8)}
.badge{display:inline-block;border-radius:999px;padding:.28rem .7rem;font-size:.78rem;font-weight:800;background:#f5d8d2;color:#923f38;margin-right:.35rem}
.evidence-card{background:#fffdf8;border:1px solid #ded6c5;border-left:4px solid var(--coral);border-radius:12px;padding:1rem;line-height:1.72}
.evidence-meta{color:#6f7262;font-size:.86rem;margin:.1rem 0 .55rem}
.forecast{background:linear-gradient(135deg,#fff9ef,#f7d8d1);border:1px solid #efb8ae;border-radius:19px;padding:1.1rem 1.25rem}
.forecast-spotlight{background:linear-gradient(120deg,#fffaf0 0%,#fde7e1 72%,#f7cbc2 100%);border:2px solid #eb604f;border-radius:22px;padding:1.35rem 1.55rem;margin:.7rem 0 1.25rem;box-shadow:0 14px 30px rgba(171,74,61,.12)}
.forecast-kicker{font-size:.78rem;font-weight:900;letter-spacing:.12em;color:#a34137;margin-bottom:.35rem}
.forecast-title{font-size:1.45rem;font-weight:900;color:#3f4534;letter-spacing:-.04em;margin-bottom:.42rem}
.forecast-copy{font-size:1.02rem;line-height:1.65;color:#4f5146}
.forecast-note{font-size:.79rem;color:#796f64;margin-top:.55rem}
div[data-testid="stMetric"]{background:#fffaf2;border:1px solid #ded6c5;border-radius:15px;padding:.8rem 1rem;box-shadow:0 7px 18px rgba(73,78,58,.05)}
.stButton>button{border-radius:999px;border:1px solid #cfc6b4;font-weight:800;transition:all .18s ease;box-shadow:0 4px 10px rgba(67,73,54,.05)}
.stButton>button:hover{border-color:var(--coral);color:#a33f36;transform:translateY(-1px);box-shadow:0 7px 16px rgba(167,70,58,.12)}
.map-focus-card{background:linear-gradient(120deg,#424936,#697052);color:#fff9ed;border-radius:17px;padding:.9rem 1.1rem;margin:.65rem 0 .8rem;box-shadow:0 10px 24px rgba(65,72,51,.14)}
.map-focus-card b{color:#ffd9d1}.map-focus-card span{opacity:.9}
.top10-kicker{font-size:.78rem;font-weight:900;letter-spacing:.11em;color:#a34137;margin-bottom:.25rem}
.top10-copy{color:#676b5b;margin-bottom:.8rem}
.timeline-head{display:flex;align-items:center;justify-content:space-between;background:#f0eadc;border:1px solid #d9cfbd;border-radius:15px;padding:.72rem 1rem;margin:.25rem 0 .65rem}
.timeline-head b{color:#424936}.timeline-head span{color:#9f433a;font-weight:850}
.chat-launch-note{text-align:right;color:#777565;font-size:.79rem;margin-top:-.25rem}
.st-key-cute_chatbot_launcher{position:fixed;left:1rem;bottom:1rem;z-index:100000;width:18rem;padding:.72rem!important;border:1px solid #d8c8b3;border-radius:22px;background:linear-gradient(135deg,#fff9ee 0%,#f6e5d9 100%);box-shadow:0 14px 34px rgba(53,58,43,.24)}
.st-key-cute_chatbot_launcher:before{content:'●';position:absolute;right:.72rem;top:.55rem;color:#ed5c54;font-size:1.05rem;text-shadow:0 0 0 rgba(237,92,84,.5);animation:botPulse 1.9s ease-out infinite}
.st-key-cute_chatbot_launcher button{min-height:66px!important;border:2px solid #fff5e7!important;border-radius:18px!important;background:linear-gradient(135deg,#596247,#444b37)!important;color:#fffaf0!important;font-size:1.02rem!important;box-shadow:0 8px 20px rgba(67,73,54,.24)!important}
.st-key-cute_chatbot_launcher button:hover{transform:translateY(-2px)!important;border-color:#f2b1a5!important;box-shadow:0 12px 25px rgba(67,73,54,.32)!important}
.st-key-cute_chatbot_launcher button span[data-testid='stIconMaterial'],.st-key-cute_chatbot_launcher button .material-symbols-rounded{font-size:2.25rem!important;color:#ffb5a7!important;margin-right:.45rem!important}
.st-key-cute_chatbot_launcher [data-testid='stCaptionContainer']{text-align:center;color:#786c5d!important;font-size:.74rem!important;margin-top:.15rem!important}
@keyframes botPulse{0%{text-shadow:0 0 0 rgba(237,92,84,.6)}70%{text-shadow:0 0 13px rgba(237,92,84,0)}100%{text-shadow:0 0 0 rgba(237,92,84,0)}}
.stProgress>div>div>div>div{background-color:var(--coral)}
h1,h2,h3{color:#404633;letter-spacing:-.04em}
@media(max-width:800px){.radar-art{width:86px;height:86px;right:1rem}.hero-content{max-width:86%}.st-key-cute_chatbot_launcher{left:.55rem;bottom:.55rem;width:15.5rem}}
</style>
""".replace("__FONT_ROOT_PX__", str(font_root_px)), unsafe_allow_html=True)

catalog_path = WEB_DIR / "issue_catalog.json"
if not catalog_path.is_file():
    st.error(f"최종 이슈 카탈로그가 없습니다: {catalog_path}")
    st.code("python -u src/34_build_final_web_bundle.py")
    st.stop()
catalog = load_catalog(str(catalog_path), catalog_path.stat().st_mtime_ns)
entries = list(catalog.get("issues") or [])
if not entries:
    st.error("최종 이슈 카탈로그가 비었습니다")
    st.stop()
payload_paths = [Path(row["payload"]).resolve() for row in entries]
all_payloads = [load_payload(str(path), path.stat().st_mtime_ns) for path in payload_paths]
payload_by_id = {
    str((row.get("issue") or {}).get("issue_id")): row for row in all_payloads
}


def choose_issue(issue_id: str) -> None:
    """대표 이슈 버튼을 선택하면 사이드바와 레이더의 기준을 같이 바꾼다."""
    st.session_state["selected_issue_id"] = issue_id
    st.session_state.pop("radar_payload", None)
    st.session_state.pop("radar_query", None)
    st.session_state.pop("radar_mode", None)


def set_font_size(mode: str) -> None:
    st.session_state["font_size_mode"] = mode


def shift_timeline(timeline_key: str, delta: int, total: int) -> None:
    current = int(st.session_state.get(timeline_key, total))
    st.session_state[timeline_key] = max(1, min(total, current + delta))


def reset_timeline(timeline_key: str) -> None:
    st.session_state[timeline_key] = 1


def render_agent_chat(
    default_payload: dict[str, Any], issue_payloads: list[dict[str, Any]]
) -> None:
    """질문 해석·근거 계산·답변 검증을 포함한 로컬 Agent 챗봇."""
    agent_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    agent_model = os.environ.get("COUNCIL_AGENT_MODEL", "qwen3:8b")
    ollama_status = load_ollama_status(agent_url, agent_model)
    status_left, status_right = st.columns([1, 2])
    if ollama_status["reachable"] and ollama_status["model_available"]:
        status_left.success("로컬 AI 연결됨")
        status_right.caption(f"Ollama · {agent_model} · 회의록 외부 전송 없음")
    elif ollama_status["reachable"]:
        status_left.warning("모델 설치 필요")
        status_right.code(f"ollama pull {agent_model}", language="bash")
    else:
        status_left.error("Ollama 미연결")
        status_right.caption("규칙 해석과 검색 근거 요약 모드로 대체합니다.")

    if "final_agent_messages" not in st.session_state:
        st.session_state["final_agent_messages"] = []
    if not st.session_state["final_agent_messages"]:
        with st.chat_message("assistant"):
            st.markdown(
                "안녕하세요. **의정이슈 AI 챗봇**입니다.  "
                "이슈의 최초 관측 의회, 지역별 흐름, 원문 근거와 향후 논의 전망을 물어보세요."
            )
    for message in st.session_state["final_agent_messages"][-8:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.form("floating_agent_form", clear_on_submit=True):
        question = st.text_area(
            "질문",
            height=90,
            placeholder="예: 저출생 대응 정책은 어느 의회에서 먼저 관측됐어?",
            label_visibility="collapsed",
        )
        ask = st.form_submit_button("전송  ➤", type="primary", use_container_width=True)

    if ask and question.strip():
        dynamic_candidate = st.session_state.get("dynamic_issue_payload")
        try:
            service = load_search_service()
        except Exception as error:
            service = None
            st.warning(f"pgvector 미연결: {error}")

        effective_payloads = list(issue_payloads)
        if dynamic_candidate:
            effective_payloads.append(dynamic_candidate)
        if default_payload and default_payload not in effective_payloads:
            effective_payloads.append(default_payload)
        tools_signature = "|".join(
            str(path.stat().st_mtime_ns) for path in payload_paths
        )
        followup_tools = load_followup_tools(
            f"{tools_signature}|{service is not None}", issue_payloads, service,
        )
        pipeline = QuestionPipeline(
            issue_payloads=effective_payloads,
            tools=followup_tools,
            search_service=service,
        )
        agent = LocalCouncilAgent(
            issue_payload={},
            search_service=service,
            ollama_url=agent_url,
            model=agent_model,
            timeout_seconds=int(os.environ.get("COUNCIL_AGENT_TIMEOUT", "90")),
            pipeline=pipeline,
            parser_model=os.environ.get("COUNCIL_PARSER_MODEL") or None,
            parser_timeout_seconds=int(
                os.environ.get("COUNCIL_PARSER_TIMEOUT", "120")
            ),
        )
        try:
            with st.spinner("질문을 해석하고 근거를 계산한 뒤 답변을 검증하는 중입니다…"):
                response = agent.ask(
                    question,
                    history=st.session_state["final_agent_messages"],
                    previous_slots=st.session_state.get("final_agent_slots"),
                )
        except Exception as error:
            response = {
                "answer": "Agent 처리 중 오류가 발생했습니다.",
                "citations": [],
                "retrieval_mode": "agent_error",
                "generation_mode": "agent_error",
                "warning": f"{type(error).__name__}: {error}",
            }
        routed_payload = response.get("issue_payload") or {}
        route = (response.get("plan") or {}).get("route") or {}
        response["route_name"] = (
            str((routed_payload.get("issue") or {}).get("issue_name"))
            if routed_payload else "자유검색 근거 모드"
        )
        response["route_score"] = float(route.get("score") or 0.0)
        if (response.get("plan") or {}).get("slots"):
            st.session_state["final_agent_slots"] = response["plan"]["slots"]
        st.session_state["final_agent_messages"].extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": response["answer"]},
        ])
        st.session_state["final_agent_response"] = response
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(response["answer"])

    response = st.session_state.get("final_agent_response")
    if response:
        st.caption(
            f"라우팅: {response.get('route_name')} · "
            f"점수 {float(response.get('route_score') or 0):.2f} · "
            f"생성: {response.get('generation_mode', '미상')}"
        )
        if response.get("warning"):
            st.warning(f"Agent 진단: {response['warning']}")
        plan = response.get("plan") or {}
        if plan:
            sent = (
                (response.get("policy") or {}).get("external_public_api_values")
                or []
            )
            st.caption(
                "외부 공공 API로 전송된 값: "
                + (", ".join(sent) if sent else "없음")
                + " · 질문 원문·회의록 원문 전송 없음"
            )
            slots = plan.get("slots") or {}
            verification = plan.get("verification") or {}
            with st.expander("질문 해석 · 실행 단계 · 답변 검증", expanded=False):
                st.markdown(
                    f"- 해석({plan.get('parser')}): "
                    f"**{INTENTS.get(slots.get('intent'), slots.get('intent'))}** · "
                    f"주제 `{slots.get('topic')}` · "
                    f"지역 {', '.join(slots.get('regions') or []) or '전체'} · "
                    f"모드 {slots.get('mode')}\n"
                    f"- 대표 이슈 연결: {response.get('route_name')} "
                    f"({(plan.get('route') or {}).get('method')})\n"
                    f"- KOSIS 지표: {', '.join(plan.get('indicators') or []) or '없음'}\n"
                    f"- 답변 검증: "
                    f"{'통과' if verification.get('passed') else ('미실행' if not verification else '실패')}"
                )
                terms = pd.DataFrame(plan.get("terms") or [])
                steps = pd.DataFrame(plan.get("steps") or [])
                if not terms.empty:
                    st.dataframe(terms, hide_index=True, use_container_width=True)
                if not steps.empty:
                    st.dataframe(steps, hide_index=True, use_container_width=True)
        if response.get("citations"):
            with st.expander("답변의 회의록 인용 근거"):
                render_evidence(response["citations"], expanded=1)

with st.sidebar:
    st.markdown("<div class='brand-team'>TEAM 꺼드럭's</div><div class='brand-title'>의정이슈레이더</div>", unsafe_allow_html=True)
    st.caption("화면 글자 크기")
    font_columns = st.columns(3)
    for font_index, font_mode in enumerate(("작게", "기본", "크게")):
        font_columns[font_index].button(
            font_mode,
            key=f"font_size_{font_mode}",
            type=(
                "primary"
                if st.session_state.get("font_size_mode") == font_mode
                else "secondary"
            ),
            use_container_width=True,
            on_click=set_font_size,
            args=(font_mode,),
        )
    issue_ids = [str(row["issue_id"]) for row in entries]
    if st.session_state.get("selected_issue_id") not in issue_ids:
        st.session_state["selected_issue_id"] = issue_ids[0]
    selected_id = st.selectbox(
        "대표 주제 TOP 10", issue_ids,
        format_func=lambda value: next(
            str(row["issue_name"]) for row in entries if row["issue_id"] == value
        ),
        key="selected_issue_id",
    )
    st.caption("2024~2025 광역 지방의회 회의록")
    st.markdown("<span class='badge'>LOCAL ONLY</span><span class='badge'>REPRESENTATIVE TOP 10</span>", unsafe_allow_html=True)
    st.divider()
    st.caption("Arctic Dense + BM25 · RRF · BGE reranker · 검수 군집 · 시간순 예측 · Qwen Agent")

payload = payload_by_id[selected_id]
issue = payload["issue"]
summary = payload["summary"]
single_forecast = payload["forecast"]
multi_forecast = payload["multi_issue_forecast"]

if hasattr(st, "dialog"):
    @st.dialog("💬 의정이슈 AI 챗봇", width="large")
    def open_agent_dialog() -> None:
        render_agent_chat(payload, all_payloads)
else:
    def open_agent_dialog() -> None:
        st.session_state["agent_panel_open"] = True

with st.sidebar:
    with st.container(key="cute_chatbot_launcher"):
        if st.button(
            "의정봇에게 물어보기", key="cute_chatbot_button",
            icon=":material/smart_toy:", type="primary",
            use_container_width=True,
        ):
            open_agent_dialog()
        st.caption("회의록 원문과 예측 근거를 바탕으로 답변합니다")

st.markdown(f"""
<div class='hero'><span class='badge'>2024–2025 DATA-DRIVEN TOP 10</span>
<div class='hero-content'><h1>대표 정책 이슈 TOP 10</h1>
<p><b>꺼드럭's 의정이슈레이더</b> · {escape(str(issue['issue_name']))}의 최초 관측, 지역 흐름, 원문 근거와 향후 전망</p></div>
<div class='radar-art' aria-hidden='true'><span class='radar-dot'></span></div></div>
""", unsafe_allow_html=True)

st.caption("지도의 흐름은 시간적 관측 순서이며 인과적 정책 전파를 뜻하지 않습니다.")
if not hasattr(st, "dialog") and st.session_state.get("agent_panel_open"):
    with st.expander("💬 의정이슈 AI 챗봇", expanded=True):
        render_agent_chat(payload, all_payloads)

tabs = st.tabs([
    "🏆 대표 주제 TOP 10", "📍 이슈 레이더", "🔎 원문 탐색",
])

with tabs[0]:
    st.subheader("데이터로 선정한 대표 정책 이슈 TOP 10")
    st.markdown(
        "<div class='top10-kicker'>SELECT A REPRESENTATIVE ISSUE</div>"
        "<div class='top10-copy'>버튼을 누르면 해당 이슈가 레이더·원문 탐색·Agent의 기준 주제로 설정됩니다.</div>",
        unsafe_allow_html=True,
    )
    for issue_start in range(0, len(entries), 2):
        issue_columns = st.columns(2)
        for issue_offset, issue_row in enumerate(entries[issue_start:issue_start + 2]):
            issue_number = issue_start + issue_offset + 1
            issue_value = str(issue_row["issue_id"])
            issue_columns[issue_offset].button(
                f"TOP {issue_number:02d}  {issue_row['issue_name']}",
                key=f"top10_issue_{issue_value}",
                type="primary" if issue_value == selected_id else "secondary",
                use_container_width=True,
                on_click=choose_issue,
                args=(issue_value,),
            )
    overview = pd.DataFrame(entries)
    overview["논의 지속 확률"] = overview["discussion_active_probability"].map(
        lambda value: f"{float(value):.1%}"
    )
    overview["예상 언급"] = overview["predicted_mention_count"].map(lambda value: f"{int(value)}건")
    overview["최초 의회"] = overview["first_observation_councils"].map(lambda value: " · ".join(value))
    st.dataframe(
        overview[["issue_name", "first_observation_date", "최초 의회", "council_count", "evidence_unit_count", "논의 지속 확률", "예상 언급"]].rename(columns={
            "issue_name": "이슈", "first_observation_date": "최초 관측",
            "council_count": "관측 의회", "evidence_unit_count": "근거 수",
        }), hide_index=True, use_container_width=True,
    )
    chart = pd.DataFrame({
        "이슈": [row["issue_name"] for row in entries],
        "예상 언급": [row["predicted_mention_count"] for row in entries],
    }).set_index("이슈")
    st.bar_chart(chart, color=["#eb604f"], height=340)

with tabs[1]:
    st.subheader("정책어로 바로 만드는 지역 이슈 레이더")
    st.caption(
        "고정 10대 이슈는 즉시 연결하고, 그 밖의 검색어는 전체 901,358개 청크에서 "
        "검색·재순위·군집화하여 검수 전 동적 레이더를 생성합니다."
    )
    with st.form("radar_search_form"):
        radar_query = st.text_input(
            "레이더 검색",
            placeholder="예: 자전거 안전 정책, 전세사기 피해 지원, 전기차 화재",
        )
        radar_search = st.form_submit_button("이슈 레이더 생성", type="primary")
    if radar_search and radar_query.strip():
        routed_payload, routed_score = route_fixed_issue(radar_query, all_payloads)
        if routed_payload is not None:
            st.session_state["radar_payload"] = routed_payload
            st.session_state["radar_query"] = radar_query.strip()
            st.session_state["radar_mode"] = "fixed"
            st.success(
                f"검수 완료 이슈로 연결했습니다. 라우팅 점수 {routed_score:.2f}"
            )
        else:
            try:
                with st.spinner(
                    "90만 청크에서 관련 발언을 찾고 지역 레이더를 만드는 중입니다…"
                ):
                    generated, radar_log = run_dynamic_analysis(radar_query)
                st.session_state["radar_payload"] = generated
                st.session_state["radar_query"] = radar_query.strip()
                st.session_state["radar_mode"] = "dynamic"
                st.session_state["radar_log"] = radar_log
                # 같은 동적 결과를 Agent에서도 바로 재사용한다.
                st.session_state["dynamic_issue_payload"] = generated
                st.session_state["dynamic_issue_log"] = radar_log
                st.success("자유검색 이슈 레이더가 완성되었습니다.")
            except subprocess.TimeoutExpired:
                st.error("레이더 생성이 30분 제한을 초과했습니다.")
            except subprocess.CalledProcessError as error:
                detail = (error.stderr or error.stdout or str(error))[-4000:]
                st.error("레이더 생성에 실패했습니다.")
                st.code(detail)
            except Exception as error:
                st.error(f"레이더 생성 실패: {type(error).__name__}: {error}")

    radar_payload = st.session_state.get("radar_payload") or payload
    radar_mode = st.session_state.get("radar_mode") or "fixed"
    radar_issue = radar_payload.get("issue") or {}
    radar_summary = radar_payload.get("summary") or {}
    radar_single_forecast = radar_payload.get("forecast") or {}
    radar_multi_forecast = radar_payload.get("multi_issue_forecast") or {}

    header_left, header_right = st.columns([4, 1])
    header_left.markdown(f"### {radar_issue.get('issue_name', '이슈 레이더')}")
    if radar_mode == "dynamic":
        header_left.warning("검수 전 자유검색 결과입니다. 고정 issue_id에는 반영되지 않습니다.")
    if header_right.button("고정 이슈로 초기화", use_container_width=True):
        for key in ("radar_payload", "radar_query", "radar_mode", "radar_log"):
            st.session_state.pop(key, None)
        radar_payload = payload
        radar_mode = "fixed"
        radar_issue = radar_payload.get("issue") or {}
        radar_summary = radar_payload.get("summary") or {}
        radar_single_forecast = radar_payload.get("forecast") or {}
        radar_multi_forecast = radar_payload.get("multi_issue_forecast") or {}

    predicted_mentions = (
        radar_multi_forecast.get("predicted_mention_count")
        if radar_multi_forecast
        else radar_single_forecast.get("predicted_mention_count")
    )
    metrics = st.columns(5)
    metrics[0].metric("최초 관측", radar_summary.get("first_observation_date") or "근거 부족")
    metrics[1].metric(
        "최초 의회",
        " · ".join(radar_summary.get("first_observation_councils") or []) or "근거 부족",
    )
    metrics[2].metric("관측 의회", f"{int(radar_summary.get('council_count') or 0)} / 17")
    metrics[3].metric("근거 단위", f"{int(radar_summary.get('evidence_unit_count') or 0)}건")
    metrics[4].metric(
        "예상 언급",
        f"{int(predicted_mentions)}건" if predicted_mentions is not None else "근거 부족",
    )

    forecast_title, forecast_copy, forecast_probability = forecast_interpretation(
        str(radar_issue.get("issue_name") or "해당 이슈"),
        radar_single_forecast,
        radar_multi_forecast,
    )
    forecast_note = (
        "검수 완료 10대 이슈의 시간순 다중 이슈 모델 결과입니다."
        if radar_multi_forecast
        else "검수 전 자유검색 결과에 적용한 조건부 기준선이며 확정적 미래 예측이 아닙니다."
    )
    st.markdown(
        "<div class='forecast-spotlight'>"
        "<div class='forecast-kicker'>NEXT DISCUSSION · 향후 논의 전망</div>"
        f"<div class='forecast-title'>{escape(forecast_title)}</div>"
        f"<div class='forecast-copy'>{escape(forecast_copy)}</div>"
        f"<div class='forecast-note'>{escape(forecast_note)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    if forecast_probability is not None:
        st.progress(max(0.0, min(1.0, forecast_probability)))

    st.subheader("대한민국 지역별 최초 관측 타임라인")
    st.caption("재생 버튼을 누르면 최초 관측부터 후속 의회까지 시간순으로 지도에 나타납니다.")
    map_nodes = sorted(
        list((radar_payload.get("map") or {}).get("nodes") or []),
        key=lambda row: int(row.get("order") or 0),
    )
    if map_nodes:
        total_steps = len(map_nodes)
        timeline_identity = str(
            radar_issue.get("issue_id")
            or hashlib.sha256(
                str(radar_issue.get("issue_name") or "dynamic").encode("utf-8")
            ).hexdigest()[:10]
        )
        timeline_key = f"timeline_step_{timeline_identity}"
        if timeline_key not in st.session_state:
            st.session_state[timeline_key] = total_steps

        control_play, control_reset, control_prev, control_next, control_speed = st.columns(
            [1.1, 1, 1, 1, 2.2]
        )
        play_timeline = control_play.button("▶  재생", type="primary", use_container_width=True)
        control_reset.button(
            "↺  처음", use_container_width=True,
            on_click=reset_timeline, args=(timeline_key,),
        )
        control_prev.button(
            "◀  이전", use_container_width=True,
            on_click=shift_timeline, args=(timeline_key, -1, total_steps),
        )
        control_next.button(
            "다음  ▶", use_container_width=True,
            on_click=shift_timeline, args=(timeline_key, 1, total_steps),
        )
        speed_label = control_speed.select_slider(
            "재생 속도",
            options=["빠름", "보통", "천천히"],
            value="보통",
            key=f"timeline_speed_{timeline_identity}",
        )
        delay_seconds = {"빠름": 0.45, "보통": 0.85, "천천히": 1.30}[speed_label]

        if play_timeline:
            playback = st.empty()
            for timeline_step in range(1, total_steps + 1):
                with playback.container():
                    current_node = map_nodes[timeline_step - 1]
                    st.markdown(
                        "<div class='timeline-head'>"
                        f"<b>{timeline_step}/{total_steps}단계 · "
                        f"{escape(str(current_node.get('council')))}</b>"
                        f"<span>{escape(str(current_node.get('first_observation_date')))}</span>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    render_timeline_panel(
                        radar_payload, timeline_step, interactive=False,
                    )
                time.sleep(delay_seconds)
            st.session_state[timeline_key] = total_steps
            playback.empty()

        timeline_step = st.slider(
            "타임라인 단계",
            min_value=1,
            max_value=total_steps,
            key=timeline_key,
            label_visibility="collapsed",
        )
        current_node = map_nodes[timeline_step - 1]
        st.markdown(
            "<div class='timeline-head'>"
            f"<b>{timeline_step}/{total_steps}단계 · "
            f"{escape(str(current_node.get('council')))}</b>"
            f"<span>{escape(str(current_node.get('first_observation_date')))} · "
            f"최초 대비 +{int(current_node.get('lag_days') or 0)}일</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        render_timeline_panel(radar_payload, timeline_step)
    else:
        st.info("지도에 표시할 지역별 관측 근거가 부족합니다.")

    st.subheader("월별 논의 강도")
    monthly = pd.DataFrame(radar_payload.get("monthly_series") or [])
    if not monthly.empty:
        monthly["month"] = pd.to_datetime(monthly["month"])
        st.line_chart(
            monthly.set_index("month")[["mention_count"]],
            color=["#eb604f"], height=300,
        )
    else:
        st.info("월별 시계열 근거가 부족합니다.")

    render_external_followup(radar_payload.get("external_followup"))

    st.subheader("대표 원문 근거")
    render_evidence(radar_payload.get("evidence") or [])

    if radar_mode == "dynamic":
        with st.expander("자유검색 실행 로그"):
            st.code(st.session_state.get("radar_log") or "")

with tabs[2]:
    st.subheader("전체 901,358개 청크에서 원문 검색")
    with st.form("final_search"):
        query = st.text_input("정책·현안", value=str(issue["issue_name"]))
        left, center, right = st.columns(3)
        council = left.text_input("의회 필터", placeholder="예: 경상남도의회")
        year = center.selectbox("연도", ["전체", 2024, 2025])
        top_k = right.slider("결과 수", 3, 20, 10)
        submitted = st.form_submit_button("pgvector 검색", type="primary")
    if submitted:
        try:
            with st.spinner("근거와 부모 문맥을 검색하는 중입니다…"):
                result = load_search_service().search(
                    query=query, top_k=top_k, candidate_k=max(50, top_k * 10),
                    filters=SearchFilters(
                        council=council.strip() or None,
                        year=None if year == "전체" else int(year),
                    ), include_parent=True,
                )
            st.session_state["final_search_result"] = result
        except Exception as error:
            st.error(f"검색 실패: {error}")
    result = st.session_state.get("final_search_result")
    if result:
        a, b, c = st.columns(3)
        a.metric("결과", f"{result['result_count']}건")
        b.metric("DB 검색", f"{result['search_ms']:.1f} ms")
        c.metric("질의 임베딩", f"{result['query_embedding_ms']:.1f} ms")
        render_evidence(result["results"], expanded=3)

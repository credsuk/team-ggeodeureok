#!/usr/bin/env python3
"""꺼드럭's 의정이슈레이더 최종 10개 이슈·검색·Agent 통합 화면."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
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


def issue_map(payload: dict[str, Any]) -> pdk.Deck:
    nodes = []
    for row in payload["map"]["nodes"]:
        node = dict(row)
        node["position"] = [row["longitude"], row["latitude"]]
        node["label"] = str(row["order"])
        node["stage_label"] = stage_label(str(row.get("stage") or ""))
        node["radius"] = min(
            45000, 15000 + max(int(row.get("mention_count") or 1), 1) ** 0.5 * 2500
        )
        nodes.append(node)
    arcs = pdk.Layer(
        "ArcLayer", data=payload["map"]["edges"],
        get_source_position="source", get_target_position="target",
        get_source_color=[105, 112, 79, 165], get_target_color=[238, 94, 76, 235],
        get_width=3, great_circle=True, pickable=True,
    )
    points = pdk.Layer(
        "ScatterplotLayer", data=nodes, get_position="position", get_radius="radius",
        get_fill_color=[233, 91, 76, 225], get_line_color=[255, 249, 238, 255],
        line_width_min_pixels=2, stroked=True, pickable=True,
    )
    labels = pdk.Layer(
        "TextLayer", data=nodes, get_position="position", get_text="label",
        get_color=[255, 255, 255, 255], get_size=17,
        get_alignment_baseline="center", get_text_anchor="middle",
    )
    center = payload["map"]["center"]
    return pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=pdk.ViewState(
            latitude=center["latitude"], longitude=center["longitude"],
            zoom=payload["map"].get("zoom", 6.0), pitch=22,
        ),
        layers=[arcs, points, labels],
        tooltip={"html": (
            "<b>{order}. {council}</b><br/>최초 관측 {first_observation_date}<br/>"
            "+{lag_days}일 · 근거 {mention_count}건<br/>{stage_label}"
        ), "style": {"backgroundColor": "#434936", "color": "#fffaf0"}},
    )


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
st.markdown("""
<style>
:root{--ivory:#f7f1e6;--paper:#fffaf1;--olive:#4b523b;--khaki:#747b59;
--coral:#eb604f;--ink:#30352b;--line:#ddd4c3}
.stApp{background:linear-gradient(180deg,#f5efe3 0,#faf7ef 34%,#f7f1e6 100%);color:var(--ink)}
.block-container{max-width:1580px;padding-top:1rem;padding-bottom:4rem}
[data-testid="stSidebar"]{background:#eae4d6;border-right:1px solid #d5cebc}
.brand-team{font-size:.73rem;font-weight:900;letter-spacing:.15em;color:#9b5047}
.brand-title{font-size:1.55rem;font-weight:900;letter-spacing:-.06em;color:#414733}
.hero{border-radius:26px;padding:1.65rem 2rem;margin:.2rem 0 1rem;background:radial-gradient(circle at 86% 0,rgba(242,103,87,.36),transparent 24%),linear-gradient(125deg,#424936,#6d7453);color:#fff9ed;box-shadow:0 20px 46px rgba(65,72,51,.16)}
.hero h1{color:#fff9ed;margin:.1rem 0 .35rem;font-size:2.35rem;letter-spacing:-.065em}.hero p{margin:0;color:#eee9d9}
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
.stProgress>div>div>div>div{background-color:var(--coral)}
h1,h2,h3{color:#404633;letter-spacing:-.04em}
</style>
""", unsafe_allow_html=True)

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

with st.sidebar:
    st.markdown("<div class='brand-team'>TEAM 꺼드럭's</div><div class='brand-title'>의정이슈레이더</div>", unsafe_allow_html=True)
    selected_id = st.selectbox(
        "분석 이슈", [row["issue_id"] for row in entries],
        format_func=lambda value: next(
            str(row["issue_name"]) for row in entries if row["issue_id"] == value
        ),
    )
    st.caption("2024~2025 광역 지방의회 회의록")
    st.markdown("<span class='badge'>LOCAL ONLY</span><span class='badge'>10 FIXED ISSUES</span>", unsafe_allow_html=True)
    st.divider()
    st.caption("Arctic Dense + BM25 · RRF · BGE reranker · 검수 군집 · 시간순 예측 · Qwen Agent")

payload = payload_by_id[selected_id]
issue = payload["issue"]
summary = payload["summary"]
single_forecast = payload["forecast"]
multi_forecast = payload["multi_issue_forecast"]
model_summary = catalog.get("model_summary") or {}

st.markdown(f"""
<div class='hero'><span class='badge'>COUNCIL ISSUE INTELLIGENCE</span>
<h1>의정이슈레이더</h1>
<p><b>{escape(str(issue['issue_name']))}</b> · 최초 관측부터 지역 흐름, 원문 근거와 다음 달 전망까지</p></div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "🏠 10대 이슈", "📍 이슈 레이더",
    "🔎 원문 탐색", "💬 AI Agent", "🧪 모델 설명",
])

with tabs[0]:
    st.subheader("검수 완료 대표 이슈 10개")
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

    st.subheader("대한민국 지역별 최초 관측 흐름")
    st.caption("연결선은 시간적 관측 순서이며 정책의 인과적 전달 경로가 아닙니다.")
    if (radar_payload.get("map") or {}).get("nodes"):
        st.pydeck_chart(
            issue_map(radar_payload), use_container_width=True, height=620
        )
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

    st.subheader("대표 원문 근거")
    render_evidence(radar_payload.get("evidence") or [])

    st.subheader("의회별 관측 순서")
    st.caption("지도에 표시된 흐름을 날짜와 시차 기준으로 정리한 표입니다.")
    table = pd.DataFrame(radar_payload.get("propagation") or [])
    if not table.empty:
        table["시차"] = table["lag_days_from_first"].map(lambda value: f"+{value}일")
        table["근거"] = table["mention_count"].map(lambda value: f"{value}건")
        st.dataframe(
            table[["observation_order", "council", "first_observation_date", "시차", "근거"]].rename(columns={
                "observation_order": "순서", "council": "의회",
                "first_observation_date": "최초 관측",
            }), hide_index=True, use_container_width=True,
        )
    else:
        st.info("의회별 관측 순서를 만들 근거가 부족합니다.")
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

with tabs[3]:
    st.subheader("고정 이슈 자동 연결 + 자유질문 로컬 Agent")
    st.caption("10개 대표 이슈와 연결되면 검수 결과를 사용하고, 그 밖의 질문은 검색 근거만 사용합니다.")
    agent_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    agent_model = os.environ.get("COUNCIL_AGENT_MODEL", "qwen3:8b")
    ollama_status = load_ollama_status(agent_url, agent_model)
    status_left, status_right = st.columns([1, 2])
    if ollama_status["reachable"] and ollama_status["model_available"]:
        status_left.success("로컬 Agent 연결됨")
        status_right.caption(f"Ollama · {agent_model} · 외부 전송 없음")
    elif ollama_status["reachable"]:
        status_left.warning("모델 설치 필요")
        status_right.code(f"ollama pull {agent_model}", language="bash")
    else:
        status_left.error("Ollama 미연결")
        status_right.caption(
            "로컬 LLM이 꺼져 있어도 검색 근거 요약으로 즉시 대체됩니다. "
            "터미널에서 `ollama serve`를 확인하세요."
        )
    with st.expander("Agent 연결 진단", expanded=False):
        st.json({
            "ollama_url": agent_url,
            "requested_model": agent_model,
            **ollama_status,
        })
    if "final_agent_messages" not in st.session_state:
        st.session_state["final_agent_messages"] = []
    for message in st.session_state["final_agent_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    with st.form("final_agent_form", clear_on_submit=True):
        question = st.text_area("질문", height=90, placeholder="전세사기 지원 논의가 어느 의회에서 먼저 관측됐고 이후 어떻게 이어졌어?")
        ask = st.form_submit_button("Agent에게 질문", type="primary")
    if ask and question.strip():
        routed, route_score = route_fixed_issue(question, all_payloads)
        dynamic_candidate = st.session_state.get("dynamic_issue_payload")
        dynamic_routed = None
        dynamic_score = 0.0
        if routed is None and dynamic_candidate:
            dynamic_routed, dynamic_score = route_fixed_issue(
                question, [dynamic_candidate]
            )
        selected_payload = routed or dynamic_routed or {}
        if dynamic_routed is not None:
            route_score = dynamic_score
        try:
            service = load_search_service()
        except Exception as error:
            service = None
            st.warning(f"pgvector 미연결: {error}")
        agent = LocalCouncilAgent(
            issue_payload=selected_payload, search_service=service,
            ollama_url=agent_url,
            model=agent_model,
            timeout_seconds=int(os.environ.get("COUNCIL_AGENT_TIMEOUT", "90")),
        )
        try:
            with st.spinner("관련 회의록을 검색하고 로컬 Agent가 답변하는 중입니다…"):
                response = agent.ask(
                    question,
                    history=st.session_state["final_agent_messages"],
                )
        except Exception as error:
            response = {
                "answer": (
                    "Agent 처리 중 오류가 발생했습니다. 아래 진단 정보를 확인해 주세요."
                ),
                "citations": [],
                "retrieval_mode": "agent_error",
                "generation_mode": "agent_error",
                "warning": f"{type(error).__name__}: {error}",
            }
        route_name = (
            str((selected_payload.get("issue") or {}).get("issue_name"))
            if selected_payload else "자유검색 근거 모드"
        )
        response["route_name"] = route_name
        response["route_score"] = route_score
        st.session_state["final_agent_messages"].extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": response["answer"]},
        ])
        st.session_state["final_agent_response"] = response
        # st.rerun()을 호출하면 Streamlit 탭이 첫 탭으로 돌아가 답변이
        # 사라진 것처럼 보일 수 있다. 현재 Agent 탭에서 새 메시지를 즉시 렌더링한다.
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(response["answer"])
    response = st.session_state.get("final_agent_response")
    if response:
        st.info(f"라우팅: {response['route_name']} · 점수 {response['route_score']:.2f} · 생성 {response['generation_mode']}")
        if response.get("warning"):
            st.warning(f"Agent 진단: {response['warning']}")
        with st.expander("Agent 인용 근거", expanded=True):
            render_evidence(response["citations"], expanded=1)

with tabs[4]:
    selection = model_summary.get("selection") or {}
    metrics = model_summary.get("metrics") or {}
    required_evaluation = model_summary.get("required_evaluation") or {}
    rolling_backtest = model_summary.get("rolling_backtest") or {}
    baseline_comparison = (
        required_evaluation.get("last_month_persistence_comparison") or {}
    )
    st.subheader("시간 누수를 막은 예측 모델 선택")
    st.markdown(f"""
- 검증 방식: `{model_summary.get('method')}`
- 분류 선택: `{selection.get('classifier')}`
- 언급량 선택: `{selection.get('count_model')}`
- 학습 행: `{model_summary.get('training_rows')}`
- 시간순 검증 예측: `{model_summary.get('validation_predictions')}`
    """)
    if required_evaluation:
        evaluation_columns = st.columns(5)
        evaluation_columns[0].metric(
            "논의 지속 F1",
            f"{float(required_evaluation.get('discussion_active_f1') or 0):.3f}",
        )
        evaluation_columns[1].metric(
            "언급량 MAE",
            f"{float(required_evaluation.get('mention_count_mae') or 0):.3f}",
        )
        evaluation_columns[2].metric(
            "언급량 RMSE",
            f"{float(required_evaluation.get('mention_count_rmse') or 0):.3f}",
        )
        evaluation_columns[3].metric(
            "방향성 정확도",
            f"{float(required_evaluation.get('directional_accuracy') or 0):.1%}",
        )
        evaluation_columns[4].metric(
            "Rolling folds",
            int(rolling_backtest.get("fold_count") or 0),
        )
        st.caption(
            "단순 전월 유지 기준선 대비 · "
            f"F1 Δ {float(baseline_comparison.get('classification_f1_delta') or 0):+.3f} · "
            f"MAE 개선율 {float(baseline_comparison.get('count_mae_improvement_percent') or 0):+.1f}% · "
            f"방향성 정확도 Δ {float(baseline_comparison.get('directional_accuracy_delta') or 0):+.3f}"
        )
    st.json(metrics, expanded=True)
    st.warning("예측은 제공된 2024~2025 회의록에 한정되며, 지역 간 화살표는 인과관계를 뜻하지 않습니다.")
    st.download_button(
        "현재 이슈 결과 JSON", json.dumps(payload, ensure_ascii=False, indent=2),
        file_name=f"{issue['issue_id']}.json", mime="application/json",
    )

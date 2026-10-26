#!/usr/bin/env python3
"""고정 이슈의 최초 관측·지역 흐름·예측·근거를 보여주는 Streamlit 데모."""

from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import pydeck as pdk
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAYLOAD = PROJECT_ROOT / "outputs" / "web" / "birth_rate_issue.json"


def configured_path() -> Path:
    return Path(
        os.environ.get("COUNCIL_ISSUE_PAYLOAD", str(DEFAULT_PAYLOAD))
    ).expanduser().resolve()


@st.cache_data(show_spinner=False)
def load_payload(path_text: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    path = Path(path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "council_issue_web_v1":
        raise ValueError("지원하지 않는 웹 결과 JSON 형식입니다")
    return payload


def percentage(value: Any) -> str:
    return f"{float(value or 0) * 100:.1f}%"


def direction_korean(value: str) -> str:
    return {
        "growth": "논의 증가",
        "persistence": "논의 지속",
        "attenuation": "관심 약화",
    }.get(value, value or "판단 보류")


def stage_korean(value: str) -> str:
    return {
        "first_observation": "최초 관측",
        "early_followup_30d": "30일 내 후속",
        "expansion_90d": "90일 내 확장",
        "later_observation": "후기 관측",
    }.get(value, value)


def make_map(payload: dict[str, Any]) -> pdk.Deck:
    nodes = []
    for row in payload["map"]["nodes"]:
        current = dict(row)
        current["position"] = [row["longitude"], row["latitude"]]
        current["label"] = str(row["order"])
        current["stage_ko"] = stage_korean(str(row.get("stage") or ""))
        current["radius"] = min(
            44000, 15000 + (max(int(row.get("mention_count") or 0), 1) ** 0.5) * 2800
        )
        nodes.append(current)
    edges = payload["map"]["edges"]
    arc_layer = pdk.Layer(
        "ArcLayer", data=edges, get_source_position="source",
        get_target_position="target", get_source_color=[111, 116, 83, 170],
        get_target_color=[231, 93, 84, 220], get_width=3,
        great_circle=True, pickable=True,
    )
    node_layer = pdk.Layer(
        "ScatterplotLayer", data=nodes, get_position="position",
        get_radius="radius", get_fill_color=[225, 98, 88, 220],
        get_line_color=[255, 248, 240, 255], line_width_min_pixels=2,
        stroked=True, pickable=True,
    )
    text_layer = pdk.Layer(
        "TextLayer", data=nodes, get_position="position", get_text="label",
        get_color=[255, 255, 255, 255], get_size=17,
        get_alignment_baseline="center", get_text_anchor="middle",
        font_weight=700, pickable=False,
    )
    center = payload["map"]["center"]
    return pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=pdk.ViewState(
            latitude=center["latitude"], longitude=center["longitude"],
            zoom=payload["map"].get("zoom", 6.0), pitch=25, bearing=0,
        ),
        layers=[arc_layer, node_layer, text_layer],
        tooltip={
            "html": (
                "<b>{order}. {council}</b><br/>"
                "최초 관측: {first_observation_date}<br/>"
                "최초 대비: +{lag_days}일<br/>"
                "근거: {mention_count}건<br/>{stage_ko}"
            ),
            "style": {"backgroundColor": "#424735", "color": "#fffaf2"},
        },
    )


def probability_bar(label: str, value: float, caption: str) -> None:
    st.markdown(f"**{label} · {value * 100:.1f}%**")
    st.progress(max(0.0, min(1.0, value)))
    st.caption(caption)


st.set_page_config(
    page_title="의정이슈레이더 | 꺼드럭's", page_icon="📡", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    :root {--ivory:#f7f3e9; --paper:#fffdf7; --khaki:#707553; --olive:#424735;
      --coral:#e76258; --blush:#f8d9d3; --line:#d9d4c3; --ink:#34372c;}
    .stApp {background: var(--ivory); color: var(--ink);}
    .block-container {max-width: 1540px; padding-top: 1.15rem; padding-bottom: 3rem;}
    [data-testid="stSidebar"] {background:#ede9dc; border-right:1px solid #d8d2c1;}
    [data-testid="stSidebar"] * {color:#414536;}
    .side-brand {padding:.25rem 0 .75rem;}
    .side-team {font-size:.72rem; font-weight:800; letter-spacing:.14em; color:#8a574f;}
    .side-title {font-size:1.42rem; font-weight:850; letter-spacing:-.045em;
      color:#424735; margin-top:.2rem;}
    .hero {position:relative; overflow:hidden; padding:1.85rem 2rem; border-radius:24px;
      color:#fffaf2; background:linear-gradient(125deg,#454a37 0%,#686f50 62%,#7c8062 100%);
      box-shadow:0 20px 45px rgba(66,71,53,.17); margin-bottom:1.15rem;}
    .hero:after {content:""; position:absolute; width:280px; height:280px; border-radius:50%;
      right:-72px; top:-125px; background:rgba(231,98,88,.26);}
    .hero h1 {font-size:2.25rem; margin:0 0 .25rem; letter-spacing:-.055em; line-height:1.15;}
    .hero-issue {display:inline-block; margin:.65rem 0 .55rem; padding:.28rem .7rem;
      border-radius:999px; background:rgba(255,250,242,.13); border:1px solid rgba(255,250,242,.27);
      font-size:.88rem; color:#fff3ec;}
    .hero p {margin:0; color:#f1ecdd; font-size:.98rem;}
    .eyebrow {font-size:.75rem; letter-spacing:.15em; font-weight:800;
      color:#ffd4ce; margin-bottom:.6rem;}
    .radar-dot {display:inline-block; width:8px; height:8px; margin-right:7px;
      border-radius:50%; background:#ff796f; box-shadow:0 0 0 5px rgba(255,121,111,.14);}
    .section-note {color:#797b6a; font-size:.88rem;}
    .forecast-box {padding:1.15rem 1.25rem; border:1px solid #efbbb3;
      background:linear-gradient(145deg,#fffaf5,#f9ded9); border-radius:17px;
      box-shadow:0 9px 22px rgba(155,78,69,.07);}
    .evidence-card {border:1px solid #ded8c7; border-left:4px solid var(--coral);
      padding:1rem 1.15rem; border-radius:12px; background:#fffdf8; margin:.55rem 0;}
    .evidence-meta {color:#686b5b; font-size:.86rem; margin-bottom:.55rem;}
    .evidence-text {color:#36392f; line-height:1.7; font-size:.95rem;}
    .pill {display:inline-block; padding:.22rem .55rem; border-radius:999px;
      background:#e3dfcf; color:#555a42; font-size:.78rem; font-weight:750;}
    div[data-testid="stMetric"] {background:#fffdf7; border:1px solid #ddd7c6;
      padding:.8rem 1rem; border-radius:14px; box-shadow:0 6px 17px rgba(77,81,61,.045);}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {color:#464b37;}
    .stProgress > div > div > div > div {background-color:var(--coral);}
    .stButton > button, .stDownloadButton > button {border-color:#85896c; color:#4a4e3b;
      background:#fffaf2; border-radius:11px;}
    .stButton > button:hover, .stDownloadButton > button:hover {border-color:var(--coral);
      color:#a3423c; background:#fff3ef;}
    h1, h2, h3 {color:#414536; letter-spacing:-.035em;}
    hr {border-color:#ddd7c7 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

payload_path = configured_path()
if not payload_path.is_file():
    st.error(f"지도 웹용 JSON이 없습니다: {payload_path}")
    st.code(
        "python -u src/24_build_web_payload.py --timeline ... --forecast ... "
        "--features ... --output outputs/web/birth_rate_issue.json"
    )
    st.stop()

try:
    payload = load_payload(str(payload_path), payload_path.stat().st_mtime_ns)
except Exception as error:
    st.exception(error)
    st.stop()

issue = payload["issue"]
summary = payload["summary"]
forecast = payload["forecast"]
aliases = [issue["issue_name"], *issue.get("aliases", [])]

with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
          <div class="side-team">TEAM 꺼드럭's</div>
          <div class="side-title">의정이슈레이더</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("지역 의정 이슈의 시작과 흐름을 읽습니다")
    query = st.text_input("이슈 검색", value="출산율")
    normalized = query.replace(" ", "").lower()
    matched = not normalized or any(
        normalized in str(alias).replace(" ", "").lower() for alias in aliases
    )
    st.markdown("---")
    st.markdown("**데이터 범위**")
    st.caption("2024~2025년 광역 지방의회 회의록")
    st.markdown("**검색·분석 환경**")
    st.caption("로컬 임베딩 · pgvector · 외부 LLM API 미사용")
    st.markdown("**고정 이슈 ID**")
    st.code(issue["issue_id"], language=None)
    st.markdown("---")
    with st.expander("분석상 주의사항"):
        for item in payload.get("disclaimers", []):
            st.caption(f"• {item}")
    st.download_button(
        "결과 JSON 다운로드",
        json.dumps(payload, ensure_ascii=False, indent=2),
        file_name=f"{issue['issue_id']}.json", mime="application/json",
        use_container_width=True,
    )

if not matched:
    st.info(
        f"현재 프로토타입에는 ‘{issue['issue_name']}’ 이슈만 등록돼 있습니다. "
        "검색어로 ‘출산율’ 또는 ‘저출생’을 입력해 주세요."
    )
    st.stop()

st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow"><span class="radar-dot"></span>TEAM 꺼드럭's · COUNCIL ISSUE RADAR</div>
      <h1>의정이슈레이더</h1>
      <div class="hero-issue">분석 이슈 · {escape(str(issue['issue_name']))}</div>
      <p>회의록 근거로 최초 관측부터 지역별 후속 논의, 월별 변화와 다음 달 전망까지 추적합니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_columns = st.columns(5)
metric_columns[0].metric("최초 관측", summary["first_observation_date"])
metric_columns[1].metric(
    "최초 의회", " · ".join(summary["first_observation_councils"])
)
metric_columns[2].metric("관측 의회", f"{summary['council_count']}곳 / 17곳")
metric_columns[3].metric("근거 단위", f"{summary['evidence_unit_count']}건")
metric_columns[4].metric("다음 달 전망", direction_korean(forecast["direction"]))

map_column, order_column = st.columns([1.45, 1], gap="large")
with map_column:
    st.subheader("지역별 최초 관측 흐름")
    st.caption(payload["map"]["interpretation"])
    st.pydeck_chart(make_map(payload), use_container_width=True)
with order_column:
    st.subheader("의회 관측 순서")
    st.caption("최초 관측일 기준 · 동일 이슈 근거 청크 집계")
    propagation = pd.DataFrame(payload["propagation"])
    propagation["순서"] = propagation["observation_order"]
    propagation["의회"] = propagation["council"]
    propagation["최초 관측"] = propagation["first_observation_date"]
    propagation["시차"] = propagation["lag_days_from_first"].map(lambda x: f"+{x}일")
    propagation["근거"] = propagation["mention_count"].map(lambda x: f"{x}건")
    st.dataframe(
        propagation[["순서", "의회", "최초 관측", "시차", "근거"]],
        hide_index=True, use_container_width=True, height=420,
    )
    first = payload["propagation"][0]
    last = payload["propagation"][-1]
    st.info(
        f"**{first['council']}**에서 최초 관측된 뒤 "
        f"**{last['council']}**까지 {last['lag_days_from_first']}일의 관측 시차가 있습니다."
    )

st.markdown("---")
trend_column, forecast_column = st.columns([1.55, 1], gap="large")
with trend_column:
    st.subheader("월별 논의 변화")
    monthly = pd.DataFrame(payload["monthly_series"])
    monthly["month"] = pd.to_datetime(monthly["month"])
    monthly = monthly.set_index("month")
    st.markdown("<span class='pill'>언급량</span>", unsafe_allow_html=True)
    st.line_chart(monthly[["mention_count"]], color=["#e76258"], height=245)
    st.markdown(
        "<span class='pill'>활성 의회 · 누적 의회</span>", unsafe_allow_html=True
    )
    st.line_chart(
        monthly[["active_council_count", "cumulative_council_count"]],
        color=["#707553", "#e76258"], height=245,
    )
with forecast_column:
    st.subheader(f"{forecast['target_month']} 전망")
    lower, upper = forecast["prediction_interval_80"]
    st.markdown(
        f"""
        <div class="forecast-box">
          <div class="eyebrow" style="color:#a84e47">EXPLAINABLE FORECAST</div>
          <h2 style="margin:.1rem 0 .35rem">{direction_korean(forecast['direction'])}</h2>
          <div style="color:#475569">예상 언급 <b>{forecast['predicted_mention_count']}건</b> ·
          80% 경험적 구간 <b>{lower}~{upper}건</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    probability_bar(
        "논의 지속", float(forecast["discussion_active_probability"]),
        "다음 달에도 동일 이슈가 관측될 상대적 가능성",
    )
    probability_bar(
        "언급 증가", float(forecast["discussion_growth_probability"]),
        "최근 3개월 평균보다 언급량이 증가할 상대적 가능성",
    )
    probability_bar(
        "신규 의회 확산", float(forecast["new_council_spread_probability"]),
        "아직 관측되지 않은 의회에서 새로 등장할 상대적 가능성",
    )
    baseline = forecast.get("selected_baseline") or {}
    st.caption(
        f"선택 기준선: {baseline.get('model', '미상')} · "
        f"백테스트 MAE {float(baseline.get('mae', 0)):.3f} · "
        "확률은 다중 이슈 학습 전 비보정 지표입니다."
    )

st.markdown("---")
st.subheader("회의록 원문 근거")
evidence_rows = payload.get("evidence", [])
council_options = ["전체", *sorted({row["council"] for row in evidence_rows})]
selected_council = st.selectbox("근거 지역", council_options, label_visibility="collapsed")
filtered_evidence = [
    row for row in evidence_rows
    if selected_council == "전체" or row["council"] == selected_council
]
for index, row in enumerate(filtered_evidence, start=1):
    title = (
        f"{index}. {row['council']} · {row['meeting_date']} · "
        f"{row['meeting_name'] or '회의록명 미상'}"
    )
    with st.expander(title, expanded=index <= 2):
        st.markdown(
            f"""
            <div class="evidence-card">
              <div class="evidence-meta"><b>위원회</b> {escape(str(row['committee'] or '미상'))} &nbsp;·&nbsp;
              <b>안건</b> {escape(str(row['agenda_title'] or '미상'))} &nbsp;·&nbsp;
              <b>군집 유사도</b> {row['similarity']:.3f}</div>
              <div class="evidence-text">{escape(str(row['text']))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
with st.expander("분석 방법과 한계"):
    st.markdown(
        """
        - 검색 청크를 로컬 임베딩으로 변환하고 pgvector에서 근거 후보를 검색했습니다.
        - 재순위된 근거를 확장·군집화하고, 검수 후 이슈 ID를 고정했습니다.
        - 최초 관측과 후속 의회 순서는 회의일 기준으로 계산했습니다.
        - 다음 달 전망은 단일 이슈 소표본에서 여러 기준선을 시간순 백테스트한 결과입니다.
        - 지도 연결선은 시간적 관측 순서이며 실제 정책 전달의 인과관계를 의미하지 않습니다.
        """
    )


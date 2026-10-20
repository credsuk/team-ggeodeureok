#!/usr/bin/env python3
"""광역 지방의회 정책 이슈 검색·전파·예측 Streamlit 데모."""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer


MODEL_NAME = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
MODEL_REVISION = "55ec6e9"
DEFAULT_TIMELINE_PATH = (
    "outputs/issue_timeline_final/issue_timelines.json"
    if Path("outputs/issue_timeline_final/issue_timelines.json").exists()
    else "examples/sample_issue_timelines.json"
)
DEFAULT_FORECAST_PATH = (
    "outputs/issue_forecast_final/issue_forecasts.json"
    if Path("outputs/issue_forecast_final/issue_forecasts.json").exists()
    else "examples/sample_issue_forecasts.json"
)
# 지도 이미지 안에서 광역의회의 위치를 0~1 좌표로 지정한다.
# 이 좌표는 화면 표시 전용이며 분석, 군집화, 예측 변수로 사용하지 않는다.
COUNCIL_MAP_POSITIONS = {
    "서울특별시의회": (0.36, 0.73),
    "인천광역시의회": (0.27, 0.72),
    "경기도의회": (0.39, 0.67),
    "강원특별자치도의회": (0.63, 0.78),
    "충청북도의회": (0.51, 0.57),
    "충청남도의회": (0.32, 0.53),
    "세종특별자치시의회": (0.40, 0.52),
    "대전광역시의회": (0.42, 0.47),
    "경상북도의회": (0.66, 0.49),
    "대구광역시의회": (0.66, 0.41),
    "전북특별자치도의회": (0.40, 0.37),
    "전라북도의회": (0.40, 0.37),
    "경상남도의회": (0.57, 0.29),
    "울산광역시의회": (0.76, 0.34),
    "부산광역시의회": (0.70, 0.25),
    "광주광역시의회": (0.31, 0.27),
    "전라남도의회": (0.29, 0.19),
    "제주특별자치도의회": (0.24, 0.045),
}
MAP_IMAGE_PATH = Path(__file__).resolve().parent / "assets" / "korea_map.png"


st.set_page_config(
    page_title="광역 지방의회 정책 흐름",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1380px;}
    h1 {color:#102a43; letter-spacing:-.04em;}
    h2, h3 {color:#183b56; letter-spacing:-.025em;}
    [data-testid="stMetric"] {background:#f8fafc; border:1px solid #e2e8f0; padding:14px; border-radius:14px;}
    [data-testid="stMetricValue"] {color:#0f2747;}
    .issue-box {background:linear-gradient(135deg,#0f2747,#183b67); color:white; padding:22px 26px; border-radius:16px; margin:10px 0 18px 0; box-shadow:0 10px 28px rgba(15,39,71,.12);}
    .issue-box h2 {margin:0 0 6px 0; font-size:1.45rem;}
    .issue-box p {margin:0; color:#dbeafe;}
    .section-note {background:#f0f7ff; border-left:4px solid #2f6fed; padding:12px 14px; border-radius:8px; color:#334e68; margin:.4rem 0 1rem 0;}
    .search-stat {display:inline-block; background:#eef4ff; color:#244b8a; padding:6px 10px; border-radius:999px; font-size:.84rem; margin:.25rem 0 .75rem 0;}
    .note {color:#64748b; font-size:.88rem;}
    .stTabs [data-baseweb="tab-list"] {gap:8px; background:#f8fafc; padding:6px; border-radius:12px;}
    .stTabs [data-baseweb="tab"] {height:44px; border-radius:9px; padding:0 18px;}
    .stTabs [aria-selected="true"] {background:white; box-shadow:0 2px 8px rgba(15,39,71,.08);}
    div.stButton > button {background:#173b67; color:white; border:0; border-radius:10px; min-height:42px; font-weight:700;}
    div.stButton > button:hover {background:#24548a; color:white;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_model(cache_dir: str, device: str):
    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        cache_folder=str(Path(cache_dir).resolve()),
        device=device,
        local_files_only=True,
        trust_remote_code=False,
    )
    model.max_seq_length = 2048
    return model


@st.cache_resource(show_spinner=False)
def load_chroma(chroma_dir: str, collection_name: str):
    client = chromadb.PersistentClient(path=str(Path(chroma_dir).resolve()))
    return client.get_collection(collection_name, embedding_function=None)


@st.cache_resource(show_spinner=False)
def load_map_image():
    if not MAP_IMAGE_PATH.exists():
        raise FileNotFoundError(f"지도 이미지가 없습니다: {MAP_IMAGE_PATH}")
    return Image.open(MAP_IMAGE_PATH)


@st.cache_data(show_spinner=False)
def load_assignments(path: str):
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    columns = [
        name for name in ("chunk_id", "embedding_row", "cluster_id", "cluster_similarity")
        if name in available
    ]
    by_chunk, by_row = {}, {}
    for batch in parquet.iter_batches(batch_size=8192, columns=columns):
        for row in batch.to_pylist():
            item = {
                "cluster_id": int(row.get("cluster_id", -1)),
                "cluster_similarity": float(row.get("cluster_similarity", -1.0)),
            }
            chunk_id = str(row.get("chunk_id") or "").strip()
            embedding_row = int(row.get("embedding_row", -1))
            if chunk_id:
                by_chunk[chunk_id] = item
            if embedding_row >= 0:
                by_row[embedding_row] = item
    return by_chunk, by_row


def lexical_score(query: str, timeline: dict) -> float:
    tokens = set(re.findall(r"[0-9A-Za-z가-힣]{2,}", query.lower()))
    text = " ".join(str(timeline.get(key, "")) for key in ("issue_name", "issue_summary", "keywords")).lower()
    return sum(token in text for token in tokens) / max(1, len(tokens))


def chroma_issue_rank(
    query: str,
    timelines: list[dict],
    chroma_dir: str,
    collection_name: str,
    assignments_path: str,
    cache_dir: str,
    device: str,
    candidate_chunks: int,
):
    model = load_model(cache_dir, device)
    kwargs = {"convert_to_numpy": True, "normalize_embeddings": True, "show_progress_bar": False}
    if "query" in (getattr(model, "prompts", {}) or {}):
        kwargs["prompt_name"] = "query"
    vector = np.asarray(model.encode([query], **kwargs), dtype=np.float32)
    collection = load_chroma(chroma_dir, collection_name)
    started = time.perf_counter()
    result = collection.query(
        query_embeddings=vector.tolist(),
        n_results=candidate_chunks,
        include=["documents", "metadatas", "distances"],
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    by_chunk, by_row = load_assignments(assignments_path)
    timeline_by_cluster = {int(row["cluster_id"]): row for row in timelines}
    grouped = {}
    unmatched = below_similarity = unreviewed = 0
    for rank, (chunk_id, document, metadata, distance) in enumerate(zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ), start=1):
        metadata = metadata or {}
        embedding_row = int(metadata.get("embedding_row", -1))
        assignment = by_chunk.get(chunk_id) or by_row.get(embedding_row)
        if not assignment:
            unmatched += 1
            continue
        cluster_similarity = assignment["cluster_similarity"]
        if cluster_similarity < 0.55:
            below_similarity += 1
            continue
        cluster_id = assignment["cluster_id"]
        if cluster_id not in timeline_by_cluster:
            unreviewed += 1
            continue
        retrieval_similarity = 1.0 - float(distance)
        contribution = (
            max(0.0, retrieval_similarity)
            * max(0.0, cluster_similarity)
            / math.log2(rank + 1.0)
        )
        hit = {
            "retrieval_rank": rank,
            "chunk_id": chunk_id,
            "retrieval_similarity": retrieval_similarity,
            "cluster_similarity": cluster_similarity,
            "score_contribution": contribution,
            "council": str(metadata.get("council") or ""),
            "committee": str(metadata.get("committee") or ""),
            "meeting_date": str(metadata.get("meeting_date") or ""),
            "meeting_name": str(metadata.get("meeting_name") or ""),
            "agenda_title": str(
                metadata.get("agenda_title_effective")
                or metadata.get("evidence_title")
                or metadata.get("agenda_title")
                or ""
            ),
            "speaker_label": str(metadata.get("speaker_label") or ""),
            "evidence_ref": str(metadata.get("evidence_ref") or ""),
            "text": re.sub(r"\s+", " ", str(document or "")).strip(),
        }
        grouped.setdefault(cluster_id, []).append(hit)

    ranked = []
    for cluster_id, hits in grouped.items():
        hits.sort(key=lambda row: (-row["score_contribution"], row["retrieval_rank"]))
        best = hits[0]["score_contribution"]
        top_support = sum(row["score_contribution"] for row in hits[:5]) / min(5, len(hits))
        support_bonus = min(0.15, 0.03 * math.log1p(len(hits)))
        score = 0.65 * best + 0.35 * top_support + support_bonus
        ranked.append((score, timeline_by_cluster[cluster_id], hits))
    ranked.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
    diagnostics = {
        "retrieved": len(result["ids"][0]),
        "matched": sum(len(item[2]) for item in ranked),
        "issues": len(ranked),
        "unmatched": unmatched,
        "below_similarity": below_similarity,
        "unreviewed": unreviewed,
        "elapsed_ms": elapsed_ms,
    }
    return ranked, diagnostics


def council_coordinate(name: str):
    if name in COUNCIL_MAP_POSITIONS:
        return COUNCIL_MAP_POSITIONS[name]
    simplified = name.replace("의회", "")
    for council, coordinate in COUNCIL_MAP_POSITIONS.items():
        if simplified and (simplified in council or council.replace("의회", "") in name):
            return coordinate
    return None


def short_council_name(name: str) -> str:
    replacements = {
        "서울특별시의회": "서울", "부산광역시의회": "부산", "대구광역시의회": "대구",
        "인천광역시의회": "인천", "광주광역시의회": "광주", "대전광역시의회": "대전",
        "울산광역시의회": "울산", "세종특별자치시의회": "세종", "경기도의회": "경기",
        "강원특별자치도의회": "강원", "충청북도의회": "충북", "충청남도의회": "충남",
        "전북특별자치도의회": "전북", "전라북도의회": "전북", "전라남도의회": "전남",
        "경상북도의회": "경북", "경상남도의회": "경남", "제주특별자치도의회": "제주",
    }
    return replacements.get(name, name.replace("의회", ""))


def propagation_map(events: list[dict]) -> go.Figure:
    located = []
    for event in events:
        coordinate = council_coordinate(str(event.get("council", "")))
        if coordinate:
            located.append((event, coordinate))
    figure = go.Figure()
    figure.add_layout_image(
        source=load_map_image(), x=0, y=1, sizex=1, sizey=1,
        xref="x", yref="y", xanchor="left", yanchor="top",
        sizing="stretch", opacity=0.92, layer="below",
    )
    for index in range(len(located) - 1):
        (_, start), (_, end) = located[index], located[index + 1]
        figure.add_trace(go.Scatter(
            x=[start[0], end[0]], y=[start[1], end[1]], mode="lines",
            line={"width": 2.5, "color": "rgba(47,111,237,.62)"},
            hoverinfo="skip", showlegend=False,
        ))
    if located:
        hover_text = [
            (
                f"<b>{index + 1}. {item[0].get('council')}</b><br>"
                f"최초 관측 {item[0].get('first_date')}<br>"
                f"기준 대비 {item[0].get('lag_days', 0)}일 · 근거 {item[0].get('mention_count', 0)}건"
            )
            for index, item in enumerate(located)
        ]
        figure.add_trace(go.Scatter(
            x=[item[1][0] for item in located],
            y=[item[1][1] for item in located],
            text=[str(index + 1) for index in range(len(located))],
            hovertext=hover_text,
            marker={
                "size": [28 if index == 0 else 22 for index in range(len(located))],
                "color": ["#f97316" if index == 0 else "#246bfd" for index in range(len(located))],
                "line": {"width": 2.5, "color": "white"},
            },
            textfont={"color": "white", "size": 11, "family": "Arial Black"},
            mode="markers+text", textposition="middle center",
            hovertemplate="%{hovertext}<extra></extra>", showlegend=False,
        ))
        first_event, first_position = located[0]
        figure.add_annotation(
            x=first_position[0], y=first_position[1],
            text=f"최초 · {short_council_name(str(first_event.get('council', '')))}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
            arrowcolor="#f97316", ax=-65, ay=-35,
            bgcolor="rgba(255,255,255,.96)", bordercolor="#fed7aa",
            borderpad=5, font={"size": 11, "color": "#9a3412"},
        )
    figure.update_layout(
        height=650, margin={"l": 8, "r": 8, "t": 8, "b": 8},
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        xaxis={"range": [0, 1], "visible": False, "fixedrange": True},
        yaxis={
            "range": [0, 1], "visible": False, "fixedrange": True,
            "scaleanchor": "x", "scaleratio": 1.5,
        },
        hoverlabel={"bgcolor": "white", "font_size": 13, "font_family": "Arial"},
    )
    return figure


def timeline_chart(series: list[dict], forecast: dict | None) -> go.Figure:
    months = [row.get("month") for row in series]
    counts = [row.get("mention_count", 0) for row in series]
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=months, y=counts, mode="lines+markers", name="관측 언급량",
        line={"color": "#2563eb", "width": 3}, marker={"size": 7},
    ))
    if forecast:
        figure.add_trace(go.Scatter(
            x=[forecast.get("forecast_month")], y=[forecast.get("predicted_mention_count", 0)],
            mode="markers", name="다음 달 예측", marker={"size": 14, "color": "#f97316", "symbol": "diamond"},
        ))
    figure.update_layout(
        height=360, margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_title="월", yaxis_title="근거 발언 수", hovermode="x unified",
        legend={"orientation": "h", "y": 1.12},
    )
    return figure


def korean_direction(value: str) -> str:
    return {
        "expansion": "증가 예상", "persistence": "유지 예상",
        "attenuation": "감소 예상", "dormant": "비활성 예상",
    }.get(value, value or "예측 없음")


st.title("광역 지방의회 정책 이슈 흐름")
st.caption("2024–2025년 제공 회의록만 이용한 최초 관측, 의회 간 후속 관측, 다음 달 흐름 예측")

with st.sidebar:
    st.header("분석 설정")
    st.caption("일반 사용자는 기본값 그대로 검색하면 됩니다.")
    with st.expander("고급 경로·모델 설정"):
        timeline_path = st.text_input("타임라인 JSON", DEFAULT_TIMELINE_PATH)
        forecast_path = st.text_input("예측 JSON", DEFAULT_FORECAST_PATH)
        chroma_dir = st.text_input("Chroma 경로", "data/chroma/arctic_policy_v1")
        collection_name = st.text_input("Chroma 컬렉션", "council_policy_arctic_v1")
        assignments_path = st.text_input(
            "군집 배정 Parquet",
            "data/index/arctic_ko_semantic_test_v2/clustering_k160/cluster_assignments.parquet",
        )
        candidate_chunks = st.slider("검색 후보 청크", 10, 100, 50, 10)
        cache_dir = st.text_input("모델 캐시", "models/huggingface")
        device = st.selectbox("검색 장치", ["cuda", "cpu"], index=0)
    st.caption("지도 좌표는 화면 배치 전용이며 분석·예측 변수로 사용하지 않습니다.")

try:
    timelines = load_json(timeline_path)
except Exception as error:
    st.error(f"타임라인 파일을 열 수 없습니다: {error}")
    st.stop()

try:
    forecasts = load_json(forecast_path) if Path(forecast_path).exists() else []
except Exception as error:
    st.error(f"예측 파일을 열 수 없습니다: {error}")
    st.stop()
forecast_by_issue = {str(row.get("issue_id")): row for row in forecasts}

if timeline_path.startswith("examples/"):
    st.info("현재 화면은 공개 저장소용 합성 예시입니다. 실제 경진대회 분석 결과가 아닙니다.")

query_col, button_col = st.columns([5, 1])
query = query_col.text_input("정책 이슈 검색", placeholder="예: 저출생, 버스노선 교통약자, 수돗물 안전")
search_clicked = button_col.button("검색", use_container_width=True)

ranked = []
if query and search_clicked:
    try:
        with st.spinner("Chroma에서 근거 발언을 찾고 검수 이슈에 연결 중입니다."):
            ranked, diagnostics = chroma_issue_rank(
                query, timelines, chroma_dir, collection_name, assignments_path,
                cache_dir, device, candidate_chunks,
            )
        st.session_state["search_diagnostics"] = diagnostics
        st.session_state["query_evidence_by_issue"] = {
            str(row[1]["issue_id"]): row[2][:5] for row in ranked[:5]
        }
        st.session_state["ranked_issue_ids"] = [row[1]["issue_id"] for row in ranked[:5]]
    except Exception as error:
        st.warning(f"Chroma 검색을 사용할 수 없어 키워드 검색으로 대체합니다: {error}")
        ranked = sorted(
            ((lexical_score(query, row), row, []) for row in timelines),
            key=lambda item: item[0], reverse=True,
        )
        st.session_state["search_diagnostics"] = {}
        st.session_state["query_evidence_by_issue"] = {}
        st.session_state["ranked_issue_ids"] = [row[1]["issue_id"] for row in ranked[:5]]

ranked_ids = st.session_state.get("ranked_issue_ids", [])
if ranked_ids:
    candidate_timelines = [row for issue_id in ranked_ids for row in timelines if row.get("issue_id") == issue_id]
    diagnostics = st.session_state.get("search_diagnostics", {})
    if diagnostics:
        st.markdown(
            f"<span class='search-stat'>Chroma 후보 {diagnostics.get('retrieved', 0)}개 · "
            f"연결 근거 {diagnostics.get('matched', 0)}개 · "
            f"검수 이슈 {diagnostics.get('issues', 0)}개 · "
            f"검색 {diagnostics.get('elapsed_ms', 0):.1f}ms</span>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("키워드 검색 상위 5개 이슈")
else:
    candidate_timelines = sorted(timelines, key=lambda row: row.get("issue_name", ""))

if not candidate_timelines:
    st.warning("검수 완료 이슈와 연결된 검색 결과가 없어 전체 이슈 목록을 표시합니다.")
    candidate_timelines = sorted(timelines, key=lambda row: row.get("issue_name", ""))

selected_name = st.selectbox(
    "분석할 이슈",
    [f"{row.get('issue_name')} ({row.get('issue_id')})" for row in candidate_timelines],
)
selected_index = [f"{row.get('issue_name')} ({row.get('issue_id')})" for row in candidate_timelines].index(selected_name)
timeline = candidate_timelines[selected_index]
forecast = forecast_by_issue.get(str(timeline.get("issue_id")))
query_evidence = st.session_state.get("query_evidence_by_issue", {}).get(
    str(timeline.get("issue_id")), []
)

st.markdown(
    f"<div class='issue-box'><h2>{timeline.get('issue_name')}</h2>"
    f"<p>{timeline.get('issue_summary', '')}</p></div>", unsafe_allow_html=True,
)

metric_columns = st.columns(5)
metric_columns[0].metric("최초 관측", timeline.get("first_observed_date", "-"))
metric_columns[1].metric("최초 의회", ", ".join(timeline.get("origin_councils", [])) or "-")
metric_columns[2].metric("관측 의회", f"{timeline.get('observed_council_count', 0)}개")
metric_columns[3].metric("관측 근거", f"{timeline.get('observation_count', 0)}건")
metric_columns[4].metric("다음 달", korean_direction((forecast or {}).get("predicted_direction", "")))

map_tab, forecast_tab, evidence_tab = st.tabs(["🗺️ 전파 지도", "📈 시계열·예측", "📚 근거 원문"])

with map_tab:
    st.markdown(
        "<div class='section-note'><b>읽는 법</b> · 주황색 1번은 제공 데이터에서의 최초 관측, "
        "파란 번호는 각 의회의 최초 후속 관측 순서입니다. 선은 시간 순서를 연결하며 인과관계를 뜻하지 않습니다.</div>",
        unsafe_allow_html=True,
    )
    propagation_rows = [
        {
            "순서": event.get("propagation_order"),
            "지역": short_council_name(str(event.get("council", ""))),
            "최초 관측일": event.get("first_date"),
            "시차": f"+{event.get('lag_days', 0)}일",
            "근거": f"{event.get('mention_count', 0)}건",
        }
        for event in timeline.get("propagation", [])
    ]
    map_column, flow_column = st.columns([1.35, 1], gap="large")
    with map_column:
        st.subheader("대한민국 광역의회 관측 흐름")
        st.plotly_chart(
            propagation_map(timeline.get("propagation", [])),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with flow_column:
        st.subheader("전파 순서표")
        st.dataframe(
            pd.DataFrame(propagation_rows),
            hide_index=True,
            use_container_width=True,
            height=610,
            column_config={
                "순서": st.column_config.NumberColumn(width="small"),
                "지역": st.column_config.TextColumn(width="medium"),
                "시차": st.column_config.TextColumn(width="small"),
                "근거": st.column_config.TextColumn(width="small"),
            },
        )

with forecast_tab:
    st.subheader("월별 논의 변화와 다음 달 예측")
    st.markdown(
        "<div class='section-note'>관측 구간은 2024–2025년 제공 회의록으로 제한됩니다. "
        "예측값은 과거 언급량·확산 범위·최근 변화량으로 계산한 보조 지표입니다.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        timeline_chart(timeline.get("monthly_series", []), forecast),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    if forecast:
        forecast_columns = st.columns(4)
        forecast_columns[0].metric("논의 증가 확률", f"{float(forecast.get('growth_probability', 0)):.1%}")
        forecast_columns[1].metric("예측 언급량", f"{forecast.get('predicted_mention_count', 0)}건")
        forecast_columns[2].metric("신규 의회 확산", f"{float(forecast.get('spread_probability', 0)):.1%}")
        forecast_columns[3].metric("예측 신뢰도", forecast.get("forecast_confidence", "-"))
        if forecast.get("forecast_confidence") == "low":
            st.warning("검증 성능 또는 표본 수가 제한되어 보수적으로 해석해야 하는 예측입니다.")
        st.info(forecast.get("forecast_narrative", ""))

with evidence_tab:
    if query_evidence:
        st.subheader("검색어와 직접 연결된 대표 근거")
        st.caption(
            "검색어와 가장 유사한 문장입니다. 최초 관측 근거와는 다를 수 있으며, "
            "최초 관측은 아래 의회별 근거에서 확인합니다."
        )
        for index, evidence in enumerate(query_evidence[:3], start=1):
            title = (
                f"{index}. {evidence.get('council')} · {evidence.get('meeting_date')} · "
                f"{evidence.get('agenda_title') or evidence.get('meeting_name') or '회의록'}"
            )
            with st.expander(title, expanded=index == 1):
                st.write(evidence.get("text") or "근거 문장이 없습니다.")
                st.caption(
                    f"검색 유사도 {float(evidence.get('retrieval_similarity', 0)):.4f} · "
                    f"군집 유사도 {float(evidence.get('cluster_similarity', 0)):.4f} · "
                    f"근거 참조 {evidence.get('evidence_ref') or '-'}"
                )

    st.subheader("최초 관측 및 의회별 전파 근거")
    evidence_count = 0
    for event in timeline.get("propagation", []):
        for evidence in event.get("evidence", []):
            evidence_count += 1
            title = (
                f"{event.get('propagation_order', '-')}번 · {event.get('council')} · "
                f"{evidence.get('meeting_date')} · {evidence.get('meeting_name') or '회의록'}"
            )
            with st.expander(title, expanded=evidence_count <= 2):
                st.write(f"안건: {evidence.get('agenda_title') or '-'}")
                st.write(f"발화자: {evidence.get('speaker_label') or '-'}")
                st.write(evidence.get("original_excerpt") or "원문 구간이 없습니다. 09번을 다시 실행하세요.")
                st.caption(
                    f"근거 참조: {evidence.get('evidence_ref') or '-'} · "
                    f"문자 범위: {evidence.get('char_start', -1)}–{evidence.get('char_end', -1)} · "
                    f"군집 유사도: {evidence.get('similarity', 0)}"
                )

st.markdown(
    "<p class='note'>전파는 제공 데이터 안에서 관측된 시간적 순서이며 실제 정책의 인과적 전파를 단정하지 않습니다. "
    "예측에는 2025-12-31 이후 외부 정보를 사용하지 않았습니다.</p>",
    unsafe_allow_html=True,
)

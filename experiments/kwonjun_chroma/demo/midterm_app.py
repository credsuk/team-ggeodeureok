#!/usr/bin/env python3
"""꺼드럭's 의정이슈레이더 중간제출 통합 Streamlit 앱."""

from __future__ import annotations

import json
import os
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

from agent.local_issue_agent import LocalCouncilAgent  # noqa: E402
from council_vector_runtime import CouncilSearchService, SearchFilters  # noqa: E402


DEFAULT_PAYLOAD = PROJECT_ROOT / "outputs" / "web" / "birth_rate_issue.json"


def project_path(environment_name: str, relative: str) -> Path:
    return Path(
        os.environ.get(environment_name, str(PROJECT_ROOT / relative))
    ).expanduser().resolve()


def discover_payloads() -> list[Path]:
    configured = project_path("COUNCIL_ISSUE_PAYLOAD", "outputs/web/birth_rate_issue.json")
    candidates = [configured, *sorted((PROJECT_ROOT / "outputs" / "web").glob("*.json"))]
    result = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved.is_file() and resolved not in seen:
            result.append(resolved)
            seen.add(resolved)
    return result


@st.cache_data(show_spinner=False)
def load_payload(path_text: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    payload = json.loads(Path(path_text).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "council_issue_web_v1":
        raise ValueError("지원하지 않는 이슈 결과 JSON입니다")
    return payload


@st.cache_resource(show_spinner="로컬 임베딩 모델과 pgvector를 연결하는 중입니다…")
def load_search_service() -> CouncilSearchService:
    return CouncilSearchService(
        backend="pgvector",
        embedding_dir=project_path(
            "COUNCIL_EMBEDDING_DIR", "data/embeddings/arctic_raw_v1"
        ),
        parent_path=project_path(
            "COUNCIL_PARENT_PATH", "data/canonical_v3/chunks_v1/parents.parquet"
        ),
        pg_dsn=os.environ.get("PGVECTOR_DSN"),
        pg_table=os.environ.get("COUNCIL_PG_TABLE", "council_arctic_raw_v1"),
        pg_ef_search=int(os.environ.get("COUNCIL_PG_EF_SEARCH", "200")),
        device=os.environ.get("COUNCIL_DEVICE", "auto"),
        model_cache=project_path("COUNCIL_MODEL_CACHE", "models/huggingface"),
        local_files_only=os.environ.get("COUNCIL_ALLOW_DOWNLOAD", "0") != "1",
    )


def stage_label(value: str) -> str:
    return {
        "first_observation": "최초 관측",
        "early_followup_30d": "30일 내 후속",
        "expansion_90d": "90일 내 확장",
        "later_observation": "후기 관측",
    }.get(value, value)


def direction_label(value: str) -> str:
    return {
        "growth": "논의 증가",
        "persistence": "논의 지속",
        "attenuation": "관심 약화",
    }.get(value, value or "판단 보류")


def issue_map(payload: dict[str, Any]) -> pdk.Deck:
    nodes = []
    for row in payload["map"]["nodes"]:
        node = dict(row)
        node["position"] = [row["longitude"], row["latitude"]]
        node["label"] = str(row["order"])
        node["stage_label"] = stage_label(str(row.get("stage") or ""))
        node["radius"] = min(
            43000, 15000 + max(int(row.get("mention_count") or 1), 1) ** 0.5 * 2500
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
                f"{escape(str(metadata.get('committee') or '미상'))} &nbsp;·&nbsp; "
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


def load_external_context() -> dict[str, Any] | None:
    path = project_path(
        "COUNCIL_EXTERNAL_CONTEXT", "outputs/external/external_context.json"
    )
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


st.set_page_config(
    page_title="의정이슈레이더 | 꺼드럭's", page_icon="📡", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
:root{--ivory:#f7f1e6;--paper:#fffaf1;--olive:#4b523b;--khaki:#747b59;
--coral:#eb604f;--blush:#f7d5ce;--ink:#30352b;--muted:#777a6c;--line:#ddd4c3}
.stApp{background:linear-gradient(180deg,#f5efe3 0,#faf7ef 34%,#f7f1e6 100%);color:var(--ink)}
.block-container{max-width:1550px;padding-top:1rem;padding-bottom:4rem}
[data-testid="stSidebar"]{background:#eae4d6;border-right:1px solid #d5cebc}
.brand{padding:.25rem 0 .55rem}.brand-team{font-size:.73rem;font-weight:900;letter-spacing:.15em;color:#9b5047}
.brand-title{font-size:1.55rem;font-weight:900;letter-spacing:-.06em;color:#414733}
.hero{position:relative;overflow:hidden;border-radius:26px;padding:1.65rem 2rem;margin:.2rem 0 1rem;
background:radial-gradient(circle at 86% 0,rgba(242,103,87,.36),transparent 24%),linear-gradient(125deg,#424936,#6d7453);
color:#fff9ed;box-shadow:0 20px 46px rgba(65,72,51,.16)}
.hero h1{color:#fff9ed;margin:.1rem 0 .35rem;font-size:2.35rem;letter-spacing:-.065em}.hero p{margin:0;color:#eee9d9}
.hero-badge,.soft-badge{display:inline-block;border-radius:999px;padding:.28rem .7rem;font-size:.78rem;font-weight:800}
.hero-badge{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.23);color:#ffe2dc}
.soft-badge{background:#f5d8d2;color:#923f38;margin-right:.35rem}.policy-badge{background:#dde2cf;color:#4d573c}
.card{background:#fffaf2;border:1px solid var(--line);border-radius:18px;padding:1rem 1.15rem;box-shadow:0 8px 25px rgba(72,78,57,.06)}
.forecast{background:linear-gradient(135deg,#fff9ef,#f7d8d1);border:1px solid #efb8ae;border-radius:19px;padding:1.1rem 1.25rem}
.evidence-card{background:#fffdf8;border:1px solid #ded6c5;border-left:4px solid var(--coral);border-radius:12px;padding:1rem;line-height:1.72}
.evidence-meta{color:#6f7262;font-size:.86rem;margin:.1rem 0 .55rem}.tiny{color:#797b6e;font-size:.84rem}
div[data-testid="stMetric"]{background:#fffaf2;border:1px solid #ded6c5;border-radius:15px;padding:.8rem 1rem;box-shadow:0 7px 18px rgba(73,78,58,.05)}
.stProgress>div>div>div>div{background-color:var(--coral)}
.stButton>button,.stDownloadButton>button{border-radius:11px;border-color:#858b6c;color:#4a5139;background:#fffaf1}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--coral);color:#a84237;background:#fff1ed}
h1,h2,h3{color:#404633;letter-spacing:-.04em}
</style>
""", unsafe_allow_html=True)

payload_paths = discover_payloads()
if not payload_paths:
    st.error(f"웹 결과 JSON이 없습니다: {DEFAULT_PAYLOAD}")
    st.code("python -u src/24_build_web_payload.py ...")
    st.stop()

with st.sidebar:
    st.markdown("<div class='brand'><div class='brand-team'>TEAM 꺼드럭's</div>"
                "<div class='brand-title'>의정이슈레이더</div></div>", unsafe_allow_html=True)
    selected_path = st.selectbox(
        "분석 이슈", payload_paths,
        format_func=lambda path: path.stem.replace("_", " "),
    )
    st.caption("2024~2025년 광역 지방의회 회의록")
    st.markdown("<span class='soft-badge policy-badge'>LOCAL ONLY</span>"
                "<span class='soft-badge'>CUTOFF 2025-12-31</span>", unsafe_allow_html=True)
    st.divider()
    st.markdown("**중간본 분석 구성**")
    st.caption("Arctic 임베딩 · pgvector HNSW · 근거 재순위 · 이슈 군집 · 시간적 전파 · 설명 가능한 기준선 예측")
    st.markdown("**보안 원칙**")
    st.caption("회의록 외부 전송 없음 · 로컬 Ollama만 허용")

payload = load_payload(str(selected_path), selected_path.stat().st_mtime_ns)
issue = payload["issue"]
summary = payload["summary"]
forecast = payload["forecast"]

st.markdown(f"""
<div class='hero'>
  <span class='hero-badge'>COUNCIL ISSUE INTELLIGENCE</span>
  <h1>의정이슈레이더</h1>
  <p><b>{escape(str(issue['issue_name']))}</b> · 최초 관측, 지역별 흐름, 다음 달 전망과 원문 근거를 한 번에 탐색합니다.</p>
</div>
""", unsafe_allow_html=True)

metrics = st.columns(5)
metrics[0].metric("최초 관측", summary["first_observation_date"])
metrics[1].metric("최초 의회", " · ".join(summary["first_observation_councils"]))
metrics[2].metric("관측 의회", f"{summary['council_count']} / 17")
metrics[3].metric("근거 단위", f"{summary['evidence_unit_count']}건")
metrics[4].metric("다음 달", direction_label(forecast["direction"]))

radar_tab, search_tab, agent_tab, external_tab, method_tab = st.tabs([
    "📍 이슈 레이더", "🔎 원문 근거 탐색", "💬 로컬 AI Agent",
    "📊 외부지표 참고", "🧪 방법·재현성",
])

with radar_tab:
    map_col, order_col = st.columns([1.5, 1], gap="large")
    with map_col:
        st.subheader("지역별 최초 관측 흐름")
        st.caption("연결선은 최초 관측의 시간 순서이며 정책 전달의 인과관계가 아닙니다.")
        st.pydeck_chart(issue_map(payload), use_container_width=True)
    with order_col:
        st.subheader("의회별 관측 순서")
        table = pd.DataFrame(payload["propagation"])
        table = table.rename(columns={
            "observation_order": "순서", "council": "의회",
            "first_observation_date": "최초 관측", "lag_days_from_first": "시차",
            "mention_count": "근거 수",
        })
        table["시차"] = table["시차"].map(lambda value: f"+{value}일")
        table["근거 수"] = table["근거 수"].map(lambda value: f"{value}건")
        st.dataframe(table[["순서", "의회", "최초 관측", "시차", "근거 수"]],
                     hide_index=True, use_container_width=True, height=430)

    trend_col, forecast_col = st.columns([1.55, 1], gap="large")
    with trend_col:
        st.subheader("월별 논의 강도")
        monthly = pd.DataFrame(payload["monthly_series"])
        monthly["month"] = pd.to_datetime(monthly["month"])
        st.line_chart(monthly.set_index("month")[["mention_count"]],
                      color=["#eb604f"], height=245)
        st.caption("월별 동일 이슈 근거 단위 수. 빈 달은 0건으로 보존합니다.")
    with forecast_col:
        st.subheader(f"{forecast['target_month']} 전망")
        lower, upper = forecast["prediction_interval_80"]
        st.markdown(
            f"<div class='forecast'><b>{direction_label(forecast['direction'])}</b><br>"
            f"예상 언급 <b>{forecast['predicted_mention_count']}건</b> · "
            f"80% 경험적 구간 <b>{lower}~{upper}건</b></div>",
            unsafe_allow_html=True,
        )
        st.progress(float(forecast["discussion_active_probability"]))
        st.caption(f"논의 지속 지표 {float(forecast['discussion_active_probability']):.1%}")
        st.progress(float(forecast["new_council_spread_probability"]))
        st.caption(f"신규 의회 확산 지표 {float(forecast['new_council_spread_probability']):.1%}")
        baseline = forecast.get("selected_baseline") or {}
        st.caption(f"시간순 백테스트 기준선: {baseline.get('model')} · MAE {float(baseline.get('mae', 0)):.3f}")

    st.subheader("대표 원문 근거")
    render_evidence(payload.get("evidence") or [])

with search_tab:
    st.subheader("전체 90만 청크에서 근거 다시 찾기")
    st.caption("검색 결과는 부모 문맥과 회의록명까지 복원하며, 분석 데이터는 로컬 pgvector를 벗어나지 않습니다.")
    with st.form("evidence_search"):
        query = st.text_input("정책·현안", value=str(issue["issue_name"]))
        first, second, third = st.columns(3)
        council_filter = first.text_input("의회 필터(선택)", placeholder="예: 경상남도의회")
        year_filter = second.selectbox("연도", ["전체", 2024, 2025])
        top_k = third.slider("결과 수", 3, 20, 10)
        submitted = st.form_submit_button("pgvector 근거 검색", type="primary")
    if submitted:
        try:
            with st.spinner("근거 후보를 검색하고 부모 문맥을 복원하는 중입니다…"):
                result = load_search_service().search(
                    query=query, top_k=top_k, candidate_k=max(50, top_k * 10),
                    filters=SearchFilters(
                        council=council_filter.strip() or None,
                        year=None if year_filter == "전체" else int(year_filter),
                    ), include_parent=True,
                )
            st.session_state["midterm_search"] = result
        except Exception as error:
            st.error(f"검색 환경을 확인해 주세요: {error}")
    search_result = st.session_state.get("midterm_search")
    if search_result:
        a, b, c = st.columns(3)
        a.metric("검색 결과", f"{search_result['result_count']}건")
        b.metric("DB 검색", f"{search_result['search_ms']:.1f} ms")
        c.metric("질의 임베딩", f"{search_result['query_embedding_ms']:.1f} ms")
        render_evidence(search_result["results"], expanded=3)
        st.download_button(
            "검색 결과 JSON", json.dumps(search_result, ensure_ascii=False, indent=2),
            file_name="evidence_search.json", mime="application/json",
        )

with agent_tab:
    st.subheader("근거를 먼저 찾고 답하는 로컬 AI Agent")
    st.markdown("<span class='soft-badge policy-badge'>Ollama localhost</span>"
                "<span class='soft-badge'>근거 인용 필수</span>"
                "<span class='soft-badge'>외부 API 차단</span>", unsafe_allow_html=True)
    st.caption("예: ‘최초 관측 이후 90일 안에 어떤 의회로 확장됐어?’, ‘주요 대응 정책을 근거와 함께 비교해줘’")
    if "agent_messages" not in st.session_state:
        st.session_state["agent_messages"] = []
    for message in st.session_state["agent_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    with st.form("agent_question", clear_on_submit=True):
        question = st.text_area("질문", height=85, placeholder="이 이슈가 처음 등장한 곳과 후속 논의 내용을 근거와 함께 설명해줘")
        use_live_search = st.checkbox("질문마다 pgvector 근거 검색", value=True)
        ask = st.form_submit_button("Agent에게 질문", type="primary")
    if ask and question.strip():
        history = list(st.session_state["agent_messages"])
        service = None
        if use_live_search:
            try:
                service = load_search_service()
            except Exception as error:
                st.warning(f"pgvector 연결 실패로 고정 이슈 근거를 사용합니다: {error}")
        agent = LocalCouncilAgent(
            issue_payload=payload, search_service=service,
            ollama_url=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
            model=os.environ.get("COUNCIL_AGENT_MODEL", "qwen3:8b"),
        )
        with st.spinner("근거를 찾고 로컬 모델이 답변을 작성하는 중입니다…"):
            response = agent.ask(question, history=history)
        st.session_state["agent_messages"].extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": response["answer"]},
        ])
        st.session_state["agent_last_response"] = response
        st.rerun()
    last_agent = st.session_state.get("agent_last_response")
    if last_agent:
        with st.expander("Agent가 사용한 원문 근거", expanded=True):
            render_evidence(last_agent["citations"], expanded=1)
        st.caption(
            f"검색: {last_agent['retrieval_mode']} · 생성: {last_agent['generation_mode']}"
        )
        if last_agent.get("warning"):
            st.info("Ollama 미연결 상태라 구조화 요약으로 대체했습니다. `ollama serve`와 모델 설치 상태를 확인하세요.")

with external_tab:
    st.subheader("외부 공개지표 참고 비교")
    st.warning("이 영역의 지표는 설명용 참고 자료입니다. 이슈 탐지·전파 계산·2026-01 예측에는 절대 입력하지 않습니다.")
    external = load_external_context()
    if external is None:
        st.info("검증된 외부지표가 아직 없습니다. 템플릿을 채운 뒤 아래 명령으로 검증하세요.")
        st.code("python src/29_validate_external_context.py --input data/external/context.csv")
        template = (PROJECT_ROOT / "config" / "external_context_template.csv").read_text(encoding="utf-8")
        st.download_button("외부지표 CSV 템플릿", template,
                           file_name="external_context_template.csv", mime="text/csv")
    else:
        st.success(
            f"검증 통과 {external['summary']['accepted_rows']:,}행 · "
            f"지표 {external['summary']['metric_count']:,}개 · 분석 기준일 {external['analysis_cutoff']}"
        )
        frame = pd.DataFrame(external["rows"])
        selected_metric = st.selectbox("참고 지표", sorted(frame["metric_name"].unique()))
        metric_frame = frame[frame["metric_name"] == selected_metric].copy()
        metric_frame["period"] = pd.to_datetime(metric_frame["period"])
        chart_left, chart_right = st.columns(2, gap="large")
        with chart_left:
            st.markdown("**외부 공개지표 원값**")
            st.line_chart(metric_frame.set_index("period")[["value"]], height=280)
        with chart_right:
            st.markdown("**회의록 언급량과 정규화 비교**")
            issue_frame = pd.DataFrame(payload["monthly_series"])[["month", "mention_count"]]
            issue_frame["period"] = pd.to_datetime(issue_frame["month"])
            issue_frame["회의록 언급 지수"] = (
                issue_frame["mention_count"]
                / max(float(issue_frame["mention_count"].max()), 1.0) * 100
            )
            external_monthly = (
                metric_frame.set_index("period")["value"].resample("MS").mean().dropna()
            )
            external_index = external_monthly / max(float(external_monthly.max()), 1e-12) * 100
            combined = pd.concat([
                issue_frame.set_index("period")["회의록 언급 지수"],
                external_index.rename("외부지표 지수"),
            ], axis=1)
            st.line_chart(combined, color=["#eb604f", "#747b59"], height=280)
        st.dataframe(metric_frame[["period", "region", "value", "unit", "source_name", "data_cutoff"]],
                     hide_index=True, use_container_width=True)
        st.caption("두 계열은 각각 최대값=100으로 정규화한 시각 비교이며 상관·인과를 주장하지 않습니다. 출처 URL과 기준일을 함께 보존하고, 제공 데이터 기간 이후 관측은 검증 단계에서 제외합니다.")

with method_tab:
    st.subheader("재현 가능한 분석 파이프라인")
    st.markdown("""
1. 수정된 TSV와 LLM 메타데이터를 `row_id`로 결합해 canonical 데이터를 생성합니다.
2. 발화 경계를 보존한 부모·자식 청크를 만들고 품질 플래그를 기록합니다.
3. Arctic 한국어 임베딩으로 901,358개 청크를 벡터화합니다.
4. pgvector HNSW에서 후보를 검색하고 부모 문맥을 복원합니다.
5. 검수 seed 중심으로 후보를 확장하고, 동일 부모·동일 원문을 제거한 뒤 군집화합니다.
6. 이슈 ID를 고정하고 최초 관측·후속 관측 순서·월별 변화를 계산합니다.
7. 시간순 백테스트로 단순 기준선을 선택하고 예측 구간과 한계를 함께 표시합니다.
8. 로컬 Agent는 검색된 원문과 고정 이슈 결과만 사용해 답변합니다.
    """)
    st.markdown("**현재 중간본의 명시적 한계**")
    for disclaimer in payload.get("disclaimers", []):
        st.caption(f"• {disclaimer}")
    st.download_button(
        "현재 이슈 결과 JSON", json.dumps(payload, ensure_ascii=False, indent=2),
        file_name=f"{issue['issue_id']}.json", mime="application/json",
    )

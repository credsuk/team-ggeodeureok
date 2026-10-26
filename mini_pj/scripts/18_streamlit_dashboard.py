# -*- coding: utf-8 -*-
"""
18_streamlit_dashboard.py

정책 이슈 검색 결과(15~17단계 산출물)를 시각화하는 Streamlit 대시보드
- 검색창
- 최초 등장 시점 및 의회
- 지역별 흐름(지도/막대)
- 월별 변화(라인/히트맵)
- AI 정책 해석 및 향후 예측
- 근거 회의록

실행:
    streamlit run 18_streamlit_dashboard.py
"""

from pathlib import Path
import json
import re
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="지방의회 정책 이슈 분석 대시보드",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(r"C:\mini_pj\results\policy_issue_search")

# =========================
# 공통 유틸
# =========================
REGION_COORDS = {
    "서울": {"lat": 37.5665, "lon": 126.9780},
    "부산": {"lat": 35.1796, "lon": 129.0756},
    "대구": {"lat": 35.8714, "lon": 128.6014},
    "인천": {"lat": 37.4563, "lon": 126.7052},
    "광주": {"lat": 35.1595, "lon": 126.8526},
    "대전": {"lat": 36.3504, "lon": 127.3845},
    "울산": {"lat": 35.5384, "lon": 129.3114},
    "세종": {"lat": 36.4800, "lon": 127.2890},
    "경기": {"lat": 37.4138, "lon": 127.5183},
    "강원": {"lat": 37.8228, "lon": 128.1555},
    "충북": {"lat": 36.8000, "lon": 127.7000},
    "충남": {"lat": 36.5184, "lon": 126.8000},
    "전북": {"lat": 35.7175, "lon": 127.1530},
    "전남": {"lat": 34.8679, "lon": 126.9910},
    "경북": {"lat": 36.4919, "lon": 128.8889},
    "경남": {"lat": 35.4606, "lon": 128.2132},
    "제주": {"lat": 33.4996, "lon": 126.5312},
}

REGION_ORDER = [
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
]


def find_issue_dirs(base_dir: Path):
    if not base_dir.exists():
        return []
    return sorted([p for p in base_dir.iterdir() if p.is_dir()], reverse=True)


def extract_issue_name(folder_name: str):
    # 예: 20261025_000801_청년_일자리 -> 청년 일자리
    m = re.match(r"^\d{8}_\d{6}_(.+)$", folder_name)
    if m:
        return m.group(1).replace("_", " ")
    return folder_name.replace("_", " ")


def normalize_region(assembly: str):
    if pd.isna(assembly):
        return "기타"
    s = str(assembly)

    mapping = {
        "서울특별시의회": "서울",
        "부산광역시의회": "부산",
        "대구광역시의회": "대구",
        "인천광역시의회": "인천",
        "광주광역시의회": "광주",
        "대전광역시의회": "대전",
        "울산광역시의회": "울산",
        "세종특별자치시의회": "세종",
        "경기도의회": "경기",
        "강원특별자치도의회": "강원",
        "충청북도의회": "충북",
        "충청남도의회": "충남",
        "전북특별자치도의회": "전북",
        "전라북도의회": "전북",
        "전라남도의회": "전남",
        "경상북도의회": "경북",
        "경상남도의회": "경남",
        "제주특별자치도의회": "제주",
    }

    return mapping.get(s, s.replace("의회", "").replace("광역시", "").replace("특별자치도", "").replace("특별자치시", "").replace("도", ""))


def read_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def safe_read_csv(path: Path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def first_nonempty(*values):
    for v in values:
        if v is not None and str(v).strip() != "":
            return v
    return None


def latest_llm_dir(trend_dir: Path):
    llm_root = trend_dir / "llm_analysis"
    if not llm_root.exists():
        return None
    candidates = [p for p in llm_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0]


@st.cache_data(show_spinner=False)
def load_issue_bundle(issue_dir_str: str):
    issue_dir = Path(issue_dir_str)
    trend_dir = issue_dir / "trend_analysis"

    summary = read_json(trend_dir / "분석_요약.json") or {}
    region_df = safe_read_csv(trend_dir / "지역별_정책언급순위.csv")
    monthly_df = safe_read_csv(trend_dir / "월별_정책언급추이.csv")
    region_month_df = safe_read_csv(trend_dir / "지역월별_정책언급추이.csv")
    first_obs_df = safe_read_csv(trend_dir / "지역별_최초관측_직접표현.csv")
    evidence_df = safe_read_csv(trend_dir / "시점별_근거후보.csv")
    expression_df = safe_read_csv(trend_dir / "검색표현별_일치건수.csv")
    review_df = safe_read_csv(trend_dir / "검색근거_검토구분.csv")

    llm_dir = latest_llm_dir(trend_dir)
    llm_result = {}
    llm_input = {}
    llm_evidence = pd.DataFrame()
    model_name = "미확인"
    if llm_dir:
        llm_result = read_json(llm_dir / "정책_LLM_분석결과.json") or {}
        llm_input = read_json(llm_dir / "LLM_입력요약.json") or {}
        llm_evidence = safe_read_csv(llm_dir / "LLM_입력근거후보.csv")
        model_name = llm_result.get("model", llm_input.get("model", llm_dir.name))

    # 지역명 표준화
    if not region_df.empty and "assembly" in region_df.columns:
        region_df["region"] = region_df["assembly"].apply(normalize_region)

    if not first_obs_df.empty and "assembly" in first_obs_df.columns:
        first_obs_df["region"] = first_obs_df["assembly"].apply(normalize_region)

    if not evidence_df.empty and "assembly" in evidence_df.columns:
        evidence_df["region"] = evidence_df["assembly"].apply(normalize_region)

    if not region_month_df.empty and "assembly" in region_month_df.columns:
        region_month_df["region"] = region_month_df["assembly"].apply(normalize_region)

    return {
        "issue_dir": issue_dir,
        "summary": summary,
        "region_df": region_df,
        "monthly_df": monthly_df,
        "region_month_df": region_month_df,
        "first_obs_df": first_obs_df,
        "evidence_df": evidence_df,
        "expression_df": expression_df,
        "review_df": review_df,
        "llm_result": llm_result,
        "llm_input": llm_input,
        "llm_evidence": llm_evidence,
        "model_name": model_name,
    }


def build_map_df(region_df: pd.DataFrame, first_obs_df: pd.DataFrame):

    if region_df.empty:
        return pd.DataFrame()

    df = region_df.copy()

    if "region" not in df.columns:
        df["region"] = df["assembly"].apply(normalize_region)

    # 지도에서 사용할 직접 표현 비율 생성
    if "direct_rate_pct" not in df.columns:

        if "direct" in df.columns and "all_chunks" in df.columns:
            df["direct_rate_pct"] = (
                pd.to_numeric(df["direct"], errors="coerce")
                / pd.to_numeric(df["all_chunks"], errors="coerce").replace(0, float("nan"))
                * 100
            )
        else:
            df["direct_rate_pct"] = 0.0

    # 지역별 최초 관측 날짜
    df["first_date"] = ""

    if not first_obs_df.empty:

        first_df = first_obs_df.copy()

        if "region" not in first_df.columns:
            first_df["region"] = first_df["assembly"].apply(normalize_region)

        date_col = next(
            (
                c for c in [
                    "first_direct_date",
                    "first_observed_date",
                    "first_date",
                    "date"
                ]
                if c in first_df.columns
            ),
            None
        )

        if date_col is not None:
            first_map = first_df.groupby("region")[date_col].min()
            df["first_date"] = df["region"].map(first_map).fillna("")

    # 지역별 지도 좌표
    df["lat"] = df["region"].map(
        lambda x: REGION_COORDS.get(x, {}).get("lat")
    )

    df["lon"] = df["region"].map(
        lambda x: REGION_COORDS.get(x, {}).get("lon")
    )

    return df.dropna(subset=["lat", "lon"])


def make_region_map(map_df: pd.DataFrame):
    if map_df.empty:
        return go.Figure()

    hover_columns = {
    "direct": True,
    "direct_rate_pct": ":.4f",
    "first_date": True,
    "lat": False,
    "lon": False,
    }

    if "all_chunks" in map_df.columns:
        hover_columns["all_chunks"] = True

    fig = px.scatter_geo(
        map_df,
        lat="lat",
        lon="lon",
        size="direct",
        color="direct_rate_pct",
        hover_name="assembly",
        hover_data=hover_columns,
        projection="mercator",
        color_continuous_scale="OrRd",
        size_max=35,
    )


    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showcountries=True,
        countrycolor="lightgray",
        lataxis_range=[32, 39],
        lonaxis_range=[124, 132],
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=520,
        coloraxis_colorbar_title="직접표현<br>비율(%)",
    )
    return fig


def make_monthly_chart(monthly_df: pd.DataFrame):
    if monthly_df.empty:
        return go.Figure()

    df = monthly_df.copy()

    x_col = "month"
    direct_col = "direct" if "direct" in df.columns else None
    rate_col = "direct_rate_pct" if "direct_rate_pct" in df.columns else None

    fig = go.Figure()
    if direct_col:
        fig.add_trace(
            go.Bar(
                x=df[x_col],
                y=df[direct_col],
                name="직접표현 건수",
                yaxis="y",
                opacity=0.65
            )
        )
    if rate_col:
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[rate_col],
                name="직접표현 비율(%)",
                mode="lines+markers",
                yaxis="y2"
            )
        )

    fig.update_layout(
        title="월별 정책 언급 추이",
        xaxis=dict(title="월"),
        yaxis=dict(title="건수"),
        yaxis2=dict(title="비율(%)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12, x=0),
        height=440,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


def make_region_bar(region_df: pd.DataFrame):
    if region_df.empty:
        return go.Figure()

    df = region_df.copy()
    if "region" not in df.columns:
        df["region"] = df["assembly"].apply(normalize_region)

    sort_col = "direct_rate_pct" if "direct_rate_pct" in df.columns else "direct"
    df = df.sort_values(sort_col, ascending=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["direct_rate_pct"] if "direct_rate_pct" in df.columns else df["direct"],
            y=df["region"],
            orientation="h",
            text=df["direct"] if "direct" in df.columns else None,
            name="직접표현 비율(%)",
        )
    )
    fig.update_layout(
        title="지역별 정책 언급 비교",
        xaxis_title="직접표현 비율(%)" if "direct_rate_pct" in df.columns else "직접표현 건수",
        yaxis_title="지역",
        height=520,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


def make_heatmap(region_month_df: pd.DataFrame):
    if region_month_df.empty:
        return go.Figure()

    df = region_month_df.copy()
    if "region" not in df.columns:
        df["region"] = df["assembly"].apply(normalize_region)

    value_col = "direct_rate_pct" if "direct_rate_pct" in df.columns else "direct"
    pivot = df.pivot_table(
        index="region",
        columns="month",
        values=value_col,
        aggfunc="sum"
    )

    ordered_index = [r for r in REGION_ORDER if r in pivot.index] + [r for r in pivot.index if r not in REGION_ORDER]
    pivot = pivot.reindex(ordered_index)

    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        labels=dict(color="비율(%)" if value_col == "direct_rate_pct" else "건수")
    )
    fig.update_layout(
        title="지역×월 정책 언급 히트맵",
        height=520,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


def render_llm_section(llm_result: dict, llm_input: dict, model_name: str):
    st.subheader("🤖 AI 정책 해석 및 향후 예측")

    if not llm_result:
        st.warning("LLM 분석 결과 파일을 찾지 못했습니다.")
        return

    st.caption(
        f"모델: {model_name} | 분석 기준일: {llm_result.get('analysis_cutoff', llm_input.get('corpus_end', '미상'))} | "
        f"생성 시간: {llm_result.get('generation_seconds', '미상')}초"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**관측 사실 요약**")
        st.info(llm_result.get("observed_summary", "없음"))

        st.markdown("**시기별 변화 해석**")
        st.write(llm_result.get("time_change_interpretation", "없음"))

        st.markdown("**지역별 해석**")
        st.write(llm_result.get("regional_interpretation", "없음"))

    with col2:
        st.markdown("**향후 예측 (조건부 전망)**")
        st.success(llm_result.get("forecast", "없음"))

        st.markdown("**예측 근거**")
        st.write(llm_result.get("forecast_basis", "없음"))

        if llm_result.get("uncertainties"):
            st.markdown("**불확실성**")
            for x in llm_result.get("uncertainties", []):
                st.write(f"- {x}")

        if llm_result.get("needs_verification"):
            st.markdown("**추가 검증 필요**")
            for x in llm_result.get("needs_verification", []):
                st.write(f"- {x}")

    with st.expander("LLM 분석 결과 JSON 보기"):
        st.json(llm_result)


def render_evidence_section(llm_evidence: pd.DataFrame, evidence_df: pd.DataFrame):
    st.subheader("📚 근거 회의록")

    tabs = st.tabs(["LLM 입력 근거", "시점별 근거 후보"])

    with tabs[0]:
        if llm_evidence.empty:
            st.info("LLM 입력 근거 파일이 없습니다.")
        else:
            st.caption("LLM이 실제로 참고한 대표 근거 후보")
            show_cols = [c for c in [
                "evidence_id", "assembly", "date", "meeting_title", "chunk_id", "text"
            ] if c in llm_evidence.columns]
            view_df = llm_evidence[show_cols].copy()
            st.dataframe(view_df, use_container_width=True, height=320)

            if "text" in llm_evidence.columns:
                for _, row in llm_evidence.head(5).iterrows():
                    label = f"{row.get('evidence_id','-')} | {row.get('assembly','-')} | {row.get('date','-')}"
                    with st.expander(label):
                        st.write(f"**회의명**: {row.get('meeting_title', '-')}")
                        st.write(f"**chunk_id**: `{row.get('chunk_id', '-')}`")
                        st.write(row.get("text", ""))

    with tabs[1]:
        if evidence_df.empty:
            st.info("시점별 근거 후보 파일이 없습니다.")
        else:
            show_cols = [c for c in [
                "month", "assembly", "date", "meeting_title", "chunk_id", "text"
            ] if c in evidence_df.columns]
            st.dataframe(evidence_df[show_cols], use_container_width=True, height=360)


# =========================
# 사이드바
# =========================
st.sidebar.title("⚙️ 설정")

if not BASE_DIR.exists():
    st.error(f"결과 폴더가 없습니다: {BASE_DIR}")
    st.stop()

issue_dirs = find_issue_dirs(BASE_DIR)
if not issue_dirs:
    st.error("정책 이슈 검색 결과 폴더를 찾지 못했습니다.")
    st.stop()

issue_label_map = {d.name: extract_issue_name(d.name) for d in issue_dirs}
all_options = [d.name for d in issue_dirs]

search_keyword = st.sidebar.text_input("검색창", value="청년 일자리", help="이슈 키워드 또는 질문을 입력하세요.")
filtered_options = [name for name in all_options if search_keyword.replace(" ", "_") in name or search_keyword.replace(" ", "") in name.replace("_", "")]
if not filtered_options:
    filtered_options = all_options

selected_folder_name = st.sidebar.selectbox(
    "결과 폴더 선택",
    filtered_options,
    index=0
)

selected_issue_dir = BASE_DIR / selected_folder_name
bundle = load_issue_bundle(str(selected_issue_dir))

summary = bundle["summary"]
region_df = bundle["region_df"]
monthly_df = bundle["monthly_df"]
region_month_df = bundle["region_month_df"]
first_obs_df = bundle["first_obs_df"]
evidence_df = bundle["evidence_df"]
expression_df = bundle["expression_df"]
review_df = bundle["review_df"]
llm_result = bundle["llm_result"]
llm_input = bundle["llm_input"]
llm_evidence = bundle["llm_evidence"]
model_name = bundle["model_name"]

query = first_nonempty(
    llm_result.get("query") if llm_result else None,
    llm_input.get("query") if llm_input else None,
    summary.get("query") if isinstance(summary, dict) else None,
    extract_issue_name(selected_folder_name)
)

# =========================
# 메인 화면
# =========================
st.title("🏛️ 지방의회 정책 이슈 분석 대시보드")
st.caption("국회도서관 지방의회 회의록(2024~2025) 기반 탐색형 분석 화면")

st.warning(
    "⚠️ 주의: 현재 화면의 '타 지방의회 흐름'은 정책 관련 표현의 관측 흐름입니다. "
    "동일 정책의 실제 전파·영향 관계는 아직 검증되지 않았습니다."
)

# KPI
col1, col2, col3, col4 = st.columns(4)

total_hits = summary.get("total_lexical_matches", None)
direct_hits = summary.get("direct_expression_matches", None)
first_date = summary.get("first_direct_observed_date", None)
first_assembly = summary.get("first_direct_observed_assembly", None)

# 백업 추출
if first_date is None and not first_obs_df.empty:
    try:
        row = first_obs_df.sort_values("first_direct_date").iloc[0]
        first_date = row.get("first_direct_date", "")
        first_assembly = row.get("assembly", "")
    except Exception:
        pass

with col1:
    st.metric("이슈 키워드", query)

with col2:
    st.metric("직접 표현 일치", f"{direct_hits:,}" if direct_hits is not None else "미상")

with col3:
    st.metric("전체 어휘 일치", f"{total_hits:,}" if total_hits is not None else "미상")

with col4:
    first_obs_text = f"{first_date}\n{first_assembly}" if first_date or first_assembly else "미상"
    st.metric("최초 등장 시점", first_obs_text)

# 검색 표현 현황
with st.expander("🔎 검색 표현 현황 보기"):
    if expression_df.empty:
        st.info("검색 표현 집계 파일이 없습니다.")
    else:
        st.dataframe(expression_df, use_container_width=True)

# 상단 개요
st.markdown("---")
left, right = st.columns([1.2, 1])

with left:
    st.subheader("🗺️ 타 지방의회 관측 흐름")
    map_df = build_map_df(region_df, first_obs_df)
    map_fig = make_region_map(map_df)
    st.plotly_chart(map_fig, use_container_width=True)

with right:
    st.subheader("📌 핵심 요약")
    if llm_result:
        st.markdown(f"**관측 사실**  \n{llm_result.get('observed_summary', '-')}")
        st.markdown(f"**지역 해석**  \n{llm_result.get('regional_interpretation', '-')}")
        st.markdown(f"**향후 예측**  \n{llm_result.get('forecast', '-')}")
    else:
        st.info("LLM 분석 결과를 찾지 못했습니다.")

    if not first_obs_df.empty:
        st.markdown("**지역별 최초 관측 일부**")
        show_cols = [c for c in ["assembly", "first_direct_date", "chunk_id"] if c in first_obs_df.columns]
        st.dataframe(first_obs_df[show_cols].head(10), use_container_width=True, height=240)

# 중단: 지역/시간 분석
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs(["지역별 비교", "시간 흐름", "AI 해석·예측", "근거 회의록"])

with tab1:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.plotly_chart(make_region_bar(region_df), use_container_width=True)

    with c2:
        if not region_df.empty:
            show_cols = [c for c in ["assembly", "direct", "all_chunks", "direct_rate_pct"] if c in region_df.columns]
            df_show = region_df[show_cols].copy()
            if "direct_rate_pct" in df_show.columns:
                df_show = df_show.sort_values("direct_rate_pct", ascending=False)
            st.dataframe(df_show, use_container_width=True, height=500)
        else:
            st.info("지역별 통계 파일이 없습니다.")

with tab2:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.plotly_chart(make_monthly_chart(monthly_df), use_container_width=True)

    with c2:
        st.plotly_chart(make_heatmap(region_month_df), use_container_width=True)

    if not monthly_df.empty:
        with st.expander("월별 통계표 보기"):
            st.dataframe(monthly_df, use_container_width=True)

with tab3:
    render_llm_section(llm_result, llm_input, model_name)

with tab4:
    render_evidence_section(llm_evidence, evidence_df)

# 하단 부가 정보
st.markdown("---")
st.subheader("🧪 분석 한계 및 해석 주의")

limitation_msgs = []
if llm_result.get("limitations"):
    limitation_msgs.extend(llm_result["limitations"])
else:
    limitation_msgs = [
        "직접 표현 일치는 정책 관련성 확정과 동일하지 않습니다.",
        "지역 간 전파·영향 관계는 아직 검증되지 않았습니다.",
        "LLM 분석은 회의록 미리보기 기반이며 원문 전체 검증이 필요합니다.",
    ]

for msg in limitation_msgs:
    st.write(f"- {msg}")

st.caption(f"현재 로드된 결과 폴더: {selected_issue_dir}")
#!/usr/bin/env python3
"""광역 지방의회 정책 근거 검색 Streamlit 데모."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from council_vector_runtime import CouncilSearchService, SearchFilters  # noqa: E402


COUNCIL_COORDINATES = {
    "서울특별시의회": (37.5665, 126.9780), "부산광역시의회": (35.1796, 129.0756),
    "대구광역시의회": (35.8714, 128.6014), "인천광역시의회": (37.4563, 126.7052),
    "광주광역시의회": (35.1595, 126.8526), "대전광역시의회": (36.3504, 127.3845),
    "울산광역시의회": (35.5384, 129.3114), "세종특별자치시의회": (36.4800, 127.2890),
    "경기도의회": (37.2636, 127.0286), "강원특별자치도의회": (37.8854, 127.7298),
    "충청북도의회": (36.6357, 127.4913), "충청남도의회": (36.6588, 126.6728),
    "전북특별자치도의회": (35.8203, 127.1088), "전라남도의회": (34.8161, 126.4629),
    "경상북도의회": (36.5760, 128.5058), "경상남도의회": (35.2383, 128.6924),
    "제주특별자치도의회": (33.4996, 126.5312),
}


def path_setting(name: str, relative: str) -> Path:
    return Path(os.environ.get(name, str(PROJECT_ROOT / relative))).expanduser().resolve()


EMBEDDING_DIR = path_setting("COUNCIL_EMBEDDING_DIR", "data/embeddings/arctic_raw_v1")
PARENT_PATH = path_setting("COUNCIL_PARENT_PATH", "data/canonical_v3/chunks_v1/parents.parquet")
CHROMA_DIR = path_setting("COUNCIL_CHROMA_DIR", "data/chroma/arctic_raw_v1")


@st.cache_data(show_spinner=False)
def filter_options(metadata_path: str):
    connection = duckdb.connect()
    councils = [row[0] for row in connection.execute(
        "SELECT DISTINCT council FROM read_parquet(?) WHERE council IS NOT NULL ORDER BY 1", [metadata_path]
    ).fetchall()]
    years = [int(row[0]) for row in connection.execute(
        "SELECT DISTINCT year FROM read_parquet(?) WHERE year IS NOT NULL ORDER BY 1", [metadata_path]
    ).fetchall()]
    committees = [row[0] for row in connection.execute(
        "SELECT committee FROM read_parquet(?) WHERE committee IS NOT NULL AND trim(committee) != '' GROUP BY 1 ORDER BY count(*) DESC LIMIT 100",
        [metadata_path],
    ).fetchall()]
    connection.close()
    return councils, years, committees


@st.cache_resource(show_spinner="검색 모델과 DB를 불러오는 중입니다…")
def load_service(backend: str):
    return CouncilSearchService(
        backend=backend, embedding_dir=EMBEDDING_DIR, parent_path=PARENT_PATH,
        chroma_dir=CHROMA_DIR,
        collection=os.environ.get("COUNCIL_CHROMA_COLLECTION", "council_arctic_raw_v1"),
        pg_dsn=os.environ.get("PGVECTOR_DSN"),
        pg_table=os.environ.get("COUNCIL_PG_TABLE", "council_arctic_raw_v1"),
        pg_ef_search=int(os.environ.get("COUNCIL_PG_EF_SEARCH", "200")),
        device=os.environ.get("COUNCIL_DEVICE", "auto"),
        model_cache=path_setting("COUNCIL_MODEL_CACHE", "models/huggingface"),
        local_files_only=os.environ.get("COUNCIL_ALLOW_DOWNLOAD", "0") != "1",
    )


def map_frame(results: list[dict]) -> pd.DataFrame:
    records = []
    for row in results:
        council = str(row["metadata"].get("council") or "")
        if council not in COUNCIL_COORDINATES:
            continue
        latitude, longitude = COUNCIL_COORDINATES[council]
        records.append({
            "council": council, "lat": latitude, "lon": longitude,
            "score": max(float(row["similarity"]), 0.01),
            "marker_size": 1200 + max(float(row["similarity"]), 0.0) * 2400,
            "rank": row["rank"],
        })
    return pd.DataFrame(records)


st.set_page_config(page_title="지방의회 정책 흐름 탐색", page_icon="🏛️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1500px;}
.evidence {border-left: 4px solid #2563eb; padding: .7rem 1rem; background:#f8fafc; border-radius:8px;}
.small-note {color:#64748b; font-size:.88rem;}
</style>
""", unsafe_allow_html=True)
st.title("🏛️ 광역 지방의회 정책 근거 탐색")
st.caption("외부 API 전송 없이 로컬 임베딩과 벡터 DB로 회의록 근거를 검색합니다.")

if not (EMBEDDING_DIR / "chunk_metadata.parquet").is_file():
    st.error(f"임베딩 metadata가 없습니다: {EMBEDDING_DIR}")
    st.stop()

councils, years, committees = filter_options(str(EMBEDDING_DIR / "chunk_metadata.parquet"))
with st.sidebar:
    st.header("검색 설정")
    backend = st.selectbox("검색 DB", ["chroma", "pgvector", "exact"], index=0)
    top_k = st.slider("결과 수", 3, 30, 10)
    candidate_k = st.slider("1차 후보 수", top_k, 300, max(100, top_k), step=10)
    council = st.selectbox("지역", ["전체", *councils])
    year = st.selectbox("연도", ["전체", *years])
    committee = st.selectbox("위원회", ["전체", *committees])
    priorities = st.multiselect("우선순위", ["high", "medium", "low"], default=["high", "medium"])
    include_parent = st.checkbox("부모 문맥 표시", value=True)

query = st.text_input("정책 또는 현안을 입력하세요", value="출산율 저하 대응 정책")
search_clicked = st.button("근거 검색", type="primary", use_container_width=True)

if search_clicked:
    try:
        with st.spinner("회의록 근거를 찾고 있습니다…"):
            result = load_service(backend).search(
                query=query, top_k=top_k, candidate_k=candidate_k,
                filters=SearchFilters(
                    council=None if council == "전체" else council,
                    year=None if year == "전체" else int(year),
                    committee=None if committee == "전체" else committee,
                    priorities=priorities or None,
                ), include_parent=include_parent,
            )
        st.session_state["last_result"] = result
    except Exception as error:
        st.exception(error)

result = st.session_state.get("last_result")
if result:
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("검색 결과", f"{result['result_count']}건")
    metric2.metric("DB 검색", f"{result['search_ms']:.1f} ms")
    metric3.metric("질의 임베딩", f"{result['query_embedding_ms']:.1f} ms")
    metric4.metric("백엔드", result["backend"])

    left, right = st.columns([1.05, 1.6], gap="large")
    with left:
        st.subheader("지역 분포")
        frame = map_frame(result["results"])
        if frame.empty:
            st.info("지도에 표시할 지역 결과가 없습니다.")
        else:
            st.map(frame, latitude="lat", longitude="lon", size="marker_size", zoom=5)
            counts = frame.groupby("council").size().sort_values(ascending=False)
            st.dataframe(counts.rename("근거 수"), use_container_width=True)
    with right:
        st.subheader("원문 근거")
        for row in result["results"]:
            meta = row["metadata"]
            title = f"{row['rank']}. {meta.get('council', '')} · {meta.get('meeting_date', '')} · {row['similarity']:.3f}"
            with st.expander(title, expanded=row["rank"] <= 3):
                st.markdown(
                    f"**회의록명** {meta.get('meeting_name', '미상')}  \n"
                    f"**위원회** {meta.get('committee', '미상')}  \n"
                    f"**안건** {meta.get('agenda_title', '미상')}"
                )
                with st.container(border=True):
                    st.write(row["text"])
                if row.get("parent_context"):
                    with st.popover("앞뒤 문맥 보기"):
                        st.write(row["parent_context"])
    st.download_button(
        "검색 결과 JSON 다운로드", json.dumps(result, ensure_ascii=False, indent=2),
        file_name="council_search_result.json", mime="application/json",
    )

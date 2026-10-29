#!/usr/bin/env python3
"""25번 다중 이슈 후보를 사람이 검수하고 결정을 저장하는 Streamlit 화면."""

from __future__ import annotations

import csv
import json
import os
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = PROJECT_ROOT / "outputs" / "issues" / "candidates_v1" / "issue_review_queue.csv"
DECISIONS = ["pending_human_review", "approve", "revise", "reject"]


def configured_queue() -> Path:
    return Path(
        os.environ.get("COUNCIL_ISSUE_REVIEW_QUEUE", str(DEFAULT_QUEUE))
    ).expanduser().resolve()


@st.cache_data(show_spinner=False)
def read_queue(path_text: str, modified_ns: int) -> list[dict[str, str]]:
    del modified_ns
    with Path(path_text).open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


@st.cache_data(show_spinner=False)
def read_cluster(path_text: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    return json.loads(Path(path_text).read_text(encoding="utf-8"))


def resolve_cluster(row: dict[str, str], queue: Path) -> Path:
    recorded = Path(str(row.get("cluster_json") or "")).expanduser()
    if recorded.is_file():
        return recorded.resolve()
    return (queue.parent / row["issue_key"] / "03_cluster.json").resolve()


def atomic_write_queue(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("저장할 검수 행이 없습니다")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def evidence_rows(cluster: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    members = list(cluster.get("members") or [])
    members.sort(
        key=lambda row: (
            not bool(row.get("is_seed")),
            -float(row.get("centroid_similarity") or 0.0),
        )
    )
    return members[:limit]


st.set_page_config(
    page_title="이슈 후보 검수 | 의정이슈레이더", page_icon="✅", layout="wide"
)
st.markdown("""
<style>
.stApp{background:#f7f2e8}.block-container{max-width:1500px;padding-top:1.2rem}
.hero{padding:1.35rem 1.6rem;border-radius:21px;background:linear-gradient(125deg,#424936,#747b59);color:#fffaf0;margin-bottom:1rem}
.hero h1{color:#fffaf0;margin:0;font-size:2rem}.hero p{margin:.3rem 0 0;color:#ede9db}
.evidence{background:#fffaf3;border:1px solid #ddd5c3;border-left:4px solid #eb604f;border-radius:12px;padding:1rem;line-height:1.65;margin:.5rem 0}
.meta{color:#717565;font-size:.85rem;margin-bottom:.45rem}
div[data-testid="stMetric"]{background:#fffaf3;border:1px solid #ded6c5;border-radius:14px;padding:.75rem 1rem}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero"><h1>이슈 후보 검수</h1>
<p>자동 군집 결과를 원문 근거로 확인하고 승인·수정·반려 결정을 저장합니다.</p></div>
""", unsafe_allow_html=True)

queue_path = configured_queue()
if not queue_path.is_file():
    st.error(f"검수 대기열이 없습니다: {queue_path}")
    st.code("export COUNCIL_ISSUE_REVIEW_QUEUE=outputs/issues/후보폴더/issue_review_queue.csv")
    st.stop()

rows = read_queue(str(queue_path), queue_path.stat().st_mtime_ns)
if not rows:
    st.error("검수 대기열이 비었습니다")
    st.stop()

status_counts = pd.Series([
    row.get("review_decision") or "pending_human_review" for row in rows
]).value_counts()
a, b, c, d = st.columns(4)
a.metric("전체 후보", len(rows))
b.metric("승인", int(status_counts.get("approve", 0)))
c.metric("수정 필요", int(status_counts.get("revise", 0)))
d.metric("반려", int(status_counts.get("reject", 0)))

with st.sidebar:
    st.markdown("### TEAM 꺼드럭's")
    st.caption("의정이슈레이더 · Human-in-the-loop")
    only_pending = st.checkbox("미검수만 보기", value=False)
    st.markdown("**승인 기준**")
    st.caption("• 대표 근거가 이슈와 직접 관련")
    st.caption("• seed coverage 80% 이상 권장")
    st.caption("• 최소 2개 의회에서 관측")
    st.caption("• 하나의 의회·안건에 과도하게 집중되지 않음")
    st.caption("• 최초 날짜 근거가 정책 본문임")
    st.download_button(
        "검수 CSV 다운로드", queue_path.read_bytes(),
        file_name=queue_path.name, mime="text/csv", use_container_width=True,
    )

visible_rows = rows
if only_pending:
    visible_rows = [
        row for row in rows
        if (row.get("review_decision") or "pending_human_review")
        == "pending_human_review"
    ]
if not visible_rows:
    st.success("조건에 맞는 미검수 후보가 없습니다")
    st.stop()

selected_key = st.selectbox(
    "검수할 이슈",
    [row["issue_key"] for row in visible_rows],
    format_func=lambda key: next(
        f"{row['planned_issue_name']} ({key})" for row in visible_rows
        if row["issue_key"] == key
    ),
)
row = next(item for item in rows if item["issue_key"] == selected_key)
cluster_path = resolve_cluster(row, queue_path)
if not cluster_path.is_file():
    st.error(f"군집 JSON이 없습니다: {cluster_path}")
    st.stop()
cluster = read_cluster(str(cluster_path), cluster_path.stat().st_mtime_ns)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("구성원", row.get("member_count", "-"))
m2.metric("Seed coverage", f"{float(row.get('seed_coverage') or 0):.1%}")
m3.metric("관측 의회", row.get("council_count", "-"))
m4.metric("최초 관측", row.get("first_date", "-"))
m5.metric("상위 의회 집중도", f"{float(row.get('top_council_share') or 0):.1%}")

flags = str(row.get("quality_flags") or "").strip()
if flags:
    st.warning(f"자동 품질 플래그: {flags}")
else:
    st.success("자동 품질 경고 없음")

left, right = st.columns([1.6, 1], gap="large")
with left:
    st.subheader("대표 원문 근거")
    for index, member in enumerate(evidence_rows(cluster), start=1):
        metadata = member.get("metadata") or {}
        seed = "SEED" if member.get("is_seed") else "EXPANDED"
        st.markdown(
            f"<div class='evidence'><div class='meta'><b>{index}. {escape(seed)}</b> · "
            f"{escape(str(metadata.get('council') or '미상'))} · "
            f"{escape(str(metadata.get('meeting_date') or '날짜 미상'))} · "
            f"유사도 {float(member.get('centroid_similarity') or 0):.3f}<br>"
            f"{escape(str(metadata.get('meeting_name') or '회의록명 미상'))} · "
            f"{escape(str(metadata.get('agenda_title') or '안건 미상'))}</div>"
            f"{escape(str(member.get('text') or ''))}</div>",
            unsafe_allow_html=True,
        )
with right:
    st.subheader("검수 결정")
    current_decision = row.get("review_decision") or "pending_human_review"
    if current_decision not in DECISIONS:
        current_decision = "pending_human_review"
    decision = st.radio(
        "결정", DECISIONS, index=DECISIONS.index(current_decision),
        format_func=lambda value: {
            "pending_human_review": "미검수",
            "approve": "승인",
            "revise": "수정 후 재검수",
            "reject": "반려",
        }[value],
    )
    note = st.text_area(
        "검수 메모", value=row.get("review_note") or "", height=150,
        placeholder="예: 지역소멸보다 저출생 근거가 과다해 질의 수정 필요",
    )
    st.caption("승인된 이슈만 30번 단계에서 registry·타임라인·예측·웹 결과로 생성됩니다.")
    if st.button("검수 결과 저장", type="primary", use_container_width=True):
        for item in rows:
            if item["issue_key"] == selected_key:
                item["review_decision"] = decision
                item["review_status"] = (
                    "pending_human_review"
                    if decision == "pending_human_review" else "human_reviewed"
                )
                item["review_note"] = note.strip()
                break
        atomic_write_queue(queue_path, rows)
        read_queue.clear()
        st.success(f"저장 완료: {selected_key} → {decision}")
        st.rerun()

st.caption(f"검수 파일: {queue_path}")

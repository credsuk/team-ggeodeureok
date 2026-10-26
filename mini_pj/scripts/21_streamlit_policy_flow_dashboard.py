# -*- coding: utf-8 -*-
"""21_streamlit_policy_flow_dashboard.py

Streamlit MVP: 기존 15~17번 이슈 분석 + 20번 정책 유사 논의 검토 결과 연결.
새 검색/LLM 호출이나 전파 확정은 수행하지 않습니다.

실행 (Windows / mini):
  streamlit run "C:\\mini_pj\\scripts\\21_streamlit_policy_flow_dashboard.py"
선택적으로:
  streamlit run ... -- --base-dir "C:\\mini_pj\\results\\policy_issue_search"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DEFAULT_BASE = Path(r"C:\mini_pj\results\policy_issue_search")

REGION_COORDS = {
    "서울": (37.5665,126.9780), "부산": (35.1796,129.0756),
    "대구": (35.8714,128.6014), "인천": (37.4563,126.7052),
    "광주": (35.1595,126.8526), "대전": (36.3504,127.3845),
    "울산": (35.5384,129.3114), "세종": (36.4800,127.2890),
    "경기": (37.4138,127.5183), "강원": (37.8228,128.1555),
    "충북": (36.8000,127.7000), "충남": (36.5184,126.8000),
    "전북": (35.7175,127.1530), "전남": (34.8679,126.9910),
    "경북": (36.4919,128.8889), "경남": (35.4606,128.2132),
    "제주": (33.4996,126.5312),
}
REGION_NAMES = {
    "서울특별시의회":"서울", "부산광역시의회":"부산", "대구광역시의회":"대구",
    "인천광역시의회":"인천", "광주광역시의회":"광주", "대전광역시의회":"대전",
    "울산광역시의회":"울산", "세종특별자치시의회":"세종", "경기도의회":"경기",
    "강원특별자치도의회":"강원", "강원도의회":"강원", "충청북도의회":"충북",
    "충청남도의회":"충남", "전북특별자치도의회":"전북", "전라북도의회":"전북",
    "전라남도의회":"전남", "경상북도의회":"경북", "경상남도의회":"경남",
    "제주특별자치도의회":"제주", "제주도의회":"제주"
}


def region_name(assembly):
    return REGION_NAMES.get(str(assembly), str(assembly))


def val(row, field, default="-"):
    x = row.get(field, default)
    if x is None or (not isinstance(x, (list, dict)) and pd.isna(x)):
        return default
    return x


def parse_date(s):
    # Supports YYYY-MM-DD and YYYYMMDD numeric dates
    ss = str(s).strip()
    return pd.to_datetime(ss, format="%Y%m%d" if re.fullmatch(r"\d{8}", ss) else None, errors="coerce")


def format_date(s):
    t = parse_date(s)
    return t.strftime("%Y-%m-%d") if not pd.isna(t) else "확인 필요"


def load_csv(path):
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_json(path):
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8-sig") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeError) as exc:
        st.warning(f"JSON 읽기 실패: {path.name} ({exc})")
        return {}


def choose_llm_folder(trend_dir):
    root = trend_dir / "llm_analysis"
    if not root.exists():
        return None
    folders = [p for p in root.iterdir() if p.is_dir() and (p / "정책_LLM_분석결과.json").exists()]
    return sorted(folders, key=lambda p:p.stat().st_mtime, reverse=True)[0] if folders else None


@st.cache_data(show_spinner=False)
def load_bundle(issue_dir_str):
    issue = Path(issue_dir_str)
    trend = issue / "trend_analysis"
    flow = issue / "flow_validation"
    llm_dir = choose_llm_folder(trend)
    return {
        "summary": load_json(trend / "분석_요약.json"),
        "monthly": load_csv(trend / "월별_정책언급추이.csv"),
        "region": load_csv(trend / "지역별_정책언급순위.csv"),
        "region_month": load_csv(trend / "지역월별_정책언급추이.csv"),
        "first": load_csv(trend / "지역별_최초관측_직접표현.csv"),
        "evidence": load_csv(trend / "시점별_근거후보.csv"),
        "flow": load_csv(flow / "정책내용_고유후보_검토결과.csv"),
        "dotted": load_csv(flow / "웹_정책내용유사_점선후보.csv"),
        "confirmed": load_csv(flow / "웹_명시적참고근거_수동확인.csv"),
        "flow_summary": load_json(flow / "검증_요약.json"),
        "manual": load_csv(flow / "수동검토_입력양식.csv"),
        "llm": load_json(llm_dir / "정책_LLM_분석결과.json") if llm_dir else {},
        "llm_evidence": load_csv(llm_dir / "LLM_입력근거후보.csv") if llm_dir else pd.DataFrame(),
        "llm_name": llm_dir.name if llm_dir else "미생성",
    }


def issue_label(folder):
    m = re.match(r"^\d{8}_\d{6}_(.+)$", folder.name)
    return (m.group(1) if m else folder.name).replace("_", " ")


def add_date_fields(df):
    out = df.copy()
    if "earlier_date" in out.columns:
        out["earlier_dt"] = pd.to_datetime(out["earlier_date"], errors="coerce")
    if "later_date" in out.columns:
        out["later_dt"] = pd.to_datetime(out["later_date"], errors="coerce")
    return out


def flow_status(row, manual_map):
    """Only explicitly reviewed evidence may be labeled confirmed.

    Existing empty 'confirmed' CSV is not itself evidence. Never infer diffusion
    from similarity scores or from automatic transfer_status strings.
    """
    gid = str(val(row, "duplicate_group_id", ""))
    m = manual_map.get(gid, {})
    c = str(val(m, "manual_content_verdict", "")).strip()
    t = str(val(m, "manual_transfer_verdict", "")).strip()
    note = str(val(m, "manual_review_note", "")).strip()
    if c in {"상이", "다름", "관련없음", "비유사", "내용상이"}:
        return "검토 결과 제외"
    # Conservative display: manually checked evidence is still a reference clue,
    # not proven causal diffusion. Never display as confirmed diffusion.
    if t in {"명시적참고확인", "참고근거확인"} and note:
        return "명시적 참고 근거 수동확인 (전파 인과 미확정)"
    if c in {"유사확인", "유사", "구체정책유사"}:
        return "정책 내용 유사 수동확인 (전파 미확인)"
    return "자동 유사논의 후보 (전파 미확인)"


def prepare_flow(bundle):
    f = bundle["flow"]
    dotted = bundle["dotted"]
    if f.empty or dotted.empty:
        return pd.DataFrame()
    if "candidate_id" not in f.columns or "candidate_id" not in dotted.columns:
        return pd.DataFrame()
    kept = set(dotted["candidate_id"].astype(str))
    f = f.loc[f["candidate_id"].astype(str).isin(kept)].copy()
    manual = bundle["manual"]
    lookup = {}
    if not manual.empty and "duplicate_group_id" in manual.columns:
        lookup = {str(r["duplicate_group_id"]):r for _,r in manual.iterrows()}
    f["status"] = f.apply(lambda r:flow_status(r, lookup), axis=1)
    f = add_date_fields(f)
    return f.sort_values(["earlier_dt", "later_dt", "candidate_id"])


def timeline_map(region_df, region_month_df, flow, first_df, current_month, color_metric="비율"):
    """Geo points + dotted undirected-looking candidate segments; no arrowheads."""
    fig = go.Figure()
    if region_df.empty or "assembly" not in region_df.columns:
        return fig
    reg = region_df.copy()
    reg["region"] = reg["assembly"].map(region_name)
    reg = reg.loc[reg["region"].isin(REGION_COORDS)].copy()
    if current_month != "전체 기간" and not region_month_df.empty:
        rm = region_month_df.loc[region_month_df["month"].astype(str) <= current_month].copy()
        # cumulative numerator and denominator across selected months
        for c in ("direct", "all_chunks"):
            if c in rm: rm[c] = pd.to_numeric(rm[c], errors="coerce").fillna(0)
        summary = rm.groupby("assembly", as_index=False)[["direct", "all_chunks"]].sum()
        reg = reg.drop(columns=["direct", "all_chunks"], errors="ignore").merge(summary,on="assembly",how="left")
    reg["direct"] = pd.to_numeric(reg.get("direct",0), errors="coerce").fillna(0)
    reg["all_chunks"] = pd.to_numeric(reg.get("all_chunks",0), errors="coerce").fillna(0)
    reg["rate"] = np.where(reg["all_chunks"] > 0, reg["direct"] / reg["all_chunks"] * 100, 0)
    reg["lat"] = reg["region"].map(lambda r:REGION_COORDS[r][0])
    reg["lon"] = reg["region"].map(lambda r:REGION_COORDS[r][1])
    observed = reg.loc[reg["direct"] > 0].copy()
    if not observed.empty:
        texts = [f"{r.assembly}<br>직접 표현 {int(r.direct):,}건<br>해당 기간 대비 {r.rate:.3f}%" for r in observed.itertuples()]
        fig.add_trace(go.Scattergeo(
            lon=observed["lon"],lat=observed["lat"],
            mode="markers+text",text=observed["region"],textposition="top center",
            hovertext=texts,hoverinfo="text",name="지역별 관측",
            marker=dict(size=np.clip(9 + np.sqrt(observed["direct"]) * 1.6, 10, 40),
                        color=observed["rate"],colorscale="YlOrRd",showscale=True,
                        colorbar=dict(title="직접 표현<br>비율 %"),line=dict(width=1,color="white"),opacity=0.9)
        ))
    if not flow.empty:
        f = flow.copy()
        if current_month != "전체 기간":
            cutoff = pd.Period(current_month, freq="M").end_time
            f = f.loc[f["later_dt"] <= cutoff]
        for _,r in f.iterrows():
            a,b=region_name(r["earlier_assembly"]),region_name(r["later_assembly"])
            if a not in REGION_COORDS or b not in REGION_COORDS: continue
            lat1,lon1=REGION_COORDS[a];lat2,lon2=REGION_COORDS[b]
            status=r["status"]
            line_color="#427fb5" if "수동확인" not in status else "#188274"
            label=f"{a} ({r['earlier_date']}) ↔ {b} ({r['later_date']})<br>{r['candidate_id']} | {status}"
            fig.add_trace(go.Scattergeo(
                lon=[lon1,lon2],lat=[lat1,lat2],mode="lines",
                line=dict(width=2,color=line_color,dash="dot"),
                hovertext=[label,label],hoverinfo="text",showlegend=False
            ))
    fig.update_geos(
        projection_type="mercator",showland=True,landcolor="#f2f4f7",
        showcoastlines=True,coastlinecolor="#9ea8b7",
        showcountries=True,countrycolor="#ccd1d8",
        lonaxis_range=[124,132],lataxis_range=[32.5,39]
    )
    fig.update_layout(height=560,margin=dict(l=0,r=0,t=10,b=0),
                      legend=dict(orientation="h",y=1.07),paper_bgcolor="rgba(0,0,0,0)")
    return fig


def evidence_panel(row):
    c1,c2=st.columns(2)
    for col, prefix,title in [(c1,"earlier","먼저 관측된 의회"),(c2,"later","후속 관측된 의회")]:
        with col:
            st.markdown(f"**{title}: {val(row,prefix+'_assembly')}**")
            st.caption(f"{format_date(val(row,prefix+'_date'))} · {val(row,prefix+'_meeting_title')}")
            st.code(str(val(row,prefix+'_chunk_id')),language=None)
            st.markdown("**관련 정책 문장(자동 추출)**")
            st.write(val(row,prefix+'_policy_excerpt',"추출 정보 없음"))
            with st.expander("원문 미리보기 보기 (최대 600자)"):
                st.write(val(row,prefix+'_text_preview',"미리보기 없음"))
    st.caption("원문 미리보기·자동 분류 기준이므로 실제 정책 동일성 및 전파 여부는 별도 원문 검증이 필요합니다.")


def trend_chart(monthly):
    if monthly.empty: return
    d=monthly.sort_values("month")
    fig=go.Figure()
    fig.add_trace(go.Bar(x=d["month"],y=d["direct"],name="직접 표현 일치 건수",opacity=.6))
    fig.add_trace(go.Scatter(x=d["month"],y=d["direct_match_rate_pct"],mode="lines+markers",
                             name="전체 청크 대비 비율(%)",yaxis="y2"))
    fig.update_layout(height=380,margin=dict(l=20,r=20,t=20,b=20),
                      xaxis_title="회의록 월",yaxis_title="청크 수",
                      yaxis2=dict(overlaying="y",side="right",title="비율(%)"),
                      legend=dict(orientation="h",y=1.15))
    st.plotly_chart(fig,use_container_width=True)


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE))
    args,_=parser.parse_known_args()
    base=Path(args.base_dir)

    st.set_page_config(page_title="지방의회 정책 흐름 탐색",page_icon="🏛️",layout="wide")
    st.title("🏛️ 지방의회 정책 이슈 · 유사 논의 흐름")
    st.caption("2024–2025년 광역 지방의회 회의록 · 검색 근거 → 시공간 관측 → 유사 정책 후보 → 조건부 전망")
    if not base.exists():
        st.error(f"15번 검색 결과 폴더를 찾을 수 없습니다: {base}")
        st.stop()
    folders=sorted([p for p in base.iterdir() if p.is_dir() and (p/"trend_analysis"/"분석_요약.json").exists()],
                   reverse=True)
    if not folders:
        st.error("16번 분석 결과가 있는 검색 폴더를 찾지 못했습니다.")
        st.stop()

    st.sidebar.header("🔎 정책 이슈 검색")
    q=st.sidebar.text_input("이슈 키워드 또는 질문",value="청년 일자리")
    matching=[p for p in folders if q.strip().replace(" ","") in issue_label(p).replace(" ","")]
    if q.strip() and not matching:
        st.sidebar.warning("해당 검색어의 사전 분석 결과가 없습니다. 새 질문을 실행하지 않고 기존 결과만 표시합니다.")
    selectable=matching if matching else folders
    selected=st.sidebar.selectbox("분석 결과 선택", selectable, format_func=lambda p:f"{issue_label(p)} · {p.name[:15]}")
    st.sidebar.caption("검색창은 기존 15~20번 산출물을 선택합니다. 새로운 키워드의 실시간 분석은 아직 연결되지 않았습니다.")
    if st.sidebar.button("📂 결과 다시 읽기"):
        load_bundle.clear()
        st.rerun()

    b=load_bundle(str(selected)); summary=b["summary"];llm=b["llm"]
    region=b["region"];monthly=b["monthly"];region_month=b["region_month"]
    first=b["first"]; flow=prepare_flow(b)
    st.info("**해석 원칙:** 지도 점은 이슈 표현의 지역별 관측, **점선은 자동 추출된 유사 정책 후속 논의 후보**입니다. 점선은 실제 정책 전파 방향이나 인과관계를 뜻하지 않습니다.")

    direct=int(summary.get("direct_expression_matches",0));total=int(summary.get("total_lexical_matches",0))
    first_date=str(summary.get("first_direct_expression_observed_date") or "미확인")
    first_assembly="확인 필요"
    if not first.empty and "first_observed_date" in first.columns:
        f=first.sort_values("first_observed_date").iloc[0]
        first_date=str(f["first_observed_date"]);first_assembly=str(f["assembly"])
    k1,k2,k3,k4=st.columns(4)
    k1.metric("정책 이슈", summary.get("query",issue_label(selected)))
    k2.metric("직접 표현 일치",f"{direct:,}건")
    k3.metric("어휘 일치 전체",f"{total:,}건")
    k4.metric("보유 자료 내 최초 관측",first_date)
    st.caption(f"최초 관측 의회: **{first_assembly}** · 실제 최초 정책 도입이나 발생 시점을 의미하지 않습니다.")

    st.divider()
    st.subheader("🗺️ 지방의회별 논의와 유사 정책 후속 관측")
    months=sorted(monthly["month"].dropna().astype(str).tolist()) if not monthly.empty else []
    a,bcol=st.columns([2,1])
    with a:
        selected_month=st.select_slider("관측 범위 (선택 월까지 누적)", options=["전체 기간"]+months,value="전체 기간") if months else "전체 기간"
    with bcol:
        st.caption("정책 유사 후보는 양쪽 원문을 비교한 **검토 대상**이며 확정된 확산 경로가 아닙니다.")
    map_col,info_col=st.columns([1.65,1])
    with map_col:
        st.plotly_chart(timeline_map(region,region_month,flow,first,selected_month),use_container_width=True)
    with info_col:
        st.markdown("**정책 유사 논의 검토 현황**")
        c1,c2=st.columns(2)
        c1.metric("중복 묶음 후",int(b["flow_summary"].get("unique_preview_pair_groups",len(b["flow"]))))
        c2.metric("지도 점선 후보",len(flow))
        st.metric("인과적 전파 확정",int(b["flow_summary"].get("confirmed_diffusion_causation",0)))
        if not flow.empty:
            summaries=flow.copy()
            summaries["경로"] = summaries["earlier_assembly"].map(region_name)+" ↔ "+summaries["later_assembly"].map(region_name)
            st.dataframe(summaries[["candidate_id","경로","specific_common_mechanisms","embedding_cosine"]],
                         hide_index=True,use_container_width=True,height=260)
        else:
            st.warning("표시 가능한 20번 정책 유사 후보가 없습니다. 검색 또는 검증 결과를 확인하세요.")

    st.divider()
    tab_flow,tab_trend,tab_llm,tab_evidence=st.tabs(["🔗 정책 내용 비교", "📈 지역·월별 분석", "🤖 정책 전망", "📚 근거 회의록"])
    with tab_flow:
        st.markdown("#### 유사 정책 후보의 양쪽 회의록 비교")
        if flow.empty:
            st.info("20번의 후보 CSV가 없거나 0건입니다. 빈 결과를 임의의 연결로 채우지 않습니다.")
        else:
            candidates=flow.copy()
            if selected_month!="전체 기간":
                cutoff=pd.Period(selected_month,freq="M").end_time
                candidates=candidates.loc[candidates["later_dt"]<=cutoff]
            if candidates.empty:
                st.info("해당 시점까지 관측된 후속 유사 논의 후보가 없습니다.")
            else:
                all_regions=sorted(set(candidates["earlier_assembly"]).union(candidates["later_assembly"]))
                selected_region=st.selectbox("관련 지역 필터",["전체 지역"]+all_regions)
                if selected_region!="전체 지역":
                    candidates=candidates.loc[(candidates["earlier_assembly"]==selected_region)|(candidates["later_assembly"]==selected_region)]
                ids=candidates["candidate_id"].astype(str).tolist()
                if ids:
                    chosen=st.selectbox("비교할 정책 후보 선택",ids,format_func=lambda cid:(
                        lambda r:f"{cid} | {region_name(r['earlier_assembly'])} {r['earlier_date']} ↔ {region_name(r['later_assembly'])} {r['later_date']}"
                    )(candidates.loc[candidates["candidate_id"].astype(str)==cid].iloc[0]))
                    row=candidates.loc[candidates["candidate_id"].astype(str)==chosen].iloc[0]
                    st.markdown(f"**공통 정책 수단:** {val(row,'specific_common_mechanisms')}")
                    st.markdown(f"**자동 내용 판정:** {val(row,'content_judgment')} · **상태:** {val(row,'status')}")
                    st.caption(f"의미 유사도 {float(val(row,'embedding_cosine',0)):.3f} · 후속 간격 {val(row,'lag_days')}일 · 중복 병합 {val(row,'merged_pair_count')}쌍")
                    evidence_panel(row)
                    if "수동확인" not in str(row["status"]):
                        st.warning("이 후보는 아직 수동 검증이 끝나지 않았습니다. 지도는 실제 전파를 주장하지 않습니다.")
                else:
                    st.info("선택 지역의 정책 후보가 없습니다.")
        if not b["manual"].empty:
            with st.expander("수동 검토 양식 및 현재 기록 확인"):
                st.dataframe(b["manual"],use_container_width=True,hide_index=True)
                st.caption("검증 기록은 20번 CSV를 별도로 편집·검증 후 다시 읽기 하세요. 이 화면에서는 판정을 저장하지 않습니다.")
    with tab_trend:
        st.markdown("#### 월별 직접 표현 일치 추이")
        trend_chart(monthly)
        if not region.empty:
            rr=region.copy()
            rr["region"]=rr["assembly"].map(region_name)
            rr=rr.sort_values("direct_match_rate_pct",ascending=True)
            fig=px.bar(rr,x="direct_match_rate_pct",y="region",orientation="h",hover_data=["direct","all_chunks"],
                       labels={"direct_match_rate_pct":"직접 표현 비율 (%)","region":"지역"})
            fig.update_layout(height=500,margin=dict(l=20,r=20,t=10,b=20))
            st.plotly_chart(fig,use_container_width=True)
        if not region_month.empty:
            x=region_month.copy();x["region"]=x["assembly"].map(region_name)
            pivot=x.pivot_table(index="region",columns="month",values="direct_match_rate_pct",aggfunc="mean")
            fig=px.imshow(pivot,aspect="auto",color_continuous_scale="YlOrRd",labels={"color":"직접 표현 비율 (%)"})
            fig.update_layout(height=520,margin=dict(l=20,r=20,t=20,b=20))
            st.plotly_chart(fig,use_container_width=True)
        st.caption("청크 건수·비율은 정책 채택 또는 지역 간 인과적 전파의 증거가 아닙니다.")
    with tab_llm:
        st.markdown("#### AI의 근거 기반 해석 및 조건부 전망")
        if not llm:
            st.info("17번 LLM 분석 결과가 없습니다.")
        else:
            st.caption(f"모델: {llm.get('model',b['llm_name'])} · 자료 기준일: {llm.get('analysis_cutoff','미상')} · 19~20번 검증 결과는 아직 이 생성 답변에 반영되지 않았습니다.")
            st.markdown("**관측 사실**")
            st.write(llm.get("observed_summary","없음"))
            st.markdown("**시기별 변화 해석**")
            st.write(llm.get("time_change_interpretation","없음"))
            st.markdown("**지역별 해석**")
            st.write(llm.get("regional_interpretation","없음"))
            st.markdown("**향후 전망 (예측 모델 검증 전 조건부 가설)**")
            st.info(llm.get("forecast","없음"))
            st.markdown("**전망 근거**")
            st.write(llm.get("forecast_basis","없음"))
            for msg in llm.get("uncertainties",[]):
                st.caption(f"• {msg}")
            st.warning("전망은 2024~2025년 회의록 미리보기 및 어휘 통계에 기반한 LLM 초안이며, 실증적 예측 정확도와 지역 간 실제 확산은 검증되지 않았습니다.")
    with tab_evidence:
        st.markdown("#### 정책 분석의 원문 근거")
        a1,a2=st.tabs(["17번 LLM 입력 근거", "16번 시점별 근거"])
        with a1:
            e=b["llm_evidence"]
            if e.empty:st.info("LLM 입력 근거 파일이 없습니다.")
            else:
                cols=[c for c in ["evidence_id","assembly","date","meeting_title","chunk_id","text_preview"] if c in e.columns]
                st.dataframe(e[cols],hide_index=True,use_container_width=True,height=400)
        with a2:
            e=b["evidence"]
            if e.empty:st.info("시점별 근거 파일이 없습니다.")
            else:
                options=["전체"]+sorted(e["assembly"].dropna().astype(str).unique().tolist())
                r=st.selectbox("근거 의회",options)
                if r!="전체":e=e.loc[e["assembly"]==r]
                cols=[c for c in ["assembly","date","meeting_title","chunk_id","matched_terms","text_preview"] if c in e.columns]
                st.dataframe(e[cols],hide_index=True,use_container_width=True,height=440)
        st.caption("표시는 원문 전체가 아닌 잘린 발언 미리보기일 수 있습니다. 원문 파일·청크 ID를 통해 추후 원문 전체 보기 연결이 필요합니다.")
    st.divider()
    st.caption(f"기존 저장 결과만 조회 · 읽기 전용 · {selected} | 21번: 화면 통합 (새 임베딩/재학습 없음)")


if __name__ == "__main__":
    main()

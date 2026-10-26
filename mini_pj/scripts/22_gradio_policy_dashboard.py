# -*- coding: utf-8 -*-
"""22_gradio_policy_dashboard.py

개인 프로젝트 / 학원 PC / 로컬 Gradio 시각화 MVP.
기존 15~20번 결과의 읽기 전용 뷰어. 새 검색, Ollama 호출, 재학습은 수행하지 않음.

실행 (Conda mini):
    python C:\\mini_pj\\scripts\\22_gradio_policy_dashboard.py
    http://127.0.0.1:7860

필요 패키지:
    pip install gradio pandas plotly numpy
"""
from __future__ import annotations

import html
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gradio as gr

BASE = Path(r"C:\mini_pj\results\policy_issue_search")
REGIONS = {
    "서울": (37.5665, 126.9780), "부산": (35.1796, 129.0756),
    "대구": (35.8714, 128.6014), "인천": (37.4563, 126.7052),
    "광주": (35.1595, 126.8526), "대전": (36.3504, 127.3845),
    "울산": (35.5384, 129.3114), "세종": (36.4800, 127.2890),
    "경기": (37.4138, 127.5183), "강원": (37.8228, 128.1555),
    "충북": (36.8000, 127.7000), "충남": (36.5184, 126.8000),
    "전북": (35.7175, 127.1530), "전남": (34.8679, 126.9910),
    "경북": (36.4919, 128.8889), "경남": (35.4606, 128.2132),
    "제주": (33.4996, 126.5312),
}
NAME = {
    "서울특별시의회": "서울", "부산광역시의회": "부산", "대구광역시의회": "대구",
    "인천광역시의회": "인천", "광주광역시의회": "광주", "대전광역시의회": "대전",
    "울산광역시의회": "울산", "세종특별자치시의회": "세종", "경기도의회": "경기",
    "강원특별자치도의회": "강원", "강원도의회": "강원",
    "충청북도의회": "충북", "충청남도의회": "충남",
    "전북특별자치도의회": "전북", "전라북도의회": "전북", "전라남도의회": "전남",
    "경상북도의회": "경북", "경상남도의회": "경남", "제주특별자치도의회": "제주",
}


def esc(s):
    if s is None or (not isinstance(s, (list, dict)) and pd.isna(s)):
        return ""
    return html.escape(str(s), quote=True)


def label(assembly):
    return NAME.get(str(assembly), str(assembly))


def read_csv(path):
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_json(path):
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f)


def issue_title(path):
    m = re.match(r"^\d{8}_\d{6}_(.+)$", path.name)
    return (m.group(1) if m else path.name).replace("_", " ")


def list_issues():
    if not BASE.exists():
        return []
    return sorted((p for p in BASE.iterdir() if p.is_dir() and
                   (p / "trend_analysis" / "분석_요약.json").exists()), reverse=True)


@lru_cache(maxsize=10)
def bundle(path_string):
    issue = Path(path_string)
    trend = issue / "trend_analysis"
    flow = issue / "flow_validation"
    models = trend / "llm_analysis"
    llm_dirs = sorted((p for p in models.iterdir() if p.is_dir() and
                       (p / "정책_LLM_분석결과.json").exists()),
                      key=lambda p: p.stat().st_mtime, reverse=True) if models.exists() else []
    llm_dir = llm_dirs[0] if llm_dirs else None
    result = {
        "summary": read_json(trend / "분석_요약.json"),
        "regions": read_csv(trend / "지역별_정책언급순위.csv"),
        "months": read_csv(trend / "월별_정책언급추이.csv"),
        "region_month": read_csv(trend / "지역월별_정책언급추이.csv"),
        "first": read_csv(trend / "지역별_최초관측_직접표현.csv"),
        "flows": read_csv(flow / "정책내용_고유후보_검토결과.csv"),
        "dotted": read_csv(flow / "웹_정책내용유사_점선후보.csv"),
        "validation": read_json(flow / "검증_요약.json"),
        "llm": read_json(llm_dir / "정책_LLM_분석결과.json") if llm_dir else {},
        "evidence": read_csv(llm_dir / "LLM_입력근거후보.csv") if llm_dir else pd.DataFrame(),
        "issue_dir": issue,
    }
    flows = result["flows"]
    dotted = result["dotted"]
    if flows.empty or dotted.empty or "candidate_id" not in flows or "candidate_id" not in dotted:
        result["flows"] = pd.DataFrame()
    else:
        ids = set(dotted["candidate_id"].astype(str))
        result["flows"] = flows.loc[flows["candidate_id"].astype(str).isin(ids)].copy()
    return result


def picker_value(files, query):
    q = str(query).strip().replace(" ", "")
    if not q:
        return files[0] if files else None
    matches = [p for p in files if q in issue_title(p).replace(" ", "")]
    return matches[0] if matches else None


def months_for(b):
    m = b["months"]
    return sorted(m["month"].dropna().astype(str).unique().tolist()) if "month" in m else []


def candidates_for(b, month):
    df = b["flows"].copy()
    if df.empty:
        return df
    if month and month != "전체 기간":
        cutoff = pd.Period(month, freq="M").end_time
        later = pd.to_datetime(df["later_date"], errors="coerce")
        df = df.loc[later <= cutoff]
    return df.sort_values(["earlier_date", "candidate_id"])


def option_label(r):
    return f"{r['candidate_id']}  ·  {label(r['earlier_assembly'])} → {label(r['later_assembly'])}  ·  {str(r['specific_common_mechanisms'])}"


def options_for(df):
    return [(option_label(r), str(r["candidate_id"])) for _, r in df.iterrows()]


def metric_html(b):
    summary = b["summary"]
    first = b["first"]
    first_day = summary.get("first_direct_expression_observed_date", "-")
    first_assembly = "의회 확인 필요"
    if not first.empty and {"first_observed_date", "assembly"}.issubset(first.columns):
        r = first.sort_values("first_observed_date").iloc[0]
        first_day = r["first_observed_date"]
        first_assembly = r["assembly"]
    direct = int(summary.get("direct_expression_matches", 0))
    total = int(summary.get("total_lexical_matches", 0))
    num_candidates = len(b["flows"])
    fields = [
        ("보유 자료 내 최초 관측", esc(first_day), esc(first_assembly)),
        ("직접 표현 일치", f"{direct:,}건", "정책 관련성 미확정"),
        ("유사 논의 검토 후보", f"{num_candidates}쌍", "지역 간 전파 미검증"),
    ]
    cards = "".join(f'<div class="stat"><div class="stat-name">{a}</div><div class="stat-value">{v}</div><div class="stat-foot">{f}</div></div>' for a,v,f in fields)
    return f'<div class="stats">{cards}</div><div class="context-line">검색 주제 <strong>{esc(summary.get("query", "-"))}</strong> · 전체 어휘 일치 {total:,}건 · 분석 기간 2024–2025년</div>'


def map_plot(b, month="전체 기간", selected_id=None):
    reg = b["regions"].copy()
    if reg.empty:
        return go.Figure().update_layout(annotations=[dict(text="지역별 분석 결과 없음", showarrow=False)])
    if month != "전체 기간" and not b["region_month"].empty:
        rm = b["region_month"]
        rm = rm.loc[rm["month"].astype(str) <= month]
        temp = rm.groupby("assembly", as_index=False)[["direct", "all_chunks"]].sum()
        reg = reg.drop(columns=["direct", "all_chunks"], errors="ignore").merge(temp, on="assembly", how="left")
    reg["direct"] = pd.to_numeric(reg.get("direct", 0), errors="coerce").fillna(0)
    reg["all_chunks"] = pd.to_numeric(reg.get("all_chunks", 0), errors="coerce").fillna(0)
    reg["rate"] = np.where(reg["all_chunks"] > 0, reg["direct"] / reg["all_chunks"] * 100, 0)
    reg["region"] = reg["assembly"].map(label)
    reg = reg.loc[reg["region"].isin(REGIONS) & (reg["direct"] > 0)].copy()
    fig = go.Figure()
    # Candidate routes as undirected dotted similarities, not causal arrows.
    flows = candidates_for(b, month)
    for _, r in flows.iterrows():
        a = label(r["earlier_assembly"]); c = label(r["later_assembly"])
        if a not in REGIONS or c not in REGIONS:
            continue
        active = str(r["candidate_id"]) == str(selected_id)
        la,lo = REGIONS[a]; lb,lob = REGIONS[c]
        fig.add_trace(go.Scattergeo(
            lat=[la,lb], lon=[lo,lob], mode="lines",
            line=dict(color="#147d87" if active else "#98a9bb", width=4 if active else 1.5, dash="dot"),
            hovertext=f"{esc(a)} ↔ {esc(c)} · 유사 논의 후보 (전파 미확인)", hoverinfo="text", showlegend=False,
        ))
    if not reg.empty:
        hover = [f"{r['assembly']}<br>직접 표현 일치 {int(r['direct']):,}건<br>전체 대비 {r['rate']:.3f}%" for _,r in reg.iterrows()]
        fig.add_trace(go.Scattergeo(
            lat=[REGIONS[s][0] for s in reg["region"]], lon=[REGIONS[s][1] for s in reg["region"]],
            mode="markers+text", text=reg["region"], textposition="top center", hovertext=hover, hoverinfo="text",
            marker=dict(size=np.clip(8 + np.sqrt(reg["direct"]) * 1.2, 9, 28), color=reg["rate"],
                        colorscale=[[0,"#d8eeed"],[.5,"#6bb8b6"],[1,"#126d79"]],
                        showscale=True, colorbar=dict(title="언급률 %", thickness=11), opacity=.91,
                        line=dict(color="white",width=1.5)), showlegend=False,
        ))
    fig.update_geos(projection_type="mercator", showland=True, landcolor="#f4f7f9",
                    showcountries=True, countrycolor="#d2dbe5", showcoastlines=True,
                    coastlinecolor="#9fb3c4", lonaxis_range=[124,132], lataxis_range=[32.7,39.1],
                    showframe=False, bgcolor="white")
    fig.update_layout(height=435, margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="white",
                      font=dict(color="#26364a",family="Arial"))
    return fig


def trend_plot(b, month="전체 기간"):
    df = b["months"].copy()
    fig = go.Figure()
    if not df.empty:
        df = df.sort_values("month")
        fig.add_bar(x=df["month"], y=df["direct"], name="직접 표현 건수", marker_color="#bfdadf", opacity=.85)
        fig.add_scatter(x=df["month"], y=df["direct_match_rate_pct"], name="전체 청크 대비 %",
                        mode="lines+markers", yaxis="y2", line=dict(color="#147d87", width=2.5))
        if month != "전체 기간":
            fig.add_vline(x=month, line_dash="dot", line_color="#ef9b53")
    fig.update_layout(height=235, margin=dict(l=40,r=35,t=12,b=25),paper_bgcolor="white",plot_bgcolor="white",
                      yaxis=dict(title="건수", gridcolor="#edf1f5"),
                      yaxis2=dict(title="비율 %",overlaying="y",side="right", showgrid=False),
                      xaxis=dict(tickangle=-40),legend=dict(orientation="h",y=1.3,x=0),
                      font=dict(color="#26364a"))
    return fig


def flow_card(r):
    mech = esc(r.get("specific_common_mechanisms", "관련 정책 수단"))
    return f'<div class="flow-summary"><strong>{esc(label(r["earlier_assembly"]))} → {esc(label(r["later_assembly"]))}</strong><div>{mech}</div><small>{esc(r["earlier_date"])} → {esc(r["later_date"])} · 내용 유사 자동 후보 · 전파 미확인</small></div>'


def evidence_box(r,prefix,title):
    text = esc(r.get(prefix+"_policy_excerpt", "")) or esc(r.get(prefix+"_text_preview", ""))
    more = esc(r.get(prefix+"_text_preview", ""))
    return f'''<section class="evidence-side"><div class="eyebrow">{title}</div>
      <h3>{esc(r[prefix+"_assembly"])}</h3><p class="quiet">{esc(r[prefix+"_date"])} · {esc(r.get(prefix+"_meeting_title",""))}</p>
      <div class="evidence-quote">{text}</div><div class="id-line">근거 ID · {esc(r.get(prefix+"_chunk_id",""))}</div>
      <details><summary>원문 미리보기 펼치기 (최대 600자)</summary><div class="full-text">{more}</div></details></section>'''


def compare_html(b, candidate_id, month):
    f = candidates_for(b, month)
    selected = f.loc[f["candidate_id"].astype(str) == str(candidate_id)] if not f.empty else pd.DataFrame()
    if selected.empty:
        return '<div class="empty-note">이 기간에 비교할 정책 후보가 없습니다.</div>'
    r = selected.iloc[0]
    top = flow_card(r)
    left = evidence_box(r,"earlier","먼저 관측")
    right = evidence_box(r,"later","후속 관측")
    return f'''{top}<div class="evidence-grid">{left}{right}</div>
      <div class="notice">표시된 유사도 {float(r['embedding_cosine']):.3f}은 정책 내용 동일성이나 지역 간 전파를 입증하지 않습니다.
      수동 원문 검증 전의 자동 후보입니다.</div>'''


def llm_html(b):
    r = b["llm"]
    if not r:
        return '<div class="empty-note">17번 LLM 분석 결과가 없습니다.</div>'
    return f'''<div class="forecast">
      <div class="eyebrow">{esc(r.get('model','qwen3:8b'))} · 2025-12-31까지의 자료 · 조건부 전망</div>
      <h3>{esc(r.get('forecast','전망 데이터 없음'))}</h3>
      <p><strong>전망 근거</strong> {esc(r.get('forecast_basis',''))}</p>
      <details><summary>관측 사실 및 분석 설명</summary>
       <p><strong>관측</strong> {esc(r.get('observed_summary',''))}</p>
       <p><strong>시기</strong> {esc(r.get('time_change_interpretation',''))}</p>
       <p><strong>지역</strong> {esc(r.get('regional_interpretation',''))}</p>
      </details><p class="quiet">이 예측은 19~20번 검증 결과를 반영하지 않은 LLM 초안이며 실증 예측 정확도는 검증되지 않았습니다.</p>
    </div>'''


def source_html(b):
    d=b["evidence"]
    if d.empty:
        return '<div class="empty-note">17번 LLM 입력 근거가 없습니다.</div>'
    pieces=[]
    for _,r in d.head(10).iterrows():
        title=f"{esc(r.get('evidence_id',''))} · {esc(r.get('assembly',''))} · {esc(r.get('date',''))}"
        snippet=esc(r.get("text_preview",""))
        pieces.append(f'''<details class="source"><summary>{title}</summary>
          <div class="quiet">{esc(r.get('meeting_title',''))} · 청크 {esc(r.get('chunk_id',''))}</div>
          <div class="full-text">{snippet}</div></details>''')
    return '<div class="sources">'+"".join(pieces)+'</div><div class="quiet">표시된 내용은 원문 전체가 아닌 잘린 미리보기일 수 있습니다.</div>'


def fresh_view(query, month="전체 기간", selected_id=None):
    files=list_issues()
    issue=picker_value(files,query)
    if not files:
        raise gr.Error(f"기존 분석 결과가 없습니다: {BASE}")
    if issue is None:
        gr.Info("입력한 키워드의 저장된 분석 결과가 없습니다. 아직 새 검색 자동 실행은 지원하지 않습니다.")
        issue=files[0]
    b=bundle(str(issue))
    mm=months_for(b)
    if month not in mm and month!="전체 기간": month="전체 기간"
    choices=options_for(candidates_for(b,month))
    ids=[v for _,v in choices]
    if selected_id not in ids: selected_id=ids[0] if ids else None
    return (str(issue),
            gr.update(choices=["전체 기간"]+mm,value=month),
            gr.update(choices=choices,value=selected_id),
            metric_html(b), map_plot(b,month,selected_id), trend_plot(b,month),
            compare_html(b,selected_id,month),llm_html(b),source_html(b),
            f"**현재 표시:** {issue_title(issue)} · 저장된 분석 결과 조회 · 실시간 새 검색 미연결")


def update_selection(issue_path,month,selected_id):
    if not issue_path: return None,None,None
    b=bundle(issue_path)
    mm=months_for(b)
    if month not in mm and month!="전체 기간":month="전체 기간"
    choices=options_for(candidates_for(b,month))
    ids=[v for _,v in choices]
    if selected_id not in ids:selected_id=ids[0] if ids else None
    return (gr.update(choices=choices,value=selected_id),map_plot(b,month,selected_id),
            compare_html(b,selected_id,month))


CSS = """
.gradio-container {max-width: 1110px !important; margin:auto !important; background:#fff; color:#22354a;}
#hero {padding:18px 0 8px;}
#hero h1 {font-size:30px;letter-spacing:-1px;font-weight:800;color:#17344c;margin:0;}
#hero p {font-size:14px;color:#6e7e8c;margin-top:4px;}
#search-row {background:#f6f9fb;padding:13px;border-radius:14px;margin-bottom:12px;}
#search-row button {background:#147d87;color:white;font-weight:700;border:0;}
.stats {display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:12px 0;}
.stat {border:1px solid #e3ebee;border-radius:12px;padding:17px 20px;background:white;}
.stat-name {font-size:13px;color:#6d8191;}.stat-value {font-weight:800;font-size:25px;color:#18344b;margin:8px 0;}
.stat-foot,.context-line,.quiet {font-size:12px;color:#7d8c99;}.context-line {margin:6px 0 14px;}
.section-title {font-size:21px;font-weight:800;color:#1c394e;margin:22px 0 5px;}
.section-sub {font-size:12px;color:#7d8c99;margin:0 0 12px;}
.flow-summary {border:1px solid #cddfdf;border-radius:10px;padding:13px 16px;background:#f6fafa;margin-bottom:12px;}
.flow-summary strong {font-size:17px;color:#176f79;}.flow-summary div {font-size:14px;margin-top:6px;}
.flow-summary small {display:block;margin-top:6px;color:#7d8c99;}
.evidence-grid {display:grid;grid-template-columns:1fr 1fr;gap:13px;}
.evidence-side {padding:17px;background:#fbfcfd;border:1px solid #e3eaef;border-radius:12px;min-width:0;}
.evidence-side h3 {margin:4px 0;color:#223f58;font-size:18px;}
.eyebrow {font-size:12px;font-weight:700;color:#147d87;}.evidence-quote {font-size:13px;line-height:1.85;max-height:180px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;margin:12px 0;}
.id-line {font-size:11px;color:#81919d;overflow-wrap:anywhere;}
details {margin-top:12px;}summary {cursor:pointer;color:#367985;font-size:13px;font-weight:650;}
.full-text {white-space:pre-wrap;font-size:12px;line-height:1.8;margin:12px 0;overflow-wrap:anywhere;}
.notice {margin:14px 0;padding:12px 16px;background:#fbf7ed;border-radius:9px;color:#8a6b30;font-size:12px;line-height:1.6;}
.forecast {border:1px solid #dce8ec;border-radius:14px;padding:21px;background:#f6fafb;}
.forecast h3 {font-size:18px;color:#234155;margin:11px 0;line-height:1.6;}.forecast p {font-size:13px;line-height:1.75;}
.source {border-bottom:1px solid #e6edf1;padding:10px 3px;}.empty-note {padding:20px;color:#728596;background:#f6f8fa;border-radius:10px;}
@media(max-width:760px){.stats,.evidence-grid {grid-template-columns:1fr;}#hero h1{font-size:23px;}}
"""


def build_app():
    gr_major = int(gr.__version__.split(".")[0])
    block_kwargs = {"title": "지방의회 AI 정책 이슈 탐색"}
    if gr_major < 6:
        block_kwargs.update(css=CSS, theme=gr.themes.Soft())
    with gr.Blocks(**block_kwargs) as demo:
        gr.HTML('''<header id="hero"><div class="eyebrow">LOCAL POLICY INTELLIGENCE</div>
            <h1>🏛️ 지방의회 AI 정책 이슈 탐색</h1>
            <p>어디서 처음 관측됐고, 어떤 지역에서 유사한 논의가 이어졌을까요?</p></header>''')
        with gr.Row(elem_id="search-row"):
            query=gr.Textbox(label="정책 이슈 검색", value="청년 일자리",placeholder="기존 분석 주제 입력 (예: 청년 일자리)",scale=5)
            search=gr.Button("검색 결과 보기",variant="primary",scale=1)
        status=gr.Markdown("저장된 기존 분석 결과를 표시합니다. 새 키워드의 실시간 검색은 아직 연결되지 않았습니다.")
        issue_state=gr.State("")
        stats=gr.HTML()
        gr.HTML('<div class="section-title">지역별 정책 논의 흐름</div><div class="section-sub">점 = 관련 표현의 관측 · 점선 = 유사 정책 논의 후보 (정책 전파 미확인)</div>')
        with gr.Row():
            with gr.Column(scale=7):
                month=gr.Dropdown(label="관측 시점",choices=["전체 기간"],value="전체 기간",interactive=True)
                map_chart=gr.Plot(label="지방의회 관측 지도",show_label=False)
            with gr.Column(scale=5):
                gr.Markdown("**월별 언급 변화** · 건수와 비율을 함께 확인")
                trend=gr.Plot(show_label=False)
                gr.HTML('<div class="notice">지도 점선은 시간 순서와 정책 내용 유사성에 따른 검토 후보입니다. 다른 의회의 정책을 채택하거나 참고했다는 증거는 아닙니다.</div>')
        gr.HTML('<div class="section-title">유사 정책 논의 비교</div><div class="section-sub">후보를 선택하면 두 의회의 회의록 핵심 구간을 나란히 비교합니다.</div>')
        candidate=gr.Dropdown(label="유사 정책 후보 선택",choices=[],interactive=True)
        comparison=gr.HTML()
        gr.HTML('<div class="section-title">AI 정책 해석과 향후 전망</div>')
        forecast=gr.HTML()
        with gr.Accordion("📚 근거 회의록 원문 미리보기",open=False):
            sources=gr.HTML()
        gr.Markdown("*분석 대상은 2024~2025년 보유 회의록입니다. 최초 관측일은 정책의 실제 최초 도입일을 의미하지 않습니다. 이 화면은 저장된 결과를 읽기만 하며 Chroma·임베딩·LLM을 재실행하지 않습니다.*")
        search.click(fn=lambda q:fresh_view(q),inputs=[query],outputs=[issue_state,month,candidate,stats,map_chart,trend,comparison,forecast,sources,status])
        query.submit(fn=lambda q:fresh_view(q),inputs=[query],outputs=[issue_state,month,candidate,stats,map_chart,trend,comparison,forecast,sources,status])
        month.change(fn=update_selection,inputs=[issue_state,month,candidate],outputs=[candidate,map_chart,comparison])
        candidate.change(fn=lambda p,m,c:(map_plot(bundle(p),m,c),compare_html(bundle(p),c,m)) if p else (None,""),inputs=[issue_state,month,candidate],outputs=[map_chart,comparison])
        demo.load(fn=lambda:fresh_view("청년 일자리"),outputs=[issue_state,month,candidate,stats,map_chart,trend,comparison,forecast,sources,status])
    return demo


if __name__=="__main__":
    if not BASE.exists():
        print(f"[오류] 결과 폴더 없음: {BASE}")
    else:
        print(f"[준비] 저장된 분석 이슈 {len(list_issues())}개")
    print("[실행] http://127.0.0.1:7860")
    launch_kwargs = dict(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
    if int(gr.__version__.split(".")[0]) >= 6:
        launch_kwargs.update(css=CSS, theme=gr.themes.Soft())
    build_app().launch(**launch_kwargs)

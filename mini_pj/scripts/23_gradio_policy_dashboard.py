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



# ========================= 23: 간결한 검색형 UI =========================
from datetime import datetime
import time


def cut_exact(value, limit):
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def day_format(value):
    raw = str(value or "").strip()
    try:
        return pd.to_datetime(raw, errors="raise").strftime("%Y년 %m월 %d일")
    except Exception:
        return raw or "확인 필요"


def first_item(b):
    first = b["first"]
    if not first.empty and {"first_observed_date", "assembly"}.issubset(first.columns):
        d = first.dropna(subset=["first_observed_date"]).sort_values("first_observed_date")
        if not d.empty:
            row = d.iloc[0]
            return str(row["first_observed_date"]), str(row["assembly"])
    return str(b["summary"].get("first_direct_expression_observed_date", "")), "의회 확인 필요"


def find_first_evidence(b, when, assembly):
    evidence = b["evidence"]
    if not evidence.empty and {"date", "assembly"}.issubset(evidence.columns):
        f = evidence.loc[(evidence["date"].astype(str) == when) & (evidence["assembly"].astype(str) == assembly)]
        if not f.empty:
            return f.iloc[0].to_dict()
    # 근거 원문이 확보되지 않았으면 최초 관측 사실만 표시하고 원문은 추정하지 않는다.
    return None


def extract_quote(text, query="", limit=100):
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if not t:
        return "근거 문장을 확인할 수 없습니다."
    terms = [query.strip()] if query else []
    terms += ["청년", "취업", "지원", "일자리"]
    pos = next((t.find(term) for term in terms if term and t.find(term) >= 0), 0)
    start = max(0, pos - 18)
    snippet = t[start:start + limit]
    # 원문에서 직접 발췌(기계적으로 잘라내기만 함)
    prefix = "…" if start else ""
    room = max(0, limit - len(prefix))
    clipped = t[start:start + room]
    if start + len(clipped) < len(t) and room > 1:
        clipped = clipped[:-1] + "…"
    return prefix + clipped


def candidate_events(b, month):
    df = candidates_for(b, month)
    out = []
    for _, r in df.iterrows():
        cid = str(r.get("candidate_id", ""))
        for side in ["earlier", "later"]:
            dt = str(r.get(side + "_date", ""))
            place = str(r.get(side + "_assembly", ""))
            mechanism = str(r.get("specific_common_mechanisms", "")).replace("nan", "")
            if not dt or not place:
                continue
            out.append({"key": f"{cid}|{side}", "date": dt, "assembly": place,
                        "action": "유사 정책 논의 관측", "mechanism": mechanism,
                        "candidate_id": cid, "side": side, "row": r})
    # 같은 회의의 반복된 후보는 타임라인에 한 번만 보이도록 제거
    unique = {}
    for ev in sorted(out, key=lambda x: (x["date"], x["assembly"], x["candidate_id"])):
        key = (ev["date"], ev["assembly"], ev["mechanism"])
        unique.setdefault(key, ev)
    return list(unique.values())


def timeline_choices(events):
    return [(f"{e['date']}  ·  {e['assembly']}  ·  {e['action']}", e["key"]) for e in events]


def detail_card(title, assembly, date, meeting, chunk, excerpt, preview="", warning=""):
    snippet = extract_quote(excerpt or preview, limit=100)
    return f"""<article class="detail-card">
      <span class="mini-label">{esc(title)}</span><h3>{esc(assembly)}</h3>
      <p class="quiet">{esc(day_format(date))} · {esc(meeting or '회의록명 확인 필요')}</p>
      <p class="quotation">“{esc(snippet)}”</p>
      <p class="quiet">근거 청크 · {esc(chunk or '확인 필요')}</p>
      {('<div class="notice">' + esc(warning) + '</div>') if warning else ''}
      <details><summary>원문 미리보기 및 상세 내용</summary><div class="full-text">{esc(preview or excerpt or '원문 미리보기를 찾을 수 없습니다.')}</div></details>
    </article>"""


def detail_for_event(b, month, event_key):
    ev = next((e for e in candidate_events(b, month) if e["key"] == str(event_key)), None)
    if not ev:
        return '<div class="empty-note">지도 오른쪽에서 의회 논의 항목을 선택하면 상세 근거가 표시됩니다.</div>'
    r = ev["row"]; side=ev["side"]
    return detail_card("정책 내용 유사 · 자동 후보 / 전파 미확인",
                       ev["assembly"], ev["date"], r.get(side + "_meeting_title", ""),
                       r.get(side + "_chunk_id", ""), r.get(side + "_policy_excerpt", ""),
                       r.get(side + "_text_preview", ""),
                       "선택한 항목은 정책 내용 유사 후보입니다. 재논의·공청회 개최나 지역 간 정책 전파는 검증되지 않았습니다.")


def first_html(b):
    date, assembly = first_item(b)
    return f"""<section class="first-card"><div class="mini-label">01 · 보유 자료 내 최초 관측</div>
      <div class="first-line"><strong>{esc(day_format(date))}</strong><span>·</span><strong>{esc(assembly)}</strong></div>
      <p class="quiet">정책 실제 최초 도입이 아닌 2024~2025년 검색 표현의 최초 관측입니다.</p></section>"""


def first_detail(b):
    date, assembly = first_item(b)
    e = find_first_evidence(b, date, assembly)
    if e:
        return detail_card("최초 관측 관련 근거", assembly, date,
                           e.get("meeting_title", ""), e.get("chunk_id", ""),
                           e.get("text_preview", ""), e.get("text_preview", ""))
    return f"""<div class="detail-card"><h3>{esc(assembly)}</h3><p>{esc(day_format(date))}</p>
      <p class="quiet">최초 관측은 집계에서 확인됐지만 현재 17번의 대표 근거 10건에서 해당 원문 전체를 찾지 못했습니다.
      15번 원본 검색 근거를 조회해 정확한 출처를 확인해야 합니다.</p></div>"""


def forecast_short(b):
    llm = b["llm"]
    if not llm:
        return '<div class="empty-note">저장된 LLM 분석 결과가 없습니다.</div>'
    original = str(llm.get("forecast", "")).strip()
    short = cut_exact(original, 150)
    note = "원문이 150자를 넘어 화면용으로 줄였습니다. 상세 설명에서 확인할 수 있습니다." if len(original) > 150 else "150자 이내 · 조건부 전망"
    return f"""<article class="forecast-simple"><div class="mini-label">03 · AI 향후 예측 <span class="tag">{esc(llm.get('model','qwen3:8b'))}</span></div>
      <p class="forecast-main">{esc(short or '전망 결과가 없습니다.')}</p>
      <p class="quiet">{esc(note)} · 2025-12-31까지 보유 자료 기준 · 예측 정확도 미검증</p>
      <details><summary>분석 근거·불확실성 상세 보기</summary>
        <p><b>전망 근거</b> {esc(llm.get('forecast_basis', ''))}</p>
        <p><b>원문 전망</b> {esc(original)}</p>
        <p><b>시기별 해석</b> {esc(llm.get('time_change_interpretation',''))}</p>
        <p><b>지역별 해석</b> {esc(llm.get('regional_interpretation',''))}</p>
        <p class="quiet">19~20번의 후보 수동 검증 결과가 LLM 생성에 반영된 것은 아닙니다.</p>
      </details></article>"""


def source_choices(b):
    d = b["evidence"]
    if d.empty: return []
    return [(f"{str(r.get('meeting_title','회의록명 없음'))[:40]}  ·  {r.get('date','')}  ·  {r.get('assembly','')}",
             str(r.get('evidence_id',i))) for i,r in d.iterrows()]


def source_rows_html(b):
    d = b["evidence"]
    if d.empty:
        return '<div class="empty-note">저장된 LLM 입력 근거가 없습니다.</div>'
    q = b["summary"].get("query", "")
    cards=[]
    for _,r in d.head(5).iterrows():
        excerpt = extract_quote(r.get("text_preview",""), query=q, limit=100)
        cards.append(f"""<div class="source-item"><div class="source-top"><strong>{esc(r.get('meeting_title','회의록명 확인 필요'))}</strong>
         <span>{esc(day_format(r.get('date','')))} · {esc(r.get('assembly',''))}</span></div>
         <div class="source-quote">“{esc(excerpt)}”</div></div>""")
    return ''.join(cards) + '<p class="quiet">위 내용은 저장된 근거 원문 미리보기에서 발췌했으며, 모든 문장의 관련성이 검증된 것은 아닙니다.</p>'


def source_detail(b, evidence_id):
    d = b["evidence"]
    if d.empty:
        return '<div class="empty-note">근거 없음</div>'
    match = d.loc[d["evidence_id"].astype(str) == str(evidence_id)] if "evidence_id" in d else pd.DataFrame()
    if match.empty:
        return '<div class="empty-note">선택한 회의록이 없습니다.</div>'
    r=match.iloc[0]
    return detail_card(str(r.get("evidence_id", "근거")), r.get("assembly", ""), r.get("date", ""),
                       r.get("meeting_title", ""), r.get("chunk_id", ""),
                       r.get("text_preview", ""), r.get("text_preview", ""),
                       "이 발언이 정책 전망을 실제로 뒷받침하는지는 별도의 내용 검증이 필요합니다.")


def resolve_issue(query):
    files=list_issues()
    if not files: raise gr.Error(f"분석 결과 폴더가 없습니다: {BASE}")
    q = re.sub(r"\s+", "", str(query or "")).strip()
    if not q:
        return files[0]
    matches=[p for p in files if q in re.sub(r"\s+", "", issue_title(p))]
    if not matches:
        # 자연어 질문은 저장된 이슈명과 직접 포함 관계일 때만 검색 결과와 연결.
        matches=[p for p in files if re.sub(r"\s+", "", issue_title(p)) in q]
    if not matches:
        raise gr.Error("아직 이 검색어/질문의 분석 결과가 저장되지 않았습니다. 현재는 15~20번에서 생성한 결과만 조회합니다.")
    return matches[0]


def load_view(query):
    start=time.perf_counter()
    issue=resolve_issue(query)
    b=bundle(str(issue)); month="전체 기간"
    events=candidate_events(b,month)
    ec=timeline_choices(events)
    key=ec[0][1] if ec else None
    sc=source_choices(b); sid=sc[0][1] if sc else None
    print(f"[{datetime.now():%H:%M:%S}] 23번 화면 로드: {issue_title(issue)} / 후보 {len(b['flows'])} / {time.perf_counter()-start:.2f}초")
    return (str(issue),
            f"**조회 이슈:** {issue_title(issue)}  ·  저장된 분석 결과 (새 검색 자동 분석 미연결)",
            first_html(b),
            gr.update(choices=["전체 기간"] + months_for(b),value=month),
            map_plot(b,month),
            gr.update(choices=ec,value=key),
            detail_for_event(b,month,key),
            forecast_short(b),
            source_rows_html(b),
            gr.update(choices=sc,value=sid),
            source_detail(b,sid))


def on_month(issue_path,month):
    if not issue_path: return None,gr.update(),""
    b=bundle(issue_path)
    events=candidate_events(b,month); ec=timeline_choices(events)
    key=ec[0][1] if ec else None
    return (map_plot(b,month), gr.update(choices=ec,value=key),detail_for_event(b,month,key))


def on_event(issue_path,month,event_key):
    if not issue_path: return "",None
    b=bundle(issue_path)
    ev=next((e for e in candidate_events(b,month) if e["key"] == event_key),None)
    selected=ev["candidate_id"] if ev else None
    return detail_for_event(b,month,event_key), map_plot(b,month,selected)


CSS = """
.gradio-container{max-width:1060px!important;margin:0 auto!important;color:#172e41;background:#fff;}
#hero{padding:14px 0 8px}#hero h1{font-size:29px;letter-spacing:-.8px;color:#183b51;margin:3px 0;}
#hero p{color:#758798;font-size:13px;}.mini-label{color:#167d86;font-size:12px;font-weight:750;letter-spacing:.2px;}
#search-row{padding:12px 14px;border:1px solid #e7edf1;border-radius:12px;background:#f8fafb;margin-bottom:5px}
#search-row button{background:#137e88;color:#fff;font-weight:700;border:none;}
.section-title{font-size:19px;font-weight:800;color:#173a52;margin:24px 0 6px;}
.section-sub{font-size:12px;color:#788c99;margin-bottom:10px;}
.first-card{border:1px solid #e2ebf0;border-radius:14px;padding:19px 22px;margin-top:12px;background:#fff;}
.first-line{display:flex;align-items:center;gap:13px;flex-wrap:wrap;font-size:20px;margin:8px 0;color:#18344b;}
.first-line span{color:#9aacb8}.quiet{font-size:12px;color:#718697;line-height:1.6}
.detail-card{border:1px solid #e2ebef;background:#f8fafb;border-radius:12px;padding:16px 20px;margin-top:8px;}
.detail-card h3{font-size:17px;margin:7px 0;}.quotation{font-size:14px;color:#243f53;line-height:1.7;margin-top:14px;}
.notice{padding:10px 12px;background:#fff8e9;border-radius:8px;color:#8b6c35;font-size:12px;margin-top:12px;}
details{margin-top:10px}summary{cursor:pointer;color:#157b85;font-size:12px;font-weight:700;}
.full-text{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.8;font-size:12px;margin-top:12px;}
.forecast-simple{background:#f2f8f9;border:1px solid #dcebed;border-radius:13px;padding:22px;margin:8px 0;}
.forecast-main{font-size:18px;font-weight:750;color:#173e50;line-height:1.75;margin:12px 0 9px;}
.tag{font-size:11px;color:#4c7780;background:#e4eff0;border-radius:20px;padding:3px 9px;margin-left:9px;}
.source-item{padding:14px 3px;border-bottom:1px solid #e5ecf0;}.source-top{display:flex;align-items:center;gap:12px;flex-wrap:wrap;}
.source-top strong{font-size:13px;color:#233a4d}.source-top span{font-size:11px;color:#83929d}
.source-quote{font-size:13px;color:#354b5c;margin-top:8px;line-height:1.65;}
.empty-note{padding:18px;color:#718798;background:#f7fafb;border-radius:10px;}
#timeline-area{border:1px solid #e5ecef;border-radius:12px;padding:10px;background:#fff;}
#timeline-area label{line-height:1.45!important;font-size:12px!important}
#brief-status{font-size:12px;color:#718497;}
@media(max-width:760px){#hero h1{font-size:22px}.first-line{font-size:16px}}
"""


def build_app():
    major = int(gr.__version__.split(".")[0])
    kw = {"title":"지방의회 AI 정책 이슈 탐색 · 간결형"}
    if major < 6:
        kw.update(css=CSS, theme=gr.themes.Soft())
    with gr.Blocks(**kw) as app:
        gr.HTML("""<header id="hero"><div class="mini-label">LOCAL POLICY INTELLIGENCE</div>
        <h1>🏛️ 지방의회 AI 정책 이슈 탐색</h1>
        <p>정책 이슈의 최초 관측 · 타 지방의회의 유사 논의 · 향후 전망 · 회의록 근거</p></header>""")
        with gr.Row(elem_id="search-row"):
            query=gr.Textbox(label="정책 키워드 또는 질문",value="청년 일자리",placeholder="예: 청년 일자리 / 청년 일자리 정책은 어디에서 처음 논의됐어?",scale=5)
            search=gr.Button("검색",variant="primary",scale=1)
        status=gr.Markdown("저장된 분석 주제만 조회합니다. 새 키워드 실시간 분석은 추후 연결합니다.",elem_id="brief-status")
        state=gr.State("")
        first=gr.HTML()
        with gr.Accordion("최초 관측 의회 · 원문 상세 보기",open=False):
            first_more=gr.HTML()
        gr.HTML('<div class="section-title">02 · 타 지방의회로의 논의 흐름</div><div class="section-sub">지도는 논의 관측 지역, 점선은 정책 내용 유사 자동 후보입니다. 실제 정책 전파가 확인된 것은 아닙니다.</div>')
        with gr.Row():
            with gr.Column(scale=6):
                month=gr.Dropdown(label="관측 시점",choices=["전체 기간"],value="전체 기간",interactive=True)
                map_chart=gr.Plot(label="지역별 논의 지도",show_label=False)
            with gr.Column(scale=5,elem_id="timeline-area"):
                gr.Markdown("**의회별 논의 관측 순서**  ·  의회를 선택하면 근거 상세 확인")
                events=gr.Radio(label="날짜 · 의회 · 의정활동",choices=[],interactive=True)
                event_detail=gr.HTML()
        gr.HTML('<div class="section-title">03 · AI 향후 예측</div><div class="section-sub">로컬 LLM 서술형 전망 · 150자 이내 표시</div>')
        forecast=gr.HTML()
        gr.HTML('<div class="section-title">04 · 근거 회의록</div><div class="section-sub">회의명 · 회의일자 · 100자 이내 원문 발췌. 회의록 선택 시 상세 보기</div>')
        source_list=gr.HTML()
        with gr.Accordion("회의록 선택하여 상세 원문 보기",open=False):
            source_picker=gr.Dropdown(label="근거 회의록 선택",choices=[],interactive=True)
            source_detail_panel=gr.HTML()
        gr.Markdown("*2024~2025년 보유 회의록 기준입니다. 정책 실제 최초 도입 및 지역 간 전파는 미확인입니다. 이 버전은 기존 분석 결과만 읽으며 새 검색·임베딩·LLM 재실행은 하지 않습니다.*")

        outputs=[state,status,first,month,map_chart,events,event_detail,forecast,source_list,source_picker,source_detail_panel]
        search.click(load_view,[query],outputs)
        query.submit(load_view,[query],outputs)
        month.change(on_month,[state,month],[map_chart,events,event_detail])
        events.change(on_event,[state,month,events],[event_detail,map_chart])
        source_picker.change(lambda p,id:source_detail(bundle(p),id) if p else "",[state,source_picker],[source_detail_panel])
        state.change(lambda p:first_detail(bundle(p)) if p else "",[state],[first_more])
        app.load(lambda:load_view("청년 일자리"), outputs=outputs)
    return app


if __name__ == "__main__":
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 23번 Gradio 시작")
    print(f"개인 프로젝트 저장 결과 폴더: {BASE}")
    print("접속: http://127.0.0.1:7860")
    kw=dict(server_name="127.0.0.1",server_port=7860,share=False,inbrowser=True)
    if int(gr.__version__.split(".")[0])>=6:
        kw.update(css=CSS,theme=gr.themes.Soft())
    build_app().launch(**kw)

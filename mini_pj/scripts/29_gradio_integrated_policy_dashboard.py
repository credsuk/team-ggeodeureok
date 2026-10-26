# -*- coding: utf-8 -*-
"""29번: 23번 Gradio 디자인을 유지하고 최신 28 V2 정책 JSON 연결 (읽기 전용).

학원 PC 개인 프로젝트용. 팀/집 PC 데이터와 혼합하지 않음.
기존 23_gradio_policy_dashboard.py 수정·덮어쓰기 없음.

실행:
 & "C:\\Users\\Admin\\miniconda3\\envs\\mini\\python.exe" "C:\\mini_pj\\scripts\\29_gradio_integrated_policy_dashboard.py"
검증:
 & "C:\\Users\\Admin\\miniconda3\\envs\\mini\\python.exe" "C:\\mini_pj\\scripts\\29_gradio_integrated_policy_dashboard.py" --dry-run
선택:
 --data "C:\\mini_pj\\results\\policy_result_integration_v2\\20261026_XXXXXX\\all_issues.json"
 --port 7861 --no-browser

중요: 최초 관측·지역간 순서는 자동 표현 관측이다. 실제 정책 도입/인과적 전파가 아니다.
데이터 생성, 임베딩, Chroma 검색, LLM 호출은 전혀 하지 않는다.
"""
from __future__ import annotations

import argparse
import html
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:\mini_pj")
V2_ROOT = ROOT / "results" / "policy_result_integration_v2"
BASELINE_23 = ROOT / "scripts" / "23_gradio_policy_dashboard.py"
ISSUES = ["지역소멸", "청년 일자리", "저출생", "고령화", "기후위기"]


def log(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def esc(value):
    return html.escape(str(value if value is not None else ""), quote=True)


def load_data(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("all_issues.json 최상위 구조는 {이슈명: 분석결과} 딕셔너리여야 합니다.")
    missing = [key for key in ISSUES if key not in data]
    if missing:
        log("주의: 누락된 이슈: " + ", ".join(missing))
    for name, item in data.items():
        if not isinstance(item, dict):
            raise ValueError(f"이슈 {name}의 데이터가 딕셔너리가 아닙니다.")
    return data


def latest_json():
    candidates = sorted(V2_ROOT.glob("*/all_issues.json"), key=lambda p: p.stat().st_mtime, reverse=True) if V2_ROOT.is_dir() else []
    if not candidates:
        raise FileNotFoundError(
            f"28 V2 결과를 찾지 못했습니다: {V2_ROOT}\\<실행시각>\\all_issues.json\n"
            "--data 옵션으로 해당 파일의 전체 경로를 지정하세요."
        )
    return candidates[0]


def find_issue(query, dataset):
    compact = "".join(str(query or "").split()).lower()
    if not compact:
        return "청년 일자리" if "청년 일자리" in dataset else next(iter(dataset), None)
    # 표현이 길어도 질문 안의 이슈 구문을 식별 (중복 부분문자열은 길이가 긴 이슈 우선)
    sorted_names = sorted(dataset, key=lambda x: len(x.replace(" ", "")), reverse=True)
    matches = [name for name in sorted_names if name.replace(" ", "").lower() in compact]
    if matches:
        return matches[0]
    # 무관한 새 질문을 예전 청년 일자리 결과로 오인하여 표시하지 않음
    return None


def inspect_summary(dataset):
    problems = []
    log(f"읽기 검증: 통합 이슈 {len(dataset)}개")
    for issue in ISSUES:
        if issue not in dataset:
            continue
        obj = dataset[issue]
        first = obj.get("first_observation") or {}
        rows = obj.get("regional_discussion_flow") or []
        ev = obj.get("evidence_records") or []
        fw = obj.get("forecast") or {}
        act = obj.get("policy_activity_cues") or []
        pair = obj.get("cross_region_similarity_candidates") or []
        log(f"  {issue}: 최초 {first.get('date', '-')}/{first.get('assembly','-')} | 지역 {len(rows)} | 근거 {len(ev)} | 전망 {fw.get('status','-')} | 활동 {len(act)} | 유사후보 {len(pair)}")
        if not rows: problems.append(f"{issue}: 지역별 근거 없음")
        if not ev: problems.append(f"{issue}: 회의록 근거 없음")
    for x in problems: log("경고: " + x)
    return not problems


def import_23(path):
    if not path.is_file():
        raise FileNotFoundError(f"23번 원본 화면 파일이 필요합니다: {path}")
    spec = importlib.util.spec_from_file_location("policy_dashboard_original_23", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"23번 모듈을 읽을 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def install_integrated_callbacks(m, dataset, json_path):
    """23번 build_app()의 레이아웃/CSS는 유지하고 데이터 처리 함수만 런타임에 교체."""
    gr = m.gr
    go = m.go

    def result(issue):
        return dataset.get(issue, {})

    def months(issue):
        periods = result(issue).get("monthly_timeline") or []
        return sorted({str(p.get("월", "")) for p in periods if p.get("월")})

    def visible_events(issue, month):
        events = []
        for i, row in enumerate(result(issue).get("regional_discussion_flow") or []):
            date = str(row.get("date", ""))
            if month != "전체 기간" and date[:7] > month:
                continue
            events.append({"key": str(i), **row})
        return sorted(events, key=lambda x: (x.get("date", ""), x.get("assembly", "")))

    def choices(issue, month):
        return [(f"{r.get('date','-')} · {r.get('assembly','-')} · {r.get('activity_label','관측 후보')}", r["key"])
                for r in visible_events(issue, month)]

    def title_and_summary(issue):
        first = result(issue).get("first_observation") or {}
        if not first or not first.get("date"):
            return '<div class="empty-note">해당 이슈의 최초 자동 관측 결과가 없습니다.</div>'
        return (f'<section class="first-card"><div class="mini-label">01 · 보유 자료 내 최초 자동 관측</div>'
                f'<div class="first-line"><strong>{esc(m.day_format(first.get("date")))}</strong><span>·</span>'
                f'<strong>{esc(first.get("assembly"))}</strong></div>'
                f'<p class="quiet">{esc(first.get("verification_status", "자동 후보 · 수동 미검증"))}. '
                '2024~2025년 보유 자료의 최초 관측이며 실제 최초 도입은 아닙니다.</p></section>')

    def show_first_detail(issue):
        first = result(issue).get("first_observation") or {}
        if not first:
            return '<div class="empty-note">최초 관측 근거가 없습니다.</div>'
        evidence = result(issue).get("evidence_records") or []
        matched = next((r for r in evidence if str(r.get("id", "")) == str(first.get("chunk_id", ""))), None)
        # 다른 청크 원문으로 대체하지 않고, 있으면 동일 날짜·의회·회의록만 제한적으로 대응
        if matched is None:
            matched = next((r for r in evidence if r.get("date") == first.get("date") and
                            r.get("assembly") == first.get("assembly") and
                            r.get("meeting_title") == first.get("meeting_title")), None)
        snippet = (matched or {}).get("excerpt_100", "")
        return m.detail_card("자동 최초 관측 · 원문", first.get("assembly", ""), first.get("date", ""),
                             first.get("meeting_title", ""), first.get("chunk_id", ""), snippet, snippet,
                             "원문에 표현은 있으나 정책 관련성과 최초 도입 여부는 수동 미검증입니다.")

    def show_detail(issue, month, key):
        item = next((r for r in visible_events(issue, month) if r["key"] == str(key)), None)
        if item is None:
            return '<div class="empty-note">오른쪽 의회를 선택하면 해당 회의록 근거가 표시됩니다.</div>'
        snippet = item.get("evidence_100") or ""
        return m.detail_card(item.get("activity_label") or "의회 관측 후보", item.get("assembly", ""),
                             item.get("date", ""), item.get("meeting_title", ""),
                             item.get("chunk_id", ""), snippet, snippet,
                             "자동 정책 표현 관측입니다. 재논의·공청회 개최·실제 정책 전파는 확정되지 않았습니다.")

    def show_forecast(issue):
        f = result(issue).get("forecast") or {}
        original = str(f.get("forecast_150") or "").strip()
        if not original or f.get("needs_150_rewrite") or len(original) > 150:
            return '<div class="empty-note">150자 이내로 검토된 전망이 없습니다. 원문 재작성 또는 검증이 필요합니다.</div>'
        return (f'<article class="forecast-simple"><div class="mini-label">03 · AI 향후 예측 '
                f'<span class="tag">{esc(f.get("model", "모델 확인 필요"))}</span></div>'
                f'<p class="forecast-main">{esc(original)}</p>'
                f'<p class="quiet">150자 이내 · {esc(f.get("verification_status", "LLM 전망 수동 미검증"))}</p>'
                '<details><summary>생성 전망 원문 보기</summary>'
                f'<div class="full-text">{esc(f.get("forecast_200_original", ""))}</div></details></article>')

    def show_sources(issue):
        rows = result(issue).get("evidence_records") or []
        if not rows:
            return '<div class="empty-note">연결된 근거 회의록이 없습니다.</div>'
        cards = []
        for row in rows[:6]:
            cards.append(f'<div class="source-item"><div class="source-top"><strong>{esc(row.get("meeting_title", ""))}</strong>'
                         f'<span>{esc(row.get("date", ""))} · {esc(row.get("assembly", ""))}</span></div>'
                         f'<div class="source-quote">“{esc(row.get("excerpt_100", ""))}”</div>'
                         f'<div class="quiet">{esc(row.get("display_id", ""))} · '
                         f'{esc(row.get("verification_status", "자동 추출"))}</div></div>')
        cards.append('<p class="quiet">정책 관련성이 수동 검증되지 않은 원문 발췌입니다.</p>')
        return "".join(cards)

    def source_options(issue):
        rows = result(issue).get("evidence_records") or []
        return [(f"{str(r.get('meeting_title', ''))[:43]} · {r.get('date', '')} · {r.get('assembly', '')}",
                 str(r.get("display_id") or r.get("id") or i)) for i, r in enumerate(rows)]

    def show_source_detail(issue, key):
        rows = result(issue).get("evidence_records") or []
        row = next((r for i, r in enumerate(rows)
                    if str(r.get("display_id") or r.get("id") or i) == str(key)), None)
        if not row:
            return '<div class="empty-note">해당 근거 회의록을 찾지 못했습니다.</div>'
        snippet = row.get("excerpt_100", "")
        return m.detail_card(row.get("display_id", "근거"), row.get("assembly", ""), row.get("date", ""),
                             row.get("meeting_title", ""), row.get("id", ""), snippet, snippet,
                             "100자 발췌만 저장되어 있습니다. 원문 전체와 의미 연결은 별도 검증이 필요합니다.")

    def render_map(issue, month, selected_key=None):
        fig = go.Figure()
        events = visible_events(issue, month)
        for row in events:
            area = m.label(row.get("assembly", ""))
            if area not in m.REGIONS:
                continue
            lat, lon = m.REGIONS[area]
            active = str(row["key"]) == str(selected_key)
            fig.add_trace(go.Scattergeo(
                lat=[lat], lon=[lon], mode="markers+text", text=[area], textposition="top center",
                hovertext=[f"{esc(row.get('assembly',''))}<br>첫 관측 후보: {esc(row.get('date',''))}<br>{esc(row.get('meeting_title',''))}"],
                hoverinfo="text", marker=dict(size=16 if active else 11,
                                                color="#0e7981" if active else "#89babc",
                                                line=dict(width=1.4, color="white")), showlegend=False))
        # Only draw candidate similarity as dotted routes, never direction/causal arrows.
        pairs = result(issue).get("cross_region_similarity_candidates") or []
        for pair in pairs:
            a = pair.get("earlier_assembly") or pair.get("source_assembly") or pair.get("assembly_a")
            b = pair.get("later_assembly") or pair.get("target_assembly") or pair.get("assembly_b")
            area_a, area_b = m.label(a), m.label(b)
            if area_a not in m.REGIONS or area_b not in m.REGIONS or area_a == area_b:
                continue
            dates = [str(pair.get(k, "")) for k in ("earlier_date", "later_date") if pair.get(k)]
            if month != "전체 기간" and dates and max(d[:7] for d in dates) > month:
                continue
            aa, bb = m.REGIONS[area_a], m.REGIONS[area_b]
            fig.add_trace(go.Scattergeo(lat=[aa[0], bb[0]], lon=[aa[1], bb[1]], mode="lines",
                                        line=dict(color="#9fb3be", width=1.5, dash="dot"),
                                        hovertext=f"{esc(area_a)} ↔ {esc(area_b)} · 정책 내용 유사 후보 / 전파 미확인",
                                        hoverinfo="text", showlegend=False))
        fig.update_geos(projection_type="mercator", showland=True, landcolor="#f4f7f9",
                        showcountries=True, countrycolor="#d2dbe5", showcoastlines=True,
                        coastlinecolor="#9fb3c4", lonaxis_range=[124, 132], lataxis_range=[32.7, 39.1],
                        showframe=False, bgcolor="white")
        fig.update_layout(height=435, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="white",
                          font=dict(color="#26364a", family="Arial"))
        if not events:
            fig.add_annotation(text="선택한 시점에 관측된 지역이 없습니다.", x=.5, y=.5,
                               xref="paper", yref="paper", showarrow=False)
        return fig

    def page(query):
        t0 = time.perf_counter()
        issue = find_issue(query, dataset)
        if issue is None:
            raise gr.Error("이 JSON에 없는 질문입니다. 지원 이슈: " + ", ".join(dataset))
        period = "전체 기간"
        event_choices = choices(issue, period)
        first_key = event_choices[0][1] if event_choices else None
        records = source_options(issue)
        evidence_key = records[0][1] if records else None
        log(f"29번 화면 조회: {issue}, 지역 {len(event_choices)}건, {time.perf_counter()-t0:.3f}초")
        return (issue,
                f"**조회 이슈:** {issue} · 28 V2 통합 JSON · 자동 추출/수동 미검증",
                title_and_summary(issue),
                gr.update(choices=["전체 기간"] + months(issue), value=period),
                render_map(issue, period, first_key),
                gr.update(choices=event_choices, value=first_key),
                show_detail(issue, period, first_key),
                show_forecast(issue), show_sources(issue),
                gr.update(choices=records, value=evidence_key),
                show_source_detail(issue, evidence_key))

    def on_month(issue, month):
        if not issue or issue not in dataset:
            return render_map("", "전체 기간"), gr.update(choices=[], value=None), ""
        e = choices(issue, month)
        key = e[0][1] if e else None
        return render_map(issue, month, key), gr.update(choices=e, value=key), show_detail(issue, month, key)

    def on_event(issue, month, key):
        return show_detail(issue, month, key), render_map(issue, month, key)

    # Keep 23's build_app() unmodified; callbacks look up names in m.__dict__ when called.
    m.load_view = page
    m.on_month = on_month
    m.on_event = on_event
    m.first_detail = lambda issue: show_first_detail(issue)
    m.source_detail = lambda issue, key: show_source_detail(issue, key)
    m.bundle = lambda issue: issue   # 23 source_picker/first_more callbacks receive selected issue key.

    # Correct the heading's misleading causal implication without changing its visual design.
    m.CSS += "\n"  # keep established styling exactly.


def main():
    parser = argparse.ArgumentParser(description="23 Gradio 디자인 + 28 V2 JSON 통합 화면")
    parser.add_argument("--data", type=Path, help="28 V2의 all_issues.json (생략 시 최신 버전)")
    parser.add_argument("--original23", type=Path, default=BASELINE_23, help="23번 기존 UI 코드 경로")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="파일 및 데이터 검증 후 Gradio를 띄우지 않음")
    args = parser.parse_args()
    t0 = time.perf_counter()
    log("29번 시작: 통합 JSON 읽기")
    path = args.data or latest_json()
    dataset = load_data(path)
    log(f"읽은 파일: {path}")
    inspect_summary(dataset)
    if args.dry_run:
        if not args.original23.is_file():
            log(f"주의: 23번 UI 파일을 찾지 못했습니다: {args.original23}")
        log(f"DRY-RUN 종료 | 총 {time.perf_counter()-t0:.3f}초")
        return
    t1 = time.perf_counter()
    original = import_23(args.original23)
    install_integrated_callbacks(original, dataset, path)
    log(f"23번 UI 연결 완료 | {time.perf_counter()-t1:.3f}초")
    log(f"주소: http://127.0.0.1:{args.port} | 23번 파일 원본 수정 없음")
    major = int(original.gr.__version__.split('.')[0])
    launch_options = dict(server_name="127.0.0.1", server_port=args.port,
                          share=False, inbrowser=not args.no_browser)
    if major >= 6:
        launch_options.update(css=original.CSS, theme=original.gr.themes.Soft())
    try:
        original.build_app().launch(**launch_options)
    finally:
        log(f"29번 종료 | 전체 경과 {time.perf_counter()-t0:.3f}초")


if __name__ == "__main__":
    main()

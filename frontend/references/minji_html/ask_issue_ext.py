"""
이슈 흐름 질의응답 + 공공데이터(function calling)로 보강한 '향후 예측'

llm_baseline/ask_issue.py 와 같은 순서로 이슈·흐름·답변·지도 보고서를 만들되,
'향후 예측' 단계만 agent.py 로 바꾼다:
  LLM 이 공공데이터 도구(법제처 자치법규, 국회 의안)를 스스로 호출 → 코드가 DB 신호 + 외부 신호로 후보 순위 → LLM 이 근거 설명

  python ask_issue_ext.py -k 고교학점제
  python ask_issue_ext.py -q "늘봄학교가 지방의회들 사이에서 어떻게 퍼져 나갔어?"
  python ask_issue_ext.py -k 고교학점제 --as-of 2024-12-31      # 과거 시점 기준으로 예측 (그 뒤 자료는 안 씀)

  ※ --llm 은 도구 호출(tools)을 지원하는 Ollama 모델이어야 함: gemma4, qwen3, qwen2.5 (exaone3.5 는 미지원)
  ※ 검색·흐름 계산 옵션(--table --model --device --rr-device --rr-fp16 …)은 ask_issue.py 와 같음
"""
from __future__ import annotations

import argparse
import contextlib
import html
import io
import json
import os
import re
import sys
import time
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "llm_baseline"))
import psycopg                       # noqa: E402
import ask_issue as AI               # noqa: E402
import flow_report as FR             # noqa: E402
import issue_timeline as IT          # noqa: E402
from retrieval import Retriever      # noqa: E402

import agent as AG                   # noqa: E402


def ext_html(pred: dict) -> str:
    e = html.escape
    rows = "".join(f"<tr><td>{i + 1}</td><td>{e(c['assembly'])}</td><td>{c['score']}</td><td>{c['db_score']}</td>"
                   f"<td>{c['ext_score']}</td><td>{c['ext']['ord_wide']}</td><td>{c['ext']['ord_basic']}</td>"
                   f"<td>{'-' if c['ext'].get('need') is None else c['ext']['need']}</td></tr>"
                   for i, c in enumerate(pred["ranking"]))
    calls = "".join(f"<li>{e(t['tool'])}({e(str(t.get('keyword')))}) → {e(t['error']) if t.get('error') else str(t.get('count')) + '건'}</li>"
                    for t in pred["tool_calls"])
    ev = pred["evidence"].get(pred["assembly"], {})
    items = "".join(f"<li>[자치법규] {e(o['promulgated'] or '')} {e(o['org'])} 「{e(o['name'])}」</li>" for o in ev.get("ord_items", []))
    items += "".join(f"<li>[지역통계] {e(n['indicator'])} {n['value']:,}{e(n['unit'])} ({e(n['period'])}), 수요 점수 {n['need_score']:.2f}"
                     f"<br><span class='muted'>{e(n['attribution'])}</span></li>" for n in ev.get("need_items", []))
    srcs = "".join(f"<li><b>{e(s_['name'])}</b> — {e(s_['provider'])}<br>버전: {e(s_['version'])}<br>"
                   f"이용 조건: {e(s_['copyright'])} (약관 {e(s_['terms'])})<br>{e(s_['attribution'])}</li>" for s_ in pred.get("sources", []))
    srcs = srcs or "<li>외부 데이터 미사용 (공공데이터 인증값이 설정되지 않아 DB 신호만으로 예측)</li>"
    changed = (f"<p class='muted'>DB 신호만 쓴 예측 1위는 <b>{e(pred['db_only_top'])}</b>였고, 외부 자료를 더해 "
               f"<b>{e(pred['assembly'])}</b>(으)로 바뀌었습니다.</p>" if pred["db_only_top"] != pred["assembly"] else "")
    return (f"{('<p>' + e(pred['explanation']) + '</p>') if pred.get('explanation') else ''}{changed}"
            "<table><tr><th>순위</th><th>의회</th><th>점수</th><th>DB</th><th>외부</th><th>광역조례</th><th>기초조례</th><th>지역통계 수요</th></tr>"
            f"{rows}</table><div class='muted'>LLM 이 호출한 도구:</div><ul class='flow'>{calls or '<li>-</li>'}</ul>"
            f"<div class='muted'>1위 후보의 공공데이터 근거 ({e(pred['assembly'])}):</div><ul class='flow'>{items or '<li>-</li>'}</ul>"
            f"<div class='muted'>사용한 외부 데이터 (명칭·버전·이용 조건·출처):</div><ul class='flow'>{srcs}</ul>")


def answer(text, llm, tool_llm, conn, tok, model, rr, base, passthrough, verbose, keyword_mode, as_of, report_dir):
    T = {}
    t = time.perf_counter()
    retr = Retriever(conn, base.table, model=model, tokenizer=tok)
    x = AI.extract_issues(llm, text, retr, keyword_mode)
    T["이슈 찾기"] = time.perf_counter() - t
    if not x["issues"]:
        print("\n질문에서 정책 이슈를 찾지 못했습니다.")
        return None
    print("\n▶ 찾은 이슈: " + ", ".join(i["issue"] for i in x["issues"]))
    t = time.perf_counter()
    results, ref = [], 1
    for it in x["issues"]:
        res = AI.run_issue(it, x, llm, conn, tok, model, rr, passthrough, verbose, ref)
        if not res["rows"]:
            print(f"    → '{it['issue']}': 논의된 회의를 찾지 못했습니다.")
            continue
        if as_of:                                   # 과거 시점 기준: 그 뒤의 흐름은 보이지 않게
            res["rows"] = [r for r in res["rows"] if r["date"] <= as_of]
            if not res["rows"]:
                continue
        res["pred_db"] = res["pred"]
        print(f"  - {it['issue']}: 외부 데이터 수집 (도구 호출)")
        res["pred"] = AG.forecast_issue(tool_llm, it["issue"], it.get("variants", []), res["payload"], res["rows"], as_of)
        results.append(res)
        ref += len(res["rows"])
    T["흐름+외부 예측"] = time.perf_counter() - t
    if not results:
        return None
    t = time.perf_counter()
    reply, src, bad = AI.compose_answer(llm, text, results)
    T["답변 작성"] = time.perf_counter() - t

    print("\n================ 답변 ================\n" + reply)
    print("\n================ 타 지방의회로의 흐름 + 향후 예측(외부 데이터) ================")
    for res in results:
        print(f"[{res['issue']}]")
        for i, r in enumerate(res["rows"]):
            print(f"- {FR.flow_sentence(r, i == 0)} [근거 {r['ref']}]")
        p = res["pred"]
        if p:
            print(f"- (예측) {p['text']}")
            print(f"  DB만 쓴 1위: {p['db_only_top']} → 외부 자료 포함 1위: {p['assembly']}")
            for i, c in enumerate(p["ranking"][:3], 1):
                print(f"    {i}. {c['assembly']} 점수 {c['score']} (DB {c['db_score']} + 외부 {c['ext_score']}) {c['ext']}")
            if p.get("explanation"):
                print(f"  설명: {p['explanation']}")
            for s_ in p.get("sources", []):
                print(f"  {s_['attribution']}  [{s_['version']}]")
            if not p.get("external_used"):
                print("  (공공데이터 인증값이 없어 외부 자료 없이 DB 신호만으로 예측했습니다)")
    name = re.sub(r"[^0-9A-Za-z가-힣]+", "_", "_".join(r["issue"] for r in results))[:40]
    out = FR.render_html(text, reply, results, {r["issue"]: r["pred"] for r in results},
                         report_dir / f"flow_ext_{time.strftime('%Y%m%d_%H%M%S')}_{name}.html", f"{llm.model} (+도구 {tool_llm.model})",
                         ("LLM 답변이 검증을 통과하지 못해 계산 결과로 작성: " + ", ".join(bad[:3])) if src == "template" and bad else "",
                         extras={r["issue"]: ("공공데이터 근거", ext_html(r["pred"])) for r in results if r["pred"]})
    print(f"\n지도 보고서: {out.resolve()}")
    print("소요: " + " / ".join(f"{k} {v:.1f}s" for k, v in T.items()))
    return {"question": text, "answer": reply, "answer_source": src, "report": str(out),
            "results": [{"issue": r["issue"], "rows": [{k: v for k, v in x_.items() if k != "evidence"} for x_ in r["rows"]],
                         "prediction_db": r["pred_db"], "prediction_ext": r["pred"]} for r in results],
            "timing_sec": {k: round(v, 2) for k, v in T.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", "--keyword")
    ap.add_argument("-q", "--question")
    ap.add_argument("--llm", default=os.getenv("OLLAMA_MODEL", "gemma4:latest"), help="답변·요약 LLM")
    ap.add_argument("--tool-llm", default=None, help="도구 호출 LLM (기본: --llm 과 같음, tools 지원 모델이어야 함)")
    ap.add_argument("--as-of", default=None, help="이 날짜 기준으로 예측 (YYYY-MM-DD, 기본: 데이터 마지막 날)")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--report-dir", type=Path, default=HERE / "reports")
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    a, passthrough = ap.parse_known_args()

    llm = AI.Ollama(a.llm, num_ctx=a.num_ctx)
    ver = llm.check()
    tool_llm = AG.ToolLLM(a.tool_llm or llm.model, num_ctx=a.num_ctx)
    base = IT.build_parser().parse_args(["--issue", "_"] + passthrough)
    tok, model, rr = IT.load_models(base)
    llm.warmup()
    print(f"준비 완료 (LLM {llm.model} / 도구 {tool_llm.model} / Ollama {ver or '?'})")
    with psycopg.connect(base.dsn, autocommit=True) as conn:
        q, kw = (a.keyword, True) if a.keyword else (a.question, False)
        if not q:
            q, kw = input("질문 또는 키워드> ").strip(), None
        try:
            last = answer(q, llm, tool_llm, conn, tok, model, rr, base, passthrough, a.verbose, kw, a.as_of, a.report_dir)
        except (RuntimeError, urllib.error.URLError) as e:
            print(f"\n오류: {e}")
            last = None
        if a.save and last:
            a.save.write_text(json.dumps(last, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            print(f"저장: {a.save}")


if __name__ == "__main__":
    main()

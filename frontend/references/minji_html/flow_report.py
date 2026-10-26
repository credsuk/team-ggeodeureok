"""
이슈 확산 흐름 → 글 + 대한민국 지도(의회 위치 마킹) HTML 보고서

  ask_issue.py 가 계산한 결과(results)를 받아
  · 타 지방의회로의 흐름 문장 ("- 2024년 2월 27일, 광주광역시의회에서 논의")
  · 지도: A(최초) → B, C … (후속) 마커 + 날짜, 다음 확산 예측 의회는 점선 마커(D 향후 N일 예측)
  · 향후 예측: 코드가 계산한 신호(관련 언급이 이미 잡힌 의회, 최근 확산 지역과의 거리, 확산 간격)만 사용
을 HTML 한 파일로 저장한다. 외부 라이브러리·인터넷 없이 열림 (지도 윤곽은 아래 좌표로 직접 그림).

예측은 '사실'이 아니라 데이터 신호에 근거한 추정이며, 보고서에도 그렇게 표시한다.
"""
from __future__ import annotations

import html
import math
import re
import statistics
from datetime import date, datetime, timedelta
from pathlib import Path

import assembly_groups as AG

# 광역의회 소재지 (위도, 경도)
COUNCIL_COORDS = {
    "서울특별시의회": (37.566, 126.978), "부산광역시의회": (35.180, 129.076), "대구광역시의회": (35.871, 128.601),
    "인천광역시의회": (37.456, 126.705), "광주광역시의회": (35.160, 126.852), "대전광역시의회": (36.350, 127.385),
    "울산광역시의회": (35.538, 129.311), "세종특별자치시의회": (36.480, 127.289), "경기도의회": (37.264, 127.029),
    "강원특별자치도의회": (37.881, 127.730), "충청북도의회": (36.642, 127.489), "충청남도의회": (36.659, 126.673),
    "전북특별자치도의회": (35.824, 127.148), "전라남도의회": (34.816, 126.463), "경상북도의회": (36.576, 128.506),
    "경상남도의회": (35.228, 128.681), "제주특별자치도의회": (33.500, 126.531),
}
SHORT = {"서울특별시의회": "서울", "부산광역시의회": "부산", "대구광역시의회": "대구", "인천광역시의회": "인천",
         "광주광역시의회": "광주", "대전광역시의회": "대전", "울산광역시의회": "울산", "세종특별자치시의회": "세종",
         "경기도의회": "경기", "강원특별자치도의회": "강원", "충청북도의회": "충북", "충청남도의회": "충남",
         "전북특별자치도의회": "전북", "전라남도의회": "전남", "경상북도의회": "경북", "경상남도의회": "경남",
         "제주특별자치도의회": "제주"}

# 남한 윤곽 (경도, 위도) — 단순화한 해안선·휴전선. 시각화용이라 섬·만은 생략
MAINLAND = [
    (126.65, 37.78), (126.75, 37.86), (126.95, 37.96), (127.10, 38.10), (127.45, 38.30), (127.80, 38.30),
    (128.15, 38.40), (128.36, 38.62), (128.47, 38.60), (128.59, 38.21), (128.90, 37.76), (129.11, 37.52),
    (129.17, 37.44), (129.33, 37.15), (129.40, 36.99), (129.42, 36.70), (129.40, 36.41), (129.37, 36.18),
    (129.40, 36.04), (129.57, 36.08), (129.50, 35.85), (129.48, 35.70), (129.43, 35.50), (129.32, 35.32),
    (129.15, 35.15), (128.95, 35.05), (128.75, 35.00), (128.62, 34.85), (128.70, 34.72), (128.45, 34.80),
    (128.40, 34.85), (128.15, 34.88), (127.95, 34.80), (127.85, 34.90), (127.70, 34.70), (127.55, 34.80),
    (127.40, 34.50), (127.20, 34.60), (127.00, 34.45), (126.75, 34.30), (126.52, 34.30), (126.40, 34.50),
    (126.30, 34.70), (126.40, 34.85), (126.30, 35.00), (126.38, 35.20), (126.40, 35.35), (126.48, 35.55),
    (126.70, 35.62), (126.62, 35.80), (126.60, 35.97), (126.55, 36.10), (126.52, 36.35), (126.48, 36.55),
    (126.15, 36.70), (126.30, 36.88), (126.60, 37.00), (126.85, 36.97), (126.70, 37.15), (126.62, 37.30),
    (126.55, 37.40), (126.50, 37.55), (126.45, 37.70), (126.55, 37.76),
]
JEJU = (126.55, 33.38, 0.34, 0.16)          # 중심 경도, 위도, 반경(경도), 반경(위도)
ULLEUNG = (130.87, 37.50, 0.06, 0.04)

LON0, LAT0, LAT1 = 124.9, 38.9, 33.0
KY = 92.0                                     # 위도 1도당 픽셀
KX = KY * math.cos(math.radians(36))          # 경도 1도당 픽셀 (위도 36도 기준 보정)
PAD = 30
W = int((131.2 - LON0) * KX) + PAD * 2
H = int((LAT0 - LAT1) * KY) + PAD * 2

ACTION_VERB = {"질의": "질의로 다룸", "5분 자유발언": "5분 자유발언으로 제기", "도정·시정 질문": "도정·시정질문으로 제기",
               "업무보고": "업무보고에서 다룸", "조례 심사": "조례안 심사", "예산 심사": "예산 심사에서 논의",
               "건의안·결의안": "건의안·결의안으로 다룸", "공청회·토론회": "공청회·토론회 실시", "기타 논의": "논의"}


def xy(lat: float, lon: float) -> tuple:
    return PAD + (lon - LON0) * KX, PAD + (LAT0 - lat) * KY


def km(a: str, b: str) -> float:
    (la1, lo1), (la2, lo2) = COUNCIL_COORDS[a], COUNCIL_COORDS[b]
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def md_bold(s: str) -> str:
    """LLM 답변의 마크다운 **굵게**·제목 기호를 HTML 로 (이미 escape 된 문자열에 적용)"""
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r"(?m)^#{1,4}\s*", "", s)


def kdate(s: str) -> str:
    d = date.fromisoformat(str(s)[:10])
    return f"{d.year}년 {d.month}월 {d.day}일"


def flow_sentence(r: dict, first: bool) -> str:
    """'- 2024년 3월 22일, 전라북도의회에서 지역 소멸 관점에서 재논의' 형태"""
    verb = ACTION_VERB.get(r.get("행동"), "논의")
    if verb == "논의" and not first:
        verb = "재논의"
    view = (r.get("관점") or "").strip()
    head = f"{kdate(r['date'])}, {r['label']}에서 "
    s = head + (f"{view} 관점에서 {verb}" if view else verb)
    if r.get("level") == "언급":
        s += " (언급 수준)"
    return s + (" — 최초" if first else "")


# ----------------------------------------------------------------------------- 예측 (코드 계산)
def rank_candidates(payload: dict, rows: list, as_of: str | None = None) -> dict | None:
    """
    아직 논의가 확인되지 않은 의회 전체에 DB 신호 점수를 매긴다 (외부 데이터 예측 llm_external 이 이 결과에 더함)
      db_score = 관련 언급 신호(검증에서 빠진 키워드 언급·의미 유사 회의 수, 최대 3) × 2
               + 최근 확산 3곳과의 거리 근접도(0~1, 400km 이상이면 0)
      시점    = 의회 간 첫 논의 간격의 중앙값 (30~180일)
    as_of 를 주면 그 날짜까지의 회의만 사용 (과거 시점 기준 백테스트용)
    반환: {"candidates": [{assembly, db_score, signal, near, dist_km}, …점수순], "days", "gap", "trend", "from_date"}
    """
    rows = [r for r in rows if as_of is None or r["date"] <= as_of]
    done = {r["assembly"] for r in rows}
    cands = [a for a in COUNCIL_COORDS if a not in done]
    if not rows or not cands:
        return None
    recent = [r["assembly"] for r in sorted(rows, key=lambda r: r["date"])[-3:] if r["assembly"] in COUNCIL_COORDS]
    signal = {}
    for m in payload.get("meetings") or []:
        if as_of and m["meeting_date"] > as_of:
            continue
        if m["assembly"] in cands and (m["level"].startswith("제외") or m["level"].startswith("의미")):
            signal[m["assembly"]] = signal.get(m["assembly"], 0) + 1
    out = []
    for a in cands:
        near = min(recent, key=lambda b: km(a, b)) if recent else None
        d = km(a, near) if near else 999
        prox = max(0.0, 1 - d / 400)
        out.append({"assembly": a, "db_score": round(min(signal.get(a, 0), 3) * 2 + prox, 3),
                    "signal": signal.get(a, 0), "near": near, "dist_km": round(d)})
    out.sort(key=lambda c: -c["db_score"])
    dates = sorted(date.fromisoformat(r["date"]) for r in rows)
    gaps = [(y - x).days for x, y in zip(dates, dates[1:]) if (y - x).days > 0]
    gap = statistics.median(gaps) if gaps else 90
    q = [x for x in (payload.get("quarterly") or []) if as_of is None or x["quarter"] <= quarter_of(as_of)]
    trend = None
    if len(q) >= 4:
        recent_m, prev_m = sum(x["meetings"] for x in q[-2:]), sum(x["meetings"] for x in q[-4:-2])
        trend = "증가" if recent_m > prev_m * 1.2 else ("감소" if recent_m < prev_m * 0.8 else "유지")
    return {"candidates": out, "days": int(min(180, max(30, round(gap / 10) * 10))), "gap": gap, "has_gaps": bool(gaps),
            "trend": trend, "from_date": as_of or payload.get("data_range", [None, None])[1]}


def quarter_of(d: str) -> str:
    """issue_timeline 의 분기 표기(예: 2025Q3)와 같은 형식"""
    x = date.fromisoformat(str(d)[:10])
    return f"{x.year}Q{(x.month - 1) // 3 + 1}"


def predict_next(payload: dict, rows: list) -> dict | None:
    """DB 신호만으로 다음 확산 후보 1곳 + 시점(향후 N일) — rank_candidates 의 1위"""
    rk = rank_candidates(payload, rows)
    if not rk:
        return None
    top = rk["candidates"][0]
    a, near, d, sig = top["assembly"], top["near"], top["dist_km"], top["signal"]
    basis = []
    if sig:
        basis.append(f"{a}에서 이미 관련 언급 {sig}건이 잡혔으나 아직 '논의' 수준은 아님")
    if near:
        basis.append(f"가장 최근 확산 지역 중 {near}와 약 {d:.0f}km")
    basis.append(f"의회 간 첫 논의 간격 중앙값 약 {rk['gap']:.0f}일" if rk["has_gaps"] else "확산 간격 자료 부족(기본 90일 가정)")
    if rk["trend"]:
        basis.append(f"최근 2분기 논의 회의 수 {rk['trend']} 추세")
    end, days = rk["from_date"], rk["days"]
    return {"assembly": a, "days": days, "from_date": end, "near": near, "dist_km": d, "signal": sig,
            "trend": rk["trend"], "basis": basis, "score": top["db_score"],
            "alternatives": [c["assembly"] for c in rk["candidates"][1:3]],
            "text": f"{end} 기준 향후 약 {days}일 안에 {a}로 확산될 가능성 (근거: {'; '.join(basis)}) ※ 추정"}


# ----------------------------------------------------------------------------- 지도 SVG
def _label_box(x, y, text, dx, dy, anchor):
    w = sum(12 if ord(ch) > 0x1100 else 7 for ch in text)
    left = x + dx - (w if anchor == "end" else (w / 2 if anchor == "middle" else 0))
    return (left, y + dy - 12, left + w, y + dy + 3)


def _overlap(b, boxes):
    return any(not (b[2] < o[0] or b[0] > o[2] or b[3] < o[1] or b[1] > o[3]) for o in boxes)


def map_svg(rows: list, pred: dict | None) -> str:
    pts = " ".join(f"{xy(la, lo)[0]:.1f},{xy(la, lo)[1]:.1f}" for lo, la in MAINLAND)
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="이슈 확산 지도" class="map">',
             f'<polygon points="{pts}" class="land"/>']
    for lo, la, rx, ry in (JEJU, ULLEUNG):
        cx, cy = xy(la, lo)
        parts.append(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx * KX:.1f}" ry="{ry * KY:.1f}" class="land"/>')
    for a, (la, lo) in COUNCIL_COORDS.items():          # 모든 의회 위치를 옅게
        cx, cy = xy(la, lo)
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" class="dot"><title>{a}</title></circle>')

    marks = [r for r in sorted(rows, key=lambda r: r["date"]) if r["assembly"] in COUNCIL_COORDS]
    path = [xy(*COUNCIL_COORDS[r["assembly"]]) for r in marks]
    if len(path) > 1:
        parts.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in path) + '" class="route"/>')
    items = [(i, r, "first" if i == 0 else "next") for i, r in enumerate(marks)]
    if pred and pred["assembly"] in COUNCIL_COORDS:
        px, py = xy(*COUNCIL_COORDS[pred["assembly"]])
        if pred.get("near") in COUNCIL_COORDS:
            nx, ny = xy(*COUNCIL_COORDS[pred["near"]])
            parts.append(f'<line x1="{nx:.1f}" y1="{ny:.1f}" x2="{px:.1f}" y2="{py:.1f}" class="predline"/>')
        items.append((len(marks), {"assembly": pred["assembly"], "date": None}, "pred"))
    # 1) 마커 위치를 먼저 모두 정한다 — 세종·대전처럼 가까운 의회는 겹치지 않게 조금 옮기고 실제 위치와 선으로 잇는다
    placed = []
    for i, r, kind in items:
        x0, y0 = xy(*COUNCIL_COORDS[r["assembly"]])
        x, y = x0, y0
        for k in range(8):
            if all(math.hypot(x - px_, y - py_) >= 25 for px_, py_ in placed):
                break
            ang = math.radians(-60 + k * 45)
            x, y = x0 + 26 * math.cos(ang), y0 + 26 * math.sin(ang)
        placed.append((x, y))
        if (x, y) != (x0, y0):
            parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y:.1f}" class="leader"/>'
                         f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="3.5" class="anchor {kind}"/>')
    # 2) 라벨은 모든 마커·앞서 놓은 라벨과 겹치지 않는 자리에
    boxes = [(x - 14, y - 14, x + 14, y + 14) for x, y in placed]
    for (i, r, kind), (x, y) in zip(items, placed):
        letter = chr(ord("A") + i) if i < 26 else str(i + 1)
        if kind == "pred":
            text = f"{letter} {SHORT[r['assembly']]} 향후 {pred['days']}일 예측"
        else:
            text = f"{letter} {r['date']} {SHORT[r['assembly']]} {'최초' if kind == 'first' else '후속'}"
        for dx, dy, anc in ((18, -10, "start"), (18, 16, "start"), (-18, -10, "end"), (-18, 16, "end"),
                            (0, -20, "middle"), (0, 30, "middle"), (18, 34, "start"), (-18, 34, "end"),
                            (18, -28, "start"), (-18, -28, "end")):
            b = _label_box(x, y, text, dx, dy, anc)
            if not _overlap(b, boxes):
                break
        boxes.append(b)
        parts.append(f'<g class="mk {kind}"><circle cx="{x:.1f}" cy="{y:.1f}" r="12"/>'
                     f'<text x="{x:.1f}" y="{y + 4.5:.1f}" class="letter">{letter}</text>'
                     f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anc}" class="lbl">{html.escape(text)}</text>'
                     f'<title>{html.escape(r["assembly"])}</title></g>')
    parts.append("</svg>")
    return "\n".join(parts)


# ----------------------------------------------------------------------------- HTML
CSS = """
:root{--bg:#fafaf7;--card:#fff;--ink:#1f2328;--muted:#6b7280;--line:#e5e7eb;--land:#d9dadc;--landedge:#c3c5c8;
--first:#c0463a;--next:#13806f;--pred:#b5651d;--route:#9aa3ad}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#16181c;--card:#1f2228;--ink:#e6e6e6;
--muted:#9aa0a6;--line:#30343b;--land:#3a3e45;--landedge:#4a4f57;--route:#6b737d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:"Malgun Gothic","Apple SD Gothic Neo",Arial,sans-serif;line-height:1.6}
main{max-width:1280px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:18px;margin:0 0 12px}.sub{color:var(--muted);font-size:13px;margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin:0 0 18px}
.answer{white-space:pre-wrap;font-size:14.5px}
.issue{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,520px);gap:20px;align-items:start}
@media (max-width:900px){.issue{grid-template-columns:1fr}}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{border:1px solid var(--line);padding:8px 10px;vertical-align:top;text-align:left}
th{background:color-mix(in srgb,var(--ink) 6%,transparent);white-space:nowrap;width:120px}
ul.flow{margin:0;padding-left:0;list-style:none}ul.flow li{margin:0 0 4px}
.tag{display:inline-block;font-size:11px;padding:1px 6px;border-radius:9px;color:#fff;margin-right:6px}
.tag.first{background:var(--first)}.tag.next{background:var(--next)}.tag.pred{background:var(--pred)}
b.pred{color:var(--pred)}.muted{color:var(--muted);font-size:12.5px}
.map{width:100%;height:auto;display:block}.land{fill:var(--land);stroke:var(--landedge);stroke-width:1}
.dot{fill:var(--muted);opacity:.35}.route{fill:none;stroke:var(--route);stroke-width:1.4;stroke-dasharray:2 4}
.predline{stroke:var(--pred);stroke-width:2;stroke-dasharray:6 5}.leader{stroke:var(--muted);stroke-width:1}
.anchor{fill:var(--muted)}.anchor.first{fill:var(--first)}.anchor.next{fill:var(--next)}.anchor.pred{fill:var(--pred)}
.mk circle{stroke:#fff;stroke-width:2}.mk.first circle{fill:var(--first)}.mk.next circle{fill:var(--next)}
.mk.pred circle{fill:var(--pred)}.letter{fill:#fff;font-size:12px;font-weight:700;text-anchor:middle}
.lbl{font-size:12px;font-weight:700;paint-order:stroke;stroke:var(--card);stroke-width:3px}
.mk.first .lbl{fill:var(--first)}.mk.next .lbl{fill:var(--next)}.mk.pred .lbl{fill:var(--pred)}
.ev{font-size:12.5px;color:var(--muted)}
"""


def render_html(question: str, answer_text: str, results: list, preds: dict, out: Path, model: str = "",
                answer_note: str = "", extras: dict | None = None) -> Path:
    """extras: {이슈: (칸 제목, HTML)} — 표에 칸을 덧붙일 때 (예: llm_external 의 외부 데이터 근거)"""
    from ask_issue import minutes_of     # 순환 import 방지: 호출 시점에 가져옴
    e = html.escape
    secs = []
    for res in results:
        rows, p = res["rows"], res["payload"]
        pred = preds.get(res["issue"])
        first = rows[0]
        flow = "".join(f'<li><span class="tag {"first" if i == 0 else "next"}">{chr(65 + i)}</span>{e(flow_sentence(r, i == 0))}'
                       f' <span class="muted">[근거 {r["ref"]}]</span></li>' for i, r in enumerate(rows))
        if pred:
            letter = chr(65 + len(rows))
            pred_html = (f'<span class="tag pred">{letter}</span><b class="pred">{e(pred["assembly"])}</b> — '
                         f'{e(pred["from_date"])} 기준 향후 약 {pred["days"]}일 안에 확산 가능성 (추정)'
                         f'<div class="muted">근거: {e("; ".join(pred["basis"]))}'
                         + (f' · 다음 후보: {e(", ".join(pred["alternatives"]))}' if pred["alternatives"] else "")
                         + '</div>')
        else:
            pred_html = '<span class="muted">모든 광역의회에서 이미 논의가 확인되어 추가 확산 예측 없음</span>'
        refs = "".join(f'<li>[근거 {r["ref"]}] {e(minutes_of(r))} · {e(r["date"])} · {e(r["label"])}'
                       + (f' · {e(r["evidence"].get("speaker") or "")}<div class="ev">…{e(r["evidence"]["excerpt"][:150])}…</div>'
                          if r.get("evidence") else "") + '</li>' for r in rows)
        secs.append(f"""
<section class="card"><h2>이슈: {e(res["issue"])}</h2>
<div class="issue"><div><table>
<tr><th>이슈키워드</th><td>{e(res["issue"])}<div class="muted">검색 표기: {e(", ".join(p["forms"][:8]))}</div></td></tr>
<tr><th>등장시점</th><td>{e(kdate(first["date"]))}, {e(first["label"])} — 「{e(minutes_of(first))}」
<div class="muted">제공 데이터 범위 {e(p["data_range"][0])} ~ {e(p["data_range"][1])} 안에서의 최초 관측{(" · ⚠ " + e(p["caveat"])) if p.get("caveat") else ""}</div></td></tr>
<tr><th>타 지방의회로의 흐름</th><td><ul class="flow">{flow}</ul></td></tr>
<tr><th>향후 예측</th><td>{pred_html}</td></tr>
{(f'<tr><th>{e(extras[res["issue"]][0])}</th><td>{extras[res["issue"]][1]}</td></tr>') if extras and res["issue"] in extras else ""}
<tr><th>근거 회의록</th><td><ul class="flow">{refs}</ul></td></tr>
</table></div>
<div>{map_svg(rows, pred)}
<div class="muted">A = 최초 논의, B 이후 = 후속 논의(시간순), 점선 마커 = 코드가 계산한 다음 확산 예측(사실 아님)</div></div>
</div></section>""")
    doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>이슈 확산 흐름</title><style>{CSS}</style></head>
<body><main><h1>이슈 확산 흐름</h1>
<p class="sub">질문: {e(question)} · 생성 {datetime.now():%Y-%m-%d %H:%M}{(" · LLM " + e(model)) if model else ""}</p>
<section class="card"><h2>답변</h2>{('<p class="muted">※ ' + e(answer_note) + '</p>') if answer_note else ""}<div class="answer">{md_bold(e(answer_text))}</div></section>
{"".join(secs)}
<p class="muted">날짜·의회·순서·근거는 회의록 DB에서 코드가 계산한 값이고, 관점·행동 요약과 답변 문장은 로컬 LLM 이 근거 발언만 보고 작성했습니다.
'향후 예측'은 확산 간격·지역 거리·관련 언급 신호로 계산한 추정입니다.</p>
</main></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out

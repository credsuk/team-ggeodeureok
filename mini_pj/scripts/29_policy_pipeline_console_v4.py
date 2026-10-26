# -*- coding: utf-8 -*-
"""29-2 정책 이슈 분석 콘솔 V4 — V3 SQLite/Chroma 인덱스 재사용.

목표: 자유 정책어 + 동의어 → 근거 검색 → 최초 '관측 후보' → 정책수단별
타 의회 후속 논의 후보 → 의정활동 표현 후보 → 로컬 EXAONE 조건부 전망.

중요: 어휘 일치/정책수단 공통/시간 순서는 실제 정책 채택·전파의 증거가 아니다.
자동 판정은 수동 검토를 위한 후보이며, 2025-12-31 이후 자료는 기본적으로 배제한다.

실행 예시 (먼저 V3에서 --build-index 완료):
  python 29_policy_pipeline_console_v4.py --check
  python 29_policy_pipeline_console_v4.py --issue "청년 일자리" --cutoff 2025-12-31
  python 29_policy_pipeline_console_v4.py --issue "청년 주거" --aliases "청년 월세,월세 지원,청년 전세" --cutoff 2025-12-31
  python 29_policy_pipeline_console_v4.py --issue "돌봄" --no-llm

필요 라이브러리: chromadb, sentence-transformers, numpy, requests (V3와 동일)
SQLite 색인 파일: C:\\mini_pj\\results\\policy_search_index\\meetings_fts.sqlite3
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import csv
import importlib.util
import json
from pathlib import Path
import re
import sqlite3
import time
from difflib import SequenceMatcher

HERE = Path(__file__).resolve().parent
V3_PATH = HERE / "29_policy_pipeline_console_v3.py"
ROOT = Path(r"C:\mini_pj")
MODEL = "exaone3.5:2.4b"

# 5개 주제는 편의상 동의어만 기본 제안. 자유 검색어를 제한하지 않음.
DEFAULT_ALIASES = {
    "청년 일자리": ["청년 일자리", "청년일자리", "청년 취업", "청년취업", "청년 고용", "청년고용", "청년 구직", "청년구직", "청년 창업", "청년창업"],
    "지역소멸": ["지역소멸", "지방소멸", "인구소멸", "인구감소지역"],
    "저출생": ["저출생", "저출산", "출산율", "출생률"],
    "고령화": ["고령화", "초고령", "고령사회", "노인 인구"],
    "기후위기": ["기후위기", "기후 위기", "기후변화", "탄소중립", "온실가스"],
}
# 정책 주제에 관계없이 구체적 '수단'을 찾기 위한 보수적 패턴.
MECHANISMS = {
    "직무교육·훈련": [r"직무\s*(?:교육|훈련)", r"직업\s*훈련", r"취업\s*훈련"],
    "인턴·일경험": [r"인턴십", r"청년\s*인턴", r"일\s*경험\s*(?:사업|지원|프로그램)"],
    "창업자금·공간": [r"창업\s*(?:자금|공간|센터|지원금|인큐베이터|보육)"],
    "채용·취업연계": [r"채용\s*연계", r"취업\s*연계", r"기업\s*매칭"],
    "구직활동·수당": [r"구직\s*활동\s*지원", r"구직\s*수당", r"취업\s*준비\s*지원"],
    "청년월세·주거비": [r"청년\s*월세", r"월세\s*(?:특별)?지원", r"전[·\s]?월세\s*자금"],
    "주거·임대주택": [r"공공\s*임대\s*주택", r"주거\s*안정\s*(?:사업|지원)"],
    "출산·양육지원": [r"출산\s*지원\s*금", r"양육\s*수당", r"육아\s*지원", r"돌봄\s*서비스"],
    "고령돌봄·의료": [r"노인\s*돌봄", r"방문\s*간호", r"치매\s*(?:예방|관리|지원)", r"스마트\s*경로당"],
    "탄소감축·중립": [r"탄소\s*중립\s*(?:실천|사업|계획)", r"온실\s*가스\s*감축", r"재생\s*에너지\s*(?:확대|보급)"],
    "지방소멸대응기금": [r"지방\s*소멸\s*대응\s*기금", r"지방\s*소멸\s*기금"],
}
ACTIVITY_CUES = {
    "조례 발의·제안 후보": [r"(?:조례안|개정조례안).{0,40}(?:발의|제안\s*설명)", r"(?:발의|제안\s*설명).{0,40}(?:조례안|개정조례안)"],
    "조례 의결 후보": [r"(?:조례안|개정조례안).{0,60}(?:원안\s*가결|수정\s*가결|의결)", r"(?:원안\s*가결|수정\s*가결).{0,40}(?:조례안|개정조례안)"],
    "공청회 언급 후보": [r"공청회"],
    "토론회 언급 후보": [r"토론회"],
    "행정사무감사 언급 후보": [r"행정\s*사무\s*감사"],
    "예산·사업 검토 후보": [r"(?:예산|사업).{0,20}(?:심사|검토|삭감|증액)"],
}
TRANSFER_PATTERNS = [r"(?:타\s*(?:시도|지역|지자체)|다른\s*(?:시도|지역|지자체)).{0,60}(?:벤치마킹|사례|참고|도입)",
                     r"(?:벤치마킹|사례|참고).{0,60}(?:타\s*(?:시도|지역)|다른\s*(?:지역|시도))"]
REVIEW_COLUMNS = ["pair_id", "issue", "earlier_date", "earlier_assembly", "earlier_id",
                  "later_date", "later_assembly", "later_id", "shared_mechanisms",
                  "similarity_candidate", "earlier_excerpt_180", "later_excerpt_180",
                  "transfer_cue_in_later", "review_status", "review_note"]


def load_v3():
    if not V3_PATH.exists():
        raise FileNotFoundError(f"기존 V3 코드가 같은 scripts 폴더에 필요합니다: {V3_PATH}")
    spec = importlib.util.spec_from_file_location("policy_console_v3_module", V3_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def normalize_aliases(issue, user_text, disable_defaults):
    xs = [issue]
    if not disable_defaults:
        xs.extend(DEFAULT_ALIASES.get(issue, []))
    xs.extend(s.strip() for s in user_text.split(",") if s.strip())
    return list(dict.fromkeys(x for x in xs if x))


def mechanisms(text):
    return sorted(name for name, patterns in MECHANISMS.items() if any(re.search(p, text, re.I) for p in patterns))


def activity_candidates(text, matched_term, limit=180):
    """정책어가 실제로 들어 있는 문맥(±300자)에서만 활동 단서를 찾는다."""
    pos = text.casefold().find(matched_term.casefold()) if matched_term else -1
    if pos < 0:
        return []
    context = text[max(0, pos-300):pos+300]
    items = []
    for label, patterns in ACTIVITY_CUES.items():
        for pattern in patterns:
            hit = re.search(pattern, context, re.I | re.S)
            if hit:
                start = max(0, hit.start() - 35)
                items.append({"activity_type": label, "cue_excerpt": context[start:start+limit],
                              "verification_status": "자동 표현 탐지·행사/조례 상태 미확정"})
                break
    return items


def clean_row(v3, row, aliases):
    text = row.get("full_text", "")
    match = row.get("matched_term") or v3.matched_alias(text, aliases)
    if not match:
        return None
    return {"id": row["id"], "date": row["date"], "assembly": row["assembly"],
            "meeting_title": row.get("meeting_title", ""), "source_id": row.get("source_id", ""),
            "matched_term": match, "excerpt_180": v3.centered_excerpt(text, match, 180),
            "policy_mechanisms": mechanisms(text), "activity_cues": activity_candidates(text, match),
            "full_text": text,
            **({"distance": row["distance"]} if "distance" in row else {})}


def without_full_text(row):
    return {k: v for k, v in row.items() if k != "full_text"}


def search_lexical_exact(index, aliases, cutoff, cap, v3):
    """SQLite 원문 부분문자열 검색. FTS 형태소 분할 누락 방지.

    날짜 범위+LIKE 검색. 검색 상한을 만나면 최초 '전체'이라고 표시하지 않음.
    """
    con = sqlite3.connect(str(index))
    con.row_factory = sqlite3.Row
    try:
        meta = dict(con.execute("SELECT name,value FROM info").fetchall())
        if meta.get("complete") != "yes":
            raise RuntimeError("SQLite 검색 인덱스가 완성되지 않았습니다")
        # SQLite LIKE는 한글 부분 문자열도 찾음. 문자열에 LIKE escape 적용.
        clauses = []
        params = []
        for word in aliases:
            escaped = word.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            clauses.append("body LIKE ? ESCAPE '\\'")
            params.append("%" + escaped + "%")
        if not clauses:
            return [], True
        sql = ("SELECT id,date,assembly,meeting_title,source_id,body FROM docs "
               "WHERE date != '' AND date <= ? AND assembly != '' AND (" + " OR ".join(clauses) + ") "
               "ORDER BY date,assembly,id LIMIT ?")
        params = [cutoff] + params + [cap+1]
        found = list(con.execute(sql, params))
        complete = len(found) <= cap
        rows = []
        for r in found[:cap]:
            text = str(r["body"] or "")
            term = v3.matched_alias(text, aliases)
            if not term:
                continue
            rows.append({"id": r["id"], "date": r["date"], "assembly": r["assembly"],
                         "meeting_title": r["meeting_title"], "source_id": r["source_id"],
                         "matched_term": term, "full_text": text})
        return rows, complete
    finally:
        con.close()


def dedup(rows):
    seen = set()
    out = []
    for row in sorted(rows, key=lambda z: (z["date"], z["assembly"], z["id"])):
        key = (row["date"], row["assembly"], row.get("source_id") or row.get("meeting_title") or row["id"])
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def compare_candidates(issue, rows, max_pairs=20, max_rows=350):
    """동일 구체적 수단 + 시간 선후 + 다른 의회인 경우만 유사 논의 후보.

    문자 유사도는 문서 단위가 아닌 표시 근거 180자끼리의 보조 점수. 실제 전파 판정 금지.
    """
    selected = rows[:max_rows]
    by_mechanism = defaultdict(list)
    for r in selected:
        for m in r["policy_mechanisms"]:
            by_mechanism[m].append(r)
    pairs, seen = [], set()
    for mech, group in by_mechanism.items():
        group = sorted(group, key=lambda z:(z["date"], z["assembly"]))
        for later_idx in range(1, len(group)):
            later = group[later_idx]
            earlier_options = [r for r in group[:later_idx] if r["assembly"] != later["assembly"]
                               and r["date"] < later["date"]]
            # 근거가 충분히 있을 때 같은 수단의 가장 이른 지역과 비교
            if not earlier_options:
                continue
            earlier = earlier_options[0]
            key = (earlier["id"], later["id"])
            if key in seen:
                continue
            seen.add(key)
            shared = sorted(set(earlier["policy_mechanisms"]) & set(later["policy_mechanisms"]))
            if not shared:
                continue
            score = round(SequenceMatcher(None, earlier["excerpt_180"], later["excerpt_180"]).ratio(), 3)
            transfer = any(re.search(pattern, later["full_text"], re.I | re.S) for pattern in TRANSFER_PATTERNS)
            pairs.append({"pair_id": "P"+str(len(pairs)+1).zfill(3), "issue": issue,
                          "earlier_date": earlier["date"], "earlier_assembly": earlier["assembly"],
                          "earlier_id": earlier["id"], "later_date": later["date"],
                          "later_assembly": later["assembly"], "later_id": later["id"],
                          "shared_mechanisms": ", ".join(shared), "similarity_candidate": score,
                          "earlier_excerpt_180": earlier["excerpt_180"],
                          "later_excerpt_180": later["excerpt_180"],
                          "transfer_cue_in_later": bool(transfer),
                          "review_status": "미검증", "review_note": ""})
    return sorted(pairs, key=lambda r:(-int(r["transfer_cue_in_later"]),-r["similarity_candidate"],r["later_date"]))[:max_pairs]


def call_ollama(url, model, prompt, max_tokens=220):
    import requests
    payload = {"model":model,"prompt":prompt,"stream":False, "think":False,
               "options":{"temperature":0,"seed":42,"num_ctx":8192,"num_predict":max_tokens}}
    res = requests.post(url.rstrip("/")+"/api/generate", json=payload, timeout=420)
    res.raise_for_status()
    j = res.json()
    return str(j.get("response") or "").strip()


def forecast_live(url, issue, cutoff, evidence, flow, pairs, monthly, model, no_llm):
    if no_llm:
        return {"status":"SKIPPED", "original":"", "rewritten":"", "forecast_150":"", "attempts":0,"seconds":0.0}
    if not evidence:
        return {"status":"NO_EVIDENCE", "original":"", "rewritten":"", "forecast_150":"", "attempts":0,"seconds":0.0}
    t0 = time.perf_counter()
    evidence_text = "\n".join(f"[E{i}] {r['date']} {r['assembly']} {r['meeting_title']} ID={r['id']}\n{r['excerpt_180']}"
                              for i, r in enumerate(evidence[:8], 1))
    flow_text = "\n".join(f"{r['date']} {r['assembly']} / {', '.join(r['policy_mechanisms']) or '정책수단 미분류'}"
                          for r in flow[:12])
    pairs_text = "\n".join(f"{x['earlier_assembly']}({x['earlier_date']}) → {x['later_assembly']}({x['later_date']}) : {x['shared_mechanisms']}"
                           for x in pairs[:8]) or "확인된 유사 수단 후보 없음"
    monthly_text = ", ".join(f"{m}: {n}건" for m, n in monthly[-12:])
    prompt = (f"지방의회 회의록 분석가다. 주제: {issue}. 분석 기준일: {cutoff}.\n"
              "다음 근거 이외의 특정 사실·사업·수치를 만들지 마라. 단순 날짜순 관측은 정책 전파의 증거가 아니다.\n"
              "유사 정책수단은 자동 후보이며 전파로 확정하지 말라. 아직 개최·가결 확인되지 않은 의정활동을 완료했다고 쓰지 마라.\n"
              "관측된 흐름을 근거로 향후 논의 가능성을 조건부로 전망하라. 반드시 한국어 150자 이내 평서문 1~2개만 출력하라.\n"
              f"지역별 최초 후보:\n{flow_text}\n유사 수단 후보:\n{pairs_text}\n월별 회의록 수:\n{monthly_text}\n근거:\n{evidence_text}\n전망:")
    original = call_ollama(url, model, prompt)
    rewritten = ""
    forecast = original if 0 < len(original) <= 150 else ""
    attempts = 1
    if not forecast and original:
        rewrite_prompt = ("아래 초안의 근거를 왜곡하지 말고, 새로운 내용이나 고유명사 없이 150자 이내 한국어 평서문 1~2개로 재작성하라.\n"
                          "미래 논의는 조건부 표현으로 쓰고, 출력은 최종 문장만.\n"+original)
        rewritten = call_ollama(url, model, rewrite_prompt, 150)
        attempts = 2
        forecast = rewritten if 0 < len(rewritten) <= 150 else ""
    return {"status":"OK" if forecast else "OVER_150_OR_EMPTY", "original":original,
            "rewritten":rewritten, "forecast_150":forecast,"attempts":attempts,
            "original_length":len(original),"rewritten_length":len(rewritten),
            "seconds":round(time.perf_counter()-t0, 2),
            "verification_status":"LLM 조건부 전망·원문 부합성 수동 미검증"}


def main():
    p = argparse.ArgumentParser(description="29 V4 지방정책 이슈 검색·유사 논의 후보·EXAONE 검증")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--issue", default="청년 일자리")
    p.add_argument("--aliases", default="", help="쉼표 구분 동의어")
    p.add_argument("--no-default-aliases", action="store_true")
    p.add_argument("--cutoff", default="2025-12-31")
    p.add_argument("--top-k", type=int, default=60)
    p.add_argument("--max-lexical", type=int, default=25000)
    p.add_argument("--max-pairs", type=int, default=20)
    p.add_argument("--max-pair-docs", type=int, default=350)
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--model", default=MODEL)
    args = p.parse_args()
    t0 = time.perf_counter()
    v3 = load_v3()
    cutoff = v3.date_ok(args.cutoff)
    if cutoff != args.cutoff:
        p.error("--cutoff는 YYYY-MM-DD 형식이어야 합니다")
    if args.top_k < 1 or args.max_lexical < 1:
        p.error("--top-k, --max-lexical은 양수여야 합니다")
    index = v3.index_path(args.root)
    if not index.exists():
        raise FileNotFoundError(f"V3 SQLite 인덱스가 없습니다: {index}\nV3 코드에서 --build-index를 먼저 실행하세요")
    issue = args.issue.strip()
    if not issue:
        p.error("--issue는 비어 있을 수 없습니다")
    aliases = normalize_aliases(issue, args.aliases, args.no_default_aliases)
    v3.log(f"이슈={issue} / 표현={aliases} / 기준일={cutoff}")
    if args.check:
        with sqlite3.connect(index) as con:
            info = dict(con.execute("SELECT name,value FROM info"))
            count = con.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        v3.log(f"SQLite 검사: 문서 {count:,}건 / {info}")
        return
    lexical_raw, complete = search_lexical_exact(index, aliases, cutoff, args.max_lexical, v3)
    v3.log(f"전체 색인 검색 후보 청크={len(lexical_raw):,}, 상한 내 완전 탐색={complete}")
    # 인덱스를 쓰는 검색에서 원문 메타데이터/시점은 반드시 검사한다.
    rows = [clean_row(v3, r, aliases) for r in lexical_raw]
    rows = [r for r in rows if r is not None]
    unique = dedup(rows)
    by_region = {}
    for r in unique:
        by_region.setdefault(r["assembly"], r)
    flow = sorted(by_region.values(), key=lambda z:(z["date"], z["assembly"]))
    first = without_full_text(flow[0]) if flow else None
    monthly = sorted(Counter(r["date"][:7] for r in unique).items())
    # 의미 검색은 증거 우선순위 선정에 사용, 최초 관측의 기준에는 쓰지 않음.
    semantic = []
    try:
        col = v3.chroma_collection(args.root)
        semantic = v3.semantic_search(col, issue, cutoff, args.top_k, aliases)
    except Exception as exc:
        v3.log(f"[안내] 의미검색 실패: {exc}. 어휘 검색 근거로 계속 진행")
    semantic_rows = [clean_row(v3, r, aliases) for r in semantic]
    semantic_rows = [r for r in semantic_rows if r is not None]
    evidence = dedup(semantic_rows)
    chosen = {r["id"] for r in evidence}
    for row in unique:
        if len(evidence) >= 8:
            break
        if row["id"] not in chosen:
            evidence.append(row)
            chosen.add(row["id"])
    evidence = evidence[:8]
    pairs = compare_candidates(issue, unique, args.max_pairs, args.max_pair_docs)
    activity = []
    for r in unique:
        for cue in r["activity_cues"]:
            activity.append({"id":r["id"],"date":r["date"],"assembly":r["assembly"],
                             "meeting_title":r["meeting_title"],"matched_term":r["matched_term"],**cue})
    activity = activity[:50]
    forecast = forecast_live(args.ollama_url, issue, cutoff, evidence, flow, pairs, monthly,
                             args.model, args.no_llm)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    outdir = args.root / "results" / "policy_console_v4" / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    result = {"schema_version":"4.0", "issue":issue, "cutoff":cutoff, "aliases":aliases,
              "search_scope":"전체 색인 어휘 후보" if complete else "부분 검색(상한 도달)",
              "search_complete":complete, "searched_chunks":len(rows), "dedup_meetings":len(unique),
              "first_observation_candidate":first,
              "first_observation_verified":False,
              "regional_followup_candidates":[without_full_text(x) for x in flow],
              "monthly_counts_by_meeting":[{"month":k,"meetings":v} for k,v in monthly],
              "policy_similarity_candidates":pairs,
              "activity_cues":activity,
              "evidence":[without_full_text(x) for x in evidence],
              "forecast":forecast,"model":args.model,
              "total_seconds":round(time.perf_counter()-t0,2),
              "limitations":["실제 정책 최초 도입 시점과 확산 인과관계는 검증되지 않음",
                             "구체적 정책수단 일치+시간순 후보도 수동 검토 필요",
                             "활동 단서는 주변 원문 표현일 뿐 실제 공청회 개최/조례 의결 확정 아님",
                             "정책수단 사전에 없는 신규 정책은 후보 쌍이 0건일 수 있음",
                             "검색 상한 도달 시 전체 최초 관측을 주장할 수 없음",
                             "과거 자료 전체에서 동의어가 빠지면 관측 시점을 놓칠 수 있음",
                             "EXAONE 전망은 형식 검사이며 예측 정확도 검증 아님"]}
    write_json(outdir/"analysis.json", result)
    with (outdir/"evidence.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["id","date","assembly","meeting_title","source_id","matched_term","excerpt_180","policy_mechanisms"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in evidence:
            item = without_full_text(r)
            item["policy_mechanisms"] = ", ".join(item["policy_mechanisms"])
            writer.writerow(item)
    with (outdir/"policy_similarity_review.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(pairs)
    with (outdir/"activity_cues.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["id","date","assembly","meeting_title","matched_term","activity_type","cue_excerpt","verification_status"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(activity)
    v3.log(f"① 최초 관측 후보: {(first or {}).get('date','없음')} {(first or {}).get('assembly','')} / 검색 완전성={complete}")
    v3.log(f"② 지역별 후속 논의 후보: {len(flow)}개 의회 / 유사 정책수단 후보: {len(pairs)}쌍 / 활동 표현 후보: {len(activity)}건")
    v3.log(f"③ EXAONE 전망: {forecast.get('status')} / {len(forecast.get('forecast_150',''))}자 / 원시출력 {len(forecast.get('original',''))}자")
    v3.log(f"④ 원문 근거 {len(evidence)}건 / 180자, ⑤ 저장: {outdir} / {time.perf_counter()-t0:.2f}초")


if __name__ == "__main__":
    main()

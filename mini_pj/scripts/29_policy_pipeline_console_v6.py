# -*- coding: utf-8 -*-
"""29 V6 — 정책 근거 4분류·근거 문맥 검증·의정활동 세분화.

기존 V5 all_evidence_review.csv의 청크 ID를 이용해 V3 SQLite docs.body 원문을
재조회한다. Chroma/임베딩/전체 색인 재구축/LLM 호출은 하지 않는다.

실행(Windows PowerShell):
 python 29_policy_pipeline_console_v6.py --check
 python 29_policy_pipeline_console_v6.py --issue "청년 일자리" --cutoff 2025-12-31
 python 29_policy_pipeline_console_v6.py --v5-dir "C:\\mini_pj\\results\\policy_console_v5\\20261026_111259_543963"
선택: --sample-per-class 25 --labels "...\\human_labels.csv"
주의: 자동 분류 결과는 검증 전 후보이며 정책 전파/채택을 증명하지 않는다.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import sqlite3
import time

ROOT = Path(r"C:\mini_pj")
CLASSES = ["정책 실행·사업", "정책 제안·개선 요구", "현황·문제 제기", "무관·판단 보류"]

# 단어 자체가 아니라 같은 근거 구간 안의 명시적인 행위와 수단을 판정한다.
OPERATION = [
    r"(?:지원사업|일자리사업|창업지원사업|취업지원사업|청년정책사업)",
    r"(?:사업|프로그램|센터|제도|지원금|수당|교육|인턴십|조례|기금).{0,35}(?:운영|시행|추진|집행|편성|지급|지원|확대|설치|개설|실시|선정|도입)",
    r"(?:운영|시행|추진|집행|편성|지급|지원|확대|설치|개설|실시|선정|도입).{0,35}(?:사업|프로그램|센터|제도|지원금|수당|교육|인턴십|조례|기금)",
    r"(?:원안|수정)\s*가결|의결하였|사업비.{0,20}(?:증액|감액)",
]
PROPOSAL = [
    r"(?:대책|개선|지원|확대|도입|신설|보완|개편|마련|강화|검토|촉구|제안|건의|요구|필요|시급).{0,20}(?:해야|하자|바랍니다|필요|촉구|제안|건의|요구|검토)",
    r"(?:해야|하자|바랍니다|필요|촉구|제안|건의|요구|검토).{0,25}(?:대책|개선|지원|확대|도입|신설|보완|개편|마련|강화)",
    r"(?:대책\s*마련|실효성\s*있는\s*대책|조례안.{0,35}(?:발의|제안)|발의.{0,35}조례안)",
]
STATUS = [
    r"고용률|취업률|실업률|인구\s*(?:감소|증가)|통계|현황|부족|어려움|격차|원인|실태|문제점|왜\s*(?:그런|그렇)|차이.{0,12}(?:왜|나)|몇\s*명|유출",
]

# V4에서 잘못 섞인 주거 지원과 사업 입주공간을 엄격히 구분한다.
MECHANISM_PATTERNS = {
    "직무교육·훈련": [r"직무\s*(?:교육|훈련)", r"직업\s*훈련", r"취업\s*(?:교육|훈련)", r"취업역량\s*강화"],
    "인턴·일경험": [r"인턴십", r"(?:청년|행정)\s*인턴", r"일\s*경험", r"직장\s*체험"],
    "기업 인건비·채용장려": [r"(?:기업|청년|근로자).{0,15}인건비.{0,12}지원", r"채용\s*장려금", r"고용\s*장려금"],
    "창업교육·보육": [r"창업\s*(?:교육|보육|인큐베이터|컨설팅)", r"예비\s*창업자.{0,20}교육"],
    "창업자금·사업공간": [r"창업\s*(?:자금|지원금|공간|입주|센터)", r"창업기업.{0,25}(?:사업화\s*자금|입주\s*공간)"],
    "채용·취업연계": [r"채용\s*연계", r"취업\s*연계", r"기업.{0,16}매칭", r"취업\s*알선"],
    "구직활동·수당": [r"구직\s*(?:활동\s*)?지원\s*(?:금|수당)", r"취업\s*준비\s*지원", r"면접비\s*지원"],
    "주거비·월세지원": [r"청년\s*월세", r"월세.{0,10}지원", r"(?:전|월)세.{0,15}(?:이자|대출|지원)", r"주거비.{0,15}지원"],
    "공공임대주택": [r"공공\s*임대\s*주택", r"청년\s*임대\s*주택"],
}
ACTIVITY_PATTERNS = {
    "조례 의결": [r"(?:조례안|개정조례안).{0,90}(?:원안|수정)\s*가결", r"(?:원안|수정)\s*가결.{0,90}(?:조례안|개정조례안)"],
    "조례 발의·제안": [r"(?:조례안|개정조례안).{0,65}(?:대표)?발의", r"(?:대표)?발의.{0,65}(?:조례안|개정조례안)"],
    "조례 심사": [r"(?:조례안|개정조례안).{0,45}(?:심사|심의)", r"(?:심사|심의).{0,45}(?:조례안|개정조례안)"],
    "정책감사·행정사무감사": [r"정책\s*감사", r"행정\s*사무\s*감사", r"감사\s*활동"],
    "예산 심사·조정": [r"(?:예산|사업비|추경).{0,30}(?:심사|증액|감액|삭감|편성)", r"(?:증액|감액|삭감).{0,30}(?:예산|사업비)"],
    "공청회·토론회": [r"공청회", r"토론회"],
}


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list, fields: list[str]):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def try_json_list(s):
    try:
        x = json.loads(s or "[]")
        return x if isinstance(x, list) else []
    except (ValueError, TypeError):
        return []


def find_v5(root: Path, v5_dir: Path | None):
    if v5_dir:
        return v5_dir
    base = root / "results" / "policy_console_v5"
    folders = sorted((p for p in base.iterdir() if p.is_dir() and (p / "all_evidence_review.csv").exists()), reverse=True) if base.exists() else []
    if not folders:
        raise FileNotFoundError(f"V5 결과를 찾을 수 없습니다: {base} (먼저 V5 실행 또는 --v5-dir 지정)")
    return folders[0]


def term_positions(body, term):
    return [m.start() for m in re.finditer(re.escape(term), body, re.I)] if term else []


def get_context(body, p, size=460):
    # 원문을 쪼개도 문맥의 시작·끝은 항상 근거와 결합해 남긴다.
    a = max(0, p - size // 2)
    b = min(len(body), p + size // 2)
    # 인접한 문장 구분자를 우선하여 무관한 전체 안건 혼합 방지
    left = max(body.rfind("\n", a, p), body.rfind(".", a, p), body.rfind("다.", a, p))
    if left >= a and p - left < 190:
        a = left + 1
    right_newline = body.find("\n", p, b)
    if right_newline >= 0 and right_newline - p >= 45:
        b = right_newline
    return a, b, body[a:b]


def near_match(patterns, text, pos, maxdist=115):
    out = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.I | re.S):
            if min(abs(pos - m.start()), abs(pos - m.end())) <= maxdist or (m.start() <= pos <= m.end()):
                out.append(m)
    return sorted(out, key=lambda m: (abs(pos - m.start()), m.start()))


def classify(text, pos, issue):
    # '직접 사업'을 표시하려면 정책어 주변에 실제 사업·집행 표현이 있어야 한다.
    ops = near_match(OPERATION, text, pos, 140)
    proposals = near_match(PROPOSAL, text, pos, 120)
    stats = near_match(STATUS, text, pos, 105)
    if proposals and (not ops or any(re.search(r"촉구|요구|필요|해야|바랍니다|제안", text[max(0,m.start()-15):m.end()+15]) for m in proposals)):
        return CLASSES[1], "정책 제안·개선 필요 표현", proposals[0]
    if ops:
        return CLASSES[0], "정책 사업·집행 명시 표현", ops[0]
    if proposals:
        return CLASSES[1], "정책 제안 표현", proposals[0]
    if stats:
        return CLASSES[2], "현황·통계·문제 제기 표현", stats[0]
    return CLASSES[3], "정책 관련 행위·현황 근거 불충분", None


def local_mechanisms(text, pos):
    results = []
    for name, patterns in MECHANISM_PATTERNS.items():
        hits = near_match(patterns, text, pos, 150)
        if hits:
            m = hits[0]
            results.append({"type": name, "cue": clean(text[max(0,m.start()-35):min(len(text),m.end()+60)]), "start_in_context":m.start()})
    return results


def local_activities(text, pos, term, full_document):
    results = []
    for name, patterns in ACTIVITY_PATTERNS.items():
        hits = near_match(patterns, text, pos, 90)
        if not hits:
            continue
        m = hits[0]
        phrase = clean(text[max(0,m.start()-55):min(len(text),m.end()+60)])
        # 명확한 조례 제목 또는 정책어와 근접하게 결합하지 않으면 자동 확정하지 않는다.
        if name.startswith("조례"):
            if not re.search(r"조례\s*(?:일부개정)?\s*조례안|[가-힣\s]{2,50}조례안", phrase):
                continue
        results.append({"type":name,"cue":phrase,"status":"검증 전 후보"})
    return results


def evaluate_one(row, body, term):
    positions = term_positions(body, term)
    if not positions:
        # V5 원문 일부만으로 새 판정을 하지 않는다.
        return {"label":CLASSES[3],"reason":"SQLite 원문에서 V5 매칭 표현을 찾지 못함",
                "span":"","span_start":"","span_end":"","mechanisms":[],"activities":[],"is_source_found":False}
    candidates = []
    for p in positions[:12]:
        a,b,text = get_context(body,p)
        rel = p-a
        label, reason, match = classify(text,rel,term)
        mechs = local_mechanisms(text,rel) if label in CLASSES[:2] else []
        activities = local_activities(text,rel,term,body) if label in CLASSES[:2] else []
        # 실제 문맥에 수단 단서가 확인된 후보를 우선하되, 정책 직접성 > 현황 > 보류
        order = {CLASSES[0]:4,CLASSES[1]:3,CLASSES[2]:2,CLASSES[3]:1}[label]
        candidates.append((order,len(mechs),len(activities),a,b,text,label,reason,mechs,activities))
    _,_,_,a,b,text,label,reason,mechs,activities = max(candidates,key=lambda x:(x[0],x[1],x[2],-x[3]))
    # 180자는 전체 근거를 대체하는 게 아니라 선택 구간의 표시용 발췌임
    return {"label":label,"reason":reason,"span":clean(text[:180]),"span_start":a,"span_end":b,
            "mechanisms":mechs,"activities":activities,"is_source_found":True}


def read_human_labels(path):
    rows = read_csv(path)
    return {r.get("id",""):r.get("human_label","").strip() for r in rows if r.get("id") and r.get("human_label") in CLASSES}


def main():
    parser=argparse.ArgumentParser(description="29 V6: V5 원문 재검증 및 4분류, 재색인 불필요")
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--v5-dir",type=Path,default=None)
    parser.add_argument("--issue",default="청년 일자리")
    parser.add_argument("--cutoff",default="2025-12-31")
    parser.add_argument("--sample-per-class",type=int,default=25)
    parser.add_argument("--labels",type=Path,default=None,help="사람이 작성한 human_labels.csv; ID별 human_label 컬럼")
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    try:
        datetime.strptime(args.cutoff,"%Y-%m-%d")
    except ValueError:
        parser.error("--cutoff 는 YYYY-MM-DD")
    index=args.root/"results"/"policy_search_index"/"meetings_fts.sqlite3"
    if not index.exists():
        raise FileNotFoundError(f"기존 SQLite 인덱스가 없습니다: {index}")
    v5_dir=find_v5(args.root,args.v5_dir)
    summary_path=v5_dir/"review_summary.json"
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("issue")!=args.issue or summary.get("cutoff")!=args.cutoff:
        raise ValueError(f"V5 실행 조건이 다릅니다. 저장된 이슈={summary.get('issue')}, 기준일={summary.get('cutoff')}. ")
    with sqlite3.connect(f"file:{index.as_posix()}?mode=ro",uri=True) as con:
        meta=dict(con.execute("SELECT name,value FROM info"))
        if meta.get("complete")!="yes":
            raise RuntimeError("SQLite 인덱스가 미완성입니다")
        count=con.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    print(f"[V6 점검] 색인={count:,}청크 / V5 폴더={v5_dir}",flush=True)
    if args.check: return
    started=time.perf_counter()
    src=read_csv(v5_dir/"all_evidence_review.csv")
    if len(src)!=summary.get("lexical_candidates"):
        print("[주의] V5 CSV 행수와 요약 통계가 다릅니다. 실제 CSV만 처리합니다.",flush=True)
    out=[]
    with sqlite3.connect(f"file:{index.as_posix()}?mode=ro",uri=True) as con:
        for n,row in enumerate(src,1):
            if row.get("date","")>args.cutoff: continue
            record=con.execute("SELECT body FROM docs WHERE id=?",(row["id"],)).fetchone()
            full=record[0] if record else ""
            assessed=evaluate_one(row,full,row.get("matched_term",""))
            out.append({"id":row["id"],"date":row.get("date",""),"assembly":row.get("assembly",""),
                        "meeting_title":row.get("meeting_title",""),"source_id":row.get("source_id",""),
                        "matched_term":row.get("matched_term",""),"v5_class":row.get("evidence_class",""),
                        "v6_class":assessed["label"],"reason":assessed["reason"],
                        "evidence_180":assessed["span"],"span_start":assessed["span_start"],
                        "span_end":assessed["span_end"],"source_found":assessed["is_source_found"],
                        "mechanisms":json.dumps(assessed["mechanisms"],ensure_ascii=False),
                        "activities":json.dumps(assessed["activities"],ensure_ascii=False),
                        "verification_status":"자동 추출·수동 미검증"})
            if n%1000==0: print(f"[V6] 문맥 재검사: {n:,}/{len(src):,}",flush=True)
    out.sort(key=lambda r:(r["date"],r["assembly"],r["id"]))
    dest=args.root/"results"/"policy_console_v6"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dest.mkdir(parents=True,exist_ok=True)
    fields=list(out[0]) if out else ["id","date","assembly","v5_class","v6_class","reason","evidence_180"]
    write_csv(dest/"all_evidence_v6.csv",out,fields)
    for label,file in zip(CLASSES,["operational_policy.csv","policy_proposals.csv","status_mentions.csv","pending_review.csv"]):
        write_csv(dest/file,[r for r in out if r["v6_class"]==label],fields)
    activity_rows=[]
    mechanism_rows=[]
    for r in out:
        for m in try_json_list(r["mechanisms"]):
            mechanism_rows.append({"id":r["id"],"date":r["date"],"assembly":r["assembly"],"mechanism":m["type"],"cue":m["cue"],"policy_evidence_180":r["evidence_180"],"verification_status":"검증 전 후보"})
        for a in try_json_list(r["activities"]):
            activity_rows.append({"id":r["id"],"date":r["date"],"assembly":r["assembly"],"activity_type":a["type"],"cue":a["cue"],"policy_evidence_180":r["evidence_180"],"verification_status":"검증 전 후보"})
    write_csv(dest/"policy_mechanism_review.csv",mechanism_rows,["id","date","assembly","mechanism","cue","policy_evidence_180","verification_status"])
    write_csv(dest/"policy_activity_review.csv",activity_rows,["id","date","assembly","activity_type","cue","policy_evidence_180","verification_status"])
    # 사람이 각 범주 25개를 판정하도록, 날짜·의회가 치우치지 않게 간격 표본 생성
    samples=[]
    for label in CLASSES:
        group=[r for r in out if r["v6_class"]==label]
        k=min(len(group),max(0,args.sample_per_class))
        indices=[int((j+0.5)*len(group)/k) for j in range(k)] if k else []
        for idx in indices:
            r=group[min(idx,len(group)-1)]
            samples.append({"id":r["id"],"date":r["date"],"assembly":r["assembly"],
                            "v5_class":r["v5_class"],"v6_class":label,"evidence_180":r["evidence_180"],
                            "human_label":"","human_comment":""})
    sample_fields=["id","date","assembly","v5_class","v6_class","evidence_180","human_label","human_comment"]
    write_csv(dest/"human_review_sample.csv",samples,sample_fields)
    counts=Counter(r["v6_class"] for r in out)
    transition=Counter((r["v5_class"],r["v6_class"]) for r in out)
    result={"schema_version":"6.0","issue":args.issue,"cutoff":args.cutoff,
            "v5_source":str(v5_dir),"sqlite_index":str(index),
            "v5_aliases":summary.get("aliases",[]),"source_rows":len(src),"processed_rows":len(out),
            "classification_counts":dict(counts),
            "transition_counts":[{"from":a,"to":b,"count":c} for (a,b),c in sorted(transition.items())],
            "mechanism_candidates":len(mechanism_rows),"activity_candidates":len(activity_rows),
            "first_policy_execution_candidate":next((r for r in out if r["v6_class"]==CLASSES[0]),None),
            "first_policy_proposal_candidate":next((r for r in out if r["v6_class"]==CLASSES[1]),None),
            "complete_within_v5_candidates":summary.get("search_complete_within_aliases",False),
            "notes":["V5 청크 후보만 재평가하므로 누락된 동의어나 의미상 관련 문서는 찾지 못함",
                     "자동 추출이며 최초 정책 채택/확산 인과/의정활동 확정은 아님",
                     "정책수단 및 활동 후보는 근거 문구와 함께 사람 검토 필요",
                     "분류 결과의 정확도는 human_review_sample.csv에 사람 레이블을 입력해야 추정 가능"],
            "elapsed_seconds":round(time.perf_counter()-started,2)}
    if args.labels:
        gold=read_human_labels(args.labels)
        judged=[r for r in out if r["id"] in gold]
        result["human_evaluation"]={"labeled_count":len(judged),
              "accuracy":round(sum(r["v6_class"]==gold[r["id"]] for r in judged)/len(judged),4) if judged else None,
              "confusions":[{"expected":a,"predicted":b,"count":n} for (a,b),n in Counter((gold[r["id"]],r["v6_class"]) for r in judged).items()]}
    (dest/"review_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print("[V6] " + " / ".join(f"{key}={counts[key]:,}" for key in CLASSES),flush=True)
    print(f"[V6] 정책수단 후보={len(mechanism_rows):,} / 의정활동 후보={len(activity_rows):,} / 검토 표본={len(samples):,}",flush=True)
    print(f"[V6] 저장: {dest} / {time.perf_counter()-started:.2f}초",flush=True)

if __name__=="__main__":
    main()

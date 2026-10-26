# -*- coding: utf-8 -*-
"""29 V5 / 정책 근거 재선정 검증기.

V3 SQLite 인덱스를 그대로 사용한다. V4 코드를 같은 폴더에 두고 실행한다.
정책 키워드가 있는 청크를 '정책 직접 근거 / 현황·문제 제기 / 판단 보류'로
재선정한다. 정책수단 및 의정활동은 관련성이 확인된 짧은 문맥만 검사한다.
규칙 기반 후보 판정이므로 실제 정책 채택/전파가 확정되지 않는다.

실행:
 python 29_policy_pipeline_console_v5.py --check
 python 29_policy_pipeline_console_v5.py --issue "청년 일자리" --cutoff 2025-12-31
 python 29_policy_pipeline_console_v5.py --issue "청년 주거" --aliases "청년 월세,월세 지원" --cutoff 2025-12-31

선택 사항: --max-lexical 25000 --min-direct 0.55 --examples 30
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import re
import sqlite3
import time

ROOT = Path(r"C:\mini_pj")
V4_FILE = Path(__file__).resolve().with_name("29_policy_pipeline_console_v4.py")

# 강한 표현은 정책의 구체적인 수단/추진·심의를 설명하는 경향이 있지만, 단독으로 확정 근거는 아니다.
POLICY_ACTION = re.compile(r"지원(?:사업|금|프로그램|책)?|추진|시행|운영|조성|확대|신설|도입|설치|\b사업\b|프로그램|예산|편성|개정|발의|심사|계획|육성|선정|연계|취업|고용|훈련|교육|채용|인턴|창업|수당|보조", re.I)
POLICY_STRONG = re.compile(r"(?:지원사업|일자리사업|창업지원|취업지원|고용지원|조례안|개정조례안|추경|본예산|직무교육|취업교육|취업연계|인턴십|일경험|구직지원|창업보육|임대주택|월세지원|채용장려금)", re.I)
STATUS_TERMS = re.compile(r"현황|통계|비율|고용률|취업률|실업률|인구|감소|증가|부족|문제|원인|어려움|현실|차이|우려|실태|조사|지적|질문|몇\s*명", re.I)

# 정책어와 분류 근거가 실제로 같이 등장하는 최대 주변 문맥을 보수적으로 제한.
WINDOW_CHARS = 230
MAX_CONTEXTS = 6


def load_v4():
    if not V4_FILE.is_file():
        raise FileNotFoundError(f"V4 파일이 없습니다. V5와 같은 폴더에 저장하세요: {V4_FILE}")
    spec = importlib.util.spec_from_file_location("policy_v4_for_v5", V4_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def trim(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def clip_context(text: str, pos: int, term_len: int, span=WINDOW_CHARS) -> str:
    """용어 주변에서 문장 경계를 선호하되, 최대 span 문자만 반환."""
    if not text:
        return ""
    center = pos + term_len // 2
    a = max(0, center - span // 2)
    b = min(len(text), a + span)
    a = max(0, b - span)
    # 텍스트가 짧을 때는 전체 사용; 긴 경우에는 용어 좌우로 제한된 문맥만 사용
    return trim(text[a:b])


def locate_terms(text, aliases):
    hits = []
    for term in aliases:
        start = 0
        while True:
            i = text.casefold().find(term.casefold(), start)
            if i < 0:
                break
            hits.append((i, term))
            start = i + max(1, len(term))
            if len(hits) > 80:
                break
    # 가까운 용어 중복: 하나의 문맥을 여러 번 세지 않는다.
    hits.sort(key=lambda z: (z[0], -len(z[1])))
    out = []
    for p, term in hits:
        if out and p - out[-1][0] < 75:
            continue
        out.append((p, term))
        if len(out) >= MAX_CONTEXTS:
            break
    return out


def classify_context(context, term):
    """문맥 단위 보수적 휴리스틱. 확률이 아닌 우선순위 점수."""
    action = bool(POLICY_ACTION.search(context))
    strong = bool(POLICY_STRONG.search(context))
    status = bool(STATUS_TERMS.search(context))
    # 정책 키워드만 있거나 정책수단이 무엇인지 설명하지 않으면 보류
    if strong and action:
        return "정책 직접 근거", 0.75, "동일 문맥에 구체적인 정책·사업 표현"
    if action and len(context) >= 55 and not (status and not strong):
        return "정책 직접 근거", 0.58, "동일 문맥에 정책 실행·지원 표현"
    if status:
        return "현황·문제 제기", 0.75, "고용·인구 등 현황/문제 진술"
    return "판단 보류", 0.0, "정책수단·활동 근거 불충분"


def local_policy_mechanisms(v4, context):
    # V4의 정책수단 사전을 '전체 청크'가 아니라 키워드 인접 문맥에 적용
    return v4.mechanisms(context)


def local_activity_cues(v4, context, term):
    """활동 용어+정책어가 같은 짧은 문맥에 등장하는 후보만 유지.

    조례는 관련 제목이 확인될 때만 남긴다. 단순 '토론회' 등은 완료로 주장하지 않는다.
    """
    cues = []
    for label, patterns in v4.ACTIVITY_CUES.items():
        matches = []
        for p in patterns:
            matches.extend(re.finditer(p, context, re.I | re.S))
        if not matches:
            continue
        # 청년 주제와 다른 조례·토론회가 섞이는 것을 방지하도록 연결 구간 축소
        # 통상 '키워드' 위치와 활동 키워드 위치가 근접해야 함
        positions = [m.start() for m in re.finditer(re.escape(term), context, re.I)]
        if not positions:
            continue
        near = [m for m in matches if min(abs(m.start() - p) for p in positions) <= 100]
        if not near:
            continue
        # 조례 표현이 실제 주제와 붙어 있는지 명확하지 않은 것은 보류
        if label.startswith("조례"):
            if not re.search(r"(?:[가-힣\s]{0,45})?(?:조례안|개정조례안)", context):
                continue
        m = near[0]
        start = max(0, m.start()-40)
        cues.append({"activity_type": label,
                     "cue_excerpt_180": trim(context[start:start+180]),
                     "verification_status": "자동 후보·실제 활동 여부 미확정"})
    return cues


def assess_row(v4, row, aliases):
    full_text = str(row.get("full_text") or "")
    best = []
    for position, term in locate_terms(full_text, aliases):
        context = clip_context(full_text, position, len(term))
        label, score, reason = classify_context(context, term)
        mechs = local_policy_mechanisms(v4, context) if label == "정책 직접 근거" else []
        activities = local_activity_cues(v4, context, term) if label == "정책 직접 근거" else []
        best.append({"label":label,"score":score,"reason":reason,"term":term,
                     "context":context,"mechanisms":mechs,"activities":activities})
    order = {"정책 직접 근거":3,"현황·문제 제기":2,"판단 보류":1}
    if not best:
        return None
    best.sort(key=lambda x:(order[x['label']],x['score'],len(x['mechanisms'])), reverse=True)
    chosen = best[0]
    # 모든 분류와 추출은 동일 문맥에서 수행했는지 출처 확인 가능하게 저장
    return {"id":row["id"],"date":row["date"],"assembly":row["assembly"],
            "meeting_title":row.get("meeting_title", ""), "source_id":row.get("source_id", ""),
            "matched_term":chosen["term"], "evidence_class":chosen["label"],
            "rule_score":chosen["score"], "selection_reason":chosen["reason"],
            "evidence_180":chosen["context"][:180],
            "policy_mechanisms":chosen["mechanisms"],"activity_cues":chosen["activities"],
            "review_status":"자동 분류·수동 미검증", "contexts_inspected":len(best)}


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            x = dict(row)
            for k, v in x.items():
                if isinstance(v, (list, dict)):
                    x[k] = json.dumps(v, ensure_ascii=False)
            writer.writerow(x)


def main():
    p = argparse.ArgumentParser(description="29 V5 / 정책 직접 근거 재선정 (인덱스 재사용)")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--issue", default="청년 일자리")
    p.add_argument("--aliases", default="", help="쉼표로 구분한 추가 동의어")
    p.add_argument("--no-default-aliases", action="store_true")
    p.add_argument("--cutoff", default="2025-12-31")
    p.add_argument("--max-lexical", type=int, default=25000)
    p.add_argument("--examples", type=int, default=30, help="등급별 검토용 CSV 최대 예시 수")
    p.add_argument("--min-direct", type=float, default=0.55)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    v4 = load_v4()
    v3 = v4.load_v3()
    if v3.date_ok(args.cutoff) != args.cutoff:
        p.error("--cutoff 형식: YYYY-MM-DD")
    if not args.issue.strip() or args.max_lexical < 1 or not 0 <= args.min_direct <= 1:
        p.error("--issue, --max-lexical, --min-direct 확인 필요")
    idx = v3.index_path(args.root)
    if not idx.exists():
        raise FileNotFoundError(f"SQLite 인덱스가 없습니다: {idx} (V3 --build-index 필요)")
    with sqlite3.connect(idx) as con:
        info = dict(con.execute("SELECT name,value FROM info"))
        count = con.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    if info.get("complete") != "yes":
        raise RuntimeError("인덱스가 미완성입니다")
    v3.log(f"V5 SQLite 연결: {count:,}청크 / 구축={info.get('built_at')}")
    if args.check:
        return
    t0 = time.perf_counter()
    aliases = v4.normalize_aliases(args.issue.strip(), args.aliases, args.no_default_aliases)
    v3.log(f"이슈={args.issue} / 동의어={len(aliases)}개 / 기준일={args.cutoff}")
    raw, complete = v4.search_lexical_exact(idx, aliases, args.cutoff, args.max_lexical, v3)
    v3.log(f"어휘 일치 후보={len(raw):,}청크 / 상한 내 완전 탐색={complete}")
    assessments = []
    for ix, row in enumerate(raw, 1):
        rec = assess_row(v4,row,aliases)
        if rec:
            if rec["evidence_class"] == "정책 직접 근거" and rec["rule_score"] < args.min_direct:
                rec["evidence_class"] = "판단 보류"
                rec["selection_reason"] += " / 설정된 최소점수 미달"
                rec["policy_mechanisms"] = []
                rec["activity_cues"] = []
            assessments.append(rec)
        if ix % 2000 == 0:
            v3.log(f"근거 재선정: {ix:,}/{len(raw):,}")
    classes = Counter(r["evidence_class"] for r in assessments)
    direct = [r for r in assessments if r["evidence_class"] == "정책 직접 근거"]
    status = [r for r in assessments if r["evidence_class"] == "현황·문제 제기"]
    pending = [r for r in assessments if r["evidence_class"] == "판단 보류"]
    # 하나의 회의록에 여러 청크가 있어도 대표 근거를 1건만 선정하되 정책 직접 근거를 우선.
    seen = set()
    dedup_direct = []
    for row in sorted(direct, key=lambda r:(r["date"],r["assembly"],-r["rule_score"],r["id"])):
        key = (row["date"],row["assembly"],row["source_id"] or row["meeting_title"] or row["id"])
        if key not in seen:
            seen.add(key)
            dedup_direct.append(row)
    first_mention = min(assessments, key=lambda r:(r["date"],r["id"])) if assessments else None
    first_policy = dedup_direct[0] if dedup_direct else None
    outdir = args.root/"results"/"policy_console_v5"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    outdir.mkdir(parents=True, exist_ok=True)
    fields = ["id","date","assembly","meeting_title","source_id","matched_term","evidence_class",
              "rule_score","selection_reason","evidence_180","policy_mechanisms","activity_cues",
              "review_status","contexts_inspected"]
    write_csv(outdir/"all_evidence_review.csv", fields, assessments)
    write_csv(outdir/"direct_policy_evidence.csv", fields, dedup_direct)
    write_csv(outdir/"status_mentions.csv", fields, status[:args.examples])
    write_csv(outdir/"pending_review.csv", fields, pending[:args.examples])
    activities = []
    for row in dedup_direct:
        for activity in row["activity_cues"]:
            activities.append({"id":row["id"],"date":row["date"],"assembly":row["assembly"],
                               "matched_term":row["matched_term"],"policy_evidence_180":row["evidence_180"],**activity})
    write_csv(outdir/"policy_activity_review.csv",
              ["id","date","assembly","matched_term","policy_evidence_180", "activity_type","cue_excerpt_180","verification_status"],activities)
    summary = {"schema_version":"5.0","issue":args.issue,"cutoff":args.cutoff,"aliases":aliases,
               "index_documents":count,"lexical_candidates":len(raw),"search_complete_within_aliases":complete,
               "classification_counts":dict(classes),"direct_meeting_evidence_count":len(dedup_direct),
               "activity_candidates":len(activities),
               "first_keyword_mention_candidate": first_mention,
               "first_direct_policy_candidate":first_policy,
               "first_policy_is_verified":False,
               "notes":["최초 날짜는 지정 동의어와 현재 색인·기준일 안에서의 후보이며 실제 최초 시행이 아님",
                        "정책수단·활동 단서는 정책어 인접 문맥의 규칙 기반 후보이며 수동 검증 필요",
                        "관련 표현이 멀리 있거나 문장이 길면 누락될 수 있음",
                        "새로운 검색어는 동의어 보강이 필요할 수 있음",
                        "V5는 근거 재선정만 수행하며 EXAONE 추론·확산 인과 검증은 하지 않음"],
               "elapsed_seconds":round(time.perf_counter()-t0,2)}
    (outdir/"review_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    v3.log(f"분류: 직접 {classes['정책 직접 근거']:,} / 현황 {classes['현황·문제 제기']:,} / 보류 {classes['판단 보류']:,}")
    v3.log(f"회의록 중복제거 직접 근거={len(dedup_direct):,} / 의정활동 후보={len(activities):,}")
    v3.log(f"최초 정책어 언급 후보: {(first_mention or {}).get('date','없음')} {(first_mention or {}).get('assembly','')}")
    v3.log(f"최초 정책 직접 근거 후보: {(first_policy or {}).get('date','없음')} {(first_policy or {}).get('assembly','')}")
    v3.log(f"저장: {outdir} / {time.perf_counter()-t0:.2f}초")


if __name__ == "__main__":
    main()

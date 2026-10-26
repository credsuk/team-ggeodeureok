# -*- coding: utf-8 -*-
r"""24_policy_relevance_validation.py — 지방의회 정책 검색 근거 관련성 검토

기존 15·16번 검색 결과를 재사용합니다. 임베딩/Chroma/BERTopic 재실행 없음.
규칙 기반 후보 분류는 *검토용*이며 사실 확정/정밀도 평가가 아닙니다.

실행:
  python C:\mini_pj\scripts\24_policy_relevance_validation.py --input-dir "C:\mini_pj\results\policy_issue_search\20261025_000801_청년_일자리"
  python C:\mini_pj\scripts\24_policy_relevance_validation.py --input-dir "..." --review-file "...\수동검토_입력양식.csv"
  python C:\mini_pj\scripts\24_policy_relevance_validation.py --input-dir "..." --output-dir "..."

입력: 16번 trend_analysis/검색근거_검토구분.csv
출력: <검색폴더>/relevance_validation/ (원본 절대 수정하지 않음)
주의: 미리보기 최대 약 600자. 완전 원문이나 법적 사실을 검증하지 않음.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path(r"C:\mini_pj\results\policy_issue_search\20261025_000801_청년_일자리")

# 특정 이슈용 '정책 대상 + 정책 수단' 사전. 다른 주제에서는 일반 규칙만 사용(보류 우선).
YOUTH_TARGET = [r"청년", r"취업\s*준비", r"미취업", r"구직\s*청년", r"예비\s*창업"]
YOUTH_MECHANISMS = {
    "채용·고용연계": [r"채용\s*(?:지원|연계|장려)",r"고용\s*(?:촉진|지원|장려)",r"일자리\s*매칭", r"지역\s*기업.{0,12}채용"],
    "취업·구직지원": [r"취업\s*(?:준비|지원|알선|상담|연계|역량)",r"구직\s*(?:활동|지원|수당)",r"자격증\s*(?:응시료|취득)",r"취업\s*프로그램",r"대학\s*일자리\s*플러스\s*센터"],
    "청년창업": [r"청년\s*창업",r"창업\s*(?:지원|보육|자금|기업|공간|센터)",r"스타트업\s*(?:지원|육성)"],
    "직업훈련·인턴": [r"직업\s*훈련",r"취업\s*훈련",r"인턴(?:십)?",r"현장\s*실습",r"일경험"],
}
POLICY_ACTION = [r"지원\s*(?:사업|계획|예산|대상|확대|센터|금)",r"(?:지원|운영|추진|시행|도입|개정|심사|검토|확대|계획|편성|실시|개최|질의|논의)",r"(?:조례|사업|프로그램|사업비|예산)"]
PROCEDURAL = [r"투표를\s*종료",r"표결\s*결과",r"가결되었음을\s*선포",r"이의\s*없으십니까",r"재석\s*의원",r"의사일정\s*제\s*\d+\s*항"]
OFF_TOPIC = [r"청년\s*예술(?:인)?\s*(?:나이|연령|지원)",r"(?:청년|미래)\s*(?:축제|문화행사|체육대회)"]
WIDE_TERM = {"취업 지원", "일자리", "지원", "청년"}
FIELDS = ["chunk_id", "date", "month", "assembly", "meeting_title", "source_file", "source_id", "topic_id",
          "matched_terms", "review_candidate_label", "relevance_score", "mechanisms_detected", "positive_cues",
          "negative_cues", "core_evidence_text_100", "evidence_offset", "text_preview", "review_status", "review_note"]
ALLOWED_REVIEW = {"직접관련", "간접관련", "무관", "판단보류"}


def stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def elapsed(t):
    s = time.perf_counter() - t
    return f"{s:.2f}초 / {s/60:.2f}분 / {s/3600:.3f}시간"


def clean(s):
    if s is None or pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def sentence_spans(text):
    """줄바꿈/종결부호 기준으로 원문 안의 위치 보존. 긴 발언에는 문장 부분창 적용."""
    for m in re.finditer(r"[^\n.!?。！？]+(?:[.!?。！？]+|$)", text):
        piece = m.group()
        if piece.strip():
            yield m.start(), m.end(), piece


def short_excerpt(text, start, end, max_chars=100, focus=None):
    # substring must be verbatim contiguous in source text
    span = text[start:end]
    if len(span) <= max_chars:
        return span.strip(), start + len(span) - len(span.lstrip())
    # Position the actual match near the centre of the excerpt when a sentence is long.
    hit = (focus if focus is not None else start)
    pre = max(start, min(hit - max_chars // 3, end - max_chars))
    return text[pre:pre + max_chars].strip(), pre


def matches_any(text, patterns):
    return any(re.search(p, text, re.I) for p in patterns)


def extract_one(text, terms, issue, youth_mode=True):
    """형식 규칙으로 검토 우선순위 산정. label은 절대 확정 판정이 아님."""
    text = str(text or "")
    pieces = list(sentence_spans(text))
    if not pieces:
        return {"review_candidate_label": "판단보류", "relevance_score": 0,
                "mechanisms_detected": "", "positive_cues": "", "negative_cues": "",
                "core_evidence_text_100": "", "evidence_offset": -1}
    is_youth = youth_mode
    raw_terms = [a.strip() for a in str(terms or "").split("|") if a.strip()]
    query_tokens = [a for a in re.findall(r"[가-힣A-Za-z0-9]{2,}", str(issue)) if len(a) >= 2]
    anchor_patterns = [re.escape(a) for a in raw_terms if a and a not in WIDE_TERM]
    if not anchor_patterns:
        anchor_patterns = [re.escape(w) for w in query_tokens]

    ranked = []
    for start, end, sentence in pieces:
        s = clean(sentence)
        target = matches_any(s, YOUTH_TARGET) if is_youth else bool(query_tokens and any(t in s for t in query_tokens))
        ms = [k for k, pp in YOUTH_MECHANISMS.items() if matches_any(s, pp)] if is_youth else []
        lexical = matches_any(s, anchor_patterns) if anchor_patterns else False
        action = matches_any(s, POLICY_ACTION)
        proced = matches_any(s, PROCEDURAL)
        off = matches_any(s, OFF_TOPIC) if is_youth else False
        # priority: contextual target + mechanism + policy action, not just a keyword
        pts = 3 * bool(target) + 4 * bool(ms) + 2 * bool(action) + 2 * bool(lexical) - 4 * bool(proced) - 3 * bool(off)
        ranked.append((pts, start, end, s, ms, target, action, lexical, proced, off))
    best = max(ranked, key=lambda item: (item[0], bool(item[4]), -item[1]))
    score, start, end, snippet, mechs, target, action, lexical, proced, off = best
    preferred = None
    for pat in list(YOUTH_MECHANISMS.values()) if is_youth else []:
        for expr in pat:
            match = re.search(expr, text[start:end])
            if match:
                preferred = start + match.start()
                break
        if preferred is not None:
            break
    if preferred is None:
        for expr in anchor_patterns:
            match = re.search(expr, text[start:end])
            if match:
                preferred = start + match.start()
                break
    excerpt, excerpt_pos = short_excerpt(text, start, end, 100, focus=preferred)
    # A meeting text may contain mechanism and target in adjacent sentences: only weak secondary context
    close = text[max(0, start - 120): min(len(text), end + 120)]
    context_target = matches_any(close, YOUTH_TARGET) if is_youth else target
    context_mechanism = bool([k for k, pp in YOUTH_MECHANISMS.items() if matches_any(close, pp)]) if is_youth else False
    if is_youth:
        if target and mechs and action and not proced:
            label = "직접관련_후보"
        elif context_target and (mechs or context_mechanism) and (lexical or action):
            label = "간접관련_후보"
        elif off or proced:
            label = "무관가능성_검토"
        else:
            label = "판단보류"
    else:
        # Unknown policy domain: don't force labeling without domain taxonomy.
        label = "간접관련_후보" if lexical and action and not proced else "판단보류"
    cues = [f"대상:{int(bool(target))}", f"수단:{','.join(mechs) or '-'}",
            f"정책행위:{int(bool(action))}", f"검색표현:{int(bool(lexical))}"]
    neg = [x for x, b in [("의사진행", proced), ("이슈외정책", off)] if b]
    return {"review_candidate_label": label, "relevance_score": max(-10, min(11, int(score))),
            "mechanisms_detected": "|".join(mechs), "positive_cues": "; ".join(cues),
            "negative_cues": "|".join(neg), "core_evidence_text_100": excerpt,
            "evidence_offset": excerpt_pos}


def load_manual(path):
    if not path:
        return {}
    d = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    expected = {"chunk_id", "manual_verdict", "manual_note"}
    if not expected.issubset(d.columns):
        raise ValueError(f"수동 검토 컬럼 누락: {expected-set(d.columns)}")
    if d.chunk_id.duplicated().any():
        raise ValueError("수동 검토 chunk_id 중복")
    bad = set(d.manual_verdict.str.strip()) - ALLOWED_REVIEW - {""}
    if bad:
        raise ValueError(f"수동 검토값이 허용 범위 밖: {bad}")
    return {str(r.chunk_id): (r.manual_verdict.strip(), r.manual_note.strip()) for r in d.itertuples(index=False)
            if r.manual_verdict.strip()}


def choose_review_set(d, quota, seed):
    # Equal representation across labels + early dates; review set is not random accuracy estimate.
    if quota <= 0:
        return d.iloc[:0].copy()
    frames = []
    categories = sorted(d.review_candidate_label.unique())
    per = max(1, quota // max(len(categories), 1))
    for label in categories:
        sub = d[d.review_candidate_label == label]
        n_early = min(len(sub), max(1, per // 4))
        early = sub.sort_values("date", kind="stable").head(n_early)
        other = sub.loc[~sub.index.isin(early.index)].sample(n=min(per-n_early, max(0, len(sub)-n_early)), random_state=seed)
        frames.extend([early, other])
    picked = pd.concat(frames).drop_duplicates("chunk_id") if frames else d.iloc[:0].copy()
    remain = quota - len(picked)
    if remain > 0:
        pool = d.loc[~d.chunk_id.isin(picked.chunk_id)]
        picked = pd.concat([picked, pool.sample(n=min(remain, len(pool)), random_state=seed)])
    return picked.sort_values(["review_candidate_label", "date", "assembly"], kind="stable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    ap.add_argument("--output-dir", default="", help="비우면 검색 폴더/relevance_validation")
    ap.add_argument("--review-file", default="", help="수동 판정이 기록된 CSV 경로")
    ap.add_argument("--review-sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.perf_counter()
    root = Path(args.input_dir)
    out = Path(args.output_dir) if args.output_dir else root / "relevance_validation"
    source = root / "trend_analysis" / "검색근거_검토구분.csv"
    runinfo = root / "검색_실행요약.json"
    if not source.is_file() or not runinfo.is_file():
        raise FileNotFoundError("15·16번 산출물이 없습니다. 입력 폴더를 확인하세요.")
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(runinfo.read_text(encoding="utf-8-sig"))
    issue = str(meta.get("query", "")).strip()
    youth_mode = bool(re.search(r"청년.*(?:일자리|취업|고용|창업)|(?:일자리|취업|고용|창업).*청년", issue))
    print(f"=== 24 정책 관련성 검토 시작: {stamp()} ===", flush=True)
    print(f"검색 이슈: {issue} / 입력: {source} / 출력: {out}", flush=True)
    if not youth_mode:
        print("주의: 청년일자리 외 주제는 사전 미구축. 일반 규칙이며 판단보류 우선입니다.", flush=True)
    t = time.perf_counter()
    d = pd.read_csv(source, encoding="utf-8-sig", dtype={"chunk_id": str, "date": str, "month": str}, low_memory=False)
    needed = {"chunk_id", "date", "assembly", "matched_terms", "text_preview"}
    if not needed.issubset(d.columns):
        raise ValueError(f"입력 컬럼 누락: {needed-set(d.columns)}")
    if d.chunk_id.isna().any() or d.chunk_id.duplicated().any():
        raise ValueError("chunk_id 누락 또는 중복")
    expected = meta.get("lexical", {}).get("lexical_matched_chunks")
    if expected is not None and len(d) != int(expected):
        raise ValueError(f"검색 건수 불일치: {len(d)} != {expected}")
    dates = pd.to_datetime(d.date, errors="coerce")
    if dates.isna().any():
        raise ValueError("잘못된 날짜 발견")
    if "month" not in d.columns:
        d["month"] = dates.dt.strftime("%Y-%m")
    if "meeting_title" not in d.columns:
        d["meeting_title"] = ""
    for col in ["source_file", "source_id", "topic_id"]:
        if col not in d.columns:
            d[col] = ""
    print(f"[1/4 파일·날짜 검증] {len(d):,}건 / {elapsed(t)}", flush=True)
    t = time.perf_counter()
    result = []
    for i, row in enumerate(d.itertuples(index=False), 1):
        obj = extract_one(row.text_preview, row.matched_terms, issue, youth_mode)
        result.append(obj)
        if i % 1000 == 0 or i == len(d):
            print(f"[2/4 문맥 검토] {i:,}/{len(d):,}건 / {elapsed(t)}", flush=True)
    classified = pd.concat([d.reset_index(drop=True), pd.DataFrame(result)], axis=1)
    # Assert evidence excerpt actually appears verbatim in preview.
    bad = classified.apply(lambda r: bool(r.core_evidence_text_100) and r.core_evidence_text_100 not in str(r.text_preview), axis=1)
    if bad.any():
        raise AssertionError(f"원문 존재 검사 실패: {int(bad.sum())}건")
    assert classified.core_evidence_text_100.fillna("").str.len().max() <= 100
    manual = load_manual(args.review_file)
    unknown = set(manual) - set(classified.chunk_id)
    if unknown:
        raise ValueError(f"수동 판정 chunk_id가 이번 검색에 없음: {list(unknown)[:3]}")
    classified["manual_verdict"] = classified.chunk_id.map(lambda x: manual.get(x, ("", ""))[0])
    classified["manual_note"] = classified.chunk_id.map(lambda x: manual.get(x, ("", ""))[1])
    classified["review_status"] = classified.manual_verdict.map(lambda x: "수동검토완료" if x else "미검토")
    # Avoid treating rule-generated labels as confirmed truth.
    classified["final_status"] = classified.apply(lambda r: ("수동확인_" + r.manual_verdict) if r.manual_verdict else "자동후보_미검증", axis=1)
    classified.to_csv(out / "전체_정책관련성_검토결과.csv", index=False, encoding="utf-8-sig")
    print(f"[3/4 전체 결과 저장] {len(classified):,}건 / {elapsed(t)}", flush=True)
    t = time.perf_counter()
    review_set = choose_review_set(classified[classified.review_status=="미검토"], args.review_sample, args.seed)
    cols = [c for c in FIELDS if c in review_set.columns]
    review_set = review_set[cols].copy()
    review_set["manual_verdict"] = ""
    review_set["manual_note"] = ""
    review_set.to_csv(out / "수동검토_입력양식.csv", index=False, encoding="utf-8-sig")
    counts = classified.groupby("review_candidate_label", dropna=False).size().reset_index(name="청크수")
    counts.to_csv(out / "관련성_자동후보분포.csv", index=False, encoding="utf-8-sig")
    confirmed = classified[classified.manual_verdict.isin(["직접관련", "간접관련"])].copy()
    confirmed.sort_values(["date", "assembly", "chunk_id"]).to_csv(out / "수동확인_관련근거.csv", index=False, encoding="utf-8-sig")
    # Earliest is explicitly *tentative* until manual review confirms relevance.
    tentative = classified[classified.review_candidate_label == "직접관련_후보"].sort_values(["date","assembly","chunk_id"])
    tentative.head(50).to_csv(out / "최초관측_자동후보_검토목록.csv", index=False, encoding="utf-8-sig")
    summary = {
        "created_at": stamp(), "query": issue, "input_dir": str(root),
        "input_rows": int(len(d)), "classified_rows": int(len(classified)),
        "auto_candidate_counts": dict(Counter(classified.review_candidate_label)),
        "manual_reviewed_count": int((classified.review_status=="수동검토완료").sum()),
        "manual_confirmed_direct": int((classified.manual_verdict=="직접관련").sum()),
        "manual_confirmed_indirect": int((classified.manual_verdict=="간접관련").sum()),
        "review_sample_rows": int(len(review_set)),
        "earliest_auto_direct_candidate": (str(tentative.iloc[0].date) if len(tentative) else None),
        "earliest_manual_confirmed": (str(confirmed.sort_values("date").iloc[0].date) if len(confirmed) else None),
        "youth_issue_rules_applied": youth_mode,
        "limitations": [
            "자동 후보 분류는 정책 관련성 확정, 정확도 또는 전파 증명이 아님",
            "각 행은 청크이며 개별 회의나 정책 수와 다름",
            "text_preview만 사용했으므로 발언 뒤쪽 문맥이 누락될 수 있음",
            "최초 시점은 2024~2025 보유 자료 내부 관측일로 실제 정책 최초 도입 시점 아님",
            "원문 발췌 100자는 연속 문자열이며 앞뒤 맥락 없이 의미가 달라질 수 있음",
            "25번 시계열은 수동 검토 범위와 자동 후보 범위를 반드시 구분해 제시해야 함",
        ],
        "elapsed_seconds": round(time.perf_counter()-t0, 3),
    }
    (out / "검토_요약.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    print(f"[4/4 검토 양식·요약 저장] {elapsed(t)}", flush=True)
    print(f"=== 24 완료 / {elapsed(t0)} / {out} ===", flush=True)
    print("자동 후보 분포:", summary["auto_candidate_counts"], flush=True)
    print("주의: 자동 '직접관련_후보'는 사람이 검증한 직접 관련 건수가 아닙니다.", flush=True)


if __name__ == "__main__":
    main()

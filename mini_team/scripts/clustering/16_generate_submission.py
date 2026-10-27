# -*- coding: utf-8 -*-
"""
16_generate_submission.py
대회 제출 '초안' 생성기: 5개 이슈 × 5개 제출 필드
실제 원본 검증 없는 최초 등장/정책 전파를 확정하지 않는다.
필수: 13단계 first-observed CSV, 14-4-2 복원 CSV, 15단계 RAG CSV.
선택: 15-3단계 AI 모의 품질 평가 CSV, 13단계 원문 근거 CSV.

PowerShell:
  python 16_generate_submission.py
  python 16_generate_submission.py --check-only
  python 16_generate_submission.py --model "qwen3:8b" --mode "B_토픽강화RAG"

원본 입력을 수정하지 않고 새로운 output 폴더로 CSV/검증 로그를 저장한다.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

BASE = Path(r"C:\mini_team\scripts\clustering")
ISSUES = ["지역소멸", "청년 일자리", "저출생", "고령화", "기후위기"]
SUBMISSION_COLS = [
    "이슈키워드", "등장시점", "타 지방의회로의 흐름", "향후 예측", "근거 회의록"
]
AUDIT_FIELDS = [
    "이슈키워드", "선정모델", "RAG방식", "예측글자수", "예측선정상태",
    "최초관측일_POLICY범위", "최초관측의회", "최초근거청크ID",
    "지역간후속논의쌍", "초기회의록명", "후속회의록명",
    "초기근거청크ID", "후속근거청크ID", "유사도", "원문확인상태",
    "회의록복원상태", "직접전파근거", "필수확인사항", "제출판정"
]

def read_csv(path, required=()):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"입력 파일 없음: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        names = set(r.fieldnames or [])
        missing = set(required) - names
        if missing:
            raise ValueError(f"{path.name}: 필요한 컬럼 누락 {sorted(missing)} / 현재 {sorted(names)}")
        return list(r)

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def norm(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()

def as_float(value, default=-9999.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def as_int(value, default=0):
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

def full_date(value):
    s = norm(value)
    m = re.search(r"(20\d{2})[-./]?(\d{2})[-./]?(\d{2})", s)
    if not m:
        return ""
    yyyy, mm, dd = map(int, m.groups())
    try:
        return f"{yyyy:04d}-{mm:02d}-{dd:02d}" if datetime(yyyy, mm, dd) else ""
    except ValueError:
        return ""

def find_prediction(predictions, scores, issue, prefer_model="", prefer_mode=""):
    rows = [r for r in predictions if norm(r.get("issue_keyword")) == issue
            and norm(r.get("status")) in {"OK", "OVER_200"}
            and norm(r.get("prediction"))]
    if prefer_model:
        rows = [r for r in rows if r.get("model") == prefer_model]
    if prefer_mode:
        rows = [r for r in rows if r.get("mode") == prefer_mode]
    if not rows:
        return None, "정상 예측 없음"

    def key(r):
        content = norm(r["prediction"])
        review = scores.get((issue, norm(r.get("model")), norm(r.get("mode")), content))
        ai_score = as_float(review.get("provisional_subtotal_0_55")) if review else -1
        under_limit = len(content) <= 200
        # only compare scores when available; prefer 200-compliant and then AI provisional 55-point score
        return (int(under_limit), ai_score,
                int(r.get("mode") == "B_토픽강화RAG"),
                -len(content))
    rows.sort(key=key, reverse=True)
    best = rows[0]
    if len(norm(best["prediction"])) > 200:
        return best, "200자 초과: 제출 예측 빈칸 처리"
    return best, "정상 답변 자동선정(수동 품질 확정 아님)"

def select_first(first_rows, issue):
    candidates = []
    for r in first_rows:
        if norm(r.get("issue")) != issue:
            continue
        day = full_date(r.get("date") or r.get("first_observed_date_in_policy_cache"))
        if day:
            candidates.append((day, norm(r.get("assembly")), norm(r.get("chunk_id")), r))
    return min(candidates, key=lambda x: (x[0], x[1], x[2])) if candidates else None

def select_flow(links, issue):
    # Limit to unique-source metadata on both ends; otherwise do not assert a meeting title.
    candidates = []
    for r in links:
        if norm(r.get("issue")) != issue:
            continue
        a, b = full_date(r.get("earlier_date")), full_date(r.get("later_date"))
        if not a or not b or a >= b or norm(r.get("earlier_assembly")) == norm(r.get("later_assembly")):
            continue
        a_ok = r.get("earlier_restore_status") == "confirmed_unique"
        b_ok = r.get("later_restore_status") == "confirmed_unique"
        title_a = norm(r.get("earlier_original_meeting_title") or r.get("earlier_meeting_title"))
        title_b = norm(r.get("later_original_meeting_title") or r.get("later_meeting_title"))
        if not (a_ok and b_ok and title_a and title_b):
            continue
        # Guard known issue of procedural and duplicate text contamination.
        flags = norm(r.get("earlier_format_flags")) + norm(r.get("later_format_flags"))
        if flags or norm(r.get("duplicate_reason")):
            continue
        # Strong cached excerpt verification from 14-3.
        ea = norm(r.get("earlier_evidence_check"))
        eb = norm(r.get("later_evidence_check"))
        # 14-4-2 inherits audit fields from 14-3.
        if ea and ea not in ("원문발췌_일치", "공백제거후_일치"):
            continue
        if eb and eb not in ("원문발췌_일치", "공백제거후_일치"):
            continue
        score = as_float(r.get("semantic_cosine"), default=0)
        candidates.append((score, r))
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p[0], -len(norm(p[1].get("earlier_evidence_180")))), reverse=True)
    return candidates[0][1]

def main():
    ap = argparse.ArgumentParser(description="16단계 대회 제출 5열 초안 생성")
    ap.add_argument("--root", type=Path, default=BASE)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--model", default="", help="예: qwen3:8b (미입력시 200자 준수/AI 잠정 점수 기준 선정)")
    ap.add_argument("--mode", default="", choices=["", "A_기본RAG", "B_토픽강화RAG"])
    args = ap.parse_args()

    root = args.root
    results = root / "results"
    first_path = results / "policy_issue_trends/issue_first_observed_by_region.csv"
    restored_path = results / "policy_meeting_metadata_v2/policy_links_meeting_metadata_restored.csv"
    prediction_path = results / "policy_prediction_comparison/policy_trend_ab_results.csv"
    quality_path = results / "policy_prediction_quality_review/15_3_ai_simulated_reviews.csv"
    output = args.out or results / "competition_submission"

    print("[1/4] 필수 입력 확인")
    for name, path in (("최초 관측", first_path), ("출처 복원", restored_path), ("예측", prediction_path)):
        print(f"  {name}: {path} / {'OK' if path.is_file() else '없음'}")
    print(f"  품질 평가: {quality_path} / {'OK' if quality_path.is_file() else '없음 (선택)'}")
    if args.check_only:
        return 0

    # Fail safely instead of fabricating earliest or provenance.
    firsts = read_csv(first_path, ["issue", "assembly", "date", "chunk_id"])
    links = read_csv(restored_path, [
        "issue", "earlier_restore_status", "later_restore_status", "earlier_date", "later_date",
        "earlier_chunk_id", "later_chunk_id"
    ])
    predictions = read_csv(prediction_path, ["issue_keyword", "model", "mode", "prediction", "status"])
    quality = read_csv(quality_path) if quality_path.is_file() else []
    scores = {(norm(r.get("issue_keyword")), norm(r.get("model")),
               norm(r.get("mode")), norm(r.get("prediction"))): r for r in quality}
    print(f"[2/4] 최초관측 {len(firsts)}행 / 유사논의 {len(links)}쌍 / 예측 {len(predictions)}행")

    submission, audit, alerts = [], [], []
    for issue in ISSUES:
        issues = []
        first = select_first(firsts, issue)
        link = select_flow(links, issue)
        pred, prediction_note = find_prediction(predictions, scores, issue, args.model, args.mode)
        if not first:
            first_display = "미확인"
            issues.append("POLICY 최초 관측자료 없음")
            day, council, first_id = "", "", ""
        else:
            day, council, first_id, first_record = first
            first_display = f"{day} / {council} (POLICY 선별 데이터 최초 관측)"
            issues.append("전체 원본 218만 청크 최초 여부 별도 검증 필요")
        flow_text = "지역 간 유사 논의 근거 미확보"
        evidence = ""
        early_title = later_title = early_id = later_id = ""
        if link:
            a = norm(link["earlier_assembly"])
            b = norm(link["later_assembly"])
            d1, d2 = full_date(link["earlier_date"]), full_date(link["later_date"])
            flow_text = (
                f"{a}({d1}) → {b}({d2}): "
                f"내용 유사 후보(정책 전파 미확인)"
            )
            early_title = norm(link.get("earlier_original_meeting_title") or link.get("earlier_meeting_title"))
            later_title = norm(link.get("later_original_meeting_title") or link.get("later_meeting_title"))
            early_id = norm(link.get("earlier_chunk_id"))
            later_id = norm(link.get("later_chunk_id"))
            ea = norm(link.get("earlier_evidence_180"))
            eb = norm(link.get("later_evidence_180"))
            evidence = (
                f"[선행] {early_title} ({d1}, {a}), 청크 {early_id}: {ea} "
                f"[후속] {later_title} ({d2}, {b}), 청크 {later_id}: {eb}"
            )
            issues.append("유사 논의 실제 전파·정책 동일성 수동 검증 필요")
        else:
            issues.append("양측 출처가 유일하게 복원된 유사 논의 후보 부족")
        # Initial appearance evidence is NOT asserted to have a verified meeting name.
        if first:
            evidence = (
                f"[최초 관측 근거] {day} {council}, 청크 {first_id}, "
                f"회의록명 추가 확인 필요. " + evidence
            )
        prediction = norm(pred.get("prediction")) if pred else ""
        if len(prediction) > 200:
            prediction = ""
            issues.append("200자 이내 예측 재작성 필요")
        if not prediction:
            issues.append(f"예측문 미확보: {prediction_note}")

        state = "검증필요" if issues else "제출가능"
        submission.append({
            "이슈키워드": issue,
            "등장시점": first_display,
            "타 지방의회로의 흐름": flow_text,
            "향후 예측": prediction,
            "근거 회의록": evidence or "확인된 회의록 근거 없음",
        })
        audit.append({
            "이슈키워드": issue, "선정모델": pred.get("model", "") if pred else "",
            "RAG방식": pred.get("mode", "") if pred else "",
            "예측글자수": len(prediction),
            "예측선정상태": prediction_note,
            "최초관측일_POLICY범위": day if first else "",
            "최초관측의회": council if first else "",
            "최초근거청크ID": first_id if first else "",
            "지역간후속논의쌍": (
                f"{link.get('earlier_assembly')}→{link.get('later_assembly')}" if link else ""),
            "초기회의록명": early_title, "후속회의록명": later_title,
            "초기근거청크ID": early_id, "후속근거청크ID": later_id,
            "유사도": link.get("semantic_cosine", "") if link else "",
            "원문확인상태": (
                f"{link.get('earlier_evidence_check','')};{link.get('later_evidence_check','')}" if link else ""),
            "회의록복원상태": "양측 confirmed_unique" if link else "미확인",
            "직접전파근거": "미검증", "필수확인사항": " / ".join(issues),
            "제출판정": state,
        })
        alerts.append({"issue": issue, "status": state, "problems": issues})
        print(f"  {issue}: 예측={len(prediction)}자 / 양측 회의록={'있음' if link else '없음'} / {state}")

    print("[3/4] 제출 초안 및 감사표 저장")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "competition_submission_draft.csv", submission, SUBMISSION_COLS)
    write_csv(output / "competition_submission_audit.csv", audit, AUDIT_FIELDS)
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "submission_status": "DRAFT_NOT_VERIFIED",
        "issue_count": len(submission),
        "input_paths": {
            "first_observed": str(first_path), "restored_links": str(restored_path),
            "predictions": str(prediction_path),
            "quality_optional": str(quality_path) if quality_path.is_file() else None,
        },
        "warnings": [
            "13단계 등장시점은 POLICY 344664개에서의 최초 관측이며 전체 원본 최초가 아닙니다.",
            "14단계 내용 유사성은 정책 전파·인과관계의 증거가 아닙니다.",
            "14-4-2 confirmed_unique는 발언의 원본 회의록을 유일하게 찾았다는 의미입니다.",
            "15-3 AI 모의 품질 점수는 실제 인간 정답/근거 검증이 아닙니다.",
            "해당 초안에는 최초 관측 회의록명, 예측 근거 타당성 등 확인 필요 항목이 남아 있습니다.",
        ],
        "issues": alerts,
    }
    (output / "competition_submission_validation.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[4/4] 완료: {output}")
    print("주의: 'competition_submission_draft.csv'는 제출 확정 파일이 아니라 검증용 초안입니다.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        sys.exit(1)

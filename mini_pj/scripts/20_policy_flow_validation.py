# -*- coding: utf-8 -*-
"""20_policy_flow_validation.py

19단계에서 얻은 *지역 간 유사 정책 후속 논의 후보*에 대해
중복 제거, 정책 내용(대상/수단/목적), 의사진행 편향, 전파 인용 단서를
검토하는 안전 우선 검증기. LLM/임베딩 재학습 없음.

핵심 원칙:
- '유사 논의' ≠ '지역 간 정책 전파' (명시적 근거와 수동 검증 필수).
- 원문은 잘린 preview일 수 있음. 자동 결과는 검토 후보이지 확정 판정이 아님.
- 동일 지역-날짜-회의-정규화된 텍스트는 중복 청크로 그룹화.
- 동일 지역쌍이라도 사업 수단이 다르면 다른 '정책 내용 후보'로 취급.
- 수동 검토 CSV를 작성하면 --review-file로 적용할 수 있음.
- 입력 파일/19단계 산출물은 절대 수정하지 않음.

실행 (PowerShell):
& "C:\\Users\\Admin\\miniconda3\\envs\\mini\\python.exe" "C:\\mini_pj\\scripts\\20_policy_flow_validation.py" --input-dir "C:\\mini_pj\\results\\policy_issue_search\\20261025_000801_청년_일자리"

테스트: python 20_policy_flow_validation.py --candidates <CSV 경로> --output-dir <저장 경로>
검증 반영: python 20_policy_flow_validation.py --input-dir <검색 폴더> --review-file <수동검토_입력양식.csv>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path(r"C:\mini_pj\results\policy_issue_search\20261025_000801_청년_일자리")
DEFAULT_SOURCE_DIR = "similarity_flow_v2_sim0.70_txt0.12"

# 본 사업 관련 직접 비교 가능한 정책 수단 (단어 1개보다 구체적인 표현 우선)
MECHANISMS = {
    "월세지원": [r"청년\s*월세", r"월세\s*(?:특별)?\s*지원"],
    "임차보증금이자지원": [r"임차\s*보증금\s*이자", r"주택\s*임차\s*보증금\s*이자"],
    "자격증취득지원": [r"자격증\s*(?:응시료|취득)\s*지원", r"자격증\s*(?:응시료|취득)"],
    "대학일자리플러스센터": [r"대학\s*일자리\s*플러스\s*센터"],
    "청년도전지원": [r"청년\s*도전\s*지원"],
    "청년인턴": [r"청년\s*인턴", r"인턴십"],
    "구직수당": [r"청년\s*드림\s*수당", r"구직\s*(?:활동)?\s*지원\s*금", r"구직\s*수당"],
    "창업지원조례": [r"청년\s*창업\s*지원\s*조례"],
    "창업지원사업": [r"청년\s*창업\s*(?:지원|보육|공간|센터|자금)"],
    "채용연계": [r"채용\s*연계", r"취업\s*연계"],
    "직업훈련": [r"직업\s*훈련", r"취업\s*훈련"],
}
# 사업과의 직접 관련성을 가늠하기 위한 범주 (자동 확정이 아닌 힌트)
PURPOSES = {
    "취업역량강화": [r"취업\s*역량", r"취업\s*서비스", r"인재\s*육성", r"구직\s*활동"],
    "주거안정": [r"주거\s*(?:안정|지원)", r"주택\s*임차", r"청년\s*월세"],
    "창업촉진": [r"청년\s*창업", r"창업\s*활성화"],
    "고용확대": [r"일자리\s*(?:창출|확대)", r"고용\s*(?:창출|지원)", r"채용\s*연계"],
}
TARGETS = {
    "청년": [r"청년", r"19세\s*에서\s*34세", r"19\s*~\s*34세"],
    "취업준비생": [r"취업\s*준비", r"구직자", r"구직\s*활동"],
    "창업청년": [r"청년\s*창업", r"예비\s*창업"],
}
# 표결 또는 형식적 예산 총괄문만 서로 비슷한 경우 과도한 연결 방지
PROCEDURAL = [r"투표를\s*종료", r"표결\s*결과", r"가결되었음을\s*선포", r"의사일정\s*제\s*\d+항", r"재석의원", r"투표해\s*주시기"]
GENERAL_BUDGET = [r"일반회계\s*세출", r"세출예산안", r"기정예산", r"예산서\s*\d+쪽", r"사업명세서"]
# 확인 단서가 명시적으로 나타나더라도 언급 문맥/참조 대상의 실재성을 별도로 검토함
TRANSFER_CUES = [r"(?:타\s*(?:시|도|지자체|지역)|다른\s*(?:지역|시도)|타\s*지방자치단체|타\s*지방의회).{0,60}(?:벤치마킹|참고|도입|사례)",
                 r"(?:벤치마킹|참고|도입).{0,60}(?:다른\s*(?:지역|시도)|타\s*(?:시|도|지역)|지자체)"]
OUTPUT_FIELDS = [
    "candidate_id", "earlier_assembly", "earlier_date", "later_assembly", "later_date", "lag_days",
    "earlier_meeting_title", "later_meeting_title", "earlier_chunk_id", "later_chunk_id",
    "duplicate_group_id", "merged_pair_count", "merged_earlier_chunk_ids", "merged_later_chunk_ids",
    "embedding_cosine", "text_ngram_cosine", "shared_policy_mechanisms",
    "specific_common_mechanisms", "shared_targets", "shared_purposes",
    "earlier_procedural_ratio", "later_procedural_ratio", "budget_style_flag",
    "later_transfer_cue_excerpt", "content_judgment", "content_reason", "transfer_status",
    "review_priority", "earlier_policy_excerpt", "later_policy_excerpt",
    "earlier_text_preview", "later_text_preview", "review_status", "review_note"
]


def log(message):
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def policy_excerpt(text, max_len=400):
    """청년/지원/사업/조례의 실제 문맥을 우선 보이게 선택 (미리보기 내부에서만)."""
    text = clean_text(text)
    matches = list(re.finditer(r"청년.{0,20}(?:월세|창업|취업|일자리|자격증|주거|도전)|(?:자격증|월세|창업|구직).{0,15}지원", text))
    if not matches:
        return text[:max_len]
    m = matches[0]
    start = max(0, m.start() - 75)
    return text[start:start + max_len]


def tags(text, dictionary):
    return sorted(k for k, patterns in dictionary.items() if any(re.search(p, text) for p in patterns))


def ratio(text, patterns):
    text = clean_text(text)
    if not text:
        return 0.0
    # 매우 긴 정책 설명도 의사진행 표현 1회로 과도하게 탈락하지 않도록 문장 기준 근사
    count = sum(len(re.findall(p, text)) for p in patterns)
    return round(min(1.0, count * 16 / max(len(text), 1)), 3)


def find_transfer_cue(text):
    for p in TRANSFER_CUES:
        m = re.search(p, text)
        if m:
            return text[max(0, m.start()-60):m.end()+60]
    return ""


def dedup_fingerprint(row):
    """source_id/청크번호 대신 의회+날짜+내용을 비교해 반복 발언 병합.

    앞/뒤 텍스트를 각각 정규화하고 SHA로 요약한다.
    중간 공백/문장부호 차이와 한글 띄어쓰기 소폭 차이 흡수.
    서로 다른 회의의 동일한 표결 텍스트는 날짜/의회로 구분한다.
    """
    def compact(t):
        return re.sub(r"[^가-힣A-Za-z0-9]", "", clean_text(t))[:1000]
    payload = "|".join([
        clean_text(row["earlier_assembly"]), clean_text(row["earlier_date"]),
        compact(row["earlier_text_preview"]),
        clean_text(row["later_assembly"]), clean_text(row["later_date"]),
        compact(row["later_text_preview"]),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def assess(row):
    a = clean_text(row["earlier_text_preview"])
    b = clean_text(row["later_text_preview"])
    mech_a, mech_b = set(tags(a, MECHANISMS)), set(tags(b, MECHANISMS))
    shared_mech = sorted(mech_a & mech_b)
    shared_purpose = sorted(set(tags(a, PURPOSES)) & set(tags(b, PURPOSES)))
    shared_target = sorted(set(tags(a, TARGETS)) & set(tags(b, TARGETS)))
    pa, pb = ratio(a, PROCEDURAL), ratio(b, PROCEDURAL)
    budget = bool(any(re.search(x, a) for x in GENERAL_BUDGET) and
                  any(re.search(x, b) for x in GENERAL_BUDGET))
    cue = find_transfer_cue(b)

    # 자동 라벨은 정책 실질 동일성을 '확정'하지 않는다.
    if pa >= 0.20 or pb >= 0.20:
        verdict = "의사진행_유사_우세"
        reason = "표결·의사진행 표현의 영향이 커서 정책 세부 내용 동일성 확인 불가"
    elif not shared_mech:
        verdict = "정책분야_유사_내용미확인"
        reason = "양쪽 미리보기에서 동일한 구체 정책 수단이 검출되지 않음"
    elif budget and len(shared_mech) == 1 and not shared_purpose:
        verdict = "예산설명_공통사업_검토"
        reason = "사업명이 함께 등장하나 포괄 예산 설명 중심; 독립 사업·국가 공통사업 가능성"
    elif shared_mech and (shared_purpose or len(shared_mech) >= 2):
        verdict = "구체정책_유사후보"
        reason = "양쪽 텍스트에 구체 정책 수단과 목적 또는 복수 사업의 공통 표현 존재; 실제 동일 사업인지는 미검증"
    else:
        verdict = "정책수단_일부공통_검토"
        reason = "일부 세부 사업명 일치; 정책 대상·목적·사업 운영 방식 추가 검증 필요"

    transfer_status = "전파미확인_명시적언급단서검토" if cue else "전파미확인_명시적단서없음"
    priority = {"구체정책_유사후보":1, "정책수단_일부공통_검토":2,
                "예산설명_공통사업_검토":3, "정책분야_유사_내용미확인":4,
                "의사진행_유사_우세":5}[verdict]
    if cue:
        priority = max(1, priority-1)
    return {
        "specific_common_mechanisms": ", ".join(shared_mech),
        "shared_targets": ", ".join(shared_target),
        "shared_purposes": ", ".join(shared_purpose),
        "earlier_procedural_ratio": pa,
        "later_procedural_ratio": pb,
        "budget_style_flag": budget,
        "later_transfer_cue_excerpt": cue,
        "content_judgment": verdict,
        "content_reason": reason,
        "transfer_status": transfer_status,
        "review_priority": priority,
        "earlier_policy_excerpt": policy_excerpt(a),
        "later_policy_excerpt": policy_excerpt(b),
    }


def load_reviews(path):
    if path is None:
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    required = {"duplicate_group_id", "manual_content_verdict", "manual_transfer_verdict", "manual_review_note"}
    if not required.issubset(df.columns):
        raise ValueError(f"수동 검토 CSV 필수 컬럼 누락: {required - set(df.columns)}")
    valid_content = {"", "구체정책유사_확인", "동일정책여부_불명", "정책내용다름", "조례제목만유사", "원문추가필요"}
    valid_transfer = {"", "전파근거미확인", "명시적참고근거_원문확인", "독립적논의", "검토중"}
    out = {}
    for _, r in df.iterrows():
        key = r["duplicate_group_id"].strip()
        if not key:
            continue
        if r["manual_content_verdict"] not in valid_content or r["manual_transfer_verdict"] not in valid_transfer:
            raise ValueError(f"잘못된 수동 라벨: {key}")
        if key in out:
            raise ValueError(f"수동 검토 중복 ID: {key}")
        out[key] = r.to_dict()
    return out


def atomic_csv(df, path):
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def run(args):
    started = time.perf_counter()
    source = Path(args.candidates) if args.candidates else Path(args.input_dir) / DEFAULT_SOURCE_DIR / "지역간_유사정책_전체후보.csv"
    output = Path(args.output_dir) if args.output_dir else Path(args.input_dir) / "flow_validation"
    if not source.is_file():
        raise FileNotFoundError(f"19번 후보 CSV 없음: {source}")
    if output.resolve() == source.parent.resolve():
        raise ValueError("입력 폴더와 출력 폴더는 달라야 합니다.")
    output.mkdir(parents=True, exist_ok=True)
    log(f"=== 20 정책 흐름 검증 시작 / 입력: {source} ===")
    log(f"출력: {output}")

    df = pd.read_csv(source, encoding="utf-8-sig", dtype=str).fillna("")
    required = ["earlier_assembly", "earlier_date", "later_assembly", "later_date",
                "earlier_text_preview", "later_text_preview", "earlier_chunk_id", "later_chunk_id"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"필수 컬럼 누락: {set(required)-set(df.columns)}")
    for c in ("embedding_cosine", "text_ngram_cosine", "lag_days"):
        if c not in df:
            df[c] = ""
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in OUTPUT_FIELDS:
        if c not in df:
            df[c] = ""
    log(f"[1/4 후보 로드] {len(df):,}건 / {time.perf_counter()-started:.2f}초")

    if len(df):
        df["duplicate_group_id"] = df.apply(dedup_fingerprint, axis=1)
        records = []
        for j, (fp, group) in enumerate(df.groupby("duplicate_group_id", sort=False), 1):
            group = group.sort_values("embedding_cosine", ascending=False)
            row = group.iloc[0].to_dict()
            row["merged_pair_count"] = len(group)
            row["merged_earlier_chunk_ids"] = ";".join(sorted(group.earlier_chunk_id.astype(str).unique()))
            row["merged_later_chunk_ids"] = ";".join(sorted(group.later_chunk_id.astype(str).unique()))
            row["candidate_id"] = f"PV{j:03d}"
            row.update(assess(row))
            row["review_status"] = "자동검토_수동확인필요"
            row["review_note"] = ""
            records.append(row)
        dedup = pd.DataFrame(records)
        # exact preview 중복 외에 동일 사건·회의·정책 태그인 후보를 다시 그룹화하지 않는다.
        # 서로 다른 발언/후속 논의가 동일 회의에서 여럿 있을 수 있기 때문.
    else:
        dedup = pd.DataFrame(columns=OUTPUT_FIELDS)
    log(f"[2/4 중복 묶음] {len(df):,}건 → {len(dedup):,}개 고유 원문쌍 / {time.perf_counter()-started:.2f}초")

    manual = load_reviews(Path(args.review_file) if args.review_file else None)
    for i, row in dedup.iterrows():
        review = manual.get(row["duplicate_group_id"])
        if review:
            dedup.at[i, "review_note"] = review.get("manual_review_note", "")
            dedup.at[i, "review_status"] = "사람검토_입력됨"
            if review.get("manual_content_verdict"):
                dedup.at[i, "content_judgment"] = review["manual_content_verdict"]
            # 실제 참조 문장과 출처까지 검토한 경우에도 인과적 전파는 자동 확정하지 않음.
            if review.get("manual_transfer_verdict"):
                dedup.at[i, "transfer_status"] = review["manual_transfer_verdict"]

    if len(dedup):
        dedup = dedup.sort_values(["review_priority", "embedding_cosine"], ascending=[True, False]).reset_index(drop=True)
    cols = [c for c in OUTPUT_FIELDS if c in dedup.columns]
    atomic_csv(dedup[cols], output / "정책내용_고유후보_검토결과.csv")

    review_template = dedup[[c for c in ["duplicate_group_id", "candidate_id", "earlier_assembly", "later_assembly",
                                             "specific_common_mechanisms", "content_judgment", "transfer_status",
                                             "earlier_policy_excerpt", "later_policy_excerpt"] if c in dedup.columns]].copy()
    for col in ("manual_content_verdict", "manual_transfer_verdict", "manual_review_note"):
        review_template[col] = review_template["duplicate_group_id"].map(lambda k: manual.get(k, {}).get(col, ""))
    atomic_csv(review_template, output / "수동검토_입력양식.csv")

    # '유사정책 후보'만 전달. 확정 전파 화살표용 데이터는 기본적으로 빈 파일.
    candidate_labels = {"구체정책_유사후보", "정책수단_일부공통_검토", "구체정책유사_확인"}
    map_candidates = dedup[dedup.content_judgment.isin(candidate_labels)].copy()
    map_candidates["display_status"] = "내용유사_검토후보_전파미확인"
    # 명시적 원문 참조가 확인되지 않은 경우 점선/비방향 후보로만 시각화해야 함.
    map_cols = [c for c in ["candidate_id", "duplicate_group_id", "earlier_assembly", "later_assembly", "earlier_date", "later_date",
                             "embedding_cosine", "specific_common_mechanisms", "content_judgment", "transfer_status", "display_status"] if c in map_candidates.columns]
    atomic_csv(map_candidates[map_cols], output / "웹_정책내용유사_점선후보.csv")
    confirmed = dedup[(dedup.review_status == "사람검토_입력됨") &
                      (dedup.transfer_status == "명시적참고근거_원문확인")].copy()
    # '명시적 언급' 검증이어도 참고 관계의 근거일 뿐 채택/전파 인과 확정이 아니다.
    confirmed["display_status"] = "원문에서_타지역참고표현_검증됨_전파인과미확정"
    confirm_cols = [c for c in ["candidate_id", "earlier_assembly", "later_assembly", "earlier_date", "later_date", "transfer_status", "review_note", "display_status"] if c in confirmed.columns]
    atomic_csv(confirmed[confirm_cols], output / "웹_명시적참고근거_수동확인.csv")

    total = time.perf_counter() - started
    counts = dedup.content_judgment.value_counts().to_dict() if len(dedup) else {}
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"), "source": str(source),
        "raw_pair_rows": len(df), "unique_preview_pair_groups": len(dedup),
        "duplicated_rows_merged": len(df)-len(dedup),
        "content_judgment_counts": {str(k):int(v) for k,v in counts.items()},
        "map_similarity_candidates": len(map_candidates),
        "manual_reference_cues_reviewed": len(confirmed),
        "confirmed_diffusion_causation": 0,
        "manual_review_applied": sum(dedup.review_status.eq("사람검토_입력됨")) if len(dedup) else 0,
        "limitations": [
            "기본 입력은 19번의 최대 600자 미리보기이며 문서 전체 사실 검증이 아님",
            "중복 제거는 양쪽 의회·날짜·정규화된 미리보기 텍스트 기준 (완전한 중복 문서 판별 아님)",
            "정책 분류는 한국어 구문 패턴 기반 초안이며 사람 검증 전에는 동일 정책 확정 불가",
            "타 지역 인용과 정책 채택/전파의 인과적 증명은 다름",
            "지도에서 점선 후보를 화살표로 확정 전파처럼 표시하면 안 됨",
        ],
        "elapsed_seconds": round(total, 3),
    }
    summary_file = output / "검증_요약.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    log(f"[3/4 정책내용 검토] {json.dumps(summary['content_judgment_counts'], ensure_ascii=False)}")
    log(f"[4/4 저장 완료] 고유후보 {len(dedup)} / 지도 점선후보 {len(map_candidates)} / 명시적참고 수동확인 {len(confirmed)}")
    log(f"=== 20 검증 종료: {total:.2f}초 / {total/60:.2f}분 / 출력: {output} ===")
    log("주의: 자동 품질 검토 결과는 정책 내용 동일성·정책 전파의 확정 근거가 아닙니다.")


def main():
    parser = argparse.ArgumentParser(description="19번 유사정책 후보의 중복·정책내용·명시적 참고 단서 검증")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--candidates", help="19번 전체후보 CSV 직접 지정 (테스트용)")
    parser.add_argument("--output-dir", help="출력 디렉터리 (기본: <input-dir>/flow_validation)")
    parser.add_argument("--review-file", help="수동 검토 결과를 입력한 CSV 경로")
    run(parser.parse_args())

if __name__ == "__main__":
    main()

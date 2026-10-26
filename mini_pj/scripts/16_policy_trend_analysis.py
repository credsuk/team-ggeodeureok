# -*- coding: utf-8 -*-
r"""16. 정책 이슈 지역/시간 변화 분석 - 15번 검색 결과만 사용 (재학습/Chroma 접근 없음).

기본: C:\mini_pj\results\policy_issue_search 아래 최신 '검색_실행요약.json' 포함 폴더 선택
사용:
  python 16_policy_trend_analysis.py
  python 16_policy_trend_analysis.py --input-dir "C:\mini_pj\results\policy_issue_search\20261025_000801_청년_일자리"
  python 16_policy_trend_analysis.py --broad-terms "취업 지원"
  python 16_policy_trend_analysis.py --broad-terms ""   # 구분하지 않고 모든 어휘 일치 사용

핵심 주의:
  - 검색어 일치 청크(전체)와 좁은 어휘(직접표현) 일치 청크를 구분.
  - '취업 지원' 단독 등 광범위한 표현은 별도 검토; 그것이 거짓 양성임을 확정하지 않음.
  - 각 지역/월의 분모는 15번이 계산한 전체 청크. 중복 청크 및 합계를 검증.
  - 최초 관측은 2024~2025년 수집 코퍼스 내 최초이며 실제 최초 등장/확산 증거가 아님.
  - Chroma 의미 검색 상위 50건은 빈도 통계에 혼합하지 않음.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\mini_pj")
SEARCH_ROOT = ROOT / "results" / "policy_issue_search"
EXPECTED_CORPUS = 684_311


def stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def duration(seconds):
    return f"{seconds:.2f}초 / {seconds/60:.2f}분 / {seconds/3600:.3f}시간"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"필수 파일 없음: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"chunk_id": "string", "date": "string", "month": "string"}, low_memory=False)


def resolve_input(p: str) -> Path:
    if p:
        path = Path(p)
        if not (path / "검색_실행요약.json").is_file():
            raise ValueError(f"15번 검색 실행 결과 폴더가 아닙니다: {path}")
        return path
    candidates = sorted((x for x in SEARCH_ROOT.iterdir() if x.is_dir() and (x / "검색_실행요약.json").exists()), key=lambda x:x.name)
    if not candidates:
        raise FileNotFoundError(f"검색 실행 폴더 없음: {SEARCH_ROOT}")
    return candidates[-1]


def save(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  저장: {path.name} / {len(df):,}행", flush=True)


def pct(n, d):
    return np.where(np.asarray(d)>0, np.asarray(n)*100/np.maximum(np.asarray(d), 1), 0).round(4)


def build(input_dir: Path, broad_terms: list[str], min_examples: int):
    begin = time.perf_counter()
    out = input_dir / "trend_analysis"
    out.mkdir(exist_ok=True)
    print(f"=== 16번 정책 트렌드 분석 시작: {stamp()} ===", flush=True)
    print(f"입력: {input_dir}\n출력: {out}", flush=True)
    report = json.loads((input_dir / "검색_실행요약.json").read_text(encoding="utf-8-sig"))
    search_terms = report.get("lexical_terms", [])
    print(f"검색 이슈: {report.get('query')} / 검색 표현: {search_terms}", flush=True)
    print(f"광범위 표현(분리만 수행): {broad_terms}", flush=True)
    t = time.perf_counter()

    hits = read_csv(input_dir / "전체_어휘일치_근거.csv")
    regions = read_csv(input_dir / "지역별_어휘일치_통계.csv")
    months = read_csv(input_dir / "월별_어휘일치_통계.csv")
    region_month = read_csv(input_dir / "지역_월별_어휘일치_추이.csv")
    required_hits = {"chunk_id","date","month","assembly","matched_terms","text_preview"}
    if not required_hits.issubset(hits):
        raise ValueError(f"어휘 근거 컬럼 누락: {required_hits - set(hits)}")
    expected = int(report.get("lexical",{}).get("lexical_matched_chunks", -1))
    if len(hits) != expected:
        raise ValueError(f"검색 기록 {expected:,}건 vs 근거 CSV {len(hits):,}건 불일치")
    if hits.chunk_id.isna().any() or hits.chunk_id.duplicated().any():
        raise ValueError("검색 근거 chunk_id 공란 또는 중복. 중단")
    if hits[["date","month","assembly"]].isna().any().any():
        raise ValueError("검색 근거 날짜/지역 누락")
    if not (hits.date.str.slice(0,7) == hits.month).all():
        raise ValueError("검색 근거 date와 month 불일치")
    if pd.to_datetime(hits.date, format="%Y-%m-%d", errors="coerce").isna().any():
        raise ValueError("검색 근거 날짜 형식 오류")
    for frame,name in [(regions,"지역별"),(months,"월별"),(region_month,"지역×월")]:
        if int(frame.matched_chunks.sum()) != len(hits):
            raise ValueError(f"{name} 어휘 집계합계가 {len(hits):,}건과 다름")
    if int(regions.all_chunks.sum()) != EXPECTED_CORPUS or int(months.all_chunks.sum()) != EXPECTED_CORPUS or int(region_month.all_chunks.sum()) != EXPECTED_CORPUS:
        raise ValueError("지역/월/지역×월 전체 분모가 684,311건과 다름")
    if region_month.duplicated(["assembly","month"]).any():
        raise ValueError("지역×월 중복 집계 행 발견")
    print(f"[1/5 입력 검증] 전체 {EXPECTED_CORPUS:,}건 / 어휘일치 {len(hits):,}건 / {duration(time.perf_counter()-t)}",flush=True)

    t = time.perf_counter()
    broad = set(broad_terms)
    term_lists = hits.matched_terms.fillna("").astype(str).str.split("|")
    hits["broad_only"] = term_lists.map(lambda a: bool(a) and all(term.strip() in broad for term in a))
    hits["strict_match"] = ~hits["broad_only"]
    # `문맥 후보`는 원문 첫 600자에 청년이라는 단어가 보이는지를 참고하는 임시 표식일 뿐 확정 분류가 아님.
    hits["broad_with_youth_preview"] = hits["broad_only"] & hits.text_preview.fillna("").astype(str).str.contains(r"청년",regex=True)
    hits["검토구분"] = np.select(
        [hits.strict_match, hits.broad_with_youth_preview],
        ["정책명시_어휘일치", "광범위단독_청년문맥후보"],
        default="광범위단독_수동검토")
    save(hits, out / "검색근거_검토구분.csv")
    terms_count = {}
    for terms in term_lists:
        for v in set(str(term).strip() for term in terms if str(term).strip()):
            terms_count[v] = terms_count.get(v,0)+1
    terms_table = pd.DataFrame([{"검색표현":term,"일치청크수_중복허용":n,
             "광범위표현여부":term in broad} for term,n in sorted(terms_count.items(),key=lambda kv:-kv[1])])
    save(terms_table, out / "검색표현별_일치건수.csv")
    print(f"[2/5 검색 표현 분류] 직접표현 {hits.strict_match.sum():,}건 / 광범위 단독 {hits.broad_only.sum():,}건 / {duration(time.perf_counter()-t)}",flush=True)

    t = time.perf_counter()
    # 15번 분모가 제공하는 지역×월 모두 유지 (검색 0건도 누락하지 않음)
    grouped = hits.groupby(["assembly","month"],dropna=False).agg(
        direct=("strict_match","sum"), broad_only=("broad_only","sum"),
        youth_context_candidate=("broad_with_youth_preview","sum"), total=("chunk_id","count")
    ).reset_index()
    rm = region_month[["assembly","month","all_chunks","matched_chunks"]].merge(
        grouped,on=["assembly","month"],how="left",validate="one_to_one")
    for col in ["direct","broad_only","youth_context_candidate","total"]:
        rm[col] = rm[col].fillna(0).astype(int)
    if not (rm.total == rm.matched_chunks).all():
        raise ValueError("원본 15번 지역×월 집계와 16번 건수 불일치")
    if not ((rm.direct+rm.broad_only) == rm.total).all():
        raise ValueError("직접표현 + 광범위 단독 건수 불일치")
    rm["all_match_rate_pct"] = pct(rm.total,rm.all_chunks)
    rm["direct_match_rate_pct"] = pct(rm.direct,rm.all_chunks)
    rm["broad_only_rate_pct"] = pct(rm.broad_only,rm.all_chunks)
    rm = rm.sort_values(["assembly","month"])
    save(rm,out / "지역월별_정책언급추이.csv")
    month_summary = rm.groupby("month",as_index=False)[["all_chunks","total","direct","broad_only","youth_context_candidate"]].sum().sort_values("month")
    month_summary["all_match_rate_pct"] = pct(month_summary.total,month_summary.all_chunks)
    month_summary["direct_match_rate_pct"] = pct(month_summary.direct,month_summary.all_chunks)
    month_summary["month_date"] = pd.to_datetime(month_summary.month+"-01")
    month_summary["year"] = month_summary.month.str[:4].astype(int)
    month_summary["month_num"] = month_summary.month.str[5:7].astype(int)
    month_summary["direct_rate_mom_change_pp"] = month_summary.direct_match_rate_pct.diff().round(4)
    prev = month_summary[["year","month_num","direct","direct_match_rate_pct"]].copy()
    prev["year"] += 1
    prev = prev.rename(columns={"direct":"prev_year_direct", "direct_match_rate_pct":"prev_year_direct_rate_pct"})
    month_summary = month_summary.merge(prev,on=["year","month_num"],how="left",validate="one_to_one")
    month_summary["direct_rate_yoy_change_pp"] = (
        month_summary.direct_match_rate_pct-month_summary.prev_year_direct_rate_pct).round(4)
    month_summary["direct_count_yoy_change_pct"] = np.where(
        month_summary.prev_year_direct>0,
        ((month_summary.direct / month_summary.prev_year_direct)-1)*100,
        np.nan).round(2)
    save(month_summary.drop(columns=["month_date"]),out / "월별_정책언급추이.csv")
    print(f"[3/5 월별·지역별 통계] 월 {len(month_summary)}개 / 지역×월 {len(rm)}개 / {duration(time.perf_counter()-t)}",flush=True)

    t = time.perf_counter()
    region_summary=rm.groupby("assembly",as_index=False)[["all_chunks","total","direct","broad_only","youth_context_candidate"]].sum()
    region_summary["all_match_rate_pct"] = pct(region_summary.total,region_summary.all_chunks)
    region_summary["direct_match_rate_pct"] = pct(region_summary.direct,region_summary.all_chunks)
    first_all=hits.groupby("assembly").date.min().rename("first_all_lexical_seen")
    first_direct=hits[hits.strict_match].groupby("assembly").date.min().rename("first_direct_seen")
    region_summary=region_summary.join(first_all,on="assembly").join(first_direct,on="assembly")
    region_summary=region_summary.sort_values(["direct_match_rate_pct","direct"],ascending=False)
    save(region_summary,out / "지역별_정책언급순위.csv")
    observed=hits[hits.strict_match].groupby(["assembly","month"],as_index=False).agg(
        first_date=("date","min"), direct=("chunk_id","size"))
    if len(observed):
        first_month=observed.sort_values(["assembly","first_date"]).groupby("assembly",as_index=False).first()
        first_month=first_month.rename(columns={"month":"first_observed_month","first_date":"first_observed_date","direct":"first_month_chunks"})
        first_month=first_month.sort_values(["first_observed_date","assembly"])
    else:
        first_month=pd.DataFrame(columns=["assembly","first_observed_month","first_observed_date","first_month_chunks"])
    save(first_month,out / "지역별_최초관측_직접표현.csv")
    # 검색 0건인 지역·월도 포함해 지역 월별 히트맵 데이터 생성
    heat = rm.pivot(index="assembly",columns="month",values="direct_match_rate_pct").fillna(0)
    heat.index.name = "assembly"
    save(heat.reset_index(),out / "시각화_지역월별_비율행렬.csv")
    print(f"[4/5 지역별 관측·시각화 데이터] 지역 {len(region_summary)}개 / {duration(time.perf_counter()-t)}",flush=True)

    t = time.perf_counter()
    selected = hits[hits.strict_match].sort_values(["date","assembly","chunk_id"])
    # 발표·보고용 근거 후보: 날짜순 처음 3건/지역 및 월별 3건씩. 자동 관련성 검증 아님
    examples = pd.concat([
        selected.groupby("assembly",group_keys=False).head(min_examples),
        selected.groupby("month",group_keys=False).head(min_examples)
    ],ignore_index=True).drop_duplicates("chunk_id") if not selected.empty else selected.copy()
    save(examples,out / "시점별_근거후보.csv")
    info={
        "started_at":stamp(),"input_dir":str(input_dir),"query":report.get("query"),
        "lexical_terms":search_terms,"broad_terms":broad_terms,
        "total_corpus_chunks":EXPECTED_CORPUS,"total_lexical_matches":int(len(hits)),
        "direct_expression_matches":int(hits.strict_match.sum()),
        "broad_only_matches":int(hits.broad_only.sum()),
        "broad_youth_context_preview_candidates":int(hits.broad_with_youth_preview.sum()),
        "first_direct_expression_observed_date":selected.date.min() if len(selected) else None,
        "regions":len(region_summary),"months":len(month_summary),
        "limitations":[
            "전체 관련 정책 발언의 수가 아니라 지정 어휘에 일치한 청크 수",
            "직접표현으로 분류된 경우도 정책 관련성은 개별 원문 검증 필요",
            "텍스트 미리보기는 잘린 문자열일 수 있어 문맥 분류의 근거로 단정 불가",
            "2024~2025 보유 코퍼스에서의 최초 관측; 정책 실제 최초 발생 아님",
            "회의록 청크는 발언 수, 회의 건수, 정책 채택 수와 다름",
            "지역별 최초 관측 순서는 지역 간 인과적 확산이나 전파의 증거가 아님",
            "전년 동월 비교는 관측 발언량과 수집량 차이·회기 요인의 영향을 받음"
        ],"elapsed":duration(time.perf_counter()-begin),"finished_at":stamp()
    }
    (out/"분석_요약.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8-sig")
    print(f"[5/5 근거 후보·요약] {len(examples):,}건 / {duration(time.perf_counter()-t)}",flush=True)
    print(f"=== 분석 완료: {stamp()} / {duration(time.perf_counter()-begin)} ===",flush=True)
    print(f"입력 보존 / 생성 위치: {out}",flush=True)
    return info


def main():
    parser=argparse.ArgumentParser(description="15번 정책 검색 결과의 지역·시간 변화 분석")
    parser.add_argument("--input-dir",default="",help="15번 특정 실행 결과 폴더 (기본: 최신 성공 실행 폴더)")
    parser.add_argument("--broad-terms",default="취업 지원",help="쉼표로 구분한 광범위 검색 표현; 빈 문자열이면 분리하지 않음")
    parser.add_argument("--examples",type=int,default=3,help="월/지역별 날짜순 근거 후보 수")
    args=parser.parse_args()
    if args.examples<1:parser.error("--examples는 1 이상")
    terms=list(dict.fromkeys(v.strip() for v in args.broad_terms.split(",") if v.strip()))
    build(resolve_input(args.input_dir),terms,args.examples)


if __name__ == "__main__":
    main()

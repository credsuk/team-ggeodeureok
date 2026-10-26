# -*- coding: utf-8 -*-
"""19. 지방의회 정책 내용 유사성·지역별 후속 논의 후보 탐색.

개인 프로젝트 전용: C:\\mini_pj / mini conda
사용:
  python 19_policy_similarity_flow.py --input-dir "C:\\mini_pj\\results\\policy_issue_search\\20261025_000801_청년_일자리"
  python 19_policy_similarity_flow.py --input-dir "..." --limit 120 --threshold 0.78

입력: 16단계 검색근거_검토구분.csv + 12단계 원본 배치 임베딩/.csv(읽기 전용).
출력: <검색폴더>/similarity_flow/
중요: 유사도 및 후속 관측은 전파 증명이 아니며, 명시적 타 지역 인용도 수작업 검증 전에는 단서일 뿐.
      텍스트는 600자 미리보기여서 실제 정책 동일성·인과관계 최종 판정 불가.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(r"C:\mini_pj")
SEARCH_ROOT = ROOT / "results" / "policy_issue_search"
EMBED_DIR = ROOT / "data" / "processed" / "embeddings_by_region"
EXPECTED_DIM = 1024

# 정책 수단·행위: 공통 불용어보다 정책의 실행 내용을 비교하기 위한 보수적 단서.
# 두 문서의 단어가 겹쳐도 동일 사업 확정은 절대 아님.
MECHANISMS = {
    "인턴십": r"인턴십|인턴제|청년인턴|현장실습",
    "채용연계": r"채용연계|채용 연계|기업연계|기업 연계|채용약정",
    "직업훈련": r"직업훈련|직무교육|직무훈련|취업교육|기술교육",
    "창업지원": r"창업지원|창업 지원|창업보육|창업공간|스타트업|창업센터",
    "주거지원": r"주거지원|주거 지원|월세지원|월세 지원|임대주택",
    "취업알선": r"취업알선|취업 알선|일자리매칭|일자리 매칭|취업상담|취업 상담",
    "고용보조": r"고용장려금|고용보조|인건비지원|인건비 지원|고용지원금",
    "일자리센터": r"일자리센터|일자리 센터|취업지원센터|취업 지원센터",
    "지역기업": r"지역기업|지역 기업|중소기업|중소 기업|기업유치|기업 유치",
    "정착지원": r"정착지원|정착 지원|지역정착|지역 정착",
    "참여수당": r"참여수당|면접수당|면접 수당|구직수당|구직 수당",
    "자격취득": r"자격증|자격취득|자격 취득|시험응시료",
    "공공일자리": r"공공일자리|공공 일자리|공공근로|공공 근로",
    "멘토링": r"멘토링|멘토링사업|컨설팅|진로상담",
}
MECH_REGEX = {k: re.compile(v) for k, v in MECHANISMS.items()}
# 탐색용 문장 단서: 맥락/인용 대상 지역까지 확인해야 하므로 자동 확정 불가.
TRANSFER_CUES = re.compile(r"벤치마킹|우수사례|타[\s·]?시[\s·]?도|타[\s·]?지자체|다른[\s]+지역|사례를[\s]+참고|사례를[\s]+도입|모범사례|선진사례")


def log(message):
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def dt(seconds):
    return f"{seconds:.2f}초 / {seconds/60:.2f}분 / {seconds/3600:.3f}시간"


def load_csv(path, **kwargs):
    if not path.is_file():
        raise FileNotFoundError(f"필수 파일 없음: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def resolve_input(value):
    if value:
        path = Path(value)
    else:
        if not SEARCH_ROOT.exists():
            raise FileNotFoundError(SEARCH_ROOT)
        candidates = sorted([p for p in SEARCH_ROOT.iterdir() if p.is_dir() and
                             (p / "trend_analysis" / "검색근거_검토구분.csv").is_file()])
        if not candidates:
            raise FileNotFoundError("16번 검색근거_검토구분.csv를 갖춘 폴더 없음")
        path = candidates[-1]
    if not (path / "trend_analysis" / "검색근거_검토구분.csv").is_file():
        raise FileNotFoundError(f"16번 결과 없는 폴더: {path}")
    return path


def mechanisms(text):
    text = str(text)
    return frozenset(k for k, rx in MECH_REGEX.items() if rx.search(text))


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def get_candidates(path):
    df = load_csv(path, dtype={"chunk_id": str, "date": str, "source_id": str})
    expected = {"chunk_id", "date", "assembly", "source_id", "meeting_title", "text_preview", "strict_match"}
    if not expected.issubset(df.columns):
        raise ValueError(f"필수 컬럼 누락: {expected - set(df.columns)}")
    strict = df["strict_match"].astype(str).str.strip().str.lower().eq("true")
    df = df.loc[strict].copy()
    if df.chunk_id.isna().any() or df.chunk_id.duplicated().any():
        raise ValueError("직접표현 청크 ID 누락/중복")
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce", format="%Y-%m-%d")
    if df["date_parsed"].isna().any():
        raise ValueError("날짜 해석 불가능한 데이터 존재")
    df["text_preview"] = df["text_preview"].fillna("").map(normalize_text)
    df["mechanism_tags"] = df["text_preview"].map(mechanisms)
    df["transfer_hint"] = df["text_preview"].str.contains(TRANSFER_CUES, regex=True)
    df["is_substantive"] = (df["text_preview"].str.len() >= 80) & df["mechanism_tags"].map(bool)
    return df.reset_index(drop=True)


def load_existing_vectors(target_ids: set[str]):
    """npy 옆 metadata.csv 순서가 맞는 경우에만 벡터 선택. 새 임베딩 X."""
    batches = sorted(EMBED_DIR.rglob("batch_*.npy"))
    if not batches:
        raise FileNotFoundError(f"임베딩 배치 없음: {EMBED_DIR}")
    vectors = {}
    touched = Counter()
    started = time.perf_counter()
    for i, npy in enumerate(batches, 1):
        meta = npy.with_name(npy.stem + "_metadata.csv")
        if not meta.is_file():
            raise FileNotFoundError(meta)
        md = load_csv(meta, usecols=["chunk_id"], dtype={"chunk_id": str})
        arr = np.load(npy, mmap_mode="r", allow_pickle=False)
        if arr.ndim != 2 or arr.shape[1] != EXPECTED_DIM or len(arr) != len(md):
            raise ValueError(f"벡터-메타데이터 길이/차원 오류: {npy}, {arr.shape}, {len(md)}")
        local = md["chunk_id"].isin(target_ids)
        idx = np.flatnonzero(local.to_numpy())
        for j in idx:
            cid = str(md.iloc[int(j)]["chunk_id"])
            if cid in vectors:
                raise ValueError(f"중복 chunk_id가 서로 다른 배치에 존재: {cid}")
            vec = np.asarray(arr[int(j)], dtype=np.float32)
            if not np.isfinite(vec).all():
                raise ValueError(f"NaN/Inf 벡터: {cid}")
            vectors[cid] = vec
        touched["npy_files"] += 1
        if i % 20 == 0 or i == len(batches):
            log(f"[기존 임베딩 연결 {i}/{len(batches)}] {len(vectors):,}/{len(target_ids):,}건 / {dt(time.perf_counter()-started)}")
    missing = target_ids.difference(vectors)
    if missing:
        raise ValueError(f"기존 임베딩에 없는 청크 {len(missing):,}개. 예: {list(sorted(missing))[:3]}")
    return vectors, touched["npy_files"]


def compare(df, vectors, args):
    """교차 지역/후속 날짜를 먼저 제한하고, 의미 유사도 조건의 병목을 진단한다."""
    n = len(df)
    x = np.stack([vectors[cid] for cid in df["chunk_id"]]).astype(np.float32)
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms <= 1e-9):
        raise ValueError("0벡터 포함")
    x /= norms[:, None]
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2,
                            max_features=60000, sublinear_tf=True)
    t = tfidf.fit_transform(df["text_preview"])
    eligible = df["is_substantive"].to_numpy(dtype=bool)
    dates = df["date_parsed"].to_numpy(dtype="datetime64[D]")
    regions = df["assembly"].astype(str).to_numpy()
    ids = df["chunk_id"].astype(str).to_numpy()
    source_ids = df["source_id"].fillna("").astype(str).to_numpy()
    tags = df["mechanism_tags"].tolist()
    texts = df["text_preview"].tolist()
    matches = df["matched_terms"].fillna("").astype(str).tolist() if "matched_terms" in df else [""]*n
    # 동일한 쌍은 '더 이른 날짜 → 더 늦은 날짜' 방향으로 한 번만 검사한다.
    # np.argpartition 전에 지역/날짜/수단 조건을 적용한다.
    pairs = []
    near_misses = []
    stage = Counter()
    total_valid_time_region = 0
    candidate_pool_sizes = []
    started = time.perf_counter()
    batch_size = args.block_size
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        scores = x[start:end] @ x.T
        for offset, i in enumerate(range(start, end)):
            if not eligible[i]:
                continue
            stage["substantive_anchor"] += 1
            lag = (dates - dates[i]).astype("timedelta64[D]").astype(np.int32)
            # 한 방향만: anchor=i 는 '앞선 회의', 비교대상 j 는 '나중 회의'
            base_mask = (eligible & (regions != regions[i]) &
                         (lag >= args.min_lag_days) & (lag <= args.max_lag_days))
            base_mask[i] = False
            if source_ids[i]:
                base_mask &= (source_ids != source_ids[i])
            js_all = np.flatnonzero(base_mask)
            if len(js_all) == 0:
                continue
            stage["anchors_with_cross_region_later"] += 1
            total_valid_time_region += len(js_all)
            # 정책 수단을 검사하기 전 의미 유사도 최고 점수를 진단용으로 보관.
            if args.diagnostic_top > 0:
                row_all = scores[offset, js_all]
                top_m = min(2, len(js_all))
                best_i = np.argpartition(row_all, -top_m)[-top_m:]
                for ix in best_i:
                    j0 = int(js_all[ix])
                    near_misses.append({
                        "earlier_chunk_id": ids[i], "earlier_assembly": regions[i],
                        "earlier_date": str(dates[i]), "earlier_text_preview": texts[i],
                        "later_chunk_id": ids[j0], "later_assembly": regions[j0],
                        "later_date": str(dates[j0]), "later_text_preview": texts[j0],
                        "embedding_cosine": round(float(scores[offset,j0]),5),
                        "shared_policy_mechanisms": "|".join(sorted(tags[i].intersection(tags[j0]))),
                        "reason": "조건 적용 전 의미유사도 상위(최종 후보/전파 판정 아님)"
                    })
            # 지역·날짜를 먼저 검증한 다음 공통 정책 수단이 있는 이웃만 선택.
            common_js = [int(j) for j in js_all if tags[i].intersection(tags[int(j)])]
            if not common_js:
                continue
            stage["anchors_with_common_mechanism"] += 1
            stage["time_region_common_mechanism_pairs"] += len(common_js)
            js = np.asarray(common_js, dtype=np.int32)
            candidate_pool_sizes.append(len(js))
            row = scores[offset, js]
            # 진단을 위해 '최고 35개 안에 있는가'가 아니라 조건 충족 후보 전체에서 순위.
            k = min(args.neighbors, len(js))
            inds = np.argpartition(row, -k)[-k:]
            inds = inds[np.argsort(row[inds])[::-1]]
            accepted = 0
            for ix in inds:
                j = int(js[ix]); sim = float(row[ix])
                if sim < args.threshold:
                    stage["failed_embedding_threshold"] += 1
                    continue
                stage["passed_embedding_threshold"] += 1
                if texts[i] == texts[j]:
                    stage["identical_preview_removed"] += 1
                    continue
                char_sim = float(t[i].multiply(t[j]).sum())
                if char_sim < args.text_threshold:
                    stage["failed_text_threshold"] += 1
                    continue
                stage["passed_text_threshold"] += 1
                shared = tags[i].intersection(tags[j])
                pairs.append({
                    "earlier_chunk_id": ids[i], "earlier_date": str(dates[i]),
                    "earlier_assembly": regions[i], "earlier_meeting_title": str(df.iloc[i]["meeting_title"]),
                    "earlier_source_id": source_ids[i], "earlier_matched_terms": matches[i],
                    "earlier_text_preview": texts[i],
                    "later_chunk_id": ids[j], "later_date": str(dates[j]),
                    "later_assembly": regions[j], "later_meeting_title": str(df.iloc[j]["meeting_title"]),
                    "later_source_id": source_ids[j], "later_matched_terms": matches[j],
                    "later_text_preview": texts[j],
                    "lag_days": int(lag[j]), "embedding_cosine": round(sim,5),
                    "text_ngram_cosine": round(char_sim,5),
                    "shared_policy_mechanisms": "|".join(sorted(shared)),
                    "transfer_cue_in_later_preview": bool(df.iloc[j]["transfer_hint"]),
                    "assessment": "타 지역 유사 정책 후속논의 후보(원문 미검증)",
                    "diffusion_confirmed": False,
                    "review_priority": round(sim * 0.70 + char_sim * 0.20 + min(len(shared), 3) * .0333, 5)
                })
                accepted += 1
                if accepted >= args.per_anchor:
                    break
        log(f"[지역 간 후보 탐색 {end:,}/{n:,}] 누적 후보 {len(pairs):,} / {dt(time.perf_counter()-started)}")
    stage["time_region_eligible_pairs"] = total_valid_time_region
    if candidate_pool_sizes:
        stage["median_common_mechanism_pool"] = float(np.median(candidate_pool_sizes))
    diagnostics = pd.DataFrame(near_misses)
    if not diagnostics.empty:
        diagnostics = (diagnostics.sort_values("embedding_cosine", ascending=False)
                       .drop_duplicates(["earlier_chunk_id","later_chunk_id"])
                       .head(args.diagnostic_top))
    # pairs are unique by construction (date strictly increases, each i visited once).
    return pairs, stage, diagnostics

def save_df(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"저장 {path.name}: {len(df):,}행")


def run(args):
    total_start = time.perf_counter()
    input_dir = resolve_input(args.input_dir)
    # output = input_dir / "similarity_flow_v2"
    output = input_dir / f"similarity_flow_v2_sim{args.threshold:.2f}_txt{args.text_threshold:.2f}"
    output.mkdir(exist_ok=True)
    log(f"=== 19 지역 간 동일·유사 정책 후속 논의 탐색 시작 ===")
    log(f"입력: {input_dir}")
    log(f"출력: {output}")
    t0 = time.perf_counter()
    df = get_candidates(input_dir / "trend_analysis" / "검색근거_검토구분.csv")
    if len(df) == 0:
        raise RuntimeError("직접 표현 일치 청크가 없습니다")
    log(f"[1/4 근거 준비] 직접 표현 일치 {len(df):,}건 / 정책 수단 사전 탐지 {int(df.is_substantive.sum()):,}건 / {dt(time.perf_counter()-t0)}")
    if int(df.is_substantive.sum()) < 2:
        raise RuntimeError("정책 수단을 식별한 발언이 너무 적습니다. MECHANISMS 검토 필요")
    if len(df) > args.max_input:
        raise ValueError(f"직접표현 {len(df):,}건 > 안전 상한 {args.max_input:,}건. --max-input 조정 필요")
    t0 = time.perf_counter()
    vectors, file_count = load_existing_vectors(set(df.chunk_id))
    log(f"[2/4 기존 벡터 연결] {len(vectors):,}건 / {file_count}개 배치 / {dt(time.perf_counter()-t0)}")
    t0 = time.perf_counter()
    pairs, stage, diagnostics = compare(df, vectors, args)
    log(f"[3/4 후보 탐색 완료] 후보 {len(pairs):,} / {dt(time.perf_counter()-t0)}")
    log("[탈락 진단] " + json.dumps(dict(stage), ensure_ascii=False))
    save_df(diagnostics, output / "진단_지역날짜기준_최상위유사쌍.csv")
    # UI 최적화: 순위 상위 순으로 추출하되 동일 두 지역의 중복 표시 상한.
    candidates = pd.DataFrame(pairs)
    if candidates.empty:
        candidates = pd.DataFrame(columns=[
            "earlier_chunk_id","earlier_date","earlier_assembly","earlier_meeting_title","earlier_source_id",
            "earlier_matched_terms","earlier_text_preview","later_chunk_id","later_date","later_assembly",
            "later_meeting_title","later_source_id","later_matched_terms","later_text_preview","lag_days",
            "embedding_cosine","text_ngram_cosine","shared_policy_mechanisms","transfer_cue_in_later_preview",
            "assessment","diffusion_confirmed","review_priority"])
    else:
        candidates = candidates.sort_values("review_priority", ascending=False).reset_index(drop=True)
    save_df(candidates, output / "지역간_유사정책_전체후보.csv")
    filtered = []
    by_edge = Counter()
    for rec in candidates.to_dict("records"):
        edge = (rec["earlier_assembly"], rec["later_assembly"])
        if by_edge[edge] >= args.per_route:
            continue
        filtered.append(rec)
        by_edge[edge] += 1
        if len(filtered) >= args.limit:
            break
    display = pd.DataFrame(filtered, columns=candidates.columns)
    save_df(display, output / "웹_지역간_유사논의후보.csv")
    # 지도에서 실선으로 '확정 전파'를 그려선 안됨. 집계 후보만 점선/별도 범례로 보여줄 것.
    if not display.empty:
        routes = (display.groupby(["earlier_assembly", "later_assembly"], as_index=False)
                  .agg(candidate_pairs=("earlier_chunk_id", "size"),
                       max_embedding_cosine=("embedding_cosine", "max"),
                       min_earlier_date=("earlier_date", "min"),
                       max_later_date=("later_date", "max"),
                       transfer_cue_candidates=("transfer_cue_in_later_preview", "sum")))
        routes["relationship_status"] = "유사논의 후보(전파 미확인)"
    else:
        routes = pd.DataFrame(columns=["earlier_assembly", "later_assembly", "candidate_pairs",
                                       "max_embedding_cosine", "min_earlier_date", "max_later_date",
                                       "transfer_cue_candidates", "relationship_status"])
    save_df(routes, output / "웹_지역간_후보경로.csv")
    clues = candidates[candidates.transfer_cue_in_later_preview.astype(str).str.lower().eq("true")].copy()
    clues["manual_check"] = "후속 발언에 타지역·벤치마킹 일반표현 탐지. 실제 참조 지역명 및 정책 동일성 원문 확인 필수"
    save_df(clues, output / "전파_언급단서_수동검토.csv")
    overview = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir), "full_matched_chunks": len(df),
        "substantive_mechanism_detected_chunks": int(df.is_substantive.sum()),
        "existing_embeddings_matched": len(vectors), "embedding_dimension": EXPECTED_DIM,
        "all_cross_region_candidate_pairs": len(candidates),
        "web_candidate_pairs": len(display), "web_candidate_routes": len(routes),
        "transfer_language_clues_need_manual_review": len(clues),
        "confirmed_policy_diffusion_pairs": 0,
        "filter_diagnostics": dict(stage),
        "diagnostic_rows": len(diagnostics),
        "parameters": {k: v for k, v in vars(args).items() if k != "input_dir"},
        "limitations": [
            "동일 키워드 언급은 동일 정책 내용 또는 정책 전파의 증명이 아님",
            "유사도+공통 정책수단을 갖는 후속 발언은 검증 전 '후보'이며, 인과 전파가 아님",
            "청크 텍스트는 최대 600자 미리보기여서 정책 내용 분류와 실제 인용 여부가 제한됨",
            "지역·후속날짜·공통 정책수단을 먼저 적용한 후 의미 유사도가 높은 이웃만 검사하므로 모든 문장쌍을 보장하지 않음",
            "정책 수단 사전은 청년 일자리 관련 시범용이며 다른 이슈에는 사전을 확장해야 함",
            "원문 발언과 출처·참조 의회를 사람이 확인하기 전에는 '확정 전파'로 분류할 수 없음",
            "원자료 기간은 2024~2025년이며 그 이전 최초 정책 도입은 평가 불가"
        ],
        "duration": dt(time.perf_counter() - total_start)
    }
    (output / "분석_요약.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    log(f"[4/4 저장 완료] 후보 {len(candidates):,}쌍 / 웹 후보 {len(display):,}쌍 / 지도 후보 경로 {len(routes):,}개 / {overview['duration']}")
    log("주의: 실제 전파 확정 0건(미검증 상태). 원문·명시적 인용 수작업 검증 후에만 판정 가능")
    log(f"=== 종료 / {output} ===")


def main():
    ap = argparse.ArgumentParser(description="기존 임베딩 기반 타 지방의회 유사 정책 후속 논의 후보 탐색")
    ap.add_argument("--input-dir", default="", help="15/16단계 결과 폴더(생략 시 최신)")
    ap.add_argument("--diagnostic-top", type=int, default=100, help="최종 필터 통과 여부와 관계없이 비교할 교차지역 상위 유사쌍 저장 건수")
    ap.add_argument("--threshold", type=float, default=0.78, help="임베딩 코사인 최소값(기본 .78)")
    ap.add_argument("--text-threshold", type=float, default=0.12, help="문자 n-gram 보조 유사도 최소값")
    ap.add_argument("--neighbors", type=int, default=35, help="청크별 가까운 벡터 이웃 수")
    ap.add_argument("--per-anchor", type=int, default=3, help="개별 청크당 최대 후보")
    ap.add_argument("--max-lag-days", type=int, default=730, help="최대 후속 시간 간격(일)")
    ap.add_argument("--min-lag-days", type=int, default=1, help="동일 날짜 제외, 최소 날짜 간격")
    ap.add_argument("--limit", type=int, default=120, help="웹 화면 후보 총개수")
    ap.add_argument("--per-route", type=int, default=5, help="같은 지역쌍 출력 상한")
    ap.add_argument("--block-size", type=int, default=128, help="코사인 배치 크기")
    ap.add_argument("--max-input", type=int, default=12000, help="최대 후보 임베딩 청크 수 안전상한")
    args = ap.parse_args()
    if not (0 < args.threshold <= 1 and 0 <= args.text_threshold <= 1):
        ap.error("유사도 임계값은 0~1 범위")
    if min(args.neighbors, args.per_anchor, args.limit, args.per_route, args.block_size) < 1 or args.diagnostic_top < 0:
        ap.error("탐색 크기 관련 옵션은 1 이상")
    run(args)


if __name__ == "__main__":
    main()

"""10단계: 개선 POLICY K=100 군집의 c-TF-IDF 토픽 키워드 추출.

실행: python 10_extract_policy_topics.py
필요: pip install chromadb pandas scikit-learn scipy
선택: pip install kiwipiepy  (한국어 형태소 기반 명사 추출; 없으면 정규식 사용)

기존 ChromaDB는 조회만 수행합니다. 결과는 clustering/results/policy_topics에 저장.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
from scipy.sparse import diags
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(r"C:\mini_team\scripts\clustering\results")
DB_PATH = Path(r"C:\mini_team\_db\chroma_snowflake_regions")
COLLECTION_NAME = "team_regions_6a0a006e92d1e93e"
FILTER_CSV = ROOT / "policy_refined" / "policy_filter_68000.csv"
ASSIGN_CSV = ROOT / "policy_kmeans" / "policy_sample_clusters_k100.csv"
OUTPUT_DIR = ROOT / "policy_topics"
CACHE_DIR = OUTPUT_DIR / "document_cache"
FETCH_BATCH = 100  # 대용량 Chroma 연결에서 안전한 요청 크기
TOP_N = 20
USE_KIWI_IF_AVAILABLE = True
MIN_DF = 2
MAX_DF = 0.90

# 회의 운영 상투어, 형태소 분석 시에도 제거할 일반어 목록 (필요하면 보완)
STOPWORDS = {
    "감사합니다", "수고하셨습니다", "말씀", "말씀을", "말씀드리겠습니다", "위원", "의원", "위원님",
    "위원장", "의장", "질의", "답변", "보고", "바랍니다", "입니다", "있습니다", "있는데",
    "이렇게", "그렇게", "그런", "그러면", "저희", "제가", "우리가", "이것", "그것",
    "내용", "부분", "관련", "대한", "통해서", "그리고", "하지만", "있는", "없는", "합니다",
    "지금", "이번", "오늘", "다음", "우리", "정말", "해서", "경우", "정도", "생각",
    "것으로", "때문에", "여러분", "주시기", "하도록", "하겠습니다", "대해서",
}


def stamp(msg: str, start: float) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg} | 경과 {time.perf_counter() - start:.1f}초", flush=True)


def choose_cluster_column(df: pd.DataFrame) -> str:
    options = ["cluster_id", "cluster", "label", "cluster_label", "pred_cluster"]
    match = [x for x in options if x in df.columns]
    if match:
        return match[0]
    raise ValueError(f"군집 ID 컬럼을 찾을 수 없습니다. CSV 컬럼: {df.columns.tolist()}")


def load_policy_assignments() -> pd.DataFrame:
    if not FILTER_CSV.is_file() or not ASSIGN_CSV.is_file():
        raise FileNotFoundError(f"입력 확인: {FILTER_CSV}\n{ASSIGN_CSV}")
    filt = pd.read_csv(FILTER_CSV, usecols=["chunk_id", "refined_label"], dtype=str)
    assign = pd.read_csv(ASSIGN_CSV, dtype={"chunk_id": str})
    cluster_col = choose_cluster_column(assign)
    if "chunk_id" not in assign:
        raise ValueError("군집 할당 파일에 chunk_id가 없습니다.")
    assign = assign[["chunk_id", cluster_col]].rename(columns={cluster_col: "cluster_id"})
    if assign.chunk_id.duplicated().any():
        raise ValueError("군집 할당 CSV에 중복 chunk_id 존재")
    df = assign.merge(filt, on="chunk_id", how="inner", validate="one_to_one")
    df = df.loc[df.refined_label == "POLICY", ["chunk_id", "cluster_id"]].copy()
    df["cluster_id"] = pd.to_numeric(df.cluster_id, errors="raise").astype(int)
    if df.empty:
        raise ValueError("POLICY 데이터가 없습니다.")
    print(f"POLICY 할당: {len(df):,}개 / 군집: {df.cluster_id.nunique()}개")
    return df.sort_values(["cluster_id", "chunk_id"]).reset_index(drop=True)


def fetch_documents(df: pd.DataFrame, start: float) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "policy_docs_by_id.csv"
    needed = set(df.chunk_id)
    # 원문 캐시가 있다면 불필요한 반복 조회 없이 재사용
    if cache_path.is_file():
        cached = pd.read_csv(cache_path, dtype=str).fillna("")
        if {"chunk_id", "document"}.issubset(cached.columns):
            cached = cached.drop_duplicates("chunk_id", keep="last")
            present = set(cached.loc[cached.document.str.len() > 0, "chunk_id"])
            missing = needed - present
            stamp(f"캐시 로드 {len(present & needed):,}개 / 미조회 {len(missing):,}개", start)
        else:
            cached = pd.DataFrame(columns=["chunk_id", "document"])
            missing = needed
    else:
        cached = pd.DataFrame(columns=["chunk_id", "document"])
        missing = needed

    if missing:
        if not (DB_PATH / "chroma.sqlite3").exists():
            raise FileNotFoundError(DB_PATH / "chroma.sqlite3")
        client = chromadb.PersistentClient(path=str(DB_PATH))
        collection = client.get_collection(COLLECTION_NAME)
        ids = sorted(missing)
        rows = []
        for offset in range(0, len(ids), FETCH_BATCH):
            part = ids[offset:offset + FETCH_BATCH]
            res = collection.get(ids=part, include=["documents"])
            for rid, doc in zip(res["ids"], res.get("documents") or []):
                rows.append({"chunk_id": rid, "document": doc or ""})
            if (offset // FETCH_BATCH + 1) % 10 == 0 or offset + len(part) == len(ids):
                stamp(f"원문 조회 {min(offset + len(part), len(ids)):,}/{len(ids):,}", start)
                # 매 1000건마다 체크포인트 저장 (재시작 안전)
                incoming = pd.DataFrame(rows)
                cached = pd.concat([cached, incoming], ignore_index=True).drop_duplicates("chunk_id", keep="last")
                cached.to_csv(cache_path, index=False, encoding="utf-8-sig")
                rows.clear()
    merged = df.merge(cached[["chunk_id", "document"]], on="chunk_id", how="left", validate="one_to_one")
    absent = merged.document.isna() | (merged.document.fillna("").str.strip() == "")
    if absent.any():
        print(f"주의: 본문 누락 {int(absent.sum())}개 (토픽 분석 제외)")
    return merged.loc[~absent].copy()


def build_tokenizer():
    if USE_KIWI_IF_AVAILABLE:
        try:
            from kiwipiepy import Kiwi
            kiwi = Kiwi()
            print("토큰화: Kiwi 명사(NNG/NNP) + 영문 명사 토큰")
            def tokenize(text):
                terms = [t.form.lower() for t in kiwi.tokenize(text) if t.tag in {"NNG", "NNP", "SL"}]
                return [w for w in terms if len(w) >= 2 and w not in STOPWORDS]
            return tokenize, "kiwi_nouns"
        except ImportError:
            print("Kiwi 미설치: 정규식 기반 공백 토큰화로 진행 (정확도 제한)")
    pattern = re.compile(r"[가-힣]{2,}|[A-Za-z]{2,}(?:-[A-Za-z]+)*")
    def tokenize(text):
        return [w for w in pattern.findall(text.lower()) if w not in STOPWORDS]
    return tokenize, "regex_words"


def calculate_topics(df: pd.DataFrame, start: float):
    tokenizer, method = build_tokenizer()
    # c-TF-IDF는 각 군집의 문서를 하나의 클래스 문서로 합쳐서 계산
    grouped = df.groupby("cluster_id", sort=True)["document"].agg(" ".join)
    vectorizer = CountVectorizer(
        tokenizer=tokenizer, token_pattern=None, lowercase=False,
        ngram_range=(1, 2), min_df=MIN_DF, max_df=MAX_DF,
        max_features=60000, dtype=np.int64,
    )
    counts = vectorizer.fit_transform(grouped.to_list()).tocsr().astype(np.float64)
    # 한 군집 = 한 문서이므로 min_df/max_df는 군집 출현 빈도를 의미
    # c-TF-IDF: tf(t,c) = count(t,c) / sum_t count(t,c)
    # idf(t) = log(1 + A / freq(t)), A = 군집별 총 단어수의 평균
    totals = np.asarray(counts.sum(axis=1)).ravel()
    normalized = diags(1.0 / np.maximum(totals, 1)).dot(counts)
    frequencies = np.asarray(counts.sum(axis=0)).ravel()
    avg_words = float(totals.mean())
    idf = np.log1p(avg_words / np.maximum(frequencies, 1))
    ctfidf = normalized.multiply(idf).tocsr()
    terms = np.array(vectorizer.get_feature_names_out())
    counts_per_cluster = df.groupby("cluster_id").size().to_dict()
    topic_rows = []
    term_rows = []
    for row_idx, cluster in enumerate(grouped.index):
        r = ctfidf.getrow(row_idx)
        order = np.argsort(r.data)[::-1][:TOP_N]
        top_terms = [(terms[r.indices[j]], float(r.data[j])) for j in order]
        topic_rows.append({
            "cluster_id": int(cluster), "policy_sample_size": int(counts_per_cluster[cluster]),
            "tokenizer": method, "topic_name_candidate": " / ".join(x[0] for x in top_terms[:5]),
            "top_keywords": ", ".join(x[0] for x in top_terms),
            "top_scores": ", ".join(f"{x[1]:.5f}" for x in top_terms),
            "human_topic": "", "topic_quality": "", "review_note": "",
        })
        for rank, (term, score) in enumerate(top_terms, 1):
            term_rows.append({"cluster_id": int(cluster), "rank": rank, "term": term, "ctfidf_score": round(score, 7)})
    stamp(f"c-TF-IDF 완료: 어휘 {len(terms):,}개 / 토픽 {len(topic_rows)}개", start)
    return pd.DataFrame(topic_rows), pd.DataFrame(term_rows), method


def main():
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp("10단계 c-TF-IDF 시작", start)
    assignments = load_policy_assignments()
    stamp("[1/4] POLICY 군집 ID 검증 완료", start)
    docs = fetch_documents(assignments, start)
    stamp(f"[2/4] 원문 확보 {len(docs):,}개", start)
    summary, terms, method = calculate_topics(docs, start)
    stamp("[3/4] 토픽 계산 완료", start)
    topic_path = OUTPUT_DIR / "policy_topics_k100.csv"
    keyword_path = OUTPUT_DIR / "policy_topic_keywords_k100.csv"
    summary.to_csv(topic_path, index=False, encoding="utf-8-sig")
    terms.to_csv(keyword_path, index=False, encoding="utf-8-sig")
    report = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "policy_ids": len(assignments), "documents_used": len(docs),
        "clusters": len(summary), "tokenizer": method,
        "total_sec": round(time.perf_counter() - start, 2),
        "warning": "키워드는 군집 설명 후보이며, 정책 주제의 정확성은 대표·무작위 발언으로 검증 필요",
    }
    (OUTPUT_DIR / "topic_extraction_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== 완료 ===")
    print("토픽 요약:", topic_path)
    print("키워드 상세:", keyword_path)
    print("분석 청크:", len(docs), "/", len(assignments))
    print("전체 소요:", report["total_sec"], "초")
    print("종료:", datetime.now())


if __name__ == "__main__":
    main()


# -*- coding: utf-8 -*-
"""
12단계: 전체 POLICY 청크 기반 c-TF-IDF 토픽 분석

입력:
  - 11단계 all_cluster_assignments.sqlite3
  - 원본 ChromaDB (읽기 전용 사용)

출력:
  - 전체 군집 토픽 요약
  - 군집별 Top-20 키워드
  - 지역·연도별 군집 분포
  - 5대 정책 이슈 관련 군집 후보
  - 실행 요약 JSON

특징:
  - POLICY 청크만 분석
  - 500개 단위 원문 조회
  - SQLite 원문 캐시를 통한 중단 후 재실행
  - Kiwi 형태소 분석
  - 메모리 절약형 군집별 단어 집계
"""

from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from math import log
import csv
import json
import re
import sqlite3
import time

import chromadb
from kiwipiepy import Kiwi


# ============================================================
# 0. 환경 설정
# ============================================================

ROOT = Path(r"C:\mini_team\scripts\clustering")
RESULTS = ROOT / "results"

ASSIGN_DB = (
    RESULTS
    / "all_cluster_assignment"
    / "all_cluster_assignments.sqlite3"
)

CHROMA_PATH = Path(
    r"C:\mini_team\_db\chroma_snowflake_regions"
)

COLLECTION_NAME = "team_regions_6a0a006e92d1e93e"

OUT = RESULTS / "all_policy_topics"
OUT.mkdir(parents=True, exist_ok=True)

CACHE_DB = OUT / "policy_text_cache.sqlite3"

BATCH_SIZE = 500
TOP_N = 20

ISSUES = {
    "지역소멸": [
        "지역소멸", "지방소멸", "인구감소",
        "인구유출", "정주", "인구",
    ],
    "청년 일자리": [
        "청년", "일자리", "고용",
        "취업", "창업", "구직",
    ],
    "저출생": [
        "저출생", "저출산", "출산",
        "출생", "난임", "양육",
    ],
    "고령화": [
        "고령화", "고령", "노인",
        "어르신", "돌봄", "요양",
    ],
    "기후위기": [
        "기후", "탄소", "탄소중립",
        "온실가스", "환경", "에너지",
    ],
}

STOPWORDS = {
    "위원", "위원장", "의원", "의회",
    "말씀", "질문", "답변", "생각",
    "경우", "부분", "관련", "사항",
    "내용", "오늘", "이번", "지금",
    "이것", "저것", "그것",
    "사실", "정도", "이유",
    "감사", "수고", "설명",
    "보고", "자료", "확인",
    "진행", "우리", "여러분",
    "때문", "필요", "문제",
    "부분", "해당", "대한",
    "현재", "작년", "내년",
    "하나", "가지",
}


def log_msg(message, start):
    elapsed = time.time() - start
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message} | 경과 {elapsed:.1f}초", flush=True)


def sql_ident(name):
    return '"' + name.replace('"', '""') + '"'


def detect_assignment_table(conn):
    """
    11단계 SQLite 스키마가 조금 달라도
    POLICY와 군집 번호가 있는 테이블을 찾아냄.
    """
    candidates = []

    tables = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()

    id_names = (
        "chunk_id", "id", "embedding_id"
    )
    label_names = (
        "refined_label", "policy_label",
        "label", "classification"
    )
    cluster_names = (
        "cluster_id", "cluster", "topic_id"
    )

    for (table,) in tables:
        cols = [
            row[1] for row in conn.execute(
                f"PRAGMA table_info({sql_ident(table)})"
            )
        ]

        def first_available(names):
            return next((c for c in names if c in cols), None)

        id_col = first_available(id_names)
        label_col = first_available(label_names)
        cluster_col = first_available(cluster_names)

        if id_col and label_col and cluster_col:
            candidates.append(
                (table, id_col, label_col, cluster_col)
            )

    if not candidates:
        details = {}
        for (table,) in tables:
            details[table] = [
                r[1] for r in conn.execute(
                    f"PRAGMA table_info({sql_ident(table)})"
                )
            ]
        raise RuntimeError(
            "군집 할당 테이블을 자동 탐지하지 못했습니다.\n"
            + json.dumps(details, ensure_ascii=False, indent=2)
        )

    # POLICY 레코드가 가장 많은 테이블 선택
    best = None
    for item in candidates:
        table, _, label_col, _ = item
        count = conn.execute(
            f"SELECT COUNT(*) FROM {sql_ident(table)} "
            f"WHERE {sql_ident(label_col)} = ?",
            ("POLICY",),
        ).fetchone()[0]

        if best is None or count > best[0]:
            best = (count, item)

    return best[1]


def create_cache(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS policy_docs (
            chunk_id TEXT PRIMARY KEY,
            cluster_id INTEGER NOT NULL,
            document TEXT NOT NULL,
            assembly TEXT,
            source_key TEXT,
            date TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_policy_cluster
        ON policy_docs(cluster_id)
    """)
    conn.commit()


def get_metadata_value(metadata, *keys):
    metadata = metadata or {}
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return str(value)
    return ""


def retrieve_documents(
    assignment_conn, cache_conn, collection,
    table, id_col, label_col, cluster_col, start
):
    log_msg("[1/5] 전체 POLICY 원문 조회 시작", start)

    cursor = assignment_conn.execute(
        f"SELECT {sql_ident(id_col)}, {sql_ident(cluster_col)} "
        f"FROM {sql_ident(table)} "
        f"WHERE {sql_ident(label_col)} = ? "
        f"AND {sql_ident(cluster_col)} IS NOT NULL",
        ("POLICY",)
    )

    total = 0
    inserted = 0
    missing = 0
    existing = 0

    while True:
        batch = cursor.fetchmany(BATCH_SIZE)
        if not batch:
            break

        total += len(batch)
        ids = [str(row[0]) for row in batch]

        # 재실행 시 이미 저장된 ID는 조회하지 않음
        placeholders = ",".join("?" for _ in ids)

        cached = {
            row[0] for row in cache_conn.execute(
                f"SELECT chunk_id FROM policy_docs "
                f"WHERE chunk_id IN ({placeholders})",
                ids
            )
        }

        remaining = [
            (str(chunk_id), int(cluster_id))
            for chunk_id, cluster_id in batch
            if str(chunk_id) not in cached
        ]

        existing += len(batch) - len(remaining)

        if remaining:
            request_ids = [r[0] for r in remaining]
            cluster_lookup = dict(remaining)

            result = collection.get(
                ids=request_ids,
                include=["documents", "metadatas"]
            )

            found_ids = result.get("ids") or []
            docs = result.get("documents") or []
            metas = result.get("metadatas") or []

            rows = []

            for i, chunk_id in enumerate(found_ids):
                document = docs[i] if i < len(docs) else ""
                metadata = metas[i] if i < len(metas) else {}

                if not document:
                    missing += 1
                    continue

                rows.append((
                    chunk_id,
                    cluster_lookup[chunk_id],
                    document,
                    get_metadata_value(
                        metadata, "assembly"
                    ),
                    get_metadata_value(
                        metadata, "source_key"
                    ),
                    get_metadata_value(
                        metadata, "date"
                    ),
                ))

            if rows:
                cache_conn.executemany("""
                    INSERT OR IGNORE INTO policy_docs
                    (chunk_id, cluster_id, document,
                     assembly, source_key, date)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, rows)
                cache_conn.commit()
                inserted += len(rows)

            # 반환되지 않은 ID도 누락에 포함
            missing += len(remaining) - len(found_ids)

        if total % 5000 == 0:
            log_msg(
                f"원문 확인 {total:,}개 | "
                f"신규 저장 {inserted:,}개 | "
                f"기존 캐시 {existing:,}개",
                start
            )

    count = cache_conn.execute(
        "SELECT COUNT(*) FROM policy_docs"
    ).fetchone()[0]

    log_msg(
        f"[1/5] 원문 캐시 완료: {count:,}개 "
        f"/ 조회 누락 {missing:,}개",
        start
    )

    return total, count, missing


def tokenize(text, kiwi):
    # 10단계와 같은 기본 원칙: Kiwi 명사 + 영문 토큰
    tokens = []

    for token in kiwi.tokenize(text):
        if token.tag in ("NNG", "NNP"):
            word = token.form.strip().lower()

            if len(word) < 2:
                continue

            if word in STOPWORDS:
                continue

            tokens.append(word)

    english = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)
    tokens.extend(w.lower() for w in english)

    return tokens


def calculate_ctfidf(cache_conn, start):
    log_msg("[2/5] Kiwi 토큰화 및 군집별 단어 집계", start)

    kiwi = Kiwi()

    cluster_counts = defaultdict(Counter)
    cluster_docs = Counter()
    cluster_lengths = Counter()
    cluster_token_totals = Counter()

    cursor = cache_conn.execute(
        "SELECT cluster_id, document "
        "FROM policy_docs"
    )

    count = 0

    while True:
        batch = cursor.fetchmany(500)

        if not batch:
            break

        for cluster_id, document in batch:
            tokens = tokenize(document or "", kiwi)

            cluster_docs[cluster_id] += 1
            cluster_counts[cluster_id].update(tokens)
            cluster_token_totals[cluster_id] += len(tokens)
            cluster_lengths[cluster_id] += len(document or "")
            count += 1

        if count % 10000 == 0:
            log_msg(f"토큰화 {count:,}개", start)

    log_msg("[2/5] 토큰화 완료", start)

    # 전체 군집의 단어 출현 빈도
    total_term_frequency = Counter()
    for word_counts in cluster_counts.values():
        total_term_frequency.update(word_counts)

    # BERTopic 계열의 class-based TF-IDF 계산 방식
    # 클래스 문서 평균 길이 + 단어 전체 빈도
    n_clusters = len(cluster_counts)

    if n_clusters == 0:
        raise RuntimeError("토픽을 계산할 POLICY 원문이 없습니다.")

    avg_class_length = (
        sum(cluster_token_totals.values()) / n_clusters
    )

    topic_rows = []
    keyword_rows = []

    for cluster_id in sorted(cluster_counts):
        counts = cluster_counts[cluster_id]
        total_tokens = cluster_token_totals[cluster_id]

        scores = {}

        for word, term_count in counts.items():
            tf = term_count / max(total_tokens, 1)

            idf = log(
                1 + avg_class_length
                / max(total_term_frequency[word], 1)
            )

            scores[word] = tf * idf

        top_keywords = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:TOP_N]

        for rank, (word, score) in enumerate(
            top_keywords, start=1
        ):
            keyword_rows.append({
                "cluster_id": cluster_id,
                "rank": rank,
                "keyword": word,
                "ctfidf_score": round(score, 8),
                "term_frequency": counts[word],
            })

        topic_rows.append({
            "cluster_id": cluster_id,
            "document_count": cluster_docs[cluster_id],
            "total_tokens": total_tokens,
            "mean_char_length": round(
                cluster_lengths[cluster_id]
                / max(cluster_docs[cluster_id], 1),
                2
            ),
            "top_keywords": ", ".join(
                word for word, _ in top_keywords
            ),
            "proposed_topic": "",
            "topic_review": "",
        })

    log_msg("[3/5] 전체 c-TF-IDF 계산 완료", start)

    return topic_rows, keyword_rows


def save_csv(path, rows):
    if not rows:
        return

    with open(
        path, "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def analyze_region_year(cache_conn, start):
    log_msg("[4/5] 지역·연도별 분포 분석", start)

    rows = cache_conn.execute("""
        SELECT cluster_id, assembly, source_key, date
        FROM policy_docs
    """)

    counts = Counter()
    no_date = 0

    for cluster_id, assembly, source_key, date in rows:
        assembly = assembly or source_key or "UNKNOWN"

        # 날짜 문자열이 YYYY 또는 YYYY-MM-DD 형태인 경우
        year = ""

        if date:
            match = re.search(r"(20\d{2})", date)
            if match:
                year = match.group(1)

        if not year and source_key:
            match = re.search(r"(20\d{2})", source_key)
            if match:
                year = match.group(1)

        if not year:
            no_date += 1
            year = "UNKNOWN"

        counts[(year, assembly, cluster_id)] += 1

    output = [
        {
            "year": year,
            "assembly": assembly,
            "cluster_id": cluster_id,
            "document_count": count
        }
        for (year, assembly, cluster_id), count
        in sorted(counts.items())
    ]

    return output, no_date


def link_policy_issues(keyword_rows):
    """
    5개 핵심 이슈와 관련된 군집 '후보' 추출.
    단순 키워드 매칭이므로 확정 분류는 아님.
    """
    by_cluster = defaultdict(list)

    for row in keyword_rows:
        by_cluster[row["cluster_id"]].append(row)

    results = []

    for issue, terms in ISSUES.items():
        for cluster_id, keywords in by_cluster.items():
            matched = []

            score = 0.0

            for row in keywords:
                keyword = row["keyword"]

                if any(
                    term in keyword or keyword in term
                    for term in terms
                ):
                    matched.append(keyword)
                    score += row["ctfidf_score"]

            if matched:
                results.append({
                    "issue": issue,
                    "cluster_id": cluster_id,
                    "matched_keywords": ", ".join(matched),
                    "keyword_score_sum": round(score, 8),
                    "status": "후보_검토필요",
                })

    return sorted(
        results,
        key=lambda r: (
            r["issue"],
            -r["keyword_score_sum"]
        )
    )


def main():
    start = time.time()

    log_msg("12단계 전체 POLICY 토픽 분석 시작", start)

    if not ASSIGN_DB.exists():
        raise FileNotFoundError(
            f"11단계 결과 DB가 없습니다: {ASSIGN_DB}"
        )

    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            f"ChromaDB가 없습니다: {CHROMA_PATH}"
        )

    # 기존 할당 DB는 읽기 전용
    assignment_conn = sqlite3.connect(
        f"file:{ASSIGN_DB.as_posix()}?mode=ro",
        uri=True
    )

    table, id_col, label_col, cluster_col = (
        detect_assignment_table(assignment_conn)
    )

    policy_count = assignment_conn.execute(
        f"SELECT COUNT(*) FROM {sql_ident(table)} "
        f"WHERE {sql_ident(label_col)} = ?",
        ("POLICY",)
    ).fetchone()[0]

    print(
        f"할당 테이블: {table}\n"
        f"POLICY 대상: {policy_count:,}개"
    )

    cache_conn = sqlite3.connect(CACHE_DB)
    create_cache(cache_conn)

    # 원본 ChromaDB는 데이터 변경 메서드 사용하지 않음
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )
    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    total, cached, missing = retrieve_documents(
        assignment_conn,
        cache_conn,
        collection,
        table,
        id_col,
        label_col,
        cluster_col,
        start
    )

    if cached != policy_count:
        raise RuntimeError(
            f"POLICY 원문 캐시 수 불일치: "
            f"{cached:,} / {policy_count:,}\n"
            "원문 누락이나 기존 캐시 상태를 확인하세요."
        )

    topic_rows, keyword_rows = calculate_ctfidf(
        cache_conn, start
    )

    region_rows, no_date = analyze_region_year(
        cache_conn, start
    )

    issue_rows = link_policy_issues(keyword_rows)

    save_csv(
        OUT / "all_policy_topics_k100.csv",
        topic_rows
    )
    save_csv(
        OUT / "all_policy_topic_keywords_k100.csv",
        keyword_rows
    )
    save_csv(
        OUT / "all_policy_topics_by_region_year.csv",
        region_rows
    )
    save_csv(
        OUT / "all_policy_issue_candidates.csv",
        issue_rows
    )

    summary = {
        "completed_at": datetime.now().isoformat(),
        "policy_target": policy_count,
        "cached_documents": cached,
        "missing_documents": missing,
        "clusters": len(topic_rows),
        "vocabulary_keyword_rows": len(keyword_rows),
        "region_year_rows": len(region_rows),
        "unknown_year_documents": no_date,
        "issue_candidate_rows": len(issue_rows),
        "elapsed_sec": round(time.time() - start, 2),
        "cache_db": str(CACHE_DB),
        "notes": (
            "키워드는 c-TF-IDF 기반 후보이며 "
            "정책 주제 확정이나 확산 증거는 아님."
        ),
    }

    with open(
        OUT / "all_policy_topics_summary.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            summary, f, ensure_ascii=False, indent=2
        )

    assignment_conn.close()
    cache_conn.close()

    log_msg("[5/5] 모든 결과 저장 완료", start)

    print("\n=== 12단계 완료 ===")
    print(f"POLICY 원문: {cached:,}")
    print(f"토픽 군집: {len(topic_rows)}")
    print(f"지역·연도 통계 행: {len(region_rows):,}")
    print(f"결과 폴더: {OUT}")
    print(
        f"전체 소요: {time.time() - start:.1f}초"
    )


if __name__ == "__main__":
    main()

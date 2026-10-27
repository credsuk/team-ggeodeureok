# -*- coding: utf-8 -*-
"""14-2 | 지방의회 간 유사 정책 논의 후보: Snowflake 의미 유사도 개선판.

- 12단계 POLICY SQLite 캐시를 읽기 전용으로 사용
- 13단계 이슈 검색 규칙 재사용
- 회의 운영/명단/조례 목록 위주 청크 제외
- 의회별 이른 기록+시간 분산+내용 품질로 후보 표본 확대
- 후보 ID만 ChromaDB에서 기존 임베딩/회의록 메타데이터 조회
- 시간 순서, 다른 의회, 중복 방지, 정책 세부 분야, 의미 유사도 재검증
- 임베딩 로컬 SQLite 캐시로 중단 이후 재실행 가능
- 기존 14단계, Chroma 원본, KMeans 모델을 수정하지 않음

주의: 날짜가 앞서고 내용이 비슷하다고 정책이 실제로 전파된 것은 아닙니다.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import chromadb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(r"C:\mini_team\scripts\clustering")
SOURCE = ROOT / "results" / "all_policy_topics" / "policy_text_cache.sqlite3"
RULE_FILE = ROOT / "13_analyze_policy_issue_trends.py"
CHROMA_PATH = Path(r"C:\mini_team\_db\chroma_snowflake_regions")
COLLECTION_NAME = "team_regions_6a0a006e92d1e93e"
OUT = ROOT / "results" / "policy_discussion_links_refined"
EMBED_CACHE = OUT / "candidate_embeddings.sqlite3"

READ_BATCH = 2000
CHROMA_BATCH = 150
MAX_PER_COUNCIL_ISSUE = 120  # 14단계 60 -> 120, 최대 10,200 ID(중복 제거 전)
EARLY_SLOTS = 20
MAX_POOL = 450
TOP_PER_ISSUE = 12
PER_COUNCIL_PAIR_LIMIT = 2
SIM_THRESHOLD = 0.55         # 후보 시작값; 실측 분포를 보고 조정 (정확도 아님)
MAX_LAG_DAYS = 365
TOP_SCAN_NEIGHBORS = 24
MIN_CONTENT_CHARS = 90
EXCERPT_CHARS = 180

# 정책 내용 / 행동과 관련된 용어 (형식 문서만 높은 점수를 받지 않도록)
CONTENT_TERMS = (
    "지원사업", "지원금", "지원센터", "사업추진", "사업계획", "개선", "확대", "도입", "구축", "운영",
    "육성", "지원", "제도", "실행", "시행", "성과", "대책", "대응", "예산", "기금", "교육",
    "양성", "취업", "고용", "시설", "돌봄", "보육", "복지", "인프라", "참여", "설치", "감축",
    "지원대상", "인력", "프로그램", "정책", "계획", "공급", "수급", "서비스"
)
# 텍스트에 이런 형식 표현이 많이 나타나면 정책 주제보다는 회의 절차에 가까울 수 있음
FORM_TERMS = (
    "이상으로 제안설명을", "질의하실 위원님", "질의하실 의원님", "감사합니다. 다음",
    "선서를 하겠습니다", "선서를 시작", "공무원 선서", "출석공무원", "간부공무원",
    "국장님 나오셔서", "일괄 상정합니다", "원안대로 가결", "이의 없으십니까",
    "의사일정 제", "의사봉", "회의를 속개", "산회를 선포"
)
AGENDA_TERMS = ("조례안", "동의안", "건의안", "결의안", "의안번호", "심사보고", "의사일정", "원안가결")
MEETING_NAMES = ("meeting_title", "meeting_name", "session_title", "meeting", "title", "회의록명", "회의명")
SUBTOPICS = {
    "지역소멸": {
        "인구정주": ["인구감소", "정주", "귀촌", "귀농", "전입", "유입", "이주"],
        "지역활성": ["균형발전", "생활인구", "지역활력", "지방소멸대응기금", "소멸대응기금"],
        "청년정착": ["청년", "청년유출", "정착", "지역일자리"],
    },
    "청년 일자리": {
        "취업고용": ["취업", "채용", "고용", "구직", "직업훈련"],
        "청년창업": ["창업", "창업지원", "스타트업", "창업기업"],
        "일자리지원": ["일자리사업", "일자리지원", "취업지원", "구직활동", "취업준비"],
    },
    "저출생": {
        "출산난임": ["출산", "난임", "임신", "출산지원", "출산장려"],
        "양육보육": ["양육", "보육", "돌봄", "육아", "어린이집"],
        "인구정책": ["저출생", "저출산", "출산율", "인구정책"],
    },
    "고령화": {
        "노인돌봄": ["노인돌봄", "돌봄", "요양", "장기요양", "방문간호"],
        "노인일자리": ["노인일자리", "시니어", "고령자고용", "사회활동"],
        "고령친화": ["고령친화", "고령화", "초고령", "노인복지"],
    },
    "기후위기": {
        "탄소감축": ["탄소중립", "온실가스", "탄소배출", "배출량", "감축"],
        "기후재난": ["폭염", "가뭄", "홍수", "기후재난", "기후위기"],
        "에너지전환": ["신재생", "재생에너지", "에너지전환", "태양광", "수소"],
    },
}


def announce(msg, t0):
    print(f"[{datetime.now():%H:%M:%S}] {msg} | 경과 {time.perf_counter()-t0:.1f}초", flush=True)


def load_rules():
    if not RULE_FILE.exists():
        raise FileNotFoundError(f"13단계 스크립트를 같은 폴더에 두세요: {RULE_FILE}")
    spec = importlib.util.spec_from_file_location("trend_stage13", RULE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def simplified(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def fingerprint(text):
    return hashlib.sha1(re.sub(r"\W+", "", text.lower()).encode("utf-8")).hexdigest()


def content_quality(text, issue):
    """보수적 휴리스틱. 원문 정책성의 '정답'으로 간주하면 안 됨."""
    t = simplified(text)
    if len(t) < MIN_CONTENT_CHARS:
        return -99, "짧은 문서"
    procedural = sum(term in t for term in FORM_TERMS)
    topical = sum(term in t for term in CONTENT_TERMS)
    agenda = sum(t.count(term) for term in AGENDA_TERMS)
    subterm = sum(any(v in t for v in variants) for variants in SUBTOPICS[issue].values())
    if procedural >= 2:
        return -99, "의사진행 형식 반복"
    if agenda >= 5 and topical <= 4:
        return -99, "다수 안건 목록"
    if procedural and topical <= 2:
        return -99, "정책 설명 부족한 회의 형식"
    if topical == 0:
        return -99, "정책 실행 내용 부족"
    score = min(topical, 12) + subterm * 2 - procedural * 4 - max(0, agenda - 2) * 2
    return score, "통과" if score >= 1 else "정책 설명 부족"


def subtopic_labels(text, issue):
    return sorted(label for label, words in SUBTOPICS[issue].items() if any(w in text for w in words))


def meaningful_span(text, anchor):
    st = max(0, anchor - 45)
    return simplified(text[st:st + EXCERPT_CHARS])[:EXCERPT_CHARS]


def choose_candidates(rows):
    """임의의 상위 60개로 편향되지 않도록 earliest+quality+calendar coverage 혼합."""
    unique = {}
    for r in sorted(rows, key=lambda r: (r["date"], r["chunk_id"])):
        unique.setdefault(r["fingerprint"], r)
    all_rows = sorted(unique.values(), key=lambda r: (r["date"], r["chunk_id"]))
    if len(all_rows) <= MAX_PER_COUNCIL_ISSUE:
        return all_rows
    chosen = {}
    for r in all_rows[:EARLY_SLOTS]:
        chosen[r["chunk_id"]] = r
    # Distinct months and quality priority preserve detail-rich documents
    months = defaultdict(list)
    for r in all_rows:
        months[r["date"][:7]].append(r)
    for month in sorted(months):
        best = max(months[month], key=lambda r: (r["quality"], -len(r["document"])))
        chosen[best["chunk_id"]] = best
    quality = sorted(all_rows, key=lambda r: (-r["quality"], r["date"]))
    for r in quality:
        if len(chosen) >= MAX_PER_COUNCIL_ISSUE:
            break
        chosen[r["chunk_id"]] = r
    return sorted(list(chosen.values())[:MAX_PER_COUNCIL_ISSUE], key=lambda r: (r["date"], r["chunk_id"]))


def cache_setup(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS vectors (
        chunk_id TEXT PRIMARY KEY, dim INTEGER, vector BLOB,
        meeting_title TEXT, metadata_json TEXT)""")
    conn.commit()


def get_candidate_embeddings(rows, collection, cache, t0):
    missing = []
    ids = list(dict.fromkeys(r["chunk_id"] for r in rows))
    result = {}
    for offset in range(0, len(ids), 800):
        block = ids[offset:offset + 800]
        marks = ",".join("?" * len(block))
        got = cache.execute(f"SELECT chunk_id, dim, vector, meeting_title, metadata_json FROM vectors WHERE chunk_id IN ({marks})", block)
        for cid, dim, buf, title, md in got:
            vector = np.frombuffer(buf, dtype=np.float32)
            if len(vector) == dim:
                result[cid] = (vector.copy(), title or "", json.loads(md or "{}"))
    unseen = [cid for cid in ids if cid not in result]
    announce(f"[2/4] 임베딩 캐시 {len(result):,}건, 신규 조회 {len(unseen):,}건", t0)
    for offset in range(0, len(unseen), CHROMA_BATCH):
        batch = unseen[offset:offset + CHROMA_BATCH]
        fetched = collection.get(ids=batch, include=["embeddings", "metadatas"])
        fids = fetched["ids"]
        arr = fetched.get("embeddings")
        metas = fetched.get("metadatas")
        # NumPy 배열은 truth-value 검사 불가: None 여부만 판정
        if arr is None:
            arr = []
        if metas is None:
            metas = []
        insert = []
        for i, cid in enumerate(fids):
            if i >= len(arr) or arr[i] is None:
                continue
            vec = np.asarray(arr[i], dtype=np.float32).reshape(-1)
            if vec.size == 0 or not np.all(np.isfinite(vec)):
                continue
            vec = vec / max(float(np.linalg.norm(vec)), 1e-12)
            md = metas[i] if i < len(metas) and metas[i] is not None else {}
            title = next((str(md[k]) for k in MEETING_NAMES if md.get(k)), "")
            result[cid] = (vec, title, md)
            insert.append((cid, int(vec.size), vec.tobytes(), title, json.dumps(md, ensure_ascii=False, default=str)))
        if insert:
            cache.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?, ?)", insert)
            cache.commit()
        missing.extend(set(batch) - set(fids))
        if (offset // CHROMA_BATCH + 1) % 10 == 0 or offset + CHROMA_BATCH >= len(unseen):
            announce(f"임베딩 조회 {min(offset + CHROMA_BATCH, len(unseen)):,}/{len(unseen):,}", t0)
    return result, missing


def council_reference(text, earlier):
    stem = earlier.replace("특별자치도", "").replace("광역시", "").replace("특별시", "").replace("도의회", "").replace("시의회", "").replace("의회", "")
    if len(stem) >= 2 and stem in text and any(w in text for w in ("참고", "벤치마킹", "사례", "도입")):
        return "타 지역 언급 및 참조표현 공존(원문 확인 필요)"
    if any(w in text for w in ("타 시도 사례", "다른 지자체 사례", "벤치마킹")):
        return "일반적 외부사례 참조 표현(대상 미확인)"
    return "확인된 직접 참조 없음"


def evaluate_issue(issue, rows, vectors, t0):
    rows = [r for r in rows if r["chunk_id"] in vectors]
    if not rows:
        return [], {"issue": issue, "sampled": 0, "pairs_tested": 0, "above_threshold": 0, "selected": 0}
    dims = Counter(len(vectors[r["chunk_id"]][0]) for r in rows)
    main_dim = dims.most_common(1)[0][0]
    rows = [r for r in rows if len(vectors[r["chunk_id"]][0]) == main_dim]
    arr = np.stack([vectors[r["chunk_id"]][0] for r in rows]).astype(np.float32)
    arr /= np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
    # A second, lexical signal prevents boilerplate with high semantic similarity dominating
    lex = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=30000, dtype=np.float32)
    try:
        lex_x = lex.fit_transform([r["document"][:1200] for r in rows])
    except ValueError:
        lex_x = None
    scored = []
    eligible = 0
    above = 0
    for i, early in enumerate(rows):
        candidates = []
        for j, later in enumerate(rows):
            if early["assembly"] == later["assembly"] or early["fingerprint"] == later["fingerprint"]:
                continue
            lag = (later["dt"] - early["dt"]).days
            if not 1 <= lag <= MAX_LAG_DAYS:
                continue
            if early["subtopics"] and later["subtopics"] and not set(early["subtopics"]) & set(later["subtopics"]):
                continue
            candidates.append(j)
        if not candidates:
            continue
        eligible += len(candidates)
        sims = arr[candidates] @ arr[i]
        highest = np.argsort(sims)[-TOP_SCAN_NEIGHBORS:][::-1]
        for ix in highest:
            j = candidates[int(ix)]
            similarity = float(sims[int(ix)])
            if similarity < SIM_THRESHOLD:
                break
            later = rows[j]
            # lexical score is only a tie-breaker: semantic cosine remains primary
            lexical = float(lex_x[i].multiply(lex_x[j]).sum()) if lex_x is not None else 0.0
            # Fewer false positives by rewarding shared policy subtopic and actual content
            same = len(set(early["subtopics"]) & set(later["subtopics"]))
            final = 0.80 * similarity + 0.12 * lexical + 0.04 * min(same, 2) + 0.04 * min(early["quality"], later["quality"], 8) / 8
            scored.append((final, similarity, lexical, early, later))
            above += 1
    scored.sort(key=lambda x: (-x[0], -x[1]))
    results = []
    council_pairs = Counter()
    seen = set()
    for rank_score, sim, lexical, early, later in scored:
        pair = (early["assembly"], later["assembly"])
        ids = (early["chunk_id"], later["chunk_id"])
        if council_pairs[pair] >= PER_COUNCIL_PAIR_LIMIT or ids in seen:
            continue
        council_pairs[pair] += 1
        seen.add(ids)
        e_title = vectors[early["chunk_id"]][1]
        l_title = vectors[later["chunk_id"]][1]
        results.append({
            "issue": issue, "rank": len(results) + 1,
            "earlier_assembly": early["assembly"], "earlier_date": early["date"],
            "earlier_meeting_title": e_title, "earlier_chunk_id": early["chunk_id"],
            "earlier_source_key": early["source_key"], "earlier_cluster_id": early["cluster_id"],
            "earlier_subtopics": "; ".join(early["subtopics"]),
            "earlier_matched_terms": "; ".join(early["matched_terms"]),
            "earlier_evidence_180": early["evidence_180"],
            "earlier_preview_600": simplified(early["document"])[:600],
            "later_assembly": later["assembly"], "later_date": later["date"],
            "later_meeting_title": l_title, "later_chunk_id": later["chunk_id"],
            "later_source_key": later["source_key"], "later_cluster_id": later["cluster_id"],
            "later_subtopics": "; ".join(later["subtopics"]),
            "later_matched_terms": "; ".join(later["matched_terms"]),
            "later_evidence_180": later["evidence_180"],
            "later_preview_600": simplified(later["document"])[:600],
            "lag_days": (later["dt"] - early["dt"]).days,
            "semantic_cosine": round(sim, 5), "lexical_cosine": round(lexical, 5),
            "combined_ranking_score": round(rank_score, 5),
            "same_kmeans_cluster": int(early["cluster_id"] == later["cluster_id"]),
            "reference_signal": council_reference(later["document"], early["assembly"]),
            "review_status": "원문_검토필요", "actual_diffusion_verified": "아니오",
            "metadata_status": "회의록명 확인 필요" if not (e_title and l_title) else "회의록명 메타데이터 존재(원본 대조 필요)",
            "interpretation": "의미상 유사한 후속 논의 후보. 정책 전파의 직접 증거 아님",
        })
        if len(results) >= TOP_PER_ISSUE:
            break
    announce(f"이슈 {issue} | 표본 {len(rows):,} | 비교대상 {eligible:,}쌍 | 통과 {above:,}쌍 | 선정 {len(results)}쌍", t0)
    return results, {"issue": issue, "sampled": len(rows), "pairs_tested": eligible,
                     "above_threshold": above, "selected": len(results)}


def save_csv(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if not SOURCE.is_file():
        raise FileNotFoundError(f"12단계 원문 캐시 없음: {SOURCE}")
    rules = load_rules()
    conn = sqlite3.connect(SOURCE.as_uri() + "?mode=ro", uri=True)
    target = conn.execute("SELECT COUNT(*) FROM policy_docs").fetchone()[0]
    announce(f"[1/4] 전체 POLICY 원문 {target:,}개 확인", t0)
    pools = defaultdict(list)
    match_counts = Counter()
    filter_counts = Counter()
    processed = 0
    cursor = conn.execute("SELECT chunk_id, cluster_id, document, assembly, source_key, date FROM policy_docs")
    while True:
        batch = cursor.fetchmany(READ_BATCH)
        if not batch:
            break
        for chunk_id, cluster_id, doc, assembly, source_key, raw_date in batch:
            processed += 1
            if not doc or not assembly:
                continue
            day, _, _, precision = rules.normalize_date(raw_date, source_key)
            if precision != "day":
                continue
            try:
                dt = date.fromisoformat(day)
            except ValueError:
                continue
            for issue, rule in rules.ISSUE_RULES.items():
                yes, anchor, terms = rules.issue_match(doc, rule)
                if not yes:
                    continue
                match_counts[issue] += 1
                quality, why = content_quality(doc, issue)
                if quality < 1:
                    filter_counts[(issue, why)] += 1
                    continue
                r = {
                    "chunk_id": str(chunk_id), "cluster_id": int(cluster_id),
                    "document": str(doc), "assembly": str(assembly).strip(),
                    "source_key": str(source_key or ""), "date": day, "dt": dt,
                    "quality": quality, "subtopics": subtopic_labels(doc, issue),
                    "matched_terms": terms, "evidence_180": meaningful_span(doc, anchor),
                    "fingerprint": fingerprint(doc),
                }
                key = (issue, r["assembly"])
                pool = pools[key]
                pool.append(r)
                if len(pool) > MAX_POOL:
                    # Retain early items and high-quality text, bounded to ~300 / group
                    pool.sort(key=lambda x: (x["date"], x["chunk_id"]))
                    keep_early = pool[:EARLY_SLOTS]
                    better = sorted(pool[EARLY_SLOTS:], key=lambda x: (-x["quality"], x["date"]))
                    dedup = {r["chunk_id"]: r for r in keep_early + better[:MAX_POOL - EARLY_SLOTS - 80]}
                    pool[:] = list(dedup.values())
        if processed % 20000 < READ_BATCH:
            announce(f"정책 검사 {processed:,}/{target:,}", t0)
    conn.close()
    sampled = defaultdict(list)
    for (issue, council), rows in pools.items():
        sampled[issue].extend(choose_candidates(rows))
    announce(f"[1/4] 후보 선정 완료 / 표본 총 {sum(map(len,sampled.values())):,}", t0)

    # SQLite embeddings remain cached across re-runs, original Chroma untouched.
    ec = sqlite3.connect(EMBED_CACHE)
    cache_setup(ec)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    all_selected = [r for group in sampled.values() for r in group]
    vectors, missing = get_candidate_embeddings(all_selected, collection, ec, t0)
    announce(f"[2/4] 임베딩 확보 {len(vectors):,}개 / 누락 {len(missing):,}개", t0)

    results, stats = [], []
    for issue in rules.ISSUE_RULES:
        links, metric = evaluate_issue(issue, sampled[issue], vectors, t0)
        metric["matched_before_filter"] = match_counts[issue]
        metric["filtered_out"] = sum(n for (i, _), n in filter_counts.items() if i == issue)
        metric["councils"] = len(set(r["assembly"] for r in sampled[issue]))
        results.extend(links)
        stats.append(metric)
    announce(f"[3/4] 5개 이슈 후보 {len(results)}쌍 선정", t0)
    columns = [
        "issue", "rank", "earlier_assembly", "earlier_date", "earlier_meeting_title",
        "earlier_chunk_id", "earlier_source_key", "earlier_cluster_id", "earlier_subtopics",
        "earlier_matched_terms", "earlier_evidence_180", "earlier_preview_600",
        "later_assembly", "later_date", "later_meeting_title", "later_chunk_id",
        "later_source_key", "later_cluster_id", "later_subtopics", "later_matched_terms",
        "later_evidence_180", "later_preview_600", "lag_days", "semantic_cosine",
        "lexical_cosine", "combined_ranking_score", "same_kmeans_cluster",
        "reference_signal", "review_status", "actual_diffusion_verified",
        "metadata_status", "interpretation"
    ]
    save_csv(OUT / "refined_policy_discussion_candidates.csv", results, columns)
    stat_columns = ["issue", "matched_before_filter", "filtered_out", "councils", "sampled", "pairs_tested", "above_threshold", "selected"]
    save_csv(OUT / "refined_policy_link_stats.csv", stats, stat_columns)
    summary = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "total_policy_chunks": target, "processed": processed, "unique_vector_ids": len(vectors),
        "missing_vectors": len(missing), "candidate_pairs": len(results),
        "issue_stats": stats, "elapsed_sec": round(time.perf_counter() - t0, 2),
        "settings": {"semantic_threshold": SIM_THRESHOLD, "max_per_council_issue": MAX_PER_COUNCIL_ISSUE,
                     "max_lag_days": MAX_LAG_DAYS, "top_per_issue": TOP_PER_ISSUE},
        "notes": "연결은 의미상 유사·시간 선후 후보이며 실제 전파를 증명하지 않음. 키워드 정책성 필터와 표본 추출로 누락 가능. Snowflake 기존 임베딩을 재사용."
    }
    (OUT / "refined_policy_discussion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    ec.close()
    announce(f"[4/4] 완료 | {OUT}", t0)
    print("후보:", OUT / "refined_policy_discussion_candidates.csv")
    print("요약:", OUT / "refined_policy_discussion_summary.json")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""14단계 | 지방의회 간 유사 정책 논의 '후보' 연결 및 원문 근거 검증.

- 12단계 원문 캐시만 사용: 원본 ChromaDB 재조회/수정 없음
- 13단계의 5대 이슈 판별 규칙 재사용
- 이슈×의회별 날짜 분산 표본 -> 한국어 문자 n-gram TF-IDF 유사도 비교
- 반드시 다른 의회, 서로 다른 날짜, 앞선 논의→후속 논의 순서
- 내용 유사+시점 선후만으로 '정책 전파'를 확정하지 않음
- '회의록명'은 캐시에 없으므로 빈 칸으로 두고 추가 확인 표시

실행: python 14_link_policy_discussions.py
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
from datetime import datetime, date
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(r"C:\mini_team\scripts\clustering")
CACHE = ROOT / "results" / "all_policy_topics" / "policy_text_cache.sqlite3"
PREVIOUS = ROOT / "13_analyze_policy_issue_trends.py"
OUT = ROOT / "results" / "policy_discussion_links"
BATCH = 2000
MIN_CHARS = 80
MAX_PER_COUNCIL_ISSUE = 60
EARLIEST_KEEP = 10
TOP_LINKS_PER_ISSUE = 12
MAX_LINKS_PER_COUNCIL_PAIR = 2
MIN_SIMILARITY = 0.38
MAX_LAG_DAYS = 365
TOP_NEIGHBORS = 12
EXCERPT_LEN = 180

def log(msg, start):
    print(f"[{datetime.now():%H:%M:%S}] {msg} | 경과 {time.perf_counter()-start:.1f}초", flush=True)

def read_rule_module():
    if not PREVIOUS.is_file():
        raise FileNotFoundError(f"13단계 코드가 필요합니다: {PREVIOUS}")
    spec = importlib.util.spec_from_file_location("stage13_policy_rules", PREVIOUS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()

def fingerprint(text):
    clean = re.sub(r"\W+", "", text.lower())
    return hashlib.sha1(clean.encode("utf-8")).hexdigest()

def to_date(s):
    try:
        return date.fromisoformat(s)
    except (TypeError, ValueError):
        return None

def excerpt(text, pos, length=EXCERPT_LEN):
    # Anchor computed on the original document text (not whitespace-collapsed text).
    text = str(text or "")
    start = max(0, pos - 45)
    part = text[start:start + length]
    return normalize_text(part)[:length]

def csv_save(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def select_sample(samples, row):
    """시간상 최초 10건 + 나머지의 날짜 분산 안정 해시표본.
    모든 후보가 아닌 최대 60개/이슈/의회만 비교하므로 대표성 제한 있음.
    """
    samples.append(row)

def finalize_samples(all_rows):
    sampled = {}
    for key, rows in all_rows.items():
        # Exact text duplicates reduced per council; a distinct ID may still duplicate.
        unique = {}
        for r in sorted(rows, key=lambda x: (x["date"], x["chunk_id"])):
            unique.setdefault(r["fingerprint"], r)
        rows = sorted(unique.values(), key=lambda x: (x["date"], x["chunk_id"]))
        if len(rows) <= MAX_PER_COUNCIL_ISSUE:
            sampled[key] = rows
            continue
        early = rows[:EARLIEST_KEEP]
        remaining = rows[EARLIEST_KEEP:]
        slots = MAX_PER_COUNCIL_ISSUE - len(early)
        # spread across the complete temporal span, not merely earliest rows
        offsets = sorted(set(min(len(remaining)-1, int((i + 0.5) * len(remaining) / slots))
                             for i in range(slots)))
        sampled[key] = sorted(early + [remaining[j] for j in offsets],
                              key=lambda x: (x["date"], x["chunk_id"]))
    return sampled

def reference_cue(later_text, earlier_council):
    """참조 표현 존재여부의 약한 신호. 정책 참고를 입증하지 못함."""
    short = earlier_council.replace("특별자치", "").replace("광역시의회", "").replace("도의회", "").replace("특별시의회", "")
    cues = ["벤치마킹", "사례를 참고", "사례 참고", "타 시도", "타 지자체", "다른 시도"]
    if any(cue in later_text for cue in cues):
        return "간접 참조 표현 존재(대상 확인 필요)"
    if len(short) >= 2 and short in later_text and ("사례" in later_text or "참고" in later_text):
        return "지역명과 참고/사례 표현 공존(직접 인용 확인 필요)"
    return "직접 참조 근거 확인되지 않음"

def compare_issue(issue, rows, start):
    """TF-IDF (char 2-4gram), compare all selected candidate documents for this issue."""
    if len(rows) < 2:
        return [], {"issue": issue, "sampled": len(rows), "pairs_tested": 0}
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                                 min_df=2, max_df=0.94,
                                 max_features=80000, sublinear_tf=True,
                                 norm="l2", dtype="float32")
    try:
        X = vectorizer.fit_transform([r["document"][:1800] for r in rows])
    except ValueError:
        return [], {"issue": issue, "sampled": len(rows), "pairs_tested": 0}

    # Sparse dot product: avoid materializing all embeddings or full Chroma.
    scored = []
    tested = 0
    for i, early in enumerate(rows):
        # Restrict comparisons to distinct councils and later dates within 365 days.
        later_idxs = [j for j, later in enumerate(rows)
                      if later["assembly"] != early["assembly"]
                      and 0 < (later["dt"] - early["dt"]).days <= MAX_LAG_DAYS
                      and later["fingerprint"] != early["fingerprint"]]
        if not later_idxs:
            continue
        tested += len(later_idxs)
        sims = cosine_similarity(X[i], X[later_idxs]).ravel()
        best_idxs = sims.argsort()[-TOP_NEIGHBORS:][::-1]
        for idx in best_idxs:
            s = float(sims[idx])
            if s < MIN_SIMILARITY:
                break
            later = rows[later_idxs[idx]]
            # This is semantic similarity; same cluster is supporting info, not required.
            scored.append((s, early, later))

    scored.sort(key=lambda r: (-r[0], r[1]["date"], r[2]["date"]))
    used_council_pairs = Counter()
    used_id_pairs = set()
    result = []
    for similarity, early, later in scored:
        cp = (early["assembly"], later["assembly"])
        id_pair = (early["chunk_id"], later["chunk_id"])
        if used_council_pairs[cp] >= MAX_LINKS_PER_COUNCIL_PAIR or id_pair in used_id_pairs:
            continue
        used_council_pairs[cp] += 1
        used_id_pairs.add(id_pair)
        lag = (later["dt"] - early["dt"]).days
        result.append({
            "issue": issue,
            "candidate_rank": len(result) + 1,
            "earlier_assembly": early["assembly"],
            "earlier_date": early["date"],
            "earlier_chunk_id": early["chunk_id"],
            "earlier_source_key": early["source_key"],
            "earlier_meeting_title": "",  # missing in stage-12 cache
            "earlier_cluster_id": early["cluster_id"],
            "earlier_matched_terms": early["matched_terms"],
            "earlier_evidence_180": early["evidence_180"],
            "earlier_preview_600": normalize_text(early["document"])[:600],
            "later_assembly": later["assembly"],
            "later_date": later["date"],
            "later_chunk_id": later["chunk_id"],
            "later_source_key": later["source_key"],
            "later_meeting_title": "",
            "later_cluster_id": later["cluster_id"],
            "later_matched_terms": later["matched_terms"],
            "later_evidence_180": later["evidence_180"],
            "later_preview_600": normalize_text(later["document"])[:600],
            "lag_days": lag,
            "text_similarity": round(similarity, 5),
            "same_cluster": int(early["cluster_id"] == later["cluster_id"]),
            "reference_signal": reference_cue(later["document"], early["assembly"]),
            "review_status": "후보_원문검증필요",
            "diffusion_confirmed": "아니오",
            "interpretation": "서로 다른 지방의회의 내용 유사·시간 선후 후보. 실제 정책 전파를 의미하지 않음.",
            "meeting_metadata_status": "회의록명 미확보(원본 메타데이터 재확인 필요)"
        })
        if len(result) >= TOP_LINKS_PER_ISSUE:
            break
    log(f"이슈 {issue}: 비교 {tested:,}쌍 / 유사도 통과 {len(scored):,}쌍 / 선정 {len(result)}쌍", start)
    return result, {"issue": issue, "sampled": len(rows), "pairs_tested": tested,
                    "passed_threshold": len(scored), "selected": len(result)}

def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if not CACHE.is_file():
        raise FileNotFoundError(f"12단계 원문 캐시가 없습니다: {CACHE}")
    rules = read_rule_module()
    conn = sqlite3.connect(CACHE.as_uri() + "?mode=ro", uri=True)
    target = conn.execute("SELECT COUNT(*) FROM policy_docs").fetchone()[0]
    log(f"[1/4] POLICY 원문 {target:,}개 읽기 시작", start)
    # Limit memory: reservoir-like deterministic hash priority per issue/council,
    # with separately retained earliest records. This avoids storing every match.
    candidates = defaultdict(list)
    counts = Counter()
    processed = 0
    cur = conn.execute(
        "SELECT chunk_id, cluster_id, document, assembly, source_key, date FROM policy_docs")
    while True:
        batch = cur.fetchmany(BATCH)
        if not batch:
            break
        for chunk_id, cluster_id, document, assembly, source_key, raw_date in batch:
            processed += 1
            day, ym, yr, precision = rules.normalize_date(raw_date, source_key)
            dt = to_date(day)
            if dt is None or not assembly or len(document or "") < MIN_CHARS:
                continue
            for issue, rule in rules.ISSUE_RULES.items():
                yes, anchor, terms = rules.issue_match(document, rule)
                if not yes:
                    continue
                counts[(issue, assembly)] += 1
                row = {
                    "chunk_id": str(chunk_id), "cluster_id": int(cluster_id),
                    "document": document, "assembly": str(assembly).strip(),
                    "source_key": str(source_key or ""), "date": day, "dt": dt,
                    "matched_terms": "; ".join(terms),
                    "evidence_180": excerpt(document, anchor),
                    "fingerprint": fingerprint(document),
                }
                key = (issue, row["assembly"])
                # keep earliest 10 and deterministic distributed candidates by hashed key.
                pool = candidates[key]
                pool.append(row)
                if len(pool) > 180:
                    pool.sort(key=lambda x: (x["date"], x["chunk_id"]))
                    early = pool[:EARLIEST_KEEP]
                    tail = pool[EARLIEST_KEEP:]
                    # Include a deterministic broad temporal sample via compact priority.
                    tail.sort(key=lambda r: hashlib.blake2b(r["chunk_id"].encode(), digest_size=8).digest())
                    candidates[key] = early + tail[:MAX_PER_COUNCIL_ISSUE * 2]
        if processed % 20000 < BATCH:
            log(f"원문 검토 {processed:,}/{target:,} ({processed/target:.1%})", start)
    conn.close()
    log(f"[2/4] 이슈×의회 {len(candidates)}그룹 판정 완료", start)
    sampled = finalize_samples(candidates)
    grouped = defaultdict(list)
    for (issue, council), rows in sampled.items():
        grouped[issue].extend(rows)
    selected = []
    stats = []
    for issue in rules.ISSUE_RULES:
        links, metric = compare_issue(issue, grouped[issue], start)
        selected.extend(links)
        metric["councils"] = len({r["assembly"] for r in grouped[issue]})
        metric["matched_documents_before_sampling"] = sum(
            n for (i, _), n in counts.items() if i == issue)
        stats.append(metric)

    fields = [
        "issue", "candidate_rank", "earlier_assembly", "earlier_date",
        "earlier_chunk_id", "earlier_source_key", "earlier_meeting_title",
        "earlier_cluster_id", "earlier_matched_terms", "earlier_evidence_180",
        "earlier_preview_600", "later_assembly", "later_date",
        "later_chunk_id", "later_source_key", "later_meeting_title",
        "later_cluster_id", "later_matched_terms", "later_evidence_180",
        "later_preview_600", "lag_days", "text_similarity", "same_cluster",
        "reference_signal", "review_status", "diffusion_confirmed",
        "interpretation", "meeting_metadata_status"
    ]
    csv_save(OUT / "policy_discussion_link_candidates.csv", selected, fields)
    csv_save(OUT / "policy_link_issue_stats.csv", stats,
             ["issue", "matched_documents_before_sampling", "councils",
              "sampled", "pairs_tested", "passed_threshold", "selected"])
    summary = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "processed_policy_chunks": processed, "source_policy_chunks": target,
        "issue_stats": stats, "candidate_pairs": len(selected),
        "settings": {
            "max_per_council_issue": MAX_PER_COUNCIL_ISSUE,
            "earliest_keep": EARLIEST_KEEP, "min_chars": MIN_CHARS,
            "similarity_threshold": MIN_SIMILARITY, "max_lag_days": MAX_LAG_DAYS,
            "top_links_per_issue": TOP_LINKS_PER_ISSUE,
            "max_links_per_council_pair": MAX_LINKS_PER_COUNCIL_PAIR,
            "method": "character 2-4gram TF-IDF cosine on limited time-diverse sample"
        },
        "elapsed_sec": round(time.perf_counter()-start, 2),
        "caveat": "후보 검색일 뿐 전체 원본의 최초 등장/실제 정책 전파는 검증되지 않음. "
                  "의회별 시간 분산 표본이며 검색 재현율은 측정하지 않음. "
                  "회의록명은 12단계 캐시에 없으므로 추가 메타데이터 결합 필요."
    }
    with (OUT / "policy_discussion_link_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"[4/4] 완료 / 후보 {len(selected)}쌍 / {OUT}", start)
    print("저장:", OUT / "policy_discussion_link_candidates.csv")
    print("요약:", OUT / "policy_discussion_link_summary.json")

if __name__ == "__main__":
    main()

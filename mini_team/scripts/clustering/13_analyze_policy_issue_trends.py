# -*- coding: utf-8 -*-
"""13단계 | 전체 POLICY 원문 캐시에서 5대 정책 이슈의 지역·시기별 흐름과 근거 추출.

입력(12단계): results/all_policy_topics/policy_text_cache.sqlite3
                results/all_policy_topics/all_policy_topic_keywords_k100.csv (선택)
출력: results/policy_issue_trends/
- issue_monthly_trends.csv / issue_yearly_trends.csv / issue_region_summary.csv
- issue_first_observed_by_region.csv / issue_evidence_180chars.csv
- issue_cluster_distribution.csv / issue_topic_candidates.csv / issue_analysis_summary.json

주의: POLICY로 분류된 청크의 표현 출현만 측정한다. 실제 정책의 최초 발생,
      의회 간 확산의 인과관계, 토픽의 확정 분류를 의미하지 않는다.
"""
from __future__ import annotations

import csv
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:\mini_team\scripts\clustering")
SOURCE = ROOT / "results" / "all_policy_topics" / "policy_text_cache.sqlite3"
KEYWORDS_CSV = ROOT / "results" / "all_policy_topics" / "all_policy_topic_keywords_k100.csv"
OUT = ROOT / "results" / "policy_issue_trends"
BATCH = 2000
EXCERPT_CHARS = 180
EVIDENCE_PER_ISSUE_REGION = 5
PROGRESS_EVERY = 20000

# 반드시 실질적 표현 조합을 확인한다. 광범위한 '인구', '환경', '청년' 단독 등은 제외.
ISSUE_RULES = {
    "지역소멸": {
        "strong": ["지방소멸", "지역소멸", "소멸위험", "인구감소지역", "인구소멸"],
        "groups": [
            (["인구감소", "인구유출", "청년유출", "인구절벽", "인구 감소", "인구 유출"],
             ["지역", "지방", "정주", "소멸", "농촌", "시군", "시·군", "마을", "균형발전"]),
        ],
    },
    "청년 일자리": {
        "strong": ["청년일자리", "청년취업", "청년고용", "청년창업", "청년 일자리", "청년 취업", "청년 고용", "청년 창업"],
        "groups": [(["청년", "청년층", "청년들"], ["일자리", "고용", "취업", "구직", "창업", "채용", "직업훈련"])],
    },
    "저출생": {
        "strong": ["저출생", "저출산", "출산율", "합계출산율", "출생률", "출산장려", "출산지원"],
        "groups": [(["출산", "출생", "난임"], ["지원", "정책", "대책", "장려", "인구", "양육", "보육", "임신"])],
    },
    "고령화": {
        "strong": ["고령화", "초고령", "고령사회", "노인복지", "노인돌봄", "어르신돌봄", "고령친화"],
        "groups": [(["노인", "어르신", "고령자", "고령층"], ["복지", "돌봄", "요양", "일자리", "빈곤", "주거", "의료", "지원"])],
    },
    "기후위기": {
        "strong": ["기후위기", "기후변화", "탄소중립", "온실가스", "기후재난", "기후 위기", "기후 변화", "탄소 중립"],
        "groups": [(["폭염", "가뭄", "홍수", "탄소", "배출", "기후"], ["대응", "감축", "적응", "위기", "환경", "온실가스", "탄소중립"])],
    },
}

# 키워드 근접성 검사: 아주 긴 한 청크에서 서로 무관한 문맥을 결합하지 않도록 제한
MAX_GROUP_DISTANCE = 200
DATE_RX = re.compile(r"(?<!\d)((?:19|20)\d{2})[-./년 ]?\s*(0?[1-9]|1[0-2])[-./월 ]?\s*(0?[1-9]|[12]\d|3[01])(?!\d)")
MONTH_RX = re.compile(r"(?<!\d)((?:19|20)\d{2})[-./년 ]?\s*(0?[1-9]|1[0-2])(?!\d)")
YEAR_RX = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def logger(msg, start):
    print(f"[{datetime.now():%H:%M:%S}] {msg} | 경과 {time.perf_counter()-start:.1f}초", flush=True)


def normalize_date(raw, source_key=""):
    """Return (YYYY-MM-DD or '', YYYY-MM or '', YYYY or '', precision)."""
    s = str(raw or "").strip()
    m = DATE_RX.search(s)
    if m:
        year, month, day = map(int, m.groups())
        try:
            d = datetime(year, month, day)
            return d.strftime("%Y-%m-%d"), d.strftime("%Y-%m"), str(year), "day"
        except ValueError:
            pass
    m = MONTH_RX.search(s)
    if m:
        yr, mo = map(int, m.groups())
        return "", f"{yr:04d}-{mo:02d}", str(yr), "month"
    m = YEAR_RX.search(s)
    if m:
        return "", "", m.group(1), "year"
    m = YEAR_RX.search(str(source_key or ""))
    if m:
        return "", "", m.group(1), "source_year"
    return "", "", "UNKNOWN", "unknown"


def positions(text, term):
    start = 0
    while True:
        idx = text.find(term, start)
        if idx == -1:
            break
        yield idx
        start = idx + max(len(term), 1)


def issue_match(text, rule):
    """Return (matched, evidence_anchor, detected_terms) on a single utterance chunk."""
    compact = str(text or "").lower()
    for term in rule["strong"]:
        i = compact.find(term.lower())
        if i >= 0:
            return True, i, [term]
    for group_a, group_b in rule["groups"]:
        aa = [(p, a) for a in group_a for p in positions(compact, a.lower())]
        bb = [(p, b) for b in group_b for p in positions(compact, b.lower())]
        if not aa or not bb:
            continue
        best = None
        for pa, a in aa:
            for pb, b in bb:
                dist = abs(pa - pb)
                if dist <= MAX_GROUP_DISTANCE and (best is None or dist < best[0]):
                    best = (dist, min(pa, pb), a, b)
        if best:
            return True, best[1], [best[2], best[3]]
    return False, -1, []


def snippet(text, pos, limit=180):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return ""
    start = min(max(0, pos - 45), max(0, len(text) - limit))
    return text[start:start + limit]


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    logger("13단계 정책 이슈 흐름 분석 시작", start)
    if not SOURCE.is_file():
        raise FileNotFoundError(f"12단계 원문 캐시가 없습니다: {SOURCE}")

    # SQLite를 읽기 전용 연결. 11단계 DB와 원본 ChromaDB는 열지 않음.
    conn = sqlite3.connect(SOURCE.as_uri() + "?mode=ro", uri=True)
    cols = {x[1] for x in conn.execute("PRAGMA table_info(policy_docs)")}
    required = {"chunk_id", "cluster_id", "document", "assembly", "source_key", "date"}
    if not required <= cols:
        raise RuntimeError(f"policy_docs 필수 컬럼 누락: {required - cols}")
    target = conn.execute("SELECT COUNT(*) FROM policy_docs").fetchone()[0]
    logger(f"[1/4] POLICY 원문 {target:,}건 확인", start)

    monthly = Counter()
    yearly = Counter()
    region = Counter()
    cluster_dist = Counter()
    date_precision = Counter()
    issue_total = Counter()
    multi_issue = 0
    first_seen = {}  # (issue, assembly) -> (valid day, chunk_id, evidence record)
    evidence = defaultdict(list)  # (issue, assembly) -> sorted earliest dated evidence, max 5
    undated_count = Counter()
    processed = 0

    cursor = conn.execute("SELECT chunk_id, cluster_id, document, assembly, source_key, date FROM policy_docs")
    while True:
        batch = cursor.fetchmany(BATCH)
        if not batch:
            break
        for chunk_id, cluster_id, doc, assembly, source_key, raw_date in batch:
            processed += 1
            council = str(assembly or "").strip() or str(source_key or "").strip() or "UNKNOWN"
            date, ym, year, precision = normalize_date(raw_date, source_key)
            date_precision[precision] += 1
            matches = 0
            for issue, rule in ISSUE_RULES.items():
                yes, anchor, words = issue_match(doc, rule)
                if not yes:
                    continue
                matches += 1
                issue_total[issue] += 1
                yearly[(issue, year, council)] += 1
                region[(issue, council)] += 1
                cluster_dist[(issue, int(cluster_id))] += 1
                if ym:
                    monthly[(issue, ym, council)] += 1
                else:
                    undated_count[issue] += 1
                row = {
                    "issue": issue, "assembly": council, "date": date,
                    "year_month": ym, "date_precision": precision,
                    "cluster_id": int(cluster_id), "chunk_id": str(chunk_id),
                    "source_key": str(source_key or ""),
                    "matched_terms": "; ".join(words),
                    "evidence_text_180": snippet(doc, anchor, EXCERPT_CHARS),
                    "interpretation": "해당 POLICY 청크에서 표현이 관측됨; 정책 최초·전파 인과 아님",
                }
                if date:
                    key = (issue, council)
                    current = first_seen.get(key)
                    if current is None or (date, str(chunk_id)) < (current[0], current[1]):
                        first_seen[key] = (date, str(chunk_id), row)
                    # 작은 리스트만 유지. 동일 발언 날짜의 증거 중복 최소화
                    keep = evidence[key]
                    if not any(x[1]["evidence_text_180"] == row["evidence_text_180"] for x in keep):
                        keep.append(((date, str(chunk_id)), row))
                        keep.sort(key=lambda item: item[0])
                        del keep[EVIDENCE_PER_ISSUE_REGION:]
            if matches > 1:
                multi_issue += 1
        if processed % PROGRESS_EVERY < BATCH:
            logger(f"[2/4] {processed:,}/{target:,} ({processed/target:.1%})", start)

    conn.close()
    if processed != target:
        raise RuntimeError(f"캐시 행수 불일치: 조회 {processed:,}, 예상 {target:,}")
    logger("[2/4] 원문별 이슈 표현 판정 완료", start)

    months = [{"issue": k[0], "year_month": k[1], "assembly": k[2], "chunk_count": v}
              for k, v in sorted(monthly.items())]
    years = [{"issue": k[0], "year": k[1], "assembly": k[2], "chunk_count": v}
             for k, v in sorted(yearly.items())]
    regions = [{"issue": k[0], "assembly": k[1], "chunk_count": v}
               for k, v in sorted(region.items())]
    dist = [{"issue": k[0], "cluster_id": k[1], "chunk_count": v}
            for k, v in sorted(cluster_dist.items())]
    first_rows = [dict(row, **{"first_observed_date_in_policy_cache": date})
                  for (_, _), (date, _, row) in sorted(first_seen.items())]
    evidence_rows = []
    for key in sorted(evidence):
        for rank, (_, row) in enumerate(evidence[key], 1):
            evidence_rows.append(dict(row, evidence_rank=rank))

    write_csv(OUT / "issue_monthly_trends.csv", months, ["issue", "year_month", "assembly", "chunk_count"])
    write_csv(OUT / "issue_yearly_trends.csv", years, ["issue", "year", "assembly", "chunk_count"])
    write_csv(OUT / "issue_region_summary.csv", regions, ["issue", "assembly", "chunk_count"])
    write_csv(OUT / "issue_cluster_distribution.csv", dist, ["issue", "cluster_id", "chunk_count"])
    ev_fields = ["issue", "assembly", "date", "year_month", "date_precision", "cluster_id",
                 "chunk_id", "source_key", "matched_terms", "evidence_text_180", "interpretation"]
    write_csv(OUT / "issue_first_observed_by_region.csv", first_rows, ev_fields + ["first_observed_date_in_policy_cache"])
    write_csv(OUT / "issue_evidence_180chars.csv", evidence_rows, ev_fields + ["evidence_rank"])

    # 추가 파일: 토픽 키워드로 군집 연결의 보조 근거만 만들고 본문 건수에 합산하지 않음
    candidates = []
    if KEYWORDS_CSV.is_file():
        with KEYWORDS_CSV.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                word = r.get("keyword", "").strip().lower()
                if not word:
                    continue
                for issue, rule in ISSUE_RULES.items():
                    if word in {w.lower() for w in rule["strong"]}:
                        candidates.append({"issue": issue, "cluster_id": r.get("cluster_id", ""),
                                           "keyword": word, "ctfidf_score": r.get("ctfidf_score", ""),
                                           "note": "직접 핵심 키워드 일치; 토픽 확정 아님"})
    write_csv(OUT / "issue_topic_candidates.csv", candidates,
              ["issue", "cluster_id", "keyword", "ctfidf_score", "note"])

    total_sec = time.perf_counter() - start
    summary = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "policy_cache_count": target, "processed_count": processed,
        "issue_chunk_counts": dict(issue_total),
        "issue_matches_may_overlap": True, "multi_issue_chunk_count": multi_issue,
        "date_precision_counts": dict(date_precision),
        "issue_count_without_month": dict(undated_count),
        "monthly_rows": len(months), "yearly_rows": len(years),
        "region_rows": len(regions), "cluster_rows": len(dist),
        "first_observed_rows": len(first_rows), "evidence_rows": len(evidence_rows),
        "elapsed_sec": round(total_sec, 2),
        "source_cache": str(SOURCE),
        "caveat": "정책 규칙으로 POLICY 분류된 청크에 명시적 이슈 표현이 있는 경우만 집계. 청크 수는 회의·발언 수와 다름. 최초 관측은 이 POLICY 캐시 안에서 날짜가 확인된 청크 기준이며 실제 최초 논의나 확산 인과관계가 아니다."
    }
    with (OUT / "issue_analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger("[3/4] 지역·월별 통계와 원문 근거 CSV 저장", start)
    logger(f"[4/4] 완료 | {total_sec:.1f}초 | {OUT}", start)
    print("\n=== 이슈별 청크 건수(중복 이슈 가능) ===")
    for issue in ISSUE_RULES:
        print(f"{issue:10s}: {issue_total[issue]:,}")
    print("결과:", OUT)


if __name__ == "__main__":
    main()

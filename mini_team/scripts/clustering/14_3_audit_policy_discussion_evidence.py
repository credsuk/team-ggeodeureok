# -*- coding: utf-8 -*-
"""
14-3: 유사 정책 후보 60쌍 근거 보강 / 중복 검토 / 제출 후보 목록
- 기존 14-2의 원본 후보를 수정하지 않음
- 12단계 SQLite 원문 캐시로 원문 발췌 근거 점검
- 14-2 임베딩 캐시(metadata_json) -> Chroma 메타데이터 순서로 회의록명 확인
- 모든 후보에 중복/형식적 발언 의심 표시. 자동 판단으로 전파 확정하지 않음
- 재실행 시 캐시 재사용. 단계별 소요시간 출력.
"""
from __future__ import annotations
import csv, hashlib, json, re, sqlite3, time
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE = Path(r"C:\mini_team\scripts\clustering")
SOURCE = BASE / "results/policy_discussion_links_refined/refined_policy_discussion_candidates.csv"
VECTOR_CACHE = BASE / "results/policy_discussion_links_refined/candidate_embeddings.sqlite3"
TEXT_CACHE = BASE / "results/all_policy_topics/policy_text_cache.sqlite3"
CHROMA = Path(r"C:\mini_team\_db\chroma_snowflake_regions")
COLLECTION = "team_regions_6a0a006e92d1e93e"
OUT = BASE / "results/policy_discussion_evidence"
CACHE = OUT / "meeting_metadata_cache.sqlite3"
META_BATCH = 80

NAME_KEYS = ("meeting_title", "meeting_name", "session_title", "meeting",
             "title", "회의록명", "회의명")
# 형식 문구는 후보 제외가 아니라 검토 플래그에만 사용.
FORMAT_TERMS = (
    "세입세출 결산", "결산 승인안", "원안대로 가결", "이의 없으십니까",
    "안건을 상정", "행정사무감사에 앞서", "증인 선서",
    "위원님 여러분", "의사일정 제", "예산안 제안설명", "간부공무원 소개"
)
MISSING = {"", "none", "null", "nan"}

def say(s, t):
    print(f"[{datetime.now():%H:%M:%S}] {s} | 경과 {time.monotonic()-t:.1f}초", flush=True)

def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()

def norm_date(s):
    s = str(s or "").strip()
    m = re.search(r"(20\d{2})[-./]?(\d{2})[-./]?(\d{2})", s)
    return "-".join(m.groups()) if m else ""

def hash_text(s):
    return hashlib.sha1(re.sub(r"[\W_]+", "", norm(s).lower()).encode()).hexdigest()

def csv_write(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def pick_meta_title(meta):
    if not isinstance(meta, dict):
        return ""
    for key in NAME_KEYS:
        val = meta.get(key)
        if val is not None and norm(val).lower() not in MISSING:
            return norm(val)
    # No fabricated title: source_file can be a useful trace even without meeting title.
    return ""

def load_texts(ids, t):
    if not TEXT_CACHE.is_file():
        raise FileNotFoundError(f"12단계 원문 캐시 없음: {TEXT_CACHE}")
    con = sqlite3.connect(f"file:{TEXT_CACHE.as_posix()}?mode=ro", uri=True)
    out = {}
    for offset in range(0, len(ids), 300):
        subset = ids[offset:offset+300]
        ph = ",".join("?" for _ in subset)
        for r in con.execute(
                f"SELECT chunk_id, document, assembly, source_key, date "
                f"FROM policy_docs WHERE chunk_id IN ({ph})", subset):
            out[r[0]] = dict(document=r[1] or "", assembly=r[2] or "",
                             source_key=r[3] or "", date=r[4] or "")
    con.close()
    say(f"[1/5] 캐시 원문 확보 {len(out)}/{len(ids)}", t)
    return out

def load_vector_metadata(ids):
    data = {}
    if not VECTOR_CACHE.is_file():
        return data
    con = sqlite3.connect(f"file:{VECTOR_CACHE.as_posix()}?mode=ro", uri=True)
    try:
        columns = {r[1] for r in con.execute("PRAGMA table_info(vectors)")}
        if not {"chunk_id", "metadata_json"}.issubset(columns):
            return data
        for off in range(0, len(ids), 300):
            sub = ids[off:off+300]
            ph = ",".join("?" for _ in sub)
            query = ("SELECT chunk_id, metadata_json, meeting_title FROM vectors "
                     if "meeting_title" in columns else
                     "SELECT chunk_id, metadata_json, NULL FROM vectors ")
            for chunk_id, raw, title in con.execute(query + f"WHERE chunk_id IN ({ph})", sub):
                try:
                    meta = json.loads(raw) if raw else {}
                except (ValueError, TypeError):
                    meta = {}
                data[chunk_id] = (meta, norm(title) or pick_meta_title(meta))
    finally:
        con.close()
    return data

def open_meta_cache():
    con = sqlite3.connect(CACHE)
    con.execute("""CREATE TABLE IF NOT EXISTS metadata (
        chunk_id TEXT PRIMARY KEY, raw_json TEXT NOT NULL, status TEXT NOT NULL)""")
    con.commit()
    return con

def metadata_for_ids(ids, t):
    cached_con = open_meta_cache()
    result = load_vector_metadata(ids)
    initial = len(result)
    for offset in range(0, len(ids), 300):
        sub = ids[offset:offset+300]
        marks = ",".join("?" for _ in sub)
        for cid, raw, status in cached_con.execute(
            f"SELECT chunk_id, raw_json, status FROM metadata WHERE chunk_id IN ({marks})", sub):
            if cid not in result:
                try:
                    result[cid] = (json.loads(raw), "")
                except (ValueError, TypeError):
                    result[cid] = ({}, "")
    missing = [id_ for id_ in ids if id_ not in result]
    say(f"[2/5] 기존 임베딩 캐시 메타데이터 {initial}개 / 추가 조회 대상 {len(missing)}개", t)
    if missing:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(CHROMA))
            col = client.get_collection(name=COLLECTION)
            for off in range(0, len(missing), META_BATCH):
                chunk = missing[off:off+META_BATCH]
                got = col.get(ids=chunk, include=["metadatas"])
                returned = set()
                for cid, meta in zip(got.get("ids") or [], got.get("metadatas") or []):
                    meta = meta or {}
                    returned.add(cid)
                    result[cid] = (meta, pick_meta_title(meta))
                    cached_con.execute(
                        "INSERT OR REPLACE INTO metadata VALUES (?,?,?)",
                        (cid, json.dumps(meta, ensure_ascii=False, default=str), "found"))
                for cid in set(chunk)-returned:
                    result[cid] = ({}, "")
                    cached_con.execute(
                        "INSERT OR REPLACE INTO metadata VALUES (?,?,?)",
                        (cid, "{}", "not_found"))
                cached_con.commit()
            say(f"[2/5] Chroma 메타데이터 조회 완료 {len(missing)}개", t)
        except Exception as ex:
            say(f"[주의] Chroma 메타데이터 추가조회 실패: {type(ex).__name__}: {ex}", t)
            # Keep proceeding using available cached metadata.
    cached_con.close()
    return result

def evidence_status(saved, full):
    if not full:
        return "원문캐시_누락"
    sample = norm(saved)
    if not sample:
        return "근거발췌_빈칸"
    if sample in norm(full):
        return "원문발췌_일치"
    # Full text may have different whitespace
    strip = re.sub(r"\s+", "", str(full))
    if re.sub(r"\s+", "", sample) in strip:
        return "공백제거후_일치"
    return "원문발췌_불일치_확인필요"

def format_flags(s):
    return [term for term in FORMAT_TERMS if term in s]

def pair_key(r):
    return (r["issue"], r["earlier_chunk_id"], r["later_chunk_id"])

def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    if not SOURCE.is_file():
        raise FileNotFoundError(f"14-2 결과 없음: {SOURCE}")
    with SOURCE.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        original_cols = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError("14-2 후보 CSV가 비어 있습니다.")
    say(f"14-3 시작: 후보 {len(rows)}쌍", start)

    ids = sorted({r[k] for r in rows for k in ("earlier_chunk_id", "later_chunk_id") if r.get(k)})
    docs = load_texts(ids, start)
    metas = metadata_for_ids(ids, start)

    enriched = []
    for row in rows:
        r = dict(row)
        problems = []
        for side in ("earlier", "later"):
            cid = r[f"{side}_chunk_id"]
            doc = docs.get(cid, {})
            full = doc.get("document", "")
            meta, meta_title = metas.get(cid, ({}, ""))
            title = norm(r.get(f"{side}_meeting_title")) or meta_title
            r[f"{side}_meeting_title"] = title
            r[f"{side}_source_file"] = norm(meta.get("source_file", ""))
            r[f"{side}_meeting_metadata_status"] = (
                "회의록명_확인후원본대조필요" if title else "회의록명_없음_원본조회필요")
            r[f"{side}_evidence_check"] = evidence_status(r.get(f"{side}_evidence_180"), full)
            r[f"{side}_original_text_characters"] = len(full)
            r[f"{side}_format_flags"] = "; ".join(format_flags(full[:1800]))
            r[f"{side}_full_text_fingerprint"] = hash_text(full) if full else ""
            if not title: problems.append(f"{side}:회의록명누락")
            if "불일치" in r[f"{side}_evidence_check"] or "누락" in r[f"{side}_evidence_check"]:
                problems.append(f"{side}:근거검증필요")
            if r[f"{side}_format_flags"]: problems.append(f"{side}:의사진행형식의심")
            if doc and norm_date(doc.get("date")) and norm_date(doc["date"]) != norm_date(r.get(f"{side}_date")):
                problems.append(f"{side}:날짜불일치")
            if doc and doc.get("assembly") and doc["assembly"] != r.get(f"{side}_assembly"):
                problems.append(f"{side}:의회불일치")
        if norm_date(r.get("earlier_date")) >= norm_date(r.get("later_date")):
            problems.append("날짜선후오류")
        if r["earlier_assembly"] == r["later_assembly"]:
            problems.append("동일의회")
        if r["earlier_full_text_fingerprint"] == r["later_full_text_fingerprint"]:
            problems.append("양측원문중복")
        r["audit_flags"] = "; ".join(problems)
        r["human_policy_similarity"] = ""
        r["human_direct_reference"] = ""
        r["human_review_notes"] = ""
        r["diffusion_confirmed"] = "아니오"
        enriched.append(r)
    say("[3/5] 원문·회의록 메타데이터 대조 완료", start)

    # 1:1 same pair, then near-identical text within same issue / same council-pair.
    # Conservative exact fingerprints only: do not silently drop distinct policies.
    seen_pairs = {}
    seen_content = {}
    kept, duplicates = [], []
    for r in sorted(enriched,
                    key=lambda x: (-float(x.get("combined_ranking_score") or 0),
                                   x["issue"], x["earlier_date"])):
        exact = pair_key(r)
        content = (r["issue"], r["earlier_assembly"], r["later_assembly"],
                   r["earlier_full_text_fingerprint"], r["later_full_text_fingerprint"])
        why = None
        if exact in seen_pairs:
            why = "동일_청크ID쌍"
        elif all(content[-2:]) and content in seen_content:
            why = "동일_원문쌍"
        if why:
            r["duplicate_reason"] = why
            duplicates.append(r)
        else:
            r["duplicate_reason"] = ""
            kept.append(r)
            seen_pairs[exact] = r
            if all(content[-2:]): seen_content[content] = r
    say(f"[4/5] 중복 {len(duplicates)}쌍 / 고유 후보 {len(kept)}쌍", start)

    extra_cols = [
        "earlier_source_file", "later_source_file",
        "earlier_meeting_metadata_status", "later_meeting_metadata_status",
        "earlier_evidence_check", "later_evidence_check",
        "earlier_original_text_characters", "later_original_text_characters",
        "earlier_format_flags", "later_format_flags",
        "earlier_full_text_fingerprint", "later_full_text_fingerprint",
        "audit_flags", "duplicate_reason",
        "human_policy_similarity", "human_direct_reference", "human_review_notes",
        "diffusion_confirmed"
    ]
    fields = original_cols + [x for x in extra_cols if x not in original_cols]
    csv_write(OUT / "policy_links_audited_all.csv", enriched, fields)
    csv_write(OUT / "policy_links_unique.csv", kept, fields)
    csv_write(OUT / "policy_links_duplicates.csv", duplicates, fields)
    # candidate-only shortlist: does not assert human validation or verified diffusion.
    clean = [r for r in kept
             if not r["earlier_format_flags"] and not r["later_format_flags"]
             and not any(term in r["audit_flags"] for term in
                         ("원문중복", "근거검증필요", "날짜선후오류", "동일의회", "날짜불일치", "의회불일치"))]
    csv_write(OUT / "policy_links_priority_review.csv", clean, fields)
    issue_summary = []
    for issue in sorted(set(r["issue"] for r in rows)):
        issue_summary.append({
            "issue": issue,
            "original": sum(r["issue"] == issue for r in enriched),
            "unique": sum(r["issue"] == issue for r in kept),
            "priority_review": sum(r["issue"] == issue for r in clean),
            "duplicates": sum(r["issue"] == issue for r in duplicates),
        })
    csv_write(OUT / "policy_link_audit_by_issue.csv", issue_summary,
              ["issue", "original", "unique", "priority_review", "duplicates"])
    summary = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "input_pairs": len(rows), "unique_source_chunk_ids": len(ids),
        "text_cache_found": len(docs),
        "metadata_available": len(metas),
        "meeting_titles_found_for_sides": sum(bool(r[f"{side}_meeting_title"])
                                               for r in kept for side in ("earlier", "later")),
        "duplicate_pairs": len(duplicates),
        "unique_pairs": len(kept),
        "priority_review_candidates": len(clean),
        "flags_count": dict(Counter(flag.strip() for r in enriched
                            for flag in r["audit_flags"].split(";") if flag.strip())),
        "issue_counts": issue_summary,
        "elapsed_sec": round(time.monotonic()-start, 2),
        "caution": ("AI/휴리스틱 후보이며 검증된 정책 전파가 아님. "
                    "회의록명은 Chroma metadata에 없으면 비워둠. "
                    "원본 JSON/HTML에서 회의록명을 추가 매칭해야 할 수 있음.")
    }
    with (OUT / "policy_link_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    say(f"[5/5] 완료 | 고유 {len(kept)} / 우선검토 {len(clean)} | {OUT}", start)

if __name__ == "__main__":
    main()

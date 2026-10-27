# -*- coding: utf-8 -*-
"""14-4 | Restore verifiable meeting names for policy-discussion evidence.

Requires stage 14-3 policy_links_unique.csv (priority-only fallback accepted).
Run:
  python 14_4_restore_meeting_metadata.py
  python 14_4_restore_meeting_metadata.py --source-root "C:\\mini_team\\data\\processed"
  python 14_4_restore_meeting_metadata.py --source-root "D:\\council_json" --source-root "E:\\meeting_exports"

Searches ORIGINAL JSON/JSONL/CSV/TXT/HTML selectively using evidence snippets;
no fabricated meeting titles, no Chroma writes, no full re-embedding.
If original fi][-les are not supplied, produces a manual metadata-mapping template.
"""
import argparse
import csv
import hashlib
import html
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:\mini_team\scripts\clustering")
AUDIT_DIR = ROOT / "results" / "policy_discussion_evidence"
OUT_DIR = ROOT / "results" / "policy_meeting_metadata"
TEXT_CACHE = ROOT / "results" / "all_policy_topics" / "policy_text_cache.sqlite3"

# Optionally fill these with directories containing *individual original meeting records*.
# Example: Path(r"D:\mini_team\data\raw_json")
SOURCE_ROOTS = []

NAME_KEYS = ("meeting_title", "meeting_name", "meetingTitle", "meetingName",
             "session_title", "session_name", "title", "회의록명", "회의명",
             "회의제목", "회의록제목", "회의명칭")
DATE_KEYS = ("date", "meeting_date", "meetingDate", "회의일자", "회의날짜", "일시")
ASSEMBLY_KEYS = ("assembly", "council", "assembly_name", "의회", "의회명")
DOC_KEYS = ("utterance", "speech", "body", "text", "content", "html", "본문", "발언", "발언내용")
KEYS_TO_IGNORE = {"embedding", "vector", "vectors", "tokens", "ids", "metadata"}
EXTENSIONS = {".json", ".jsonl", ".ndjson", ".csv", ".txt", ".html", ".htm"}
MAX_FILE_MB = 150  # avoid accidentally loading giant JSON dumps in memory
MAX_RECORDS_PER_FILE = 200000
MAX_RECURSION = 12
STRONG_ANCHOR_CHARS = 58


def log(msg, start):
    print(f"[{datetime.now():%H:%M:%S}] {msg} | 경과 {time.perf_counter()-start:.1f}초", flush=True)


def clean(s):
    s = html.unescape(str(s or ""))
    s = re.sub(r"<[^>]*>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def compact(s):
    return re.sub(r"\s+", "", clean(s))


def date_norm(s):
    m = re.search(r"(20\d{2})\D?(\d{2})\D?(\d{2})", str(s or ""))
    return "-".join(m.groups()) if m else ""


def nonempty(obj, keys):
    if not isinstance(obj, dict):
        return ""
    for k in keys:
        v = obj.get(k)
        if isinstance(v, (str, int, float)) and clean(v):
            return clean(v)
    return ""


def input_csv():
    for name in ("policy_links_unique.csv", "policy_links_priority_review.csv",
                 "policy_links_audited_all.csv"):
        p = AUDIT_DIR / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"14-3 결과를 찾을 수 없습니다: {AUDIT_DIR}")


def load_candidates(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        cols = list(rd.fieldnames or [])
        rows = list(rd)
    if not rows:
        raise ValueError(f"CSV가 비어 있습니다: {path}")
    ids = sorted({r.get(side + "_chunk_id", "") for r in rows
                  for side in ("earlier", "later") if r.get(side + "_chunk_id")})
    return cols, rows, ids


def get_cache_text(ids):
    if not TEXT_CACHE.exists():
        raise FileNotFoundError(f"원문 캐시 없음: {TEXT_CACHE}")
    db = sqlite3.connect(f"file:{TEXT_CACHE.as_posix()}?mode=ro", uri=True)
    docs = {}
    for i in range(0, len(ids), 200):
        sub = ids[i:i+200]
        query = "SELECT chunk_id, document, assembly, source_key, date FROM policy_docs WHERE chunk_id IN (" + ",".join("?" for _ in sub) + ")"
        for cid, txt, assembly, key, day in db.execute(query, sub):
            docs[cid] = {"document": txt or "", "assembly": assembly or "",
                         "source_key": key or "", "date": date_norm(day)}
    db.close()
    return docs


def anchor_segments(s):
    s = compact(s)
    if len(s) < 35:
        return []
    pieces = []
    for f in (0.05, 0.40, 0.7):
        pos = int(max(0, len(s) - STRONG_ANCHOR_CHARS) * f)
        a = s[pos:pos+STRONG_ANCHOR_CHARS]
        if len(a) >= 35 and a not in pieces:
            pieces.append(a)
    return pieces


def get_targets(rows, docs):
    targets = {}
    for row in rows:
        for side in ("earlier", "later"):
            cid = row.get(side + "_chunk_id", "")
            if not cid or cid in targets:
                continue
            doc = docs.get(cid, {})
            raw = doc.get("document", "")
            anchor = row.get(side + "_evidence_180", "")
            # Prefer explicit evidence, but use full cache for additional anchors.
            fragments = anchor_segments(anchor)
            if not fragments:
                fragments = anchor_segments(raw)
            key = row.get(side + "_source_key", "") or doc.get("source_key", "")
            targets[cid] = {
                "chunk_id": cid, "source_key": key,
                "date": date_norm(row.get(side + "_date")) or doc.get("date", ""),
                "assembly": row.get(side + "_assembly", "") or doc.get("assembly", ""),
                "anchors": fragments, "full_text": raw,
            }
    return targets


def text_fields(record):
    if isinstance(record, str):
        return [record]
    if not isinstance(record, dict):
        return []
    vals = []
    for k in DOC_KEYS:
        v = record.get(k)
        if isinstance(v, str) and len(v) >= 35:
            vals.append(v)
    return vals


def nested_records(node, inherited=None, depth=0):
    if depth > MAX_RECURSION:
        return
    inherited = inherited or {}
    if isinstance(node, dict):
        title = nonempty(node, NAME_KEYS) or inherited.get("title", "")
        date = date_norm(nonempty(node, DATE_KEYS)) or inherited.get("date", "")
        assembly = nonempty(node, ASSEMBLY_KEYS) or inherited.get("assembly", "")
        ctx = {"title": title, "date": date, "assembly": assembly}
        texts = text_fields(node)
        if texts:
            for value in texts:
                yield value, ctx
        for k, v in node.items():
            if k in KEYS_TO_IGNORE or k in DOC_KEYS:
                continue
            if isinstance(v, (dict, list)):
                yield from nested_records(v, ctx, depth+1)
    elif isinstance(node, list):
        for val in node:
            yield from nested_records(val, inherited, depth+1)


def read_records(path):
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        with path.open(encoding="utf-8-sig", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= MAX_RECORDS_PER_FILE:
                    break
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                yield from nested_records(obj)
    elif suffix == ".csv":
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if i >= MAX_RECORDS_PER_FILE:
                    break
                yield from nested_records(row)
    elif suffix == ".json":
        with path.open(encoding="utf-8-sig", errors="replace") as f:
            obj = json.load(f)
        yield from nested_records(obj)
    else:
        # For text/html, title isn't inferred from filename. Prove exact original
        # fragment match first; user can subsequently enrich manually.
        with path.open(encoding="utf-8-sig", errors="replace") as f:
            value = f.read()
        yield value, {"title": "", "date": "", "assembly": ""}


def scan_originals(roots, targets, start):
    matched = defaultdict(list)
    if not roots:
        log("원본 자료 경로가 지정되지 않아 자동 복원은 건너뜀", start)
        return matched, {"scanned_files": 0, "skipped_large": 0, "errors": []}

    # Filter by known source_key's year+council, when filename provides clues.
    by_group = defaultdict(list)
    for t in targets.values():
        by_group[t["source_key"]].append(t)
    all_targets = list(targets.values())
    files = sorted({p for root in roots for p in root.rglob("*")
                    if p.is_file() and p.suffix.lower() in EXTENSIONS})
    visited = 0
    skipped_large = 0
    errors = []
    for path in files:
        try:
            size = path.stat().st_size
            if size > MAX_FILE_MB * 1024 * 1024:
                skipped_large += 1
                continue
            stem = path.stem
            relevant = [t for t in all_targets
                        if t["assembly"] and t["assembly"] in str(path)]
            if not relevant:
                relevant = all_targets
            # For potentially shared source files, require evidence matching.
            for recno, (txt, metadata) in enumerate(read_records(path)):
                if not isinstance(txt, str) or len(txt) < 35:
                    continue
                body = compact(txt)
                if len(body) < 35:
                    continue
                for t in relevant:
                    if not t["anchors"]:
                        continue
                    count = sum(a in body for a in t["anchors"])
                    if not count:
                        continue
                    dated = metadata.get("date", "")
                    # If an explicit date conflicts, keep the evidence as a
                    # conflict (never silently label as verified).
                    conflict = bool(dated and t["date"] and dated != t["date"])
                    title = metadata.get("title", "")
                    assembly = metadata.get("assembly", "")
                    assembly_conflict = bool(assembly and t["assembly"] and assembly != t["assembly"])
                    matched[t["chunk_id"]].append({
                        "source_path": str(path), "record_index": recno,
                        "meeting_title": title, "meeting_date": dated,
                        "assembly": assembly, "matched_anchors": count,
                        "date_conflict": conflict,
                        "assembly_conflict": assembly_conflict,
                        "evidence_confirmed": bool(count >= 2),
                    })
            visited += 1
            if visited % 100 == 0:
                log(f"원본 파일 검토 {visited:,}/{len(files):,}", start)
        except Exception as e:
            errors.append({"path": str(path), "error": str(e)[:200]})
    log(f"원본 검색 완료 | 검사 {visited:,} / 후보 매칭 ID {len(matched):,}", start)
    return matched, {"scanned_files": visited, "skipped_large": skipped_large, "errors": errors[:100]}


def choose_matches(matches, target):
    valid = [m for m in matches if not m["date_conflict"] and not m["assembly_conflict"]
             and m["meeting_title"] and m["matched_anchors"] >= 2]
    # Multiple references to the same meeting title/date are fine, but
    # conflicting title/date combinations are not auto-resolved.
    distinct = {(m["meeting_title"], m["meeting_date"]) for m in valid}
    if len(distinct) == 1:
        best = max(valid, key=lambda m: m["matched_anchors"])
        return best, "자동매칭_원문및메타데이터확인_최종원본대조필요"
    if len(distinct) > 1:
        return {}, "다중원본후보_검토필요"
    if matches:
        return {}, "근거일부일치_회의명검증필요"
    return {}, "원본매칭없음_수동확인필요"


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", action="append", default=[],
                   help="원본 JSON/JSONL/CSV/HTML 저장 폴더 (복수 지정 가능)")
    p.add_argument("--max-file-mb", type=int, default=MAX_FILE_MB,
                   help="매우 큰 JSON 파일은 건너뛰므로 원본을 나누거나 적절히 상향")
    args = p.parse_args()
    # Update scanning limit from CLI without modifying original files.
    globals()["MAX_FILE_MB"] = args.max_file_mb
    start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inp = input_csv()
    columns, rows, ids = load_candidates(inp)
    docs = get_cache_text(ids)
    targets = get_targets(rows, docs)
    log(f"14-4 시작 | 후보 {len(rows)}쌍 / 고유 청크 {len(ids)} / 캐시 원문 {len(docs)}", start)
    roots = [Path(x) for x in args.source_root] or SOURCE_ROOTS
    roots = [Path(x) for x in roots]
    for root in roots:
        if not root.is_dir():
            raise NotADirectoryError(f"원본 경로 오류: {root}")
    matches, scaninfo = scan_originals(roots, targets, start)
    resolved = {}
    for cid, t in targets.items():
        choice, status = choose_matches(matches.get(cid, []), t)
        resolved[cid] = (choice, status)

    details = []
    result = []
    for r in rows:
        copy = dict(r)
        for side in ("earlier", "later"):
            cid = r.get(side + "_chunk_id", "")
            chosen, status = resolved.get(cid, ({}, "ID누락"))
            title = chosen.get("meeting_title", "")
            # Prefer original source title from exact matching, never invent.
            if title:
                copy[f"{side}_meeting_title"] = title
            copy[f"{side}_meeting_restore_status"] = status
            copy[f"{side}_original_path"] = chosen.get("source_path", "")
            copy[f"{side}_original_record_index"] = chosen.get("record_index", "")
            copy[f"{side}_matching_anchors"] = chosen.get("matched_anchors", "")
            copy[f"{side}_metadata_source"] = "원본_본문증거매칭" if title else "미확인"
            details.append({
                "chunk_id": cid, "side": side, "issue": r.get("issue", ""),
                "assembly": r.get(f"{side}_assembly", ""),
                "date": r.get(f"{side}_date", ""),
                "meeting_title": copy.get(f"{side}_meeting_title", ""),
                "status": status, "original_path": copy[f"{side}_original_path"],
                "record_index": copy[f"{side}_original_record_index"],
                "matched_anchors": copy[f"{side}_matching_anchors"],
            })
        copy["submission_evidence_status"] = (
            "회의록명_양측복원_원본대조필요"
            if all(copy.get(f"{s}_meeting_title") for s in ("earlier", "later"))
            else "회의록명_미완성_제출보류"
        )
        result.append(copy)

    extras = [f"{side}_{field}" for side in ("earlier", "later")
              for field in ("meeting_restore_status", "original_path", "original_record_index",
                            "matching_anchors", "metadata_source")]
    augmented_columns = columns + [c for c in extras if c not in columns] + ["submission_evidence_status"]
    write_csv(OUT_DIR / "policy_links_meeting_metadata_restored.csv", result, augmented_columns)
    manual = {}
    for rec in details:
        manual.setdefault(rec["chunk_id"], rec)
    manual_rows = sorted(manual.values(), key=lambda x: (x["issue"], x["assembly"], x["date"]))
    for r in manual_rows:
        r["manual_meeting_title"] = ""
        r["manual_original_file"] = ""
        r["manual_source_url"] = ""
        r["manual_verification_note"] = ""
    write_csv(OUT_DIR / "meeting_title_manual_mapping_template.csv", manual_rows,
              ["chunk_id", "side", "issue", "assembly", "date", "meeting_title",
               "status", "original_path", "record_index", "matched_anchors",
               "manual_meeting_title", "manual_original_file", "manual_source_url", "manual_verification_note"])
    ambiguous = []
    for cid, vals in matches.items():
        for v in vals:
            ambiguous.append({"chunk_id": cid, **v})
    write_csv(OUT_DIR / "original_match_candidates.csv", ambiguous,
              ["chunk_id", "source_path", "record_index", "meeting_title",
               "meeting_date", "assembly", "matched_anchors", "date_conflict",
               "assembly_conflict", "evidence_confirmed"])
    summary = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "input_file": str(inp), "input_pairs": len(rows), "unique_chunk_ids": len(ids),
        "cache_text_found": len(docs), "original_roots": [str(r) for r in roots],
        "confirmed_title_candidates": sum(bool(resolved[c][0].get("meeting_title")) for c in resolved),
        "unresolved_chunk_ids": sum(not bool(resolved[c][0].get("meeting_title")) for c in resolved),
        "pairs_with_both_titles": sum(x["submission_evidence_status"].startswith("회의록명_양측") for x in result),
        "match_candidate_rows": len(ambiguous), "scan": scaninfo,
        "elapsed_sec": round(time.perf_counter()-start, 2),
        "caution": "자동매칭은 제출 확정이 아님. 동일 발언/예산 문구의 반복 가능성 있어 최종 원본 대조 필요."
    }
    with (OUT_DIR / "meeting_metadata_restore_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"완료 | 제목 복원 후보 {summary['confirmed_title_candidates']}/{len(ids)} | 양측 복원 {summary['pairs_with_both_titles']}/{len(rows)}", start)
    print("결과:", OUT_DIR)
    if not roots:
        print("주의: --source-root 미지정. 원본 JSON 폴더 경로를 지정하면 회의록명 자동 매칭을 시도합니다.")


if __name__ == "__main__":
    main()

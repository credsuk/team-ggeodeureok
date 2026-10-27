"""11단계: 전체 ChromaDB 청크 -> 정책 관련성 분류 -> POLICY에만 K=100 군집 할당.

- 원본 ChromaDB와 학습 모델은 읽기 전용으로 취급합니다.
- 결과 SQLite에 배치 단위로 저장하고 마지막 페이지 offset을 원자적으로 체크포인트합니다.
- 중단 후 같은 파일로 다시 실행하면 체크포인트부터 이어집니다.
- REVIEW/PROCEDURAL/MISSING은 cluster_id를 NULL로 보존합니다.
- 중요: ChromaDB가 실행 도중 수정되면 offset 페이지 탐색의 일관성이 깨질 수 있으므로
  이 스크립트 실행 중에는 DB 임베딩 추가/삭제/수정을 하지 마세요.

실행: python 11_assign_all_clusters.py
출력: C:\\mini_team\\scripts\\clustering\\results\\all_cluster_assignment\\
"""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import chromadb
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

# 같은 폴더의 07단계 정책 필터 규칙을 그대로 사용합니다 (규칙 불일치 방지).
from importlib.util import spec_from_file_location, module_from_spec

ROOT = Path(r"C:\mini_team\scripts\clustering")
DB_PATH = Path(r"C:\mini_team\_db\chroma_snowflake_regions")
COLLECTION = "team_regions_6a0a006e92d1e93e"
MODEL_PATH = ROOT / "results" / "policy_kmeans" / "policy_minibatch_kmeans_k100.joblib"
REFINE_SCRIPT = ROOT / "07_refine_policy_filter.py"
OUT = ROOT / "results" / "all_cluster_assignment"
STATE_DB = OUT / "all_cluster_assignments.sqlite3"
BATCH_SIZE = 250
LOG_EVERY_BATCHES = 20             # 5,000개마다 진행상황
THREADS = 4
EXPORT_CSV_AT_END = True           # 완주 후 약 218만 줄 CSV 파일 생성
CSV_CHUNK_SIZE = 20000
FIELDS = ["chunk_id", "source_key", "date", "assembly", "refined_label",
          "cluster_id", "policy_score", "procedural_hits", "policy_hits",
          "strong_hits", "char_length", "is_substantive", "reason"]


def stamp(msg, start):
    print(f"[{datetime.now():%H:%M:%S}] {msg} | 경과 {time.perf_counter()-start:.1f}초", flush=True)


def load_refine():
    if not REFINE_SCRIPT.is_file():
        raise FileNotFoundError(f"07단계 파일이 없습니다: {REFINE_SCRIPT}")
    spec = spec_from_file_location("refine_policy_stage07", REFINE_SCRIPT)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "refine"):
        raise RuntimeError("07단계 refine(text, substantive) 함수를 찾지 못했습니다")
    return mod.refine


def init_state(conn, expected_count, model_path):
    conn.execute("""CREATE TABLE IF NOT EXISTS progress
      (key TEXT PRIMARY KEY, value TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS assignments (
      chunk_id TEXT PRIMARY KEY,
      source_key TEXT, date TEXT, assembly TEXT,
      refined_label TEXT NOT NULL,
      cluster_id INTEGER,
      policy_score INTEGER, procedural_hits INTEGER,
      policy_hits INTEGER, strong_hits INTEGER,
      char_length INTEGER, is_substantive TEXT, reason TEXT
    )""")
    existing = dict(conn.execute("SELECT key,value FROM progress"))
    if existing:
        if int(existing.get("collection_count", -1)) != expected_count:
            raise RuntimeError("체크포인트의 전체 DB 개수와 현재 개수가 다릅니다. 데이터 변경 여부 확인하세요")
        if existing.get("model_path") != str(model_path.resolve()):
            raise RuntimeError("기존 체크포인트와 사용 모델 경로가 다릅니다. 기존 결과 재사용 불가")
        if int(existing.get("batch_size", -1)) != BATCH_SIZE:
            raise RuntimeError("기존 실행과 BATCH_SIZE가 다릅니다. 배치 크기를 원래 값으로 돌리세요")
    else:
        with conn:
            conn.executemany("INSERT INTO progress(key,value) VALUES (?,?)", [
                ("collection_count", str(expected_count)),
                ("model_path", str(model_path.resolve())),
                ("batch_size", str(BATCH_SIZE)),
                ("next_offset", "0"),
                ("started_at", datetime.now().isoformat(timespec="seconds"))
            ])
    return int(dict(conn.execute("SELECT key,value FROM progress"))["next_offset"])


def get_page(coll, offset):
    return coll.get(limit=BATCH_SIZE, offset=offset,
                    include=["embeddings", "documents", "metadatas"])


def page_to_rows(page, model, refine):
    ids = page.get("ids") or []
    if not ids:
        return []
    emb = page.get("embeddings")
    if emb is None or len(emb) != len(ids):
        raise ValueError(f"조회한 {len(ids)}개 청크에서 임베딩이 누락되었습니다")
    x = np.asarray(emb, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != model.n_features_in_ or not np.isfinite(x).all():
        raise ValueError(f"임베딩 차원 또는 수치 오류: {x.shape}")
    docs = page.get("documents")
    metas = page.get("metadatas")
    if docs is None: docs = [None] * len(ids)
    if metas is None: metas = [None] * len(ids)
    if len(docs) != len(ids) or len(metas) != len(ids):
        raise ValueError("본문 또는 메타데이터 개수가 ID와 다릅니다")

    rows, policy_positions = [], []
    for idx, (cid, doc, meta) in enumerate(zip(ids, docs, metas)):
        meta = meta or {}
        label, score, proc, noun, strong, reason = refine(doc, meta.get("is_substantive"))
        if label == "POLICY":
            policy_positions.append(idx)
        skey = str(meta.get("source_key") or "")
        date = str(meta.get("date") or "")
        assembly = str(meta.get("assembly") or "")
        if not skey and assembly and date[:4].isdigit():
            skey = date[:4] + "_" + assembly
        rows.append((str(cid), skey, date, assembly, label, None, int(score),
                     int(proc), int(noun), int(strong), len(str(doc or "").strip()),
                     str(meta.get("is_substantive", "")), reason))
    if policy_positions:
        with threadpool_limits(limits=THREADS):
            labels = model.predict(np.ascontiguousarray(x[policy_positions]))
        for j, cluster in zip(policy_positions, labels):
            row = list(rows[j])
            row[5] = int(cluster)
            rows[j] = tuple(row)
    return rows


def save_batch(conn, rows, offset):
    with conn:
        conn.executemany("""INSERT INTO assignments (
          chunk_id,source_key,date,assembly,refined_label,cluster_id,policy_score,
          procedural_hits,policy_hits,strong_hits,char_length,is_substantive,reason)
          VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
        conn.execute("UPDATE progress SET value=? WHERE key='next_offset'", (str(offset),))


def export_results(conn, total, started):
    stamp("[4/4] 결과 집계 및 CSV 내보내기", started)
    counts = dict(conn.execute("SELECT refined_label, COUNT(*) FROM assignments GROUP BY refined_label"))
    n = conn.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
    if n != total:
        raise RuntimeError(f"총 {total:,}개 중 {n:,}개만 저장됨. 전체 CSV 생성을 중단합니다")
    assigned = conn.execute("SELECT COUNT(*) FROM assignments WHERE cluster_id IS NOT NULL").fetchone()[0]
    if assigned != counts.get("POLICY", 0):
        raise RuntimeError("POLICY 개수와 군집 할당 개수 불일치")
    print("\n=== 전체 정책 필터 및 군집 할당 ===")
    for label in ("POLICY", "REVIEW", "PROCEDURAL", "MISSING"):
        v = counts.get(label, 0)
        print(f"{label:<12}: {v:>10,} ({v/total*100:5.2f}%)")
    print(f"군집이 부여된 청크: {assigned:,}개")

    with (OUT / "all_cluster_sizes_k100.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "policy_count"])
        sizes = dict(conn.execute("SELECT cluster_id, COUNT(*) FROM assignments WHERE cluster_id IS NOT NULL GROUP BY cluster_id"))
        for i in range(100):
            w.writerow([i, sizes.get(i, 0)])

    # source_key 없는 메타데이터가 얼마나 있는지 별도 확인
    with (OUT / "all_policy_filter_by_region.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_key", "refined_label", "count"])
        w.writerows(conn.execute("SELECT source_key, refined_label, COUNT(*) FROM assignments GROUP BY source_key,refined_label ORDER BY source_key,refined_label"))

    if EXPORT_CSV_AT_END:
        dest = OUT / "all_cluster_assignments_k100.csv"
        tmp = dest.with_suffix(".csv.part")
        with tmp.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FIELDS)
            cur = conn.execute("SELECT " + ",".join(FIELDS) + " FROM assignments ORDER BY rowid")
            exported = 0
            while True:
                rows = cur.fetchmany(CSV_CHUNK_SIZE)
                if not rows: break
                writer.writerows(rows)
                exported += len(rows)
                if exported % 200000 < CSV_CHUNK_SIZE or exported == total:
                    stamp(f"결과 CSV {exported:,}/{total:,}", started)
        if exported != total:
            raise RuntimeError("CSV 행 개수 불일치")
        tmp.replace(dest)
    summary = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "collection_count": total,
        "saved_count": n,
        "label_counts": counts,
        "clusters": 100,
        "policy_cluster_assigned": assigned,
        "elapsed_sec": round(time.perf_counter()-started, 2),
        "resume_db": str(STATE_DB),
        "note": "07 규칙 재사용. POLICY에만 K100 할당; REVIEW/PROCEDURAL은 NULL. 전체 데이터 원본 불변."
    }
    (OUT / "all_cluster_assignment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    stamp("완료", started)
    print("결과 폴더:", OUT)


def main():
    started = time.perf_counter()
    print("시작:", datetime.now().isoformat(timespec="seconds"), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (DB_PATH / "chroma.sqlite3", MODEL_PATH, REFINE_SCRIPT):
        if not path.is_file():
            raise FileNotFoundError(f"필수 입력 파일이 없습니다: {path}")
    refine = load_refine()
    model = joblib.load(MODEL_PATH)  # 본인이 직접 학습하여 저장한 신뢰 가능한 모델만 로드
    if int(model.n_clusters) != 100 or int(model.n_features_in_) != 1024:
        raise ValueError("K=100/1024차원 모델이 아닙니다")
    stamp("[1/4] 모델 및 07 정책 필터 로드", started)

    client = chromadb.PersistentClient(path=str(DB_PATH))
    coll = client.get_collection(name=COLLECTION)
    total = coll.count()
    stamp(f"[2/4] ChromaDB 연결 완료: {total:,} 청크", started)

    conn = sqlite3.connect(STATE_DB)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        offset = init_state(conn, total, MODEL_PATH)
        saved = conn.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
        if saved != offset:
            raise RuntimeError(f"체크포인트 불일치: saved={saved} / offset={offset}")
        stamp(f"[3/4] 처리 시작: 기존 {saved:,} / 남은 {total-saved:,}", started)
        batch_n = 0
        while offset < total:
            t = time.perf_counter()
            page = get_page(coll, offset)
            ids = page.get("ids") or []
            if not ids:
                raise RuntimeError(f"ChromaDB 페이지가 비어 있음 (offset={offset}, total={total})")
            rows = page_to_rows(page, model, refine)
            if len(rows) != len(ids):
                raise RuntimeError("배치 처리 수 불일치")
            save_batch(conn, rows, offset + len(rows))
            offset += len(rows)
            batch_n += 1
            if batch_n == 1 or batch_n % LOG_EVERY_BATCHES == 0 or offset == total:
                dt = time.perf_counter() - started
                r = offset / max(dt, 0.001)
                stamp(f"처리 {offset:,}/{total:,} ({offset/total:.1%}) | 평균 {r:.1f}개/초 | 최근 배치 {time.perf_counter()-t:.2f}초", started)
        export_results(conn, total, started)
    finally:
        conn.close()
    print("종료:", datetime.now().isoformat(timespec="seconds"))
    print("주의: 전체 ChromaDB는 처리 중 추가/삭제/수정하지 않아야 offset 스캔이 안전합니다.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n중단됨: 마지막 커밋 배치까지 저장되어 있습니다. 같은 명령으로 재실행하세요.")
        sys.exit(130)

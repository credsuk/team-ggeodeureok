
import time
import csv
import sqlite3
from pathlib import Path
from datetime import datetime

import numpy as np
import chromadb

# =====================================
# 1. 기본 설정
# =====================================

DB_PATH = Path(
    r"C:\mini_team\_db\chroma_snowflake_regions"
)

COLLECTION_NAME = "team_regions_6a0a006e92d1e93e"

OUTPUT_DIR = Path(
    r"C:\mini_team\scripts\clustering\results"
)

SAMPLE_PER_GROUP = 2000
BATCH_SIZE = 500
EXPECTED_DIM = 1024

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================
# 2. 시작 시간
# =====================================

total_start = time.perf_counter()

print("시작:", datetime.now())

# =====================================
# 3. SQLite에서 지역별 청크 수 확인
# =====================================

start = time.perf_counter()

db_file = DB_PATH / "chroma.sqlite3"

with sqlite3.connect(
    db_file.as_uri() + "?mode=ro",
    uri=True
) as conn:

    groups = conn.execute("""
        SELECT
            string_value,
            COUNT(*)
        FROM embedding_metadata
        WHERE key = 'source_key'
          AND string_value IS NOT NULL
        GROUP BY string_value
        ORDER BY string_value
    """).fetchall()

print("지역·연도 조합:", len(groups))
print("그룹 확인 시간:",
      round(time.perf_counter() - start, 2), "초")

for name, count in groups:
    print(name, f"{count:,}개")

# =====================================
# 4. ChromaDB 연결
# =====================================

start = time.perf_counter()

client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("ChromaDB 연결:",
      round(time.perf_counter() - start, 2), "초")

# =====================================
# 5. 균형 표본 추출
# =====================================

all_vectors = []
all_metadata = []
seen = set()

for group_idx, (source_key, group_count) in enumerate(
    groups, 1
):

    group_start = time.perf_counter()

    target = min(SAMPLE_PER_GROUP, group_count)

    # 그룹 내 여러 위치를 분산 추출
    offsets = np.linspace(
        0,
        max(0, group_count - BATCH_SIZE),
        max(1, int(np.ceil(target / BATCH_SIZE))),
        dtype=int
    )

    selected = 0

    for offset in offsets:

        remaining = target - selected

        if remaining <= 0:
            break

        limit = min(BATCH_SIZE, remaining)

        result = collection.get(
            where={"source_key": source_key},
            limit=limit,
            offset=int(offset),
            include=["embeddings", "metadatas"]
        )

        vectors = np.asarray(
            result["embeddings"],
            dtype=np.float32
        )

        if vectors.size == 0:
            continue

        if vectors.ndim != 2 or vectors.shape[1] != EXPECTED_DIM:
            raise ValueError(
                f"{source_key} 차원 오류: {vectors.shape}"
            )

        if not np.isfinite(vectors).all():
            raise ValueError(
                f"{source_key}: NaN/Inf 발견"
            )

        for j, chunk_id in enumerate(result["ids"]):

            if chunk_id in seen:
                continue

            seen.add(chunk_id)

            meta = result["metadatas"][j] or {}

            all_vectors.append(vectors[j])

            all_metadata.append({
                "chunk_id": chunk_id,
                "source_key": source_key,
                "date": meta.get("date", ""),
                "assembly": meta.get("assembly", ""),
                "source_file": meta.get("source_file", ""),
                "speaker": meta.get("speaker", ""),
                "agenda": meta.get("agenda", ""),
                "policy_category": meta.get("policy_category", "")
            })

            selected += 1

    elapsed = time.perf_counter() - group_start

    print(
        f"[{group_idx}/{len(groups)}] "
        f"{source_key} | "
        f"{selected:,}/{target:,}개 | "
        f"{elapsed:.2f}초"
    )

# =====================================
# 6. 저장
# =====================================

start = time.perf_counter()

vectors = np.stack(all_vectors).astype(np.float32)

npz_path = OUTPUT_DIR / "cluster_sample_balanced.npz"
csv_path = OUTPUT_DIR / "cluster_sample_balanced_metadata.csv"

np.savez(
    npz_path,
    embeddings=vectors,
    ids=np.array(
        [x["chunk_id"] for x in all_metadata]
    )
)

with csv_path.open(
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=all_metadata[0].keys()
    )

    writer.writeheader()
    writer.writerows(all_metadata)

print("\n=== 최종 결과 ===")
print("표본 수:", len(all_metadata))
print("벡터 형태:", vectors.shape)
print("NPZ:", npz_path)
print("CSV:", csv_path)
print("저장 시간:",
      round(time.perf_counter() - start, 2), "초")

print("전체 소요 시간:",
      round(time.perf_counter() - total_start, 2), "초")

print("종료:", datetime.now())


# 표본 추출 목표
# 지역·연도
# 34개

# 조합당 표본
# 2,000개

# 전체 목표
# 68,000개
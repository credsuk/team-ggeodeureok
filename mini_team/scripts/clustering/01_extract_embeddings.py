
from pathlib import Path
from datetime import datetime
import csv
import time
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

SAMPLE_SIZE = 5000
BATCH_SIZE = 500
EXPECTED_DIM = 1024


# =====================================
# 2. 임베딩 표본 추출
# =====================================

def main():

    total_start = time.perf_counter()

    print("시작:", datetime.now())

    # DB 파일 확인
    if not (DB_PATH / "chroma.sqlite3").exists():
        raise FileNotFoundError(DB_PATH)

    # ChromaDB 연결
    start = time.perf_counter()

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    total = collection.count()

    print("전체 청크:", f"{total:,}")
    print("DB 연결 시간:",
          round(time.perf_counter() - start, 3), "초")

    if total < SAMPLE_SIZE:
        raise ValueError("표본보다 DB 청크 수가 적습니다.")

    vectors_list = []
    metadata_list = []
    seen = set()

    # 전체 DB의 여러 위치에서 표본 추출
    batch_count = SAMPLE_SIZE // BATCH_SIZE

    offsets = np.linspace(
        0,
        total - BATCH_SIZE,
        batch_count,
        dtype=int
    )

    # =====================================
    # 3. 배치별 데이터 추출
    # =====================================

    for i, offset in enumerate(offsets, 1):

        start = time.perf_counter()

        result = collection.get(
            limit=BATCH_SIZE,
            offset=int(offset),
            include=["embeddings", "metadatas"]
        )

        vectors = np.asarray(
            result["embeddings"],
            dtype=np.float32
        )

        if vectors.ndim != 2 or vectors.shape[1] != EXPECTED_DIM:
            raise ValueError(f"벡터 차원 오류: {vectors.shape}")

        if not np.isfinite(vectors).all():
            raise ValueError("NaN 또는 Inf 발견")

        for j, chunk_id in enumerate(result["ids"]):

            if chunk_id in seen:
                continue

            seen.add(chunk_id)

            meta = result["metadatas"][j] or {}

            vectors_list.append(vectors[j])

            date = str(meta.get("date", ""))

            metadata_list.append({
                "chunk_id": chunk_id,
                "year": date[:4],
                "date": date,
                "assembly": meta.get("assembly", ""),
                "source_key": meta.get("source_key", ""),
                "source_file": meta.get("source_file", ""),
                "speaker": meta.get("speaker", ""),
                "agenda": meta.get("agenda", "")
            })

        elapsed = time.perf_counter() - start

        print(
            f"[{i}/{batch_count}] "
            f"offset={offset:,} / "
            f"{len(result['ids']):,}개 / "
            f"{elapsed:.2f}초"
        )

    # =====================================
    # 4. 결과 검증
    # =====================================

    vectors = np.stack(vectors_list)

    print("\n=== 표본 추출 결과 ===")
    print("추출 청크:", len(metadata_list))
    print("벡터 형태:", vectors.shape)

    groups = {
        (r["year"], r["assembly"])
        for r in metadata_list
    }

    print("연도·의회 조합:", len(groups))

    # =====================================
    # 5. 파일 저장
    # =====================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    npz_path = OUTPUT_DIR / "cluster_sample_5000.npz"

    csv_path = OUTPUT_DIR / "cluster_sample_metadata.csv"

    np.savez(
        npz_path,
        embeddings=vectors,
        ids=np.array([
            r["chunk_id"] for r in metadata_list
        ])
    )

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=metadata_list[0].keys()
        )

        writer.writeheader()
        writer.writerows(metadata_list)

    print("\nNPZ 저장:", npz_path)
    print("CSV 저장:", csv_path)

    print(
        "전체 소요 시간:",
        round(time.perf_counter() - total_start, 2),
        "초"
    )

    print("종료:", datetime.now())


if __name__ == "__main__":
    main()

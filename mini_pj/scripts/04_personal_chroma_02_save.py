import numpy as np
import pandas as pd
import chromadb
# pip install chromadb
from pathlib import Path

# ==================================================
# 1. 경로 및 저장 설정
# ==================================================

EMBEDDING_DIR = Path(
    r"C:\mini_pj\data\processed\embeddings_by_region"
)

DB_DIR = Path(
    r"C:\mini_pj\_db\chroma_personal"
)

COLLECTION_NAME = "personal_policy_chunks"

BATCH_SIZE = 250

DB_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# 2. Chroma DB 연결
# ==================================================

client = chromadb.PersistentClient(
    path=str(DB_DIR)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

print("DB 경로:", DB_DIR)
print("컬렉션:", collection.name)
print("기존 저장 청크:", collection.count())

# ==================================================
# 3. 임베딩 파일 목록
# ==================================================

npy_files = sorted(
    EMBEDDING_DIR.rglob("batch_*.npy")
)
# npy_files = npy_files[:1]  # 첫 번째 배치만 테스트

print("전체 임베딩 배치 파일:", len(npy_files))

# ==================================================
# 4. 배치별 Chroma DB 저장
# ==================================================

for file_idx, npy_file in enumerate(npy_files, 1):

    metadata_file = npy_file.with_name(
        npy_file.stem + "_metadata.csv"
    )

    if not metadata_file.exists():
        print("메타데이터 없음:", npy_file)
        continue

    print(
        f"\n[{file_idx}/{len(npy_files)}] "
        f"{npy_file.parent.name} / {npy_file.name}"
    )

    embeddings = np.load(
        npy_file,
        mmap_mode="r"
    )

    df = pd.read_csv(
        metadata_file,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False
    )

    # 데이터 검증
    if len(embeddings) != len(df):
        print("벡터와 메타데이터 수 불일치")
        continue

    if embeddings.ndim != 2 or embeddings.shape[1] != 1024:
        print("임베딩 차원 오류")
        continue

    if "chunk_id" not in df.columns or "text" not in df.columns:
        print("필수 컬럼 누락")
        continue

    if df["chunk_id"].eq("").any():
        print("빈 chunk_id 발견")
        continue

    if df["chunk_id"].duplicated().any():
        print("배치 내부 중복 chunk_id 발견")
        continue

    if not np.isfinite(embeddings).all():
        print("NaN 또는 무한대 벡터 발견")
        continue

    # Chroma DB 저장
    for start in range(0, len(df), BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            len(df)
        )

        batch_df = df.iloc[start:end]

        ids = batch_df["chunk_id"].tolist()
        documents = batch_df["text"].tolist()

        metadata_cols = [
            c for c in batch_df.columns
            if c != "text"
        ]

        metadatas = batch_df[
            metadata_cols
        ].to_dict(orient="records")

        vectors = np.asarray(
            embeddings[start:end],
            dtype=np.float32
        ).tolist()

        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas
        )

        print(
            f"  저장: {end:,}/{len(df):,}",
            flush=True
        )

    print(
        "누적 DB 청크:",
        f"{collection.count():,}"
    )

# ==================================================
# 5. 최종 결과
# ==================================================

print("\n==============================")
print("Chroma DB 저장 작업 종료")
print("최종 저장 청크:", f"{collection.count():,}")
print("DB 경로:", DB_DIR)
print("==============================")


# 전체 임베딩 청크 수와 DB 저장 청크 수 비교
# 지역·연도별 메타데이터가 정확히 들어갔는지 확인
# 실제 문장을 입력해 유사도 검색이 작동하는지 확인

# ==============================
# Chroma DB 저장 작업 종료
# 최종 저장 청크: 684,311
# DB 경로: C:\mini_pj\_db\chroma_personal
# ==============================
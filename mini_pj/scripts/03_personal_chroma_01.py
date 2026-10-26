# from pathlib import Path
# import chromadb

# # 기존 임베딩 경로
# EMBEDDING_DIR = Path(
#     r"C:\mini_pj\data\processed\embeddings_by_region"
# )

# # 개인 Chroma DB 저장 경로
# DB_DIR = Path(
#     r"C:\mini_pj\_db\chroma_personal"
# )

# DB_DIR.mkdir(parents=True, exist_ok=True)

# client = chromadb.PersistentClient(path=str(DB_DIR))

# collection = client.get_or_create_collection(
#     name="personal_policy_chunks",
#     metadata={"hnsw:space": "cosine"}
# )

# print("임베딩 경로:", EMBEDDING_DIR)
# print("Chroma DB 경로:", DB_DIR)
# print("현재 저장 청크:", collection.count())


# from pathlib import Path

# EMBEDDING_DIR = Path(
#     r"C:\mini_pj\data\processed\embeddings_by_region"
# )

# files = sorted(EMBEDDING_DIR.rglob("*"))

# for f in files[:20]:
#     if f.is_file():
#         print(f.name, "|", round(f.stat().st_size / 1024**2, 2), "MB")



import numpy as np
import pandas as pd
import json
from pathlib import Path

EMBEDDING_DIR = Path(
    r"C:\mini_pj\data\processed\embeddings_by_region"
)

# 첫 번째 npy 파일
npy_file = sorted(EMBEDDING_DIR.rglob("*.npy"))[0]

# 같은 배치의 메타데이터와 정보 파일
metadata_file = npy_file.with_name(
    npy_file.stem + "_metadata.csv"
)

info_file = npy_file.with_name(
    npy_file.stem + "_info.json"
)

# 임베딩 로드
embeddings = np.load(npy_file, mmap_mode="r")

# 메타데이터 로드
df = pd.read_csv(metadata_file, encoding="utf-8-sig")

print("임베딩 파일:", npy_file)
print("임베딩 shape:", embeddings.shape)
print("임베딩 dtype:", embeddings.dtype)

print("\n메타데이터 shape:", df.shape)
print("메타데이터 컬럼:", df.columns.tolist())

print("\n첫 번째 메타데이터:")
print(df.iloc[0].to_string())

print("\n벡터 수와 메타데이터 수 일치:",
      len(embeddings) == len(df))

if info_file.exists():
    with open(info_file, "r", encoding="utf-8") as f:
        info = json.load(f)

    print("\n배치 정보:", info)
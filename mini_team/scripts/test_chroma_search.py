
# test_chroma_search.py

import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# =====================================
# 1. 기본 설정
# =====================================

DB_PATH = Path(
    r"C:\mini_team\_db\chroma_snowflake_regions"
)

COLLECTION_NAME = "team_regions_6a0a006e92d1e93e"

# 기존 임베딩 코드와 동일한 모델명 입력
MODEL_NAME = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"

TOP_K = 5

QUERY = "지방의회에서 논의된 청년 일자리 지원 정책은 무엇인가?"

# =====================================
# 2. ChromaDB 연결
# =====================================

print("\n========== 1. DB 연결 ==========")

start = time.perf_counter()

if not (DB_PATH / "chroma.sqlite3").is_file():
    raise FileNotFoundError("ChromaDB 파일을 찾을 수 없습니다.")

client = chromadb.PersistentClient(path=str(DB_PATH))

collection = client.get_collection(
    name=COLLECTION_NAME
)

db_time = time.perf_counter() - start

print("컬렉션:", collection.name)
print("전체 청크:", f"{collection.count():,}")
print("DB 연결 시간:", round(db_time, 3), "초")

# =====================================
# 3. 임베딩 모델 로드
# =====================================

start = time.perf_counter()

model = SentenceTransformer(MODEL_NAME)

print("모델:", MODEL_NAME)
print("모델 로드 시간:",
      round(time.perf_counter() - start, 3), "초")

# =====================================
# 4. 질문 임베딩
# =====================================

print("\n========== 3. 질문 임베딩 ==========")

print("질문:", QUERY)

start = time.perf_counter()

query_embedding = model.encode(
    QUERY,
    prompt_name="query",
    normalize_embeddings=True,
    convert_to_numpy=True
).tolist()

# 옵션	권장 설정	이유
# normalize_embeddings	True	벡터 길이를 1로 정규화
# prompt_name	"query"	질문용 임베딩 방식 적용
# batch_size	1	질문 1개 검색
# convert_to_numpy	True	ChromaDB에 전달하기 편리

embedding_time = time.perf_counter() - start

print("벡터 차원:", len(query_embedding))
print("질문 임베딩 시간:", round(embedding_time, 3), "초")

# =====================================
# 5. ChromaDB 유사도 검색
# =====================================

print("\n========== 4. 유사도 검색 ==========")

start = time.perf_counter()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=TOP_K,
    include=[
        "documents",
        "metadatas",
        "distances"
    ]
)

search_time = time.perf_counter() - start

documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]
ids = results["ids"][0]

print("검색 결과 수:", len(documents))
print("검색 시간:", round(search_time, 3), "초")

# =====================================
# 6. 검색 결과 출력
# =====================================

print("\n========== 5. 검색 결과 ==========")

for i, (doc, meta, dist, chunk_id) in enumerate(
    zip(documents, metadatas, distances, ids), start=1
):
    meta = meta or {}

    print(f"\n{'=' * 60}")
    print(f"[검색 결과 {i}]")
    print(f"청크 ID: {chunk_id}")
    print(f"거리: {dist:.4f}")

    print("\n[메타데이터]")
    for key, value in meta.items():
        print(f"{key}: {value}")

    print("\n[발언 원문]")
    print(doc if doc else "(본문 없음)")

# =====================================
# 7. 검색 성능 요약
# =====================================

print("\n========== 6. 성능 요약 ==========")

print("전체 DB 청크:", f"{collection.count():,}")
print("검색 요청:", TOP_K)
print("검색 반환:", len(documents))
print("질문 임베딩:", round(embedding_time, 3), "초")
print("DB 검색:", round(search_time, 3), "초")
print("임베딩 + 검색:",
      round(embedding_time + search_time, 3), "초")

print("\n검색 테스트 완료!")


# 검증 항목	출력 결과
# DB 연결	연결 성공 여부, 전체 청크 수
# 유사도 검색	정책 질문에 대한 Top-5 검색
# 검색 결과	관련 발언 원문 5개
# 메타데이터	지역·연도·발언자·출처
# 검색 속도	질문 임베딩 시간, DB 검색 시간
# 유사도 거리	검색 결과별 거리 값
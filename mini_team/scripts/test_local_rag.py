
import time
import requests
import chromadb

from pathlib import Path
from sentence_transformers import SentenceTransformer

# =====================================
# 1. 기본 설정
# =====================================

DB_PATH = Path(
    r"C:\mini_team\_db\chroma_snowflake_regions"
)

COLLECTION_NAME = "team_regions_6a0a006e92d1e93e"

EMBED_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
LLM_MODEL = "qwen3:8b"

OLLAMA_URL = "http://localhost:11434/api/chat"

QUERY = "지방의회에서 논의된 청년 일자리 지원 정책은 무엇인가?"

SEARCH_K = 10
TOP_K = 5

# =====================================
# 2. ChromaDB 연결
# =====================================

print("\n========== 1. ChromaDB 연결 ==========")

start = time.perf_counter()

client = chromadb.PersistentClient(path=str(DB_PATH))

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("컬렉션:", collection.name)
print("전체 청크:", f"{collection.count():,}")
print("DB 연결:", round(time.perf_counter() - start, 3), "초")

# =====================================
# 3. 질문 임베딩
# =====================================

print("\n========== 2. 질문 임베딩 ==========")

model = SentenceTransformer(EMBED_MODEL)

start = time.perf_counter()

query_vector = model.encode(
    QUERY,
    prompt_name="query",
    normalize_embeddings=True
).tolist()

embed_time = time.perf_counter() - start

print("질문:", QUERY)
print("임베딩 차원:", len(query_vector))
print("임베딩 시간:", round(embed_time, 3), "초")

# =====================================
# 4. ChromaDB 검색
# =====================================

print("\n========== 3. ChromaDB 검색 ==========")

start = time.perf_counter()

results = collection.query(
    query_embeddings=[query_vector],
    n_results=SEARCH_K,
    include=["documents", "metadatas", "distances"]
)

search_time = time.perf_counter() - start

# =====================================
# 5. 중복 청크 제거
# =====================================

unique_docs = []
seen = set()

for doc, meta, dist, chunk_id in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
    results["ids"][0]
):
    if not doc:
        continue

    meta = meta or {}

    # 동일 원문 + 출처를 기준으로 중복 제거
    fingerprint = meta.get("source_fingerprint")
    key = (fingerprint, doc.strip())

    if key in seen:
        continue

    seen.add(key)

    unique_docs.append({
        "id": chunk_id,
        "document": doc,
        "metadata": meta,
        "distance": dist
    })

    if len(unique_docs) >= TOP_K:
        break

print("검색 시간:", round(search_time, 3), "초")
print("중복 제거 후:", len(unique_docs), "개")

if not unique_docs:
    raise RuntimeError("검색된 근거 문서가 없습니다.")

# =====================================
# 6. LLM 전달 근거 구성
# =====================================

print("\n========== 4. 검색 근거 ==========")

context_parts = []

for i, item in enumerate(unique_docs, start=1):

    meta = item["metadata"]

    assembly = meta.get("assembly", "미상")
    date = meta.get("date", "미상")
    speaker = meta.get("speaker", "미상")
    agenda = meta.get("agenda", "미상")

    print(f"\n[근거 {i}]")
    print("지역:", assembly)
    print("날짜:", date)
    print("발언자:", speaker)
    print("거리:", round(item["distance"], 4))
    print("본문:", item["document"][:250], "...")

    context_parts.append(
        f"""
[근거 {i}]
지역: {assembly}
날짜: {date}
발언자: {speaker}
안건: {agenda}
청크 ID: {item['id']}

발언 원문:
{item['document']}
"""
    )

context = "\n".join(context_parts)

# =====================================
# 7. LLM 프롬프트
# =====================================

system_prompt = """
당신은 지방의회 회의록 기반 정책 분석 전문가입니다.

규칙:
1. 반드시 제공된 회의록 근거만 사용하세요.
2. 근거에 없는 정책이나 사실을 만들어내지 마세요.
3. 정책별로 지역, 핵심 내용, 근거 번호를 제시하세요.
4. 근거가 부족하면 확인할 수 없다고 답하세요.
5. 한국어로 명확하고 간결하게 답하세요.
6. 근거 번호는 제공된 [근거 N]만 인용하세요.
"""

user_prompt = f"""
다음 회의록을 참고하여 질문에 답하세요.

[회의록 검색 결과]
{context}

[질문]
{QUERY}

[출력 형식]
1. 주요 정책 요약
2. 지역별 정책 내용과 근거 번호
3. 종합 분석
"""

# =====================================
# 8. Ollama LLM 연결
# =====================================

print("\n========== 5. Ollama 답변 생성 ==========")

start = time.perf_counter()

response = requests.post(
    OLLAMA_URL,
    json={
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": 700
        }
    },
    timeout=300
)

response.raise_for_status()

result = response.json()

llm_time = time.perf_counter() - start

answer = result.get("message", {}).get("content", "")

# =====================================
# 9. 답변 출력
# =====================================

print("\n========== 6. 최종 RAG 답변 ==========\n")

print(answer)

# =====================================
# 10. 처리 시간
# =====================================

print("\n========== 7. 성능 요약 ==========")

print("LLM 모델:", LLM_MODEL)
print("임베딩 시간:", round(embed_time, 3), "초")
print("검색 시간:", round(search_time, 3), "초")
print("LLM 답변 시간:", round(llm_time, 3), "초")

print(
    "전체 처리 시간:",
    round(embed_time + search_time + llm_time, 3),
    "초"
)

print("검색 근거 수:", len(unique_docs))
print("생성 토큰 수:", result.get("eval_count", "미확인"))

print("\n로컬 RAG 테스트 완료!")


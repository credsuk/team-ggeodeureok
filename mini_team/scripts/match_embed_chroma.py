"""TXT 헤더를 제외한 데이터 행 번호와 JSONL row_id를 연결해 청킹·임베딩합니다."""
import csv
import json
import re
from pathlib import Path

# 1. 이 두 경로를 본인 컴퓨터 위치로 수정하세요.
TXT_PATH = Path(r"C:/mini_10/cleaned_2025_광주광역시의회.txt")
DB_PATH = Path(r"C:/mini_10/_db/chroma_team_snowflake")
CHUNK_TOKENS = 450
OVERLAP_TOKENS = 50
SAVE_BATCH = 128
META_PATH = Path(r"C:/mini_10/cleaned_2025_광주광역시의회(1).jsonl")
MODEL_NAME = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
BATCH_SIZE = 8  # GPU 메모리가 부족하면 16 또는 8


def norm(value):
    return re.sub(r"\s+", "", str(value or ""))


def load_and_match(txt_path, meta_path):
    csv.field_size_limit(100_000_000)
    with txt_path.open(encoding="utf-8-sig", newline="") as f:
        # 컬럼명인 첫 줄은 자동으로 제외됩니다.
        texts = list(csv.DictReader(f, delimiter="\t"))
    with meta_path.open(encoding="utf-8-sig") as f:
        metas = [json.loads(line) for line in f if line.strip()]
    if len(texts) != len(metas) or not texts:
        raise ValueError(f"행 수 확인 필요: TXT={len(texts)}, JSONL={len(metas)}")
    by_id = {}
    for meta in metas:
        rid = meta.get("row_id")
        if type(rid) is not int or rid in by_id:
            raise ValueError(f"row_id 자료형 또는 중복 확인: {rid}")
        by_id[rid] = meta
    if set(by_id) != set(range(1, len(texts) + 1)):
        raise ValueError("row_id가 TXT 데이터 행 번호 1~N과 일치하지 않습니다.")
    matched = []
    for rid, row in enumerate(texts, 1):
        meta = by_id[rid]
        checks = [(row["MTG_DE"], meta["date"]),
                  (row["RASMBLY_NM"], meta["assembly"]),
                  (row["agenda_title"], meta["agenda"]),
                  (row["speaker_role"] + row["speaker_name"], meta["speaker"])]
        if not all(norm(a) == norm(b) for a, b in checks):
            raise ValueError(f"row_id={rid}: 날짜·지역·안건·발언자 불일치")
        if not row["utterance"].strip():
            raise ValueError(f"row_id={rid}: 본문이 비어 있습니다.")
        matched.append((row["utterance"], meta))
    print(f"연결 검증 완료: {len(matched):,}개 발언")
    return matched



def main():
    # TXT 첫 줄(컬럼명)을 제외하고 row_id로 연결·검증합니다.
    matched = load_and_match(TXT_PATH, META_PATH)
    import hashlib
    import torch
    import chromadb
    from sentence_transformers import SentenceTransformer
    if not 0 <= OVERLAP_TOKENS < CHUNK_TOKENS:
        raise ValueError("겹침 크기는 청크 크기보다 작아야 합니다.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("실행 장치:", device)
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = CHUNK_TOKENS + 2
    tokenizer = model.tokenizer
    if not tokenizer.is_fast:
        raise ValueError("본문 위치 기반 청킹에는 fast tokenizer가 필요합니다.")

    # 파일·모델·청킹 설정이 같으면 같은 컬렉션을 사용합니다.
    # 변경되면 새 컬렉션으로 분리해 이전 자료와 섞이지 않습니다.
    signature = hashlib.sha256()
    for p in (TXT_PATH, META_PATH):
        signature.update(hashlib.sha256(p.read_bytes()).digest())
    signature.update(f"{MODEL_NAME}|{CHUNK_TOKENS}|{OVERLAP_TOKENS}|v1".encode())
    collection_name = "team_arctic_" + signature.hexdigest()[:20]
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=None,
        metadata={"hnsw:space": "cosine", "embedding_model": MODEL_NAME,
                  "chunk_tokens": CHUNK_TOKENS, "overlap_tokens": OVERLAP_TOKENS})
    print("컬렉션:", collection_name, "/ 기존 저장:", collection.count())
    pending = []
    expected = 0

    def save_batch():
        if not pending:
            return
        # 중단 후 재실행 시 저장된 청크는 임베딩을 건너뜁니다.
        existing = set(collection.get(ids=[r["id"] for r in pending], include=[]) ["ids"])
        new = [r for r in pending if r["id"] not in existing]
        if new:
            # 이 모델의 문서 임베딩에는 E5용 passage: 접두어를 붙이지 않습니다.
            vectors = model.encode([r["text"] for r in new], prompt="",
                                   batch_size=BATCH_SIZE, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
            collection.upsert(ids=[r["id"] for r in new],
                              documents=[r["text"] for r in new],
                              metadatas=[r["metadata"] for r in new],
                              embeddings=vectors.tolist())
        print(f"처리 청크: {expected:,} / DB 저장: {collection.count():,}", flush=True)
        pending.clear()

    for text, meta in matched:
        offsets = tokenizer(text, add_special_tokens=False, truncation=False,
                            return_offsets_mapping=True)["offset_mapping"]
        if not offsets:
            raise ValueError(f"row_id={meta['row_id']}: 토큰화 결과가 비어 있습니다.")
        for chunk_no, start in enumerate(range(0, len(offsets), CHUNK_TOKENS - OVERLAP_TOKENS), 1):
            end = min(start + CHUNK_TOKENS, len(offsets))
            chunk = text[offsets[start][0]:offsets[end-1][1]]
            # 키워드 목록 등은 JSON 문자열로 보존해 Chroma 버전 간 호환성을 높입니다.
            metadata = {k: (v if isinstance(v, (str, int, float, bool)) else
                            json.dumps(v, ensure_ascii=False))
                        for k, v in meta.items() if v is not None}
            metadata.update({"chunk_no": chunk_no, "source_file": TXT_PATH.name,
                             "embedding_model": MODEL_NAME,
                             "char_start": offsets[start][0], "char_end": offsets[end-1][1]})
            pending.append({"id": f"row_{meta['row_id']}_chunk_{chunk_no}",
                            "text": chunk, "metadata": metadata})
            expected += 1
            if len(pending) >= SAVE_BATCH:
                save_batch()
            if end == len(offsets):
                break
    save_batch()
    if collection.count() != expected:
        raise ValueError(f"저장 개수 불일치: 예상={expected}, DB={collection.count()}")
    print("완료. DB 경로:", DB_PATH)
    print("컬렉션 이름:", collection_name)
    print("총 저장 청크:", expected)
    # 같은 모델과 query 프롬프트로 검색까지 확인합니다.
    query = "광주시 소상공인 지원 정책"
    query_vector = model.encode([query], prompt_name="query", normalize_embeddings=True)
    result = collection.query(query_embeddings=query_vector.tolist(), n_results=min(3, expected))
    for document, metadata in zip(result["documents"][0], result["metadatas"][0]):
        print("검색 결과:", metadata["row_id"], metadata["speaker"], document[:150])


if __name__ == "__main__":
    main()

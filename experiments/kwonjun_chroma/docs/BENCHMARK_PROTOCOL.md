# ChromaDB vs PostgreSQL+pgvector 공정 비교 규칙

## 바꾸면 안 되는 조건

두 시스템은 아래 파일을 그대로 공유한다.

1. `corpus.parquet`: 동일한 행, `chunk_id`, 본문 및 metadata
2. `embeddings.f32.npy`: 한 번만 생성한 동일한 float32 정규화 벡터
3. `query_embeddings.f32.npy`: 한 번만 생성한 동일한 질의 벡터
4. `eval_queries.csv`: 동일한 질의와 정답
5. 거리 함수: cosine
6. top-k, metadata filter 및 검색 반복 횟수

DB별로 본문을 다시 정제하거나 임베딩하면 DB 성능이 아니라 데이터·모델 차이를 비교하게 되므로 금지한다.

## 측정 항목

| 영역 | 지표 |
|---|---|
| 정확도 | FAISS exact 대비 Recall@10/20/50, nDCG@10, MRR |
| 속도 | warm latency p50/p95/p99, QPS |
| 구축 | 적재 시간, 인덱스 생성 시간, 디스크 사용량 |
| 자원 | 최대 RAM/VRAM, CPU 사용량 |
| 기능 | 의회·기간·위원회 복합 필터 정확성 |
| 운영 | 재시작, 추가/수정/삭제, 백업 및 재현성 |

## 실행 원칙

- 두 DB는 반드시 같은 컴퓨터, 같은 SSD, 같은 질의 순서에서 측정한다.
- 최초 1회는 cold 참고치로 분리하고, 워밍업 뒤 최소 5회 반복한다.
- HNSW 비교 시 가능한 한 `M`, `ef_construction`, `ef_search`를 맞춘다.
- 정확도 비교에는 동일 벡터의 FAISS exact 결과를 정답 기준으로 남긴다.
- 개발 중 25,340개 정책 청크로 결정하고, 최종 구성만 전체 청크로 확장한다.


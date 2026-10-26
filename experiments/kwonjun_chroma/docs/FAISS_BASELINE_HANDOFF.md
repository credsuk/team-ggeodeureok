# 기존 FAISS 기준선 보존 기록

이 문서는 새 ChromaDB/PostgreSQL 비교 실험과 기존 파이프라인을 혼동하지 않기 위한 인수인계 기록이다. 기존 파일은 삭제하거나 덮어쓰지 않는다.

## 기존 처리 결과

- 원천: 2024~2025년 광역 지방의회 JSON 36개, 약 3GB
- 정제 문서: 9,678개, 약 5.41억 자
- 전체 부모/검색 청크: 460,431개
- 임베딩 대상(high/medium): 438,708개
- 정책 의미 청크: 25,340개
- 임베딩 모델: `dragonkue/snowflake-arctic-embed-l-v2.0-ko`
- 모델 revision: `55ec6e9`
- 차원: 1,024
- 벡터: L2 정규화, float16 저장
- 실행 장치: NVIDIA GeForce RTX 4060 Laptop GPU
- 실행 정밀도/배치: bf16, batch 8
- 전체 임베딩 소요 시간: 약 11,186초(3시간 6분)
- 검색 기준선: FAISS `IndexFlatIP` 정확 검색(정규화 벡터의 내적 = cosine)

## 검색 실험 결과

| 구성 | nDCG | Recall | MRR |
|---|---:|---:|---:|
| Arctic dense | 0.9908 | 0.2851 | 1.0000 |
| Dense RRF | 0.9899 | 0.2851 | 1.0000 |
| KURE | 0.9813 | 0.2841 | 1.0000 |
| Hybrid RRF | 0.9684 | 0.2803 | 1.0000 |
| Hybrid rerank | 0.9642 | 0.2813 | 0.9896 |
| BM25 | 0.8568 | 0.2675 | 0.9896 |

기존 25,340개 정책 청크를 Chroma에 적재했을 때 FAISS와 `overlap@10=1.0`, Top-1 일치를 확인한 적이 있다. 이 결과는 새 통일 데이터 비교의 성능 결과가 아니라 과거 기준선이다.

## 재사용 원칙

새 실험은 별도 폴더에서 수행한다. 나중에 FAISS 기준선이 필요하면 동일한 canonical embedding을 `IndexFlatIP`에 적재하여 Chroma 및 pgvector의 recall 기준으로 사용한다.


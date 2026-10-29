# 현재 사용 파이프라인

기준일: 2026-10-23

이 문서는 현재 실제로 사용하는 코드와 이전 실험 코드를 구분한다. 기존 파일은
삭제하지 않고 `archive/legacy_20261023/`에 보존한다.

## 확정 구성

| 항목 | 선택 |
|---|---|
| canonical 데이터 | `data/canonical_v3/corpus.parquet` |
| 검색 청크 | `data/canonical_v3/chunks_v1/chunks.parquet` |
| 부모 청크 | `data/canonical_v3/chunks_v1/parents.parquet` |
| 검색 청크 수 | 901,358 |
| 임베딩 | `dragonkue/snowflake-arctic-embed-l-v2.0-ko` |
| 임베딩 입력 | raw text |
| 벡터 | 1024차원, 정규화, float16 저장 |
| 벡터 아티팩트 | `data/embeddings/arctic_raw_v1/` |
| 최종 검색 DB | PostgreSQL + pgvector |
| 비교 기준 DB | ChromaDB |
| pgvector 설정 | HNSW, cosine, `ef_search=200` |
| 검색 구성 | Dense + BM25 → RRF Top-50 → BGE 재순위 → 최종 Top-10 |

## 현재 실행 순서

### A. 데이터와 임베딩 재현

| 순서 | 파일 | 역할 |
|---:|---|---|
| 1 | `src/01_restart_inventory.py` | 신규 데이터 파일 및 metadata 검사 |
| 2 | `src/02_probe_received_metadata.py` | 수신 metadata 구조 검사 |
| 3 | `src/03_validate_row_id_join.py` | 발화와 metadata 결합키 검증 |
| 4 | `src/04_build_direct_canonical.py` | canonical corpus 생성 |
| 5 | `src/05_build_parent_child_chunks.py` | 부모·자식 청킹 |
| 6 | `src/05b_validate_chunks.py` | 청크 무결성 검사 |
| 7 | `src/06_make_embedding_eval_sample.py` | 임베딩 평가 표본 생성 |
| 8 | `src/07_embed_eval_sample.py` | 평가 표본 임베딩 |
| 9 | `src/08_build_blind_retrieval_pool.py` | blind 검색 평가 풀 생성 |
| 10 | `src/09_local_llm_relevance_judge.py` | 로컬 LLM 관련도 라벨 생성 |
| 11 | `src/10_compare_embedding_variants.py` | raw/enriched 비교 |
| 12 | `src/11_build_embedding_model_pool.py` | Arctic/KURE 후보 풀 생성 |
| 13 | `src/12_compare_embedding_models.py` | 임베딩 모델 비교 |
| 14 | `src/08_embed_full_corpus.py` | 확정 구성 전체 임베딩 |

`08_embed_full_corpus.py`는 역사적으로 파일 번호가 08로 정해졌기 때문에 이름을
억지로 바꾸지 않는다. 발표와 문서에서는 "전체 임베딩" 단계로 표현한다.

### B. 최종 DB 파이프라인

| 순서 | 파일 | 역할 |
|---:|---|---|
| 15 | `src/20_validate_embedding_artifact.py` | 전체 임베딩 무결성 검사 |
| 16 | `src/15_load_chroma.py` | 동일 벡터를 ChromaDB에 적재 |
| 17 | `src/16_load_pgvector.py` | 동일 벡터를 pgvector에 적재 |
| 18 | `src/17_benchmark_vector_databases.py` | exact 기준 두 DB 비교 |
| 18.5 | `src/35_build_bm25_index.py` | 확정 청크로 SQLite FTS5 BM25 인덱스 생성 |
| 19 | `src/23_search_evidence.py` | 검색 결과와 원문·부모 문맥 결합 |
| 공통 | `src/council_vector_runtime.py` | 임베딩 로드·질의 인코딩·검색 공통 코드 |

## 확정된 DB 비교 결과

901,358개 벡터, 10개 질의, Top-10, pgvector `ef_search=200` 조건이다.

| 조건 | DB | p50 | p95 | Recall@10 | Top-1 |
|---|---|---:|---:|---:|---:|
| 일반 | ChromaDB | 1.42 ms | 2.14 ms | 0.87 | 0.80 |
| 일반 | pgvector | 2.75 ms | 4.31 ms | 0.95 | 1.00 |
| 지역+연도 | ChromaDB | 586.29 ms | 861.08 ms | 1.00 | 0.90 |
| 지역+연도 | pgvector | 3.97 ms | 8.18 ms | 1.00 | 0.90 |

최종 서비스는 필터 검색 성능과 SQL 기반 후속 분석을 고려해 pgvector를 사용한다.
ChromaDB는 삭제하지 않고 비교 기준선으로 보존한다.

## 다음 개발 순서

1. pgvector Dense 후보와 SQLite FTS5 BM25 후보를 각각 검색한다.
2. 두 순위를 RRF로 결합해 후보 50개를 만든다.
3. BGE 재순위로 최종 근거 10개를 선택한다.
4. `parent_chunk_id`로 부모 문맥을 복원한다.
5. 검색 청크를 고정 `issue_id`와 연결한다.
6. 최초 등장일과 지방의회 관측 순서를 계산한다.
7. 월별 언급량·참여 의회 수·증가율·확산 수를 특징으로 만든다.
8. Logistic/Poisson 모델과 기준선으로 다음 달 흐름을 예측한다.
9. 로컬 LLM은 계산 결과를 변경하지 않고 근거 기반 설명만 작성한다.

## 대용량 파일과 Git

다음 파일은 로컬에 보관하고 GitHub에는 올리지 않는다.

```text
data/input/
data/canonical_v3/
data/embeddings/
data/bm25/
data/chroma/
data/postgres/
outputs/**/*.json
outputs/**/*.parquet
*.npy
*.parquet
```

GitHub에는 코드, 설정, 작은 평가 질의, 실행 문서 및 요약 결과만 올린다.


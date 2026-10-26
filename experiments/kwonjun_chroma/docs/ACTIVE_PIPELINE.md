# 의정이슈레이더 최종 활성 파이프라인

기준일: 2026-10-26

## 1. 확정 구성

| 항목 | 최종 선택 |
|---|---|
| canonical 발화 | 2,057,214행 |
| 부모·자식 검색 청크 | 901,358개 |
| 임베딩 입력 | raw 발언 텍스트 |
| 임베딩 | `dragonkue/snowflake-arctic-embed-l-v2.0-ko` |
| 임베딩 revision | `55ec6e9358a56d56af759bc8372e970caf8c305f` |
| 벡터 | 1,024차원, L2 정규화, float16 저장 |
| 서비스 DB | PostgreSQL + pgvector HNSW |
| 키워드 검색 | SQLite FTS5 BM25 |
| 검색 결합 | Dense + BM25 → RRF |
| 재순위 | `BAAI/bge-reranker-v2-m3` |
| 이슈 군집 | 코사인 유사도 그래프 연결요소 + Union-Find |
| 예측 | 활동률 기준선 + Poisson Regression |
| 로컬 Agent | Ollama `qwen3:8b` |
| 화면 | Streamlit + PyDeck |

## 2. 전체 흐름

```text
제공 회의록·metadata
→ canonical 결합
→ 부모·자식 청킹
→ 임베딩 모델·텍스트 표현 비교
→ 전체 Arctic raw 임베딩
→ pgvector·BM25 색인
→ Dense/BM25 후보 검색
→ RRF 순위 결합
→ BGE 재순위
→ 부모 문맥 복원·중복 제거
→ seed 중심 후보 확장
→ 유사도 그래프 군집화
→ 사람 검수·issue_id 고정
→ 최초 관측·의회별 시간 순서
→ 월별 특징·시간순 예측
→ 지도·원문 근거·로컬 Agent
```

## 3. 활성 코드

### 데이터·임베딩

| 단계 | 코드 |
|---|---|
| 입력 검사 | `src/01_restart_inventory.py` |
| metadata 검사 | `src/02_probe_received_metadata.py` |
| 결합키 검증 | `src/03_validate_row_id_join.py` |
| canonical 생성 | `src/04_build_direct_canonical.py` |
| 부모·자식 청킹 | `src/05_build_parent_child_chunks.py` |
| 청크 검증 | `src/05b_validate_chunks.py` |
| 임베딩 표본 | `src/06_make_embedding_eval_sample.py` |
| 표본 임베딩 | `src/07_embed_eval_sample.py` |
| 검색 평가 풀 | `src/08_build_blind_retrieval_pool.py` |
| 로컬 판정 | `src/09_local_llm_relevance_judge.py` |
| raw/enriched 비교 | `src/10_compare_embedding_variants.py` |
| Arctic/KURE 비교 | `src/11_build_embedding_model_pool.py`, `src/12_compare_embedding_models.py` |
| 전체 임베딩 | `src/08_embed_full_corpus.py` |
| 임베딩 검증 | `src/20_validate_embedding_artifact.py` |

### 검색·이슈 분석

| 단계 | 코드 |
|---|---|
| Chroma 적재 | `src/15_load_chroma.py` |
| pgvector 적재 | `src/16_load_pgvector.py` |
| DB 비교 | `src/17_benchmark_vector_databases.py` |
| 하이브리드 검색 | `src/18_pgvector_retrieval.py` |
| BGE 재순위 | `src/19_rerank_candidates.py` |
| 이슈 확장·군집 | `src/20_expand_cluster_issue.py` |
| 최초 관측·타임라인 | `src/21_build_issue_timeline.py` |
| 예측 특징 | `src/22_build_forecast_features.py` |
| 단일 이슈 예측 | `src/23_forecast_issue.py` |
| 웹 결과 생성 | `src/24_build_web_payload.py` |
| 대표 이슈 후보 | `src/25_build_issue_candidates.py` |
| 군집 임계값 평가 | `src/26_cluster_threshold_sweep.py` |
| 검수 결과 확정 | `src/30_finalize_reviewed_issues.py` |
| 다중 이슈 패널 | `src/31_build_multi_issue_panel.py` |
| 다중 이슈 예측 | `src/32_train_multi_issue_forecast.py` |
| 자유검색 전체 분석 | `src/33_analyze_free_query.py` |
| 최종 웹 번들 | `src/34_build_final_web_bundle.py` |
| BM25 색인 | `src/35_build_bm25_index.py` |

## 4. 검색 원리

1. Arctic가 질의를 1,024차원 벡터로 변환한다.
2. pgvector HNSW가 의미 기반 Dense 후보를 찾는다.
3. SQLite FTS5 BM25가 정책명·조례명·기관명 기반 후보를 찾는다.
4. 두 순위를 RRF로 결합한다.
5. BGE Cross-Encoder가 질의와 후보 문서를 함께 읽고 재순위한다.
6. 동일 본문 및 동일 부모 청크를 제한한다.
7. `parent_chunk_id`로 앞뒤 문맥을 복원한다.

## 5. 이슈와 시간 흐름

- 재순위 상위 발언을 seed로 사용한다.
- seed 벡터 평균으로 중심 벡터를 만든다.
- 전체 DB에서 유사 발언을 확장한다.
- 코사인 유사도 임계값 이상의 발언을 그래프로 연결한다.
- Union-Find 연결요소 중 seed를 포함한 하위 군집을 이슈로 선택한다.
- 구성원 `chunk_id`로 재현 가능한 임시 `issue_id`를 만들고 검수 후 고정한다.
- 의회별 최초 관측일을 계산해 시간 순서로 정렬한다.

지도 화살표는 시간적 관측 순서이며 정책의 인과적 전달을 증명하지 않는다.

## 6. 예측

- 무작위 분할을 사용하지 않는다.
- 각 검증 월보다 이전 월만 학습하는 expanding-window rolling backtest를 사용한다.
- 논의 지속 여부와 다음 달 언급량을 분리해 평가한다.
- Qwen은 수치 예측에 사용하지 않고 계산 결과의 근거 기반 설명만 생성한다.

| 지표 | 값 |
|---|---:|
| 논의 지속 F1 | 0.8814 |
| 언급량 MAE | 7.1224 |
| 언급량 RMSE | 10.0425 |
| 방향성 정확도 | 0.7600 |
| 전월 유지 대비 MAE 개선 | 29.83% |
| 전월 유지 대비 RMSE 개선 | 24.31% |

## 7. 실행

```bash
bash scripts/build_bm25_hybrid.sh
bash scripts/run_representative_issues.sh candidates
bash scripts/run_issue_review.sh
bash scripts/run_representative_issues.sh finalize
bash scripts/run_forecast_evaluation.sh
bash scripts/run_final_app.sh
```

## 8. 공개 저장소 제외 대상

```text
.env
.venv/
data/input/
data/canonical_v3/
data/embeddings/
data/bm25/
data/chroma/
data/postgres/
models/huggingface/
outputs/
```


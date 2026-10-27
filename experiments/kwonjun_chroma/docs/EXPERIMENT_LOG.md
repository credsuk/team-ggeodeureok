# 광역 지방의회 회의록 AI 분석 — 실험 원장

이 문서는 최종 발표, 결과보고서, README 및 질의응답에 사용할 단일 기준 기록이다.
실험 결과를 전달받을 때마다 수치, 실행 조건, 판단, 실패 사례를 함께 누적한다.

## 기록 원칙

- 모델명뿐 아니라 정확한 revision, 입력 데이터, 텍스트 표현, 장치와 dtype을 기록한다.
- 비교 실험은 동일한 청크, 질의, top-k, 평가 라벨을 사용한다.
- 검색 모델 비교는 exact cosine/FAISS 기준으로 수행해 벡터 DB 오차를 분리한다.
- ChromaDB와 pgvector 비교에는 최종 우승 모델의 동일 벡터를 사용한다.
- 자동 평가와 사람 검수 결과를 구분한다.
- 최고 수치만 남기지 않고 실패 실험과 제외 사유도 기록한다.
- 분석 기준일 이후의 외부 정보는 분석·예측 특징에 사용하지 않는다.

## 1. 데이터 및 전처리 기준선

| 항목 | 확정값 |
|---|---:|
| 의회×연도 파일 | 34개 |
| canonical 발화 행 | 2,057,214 |
| canonical 열 | 32개 |
| 최종 검색 청크 | 901,358 |
| 빈 발언 | 0 |
| 깨진 문자 | 0 |
| canonical 생성 | PASS |

canonical 핵심 메타데이터:

- 구조 메타데이터: `council`, `meeting_date`, `committee`, `meeting_name`, `agenda_title`, `speaker`
- LLM 메타데이터: `core_issue`, `keywords_json`, `policy_category`, `summary`, `is_substantive`
- 추적 필드: `chunk_id`, `parent_chunk_id`, `document_id`, `source_key`, 원본 행 범위

## 2. 청킹 기준선

- 부모·자식 청킹 사용
- 자식 청크 최대 길이: 약 950자
- 자식 청크: 검색과 원문 근거
- 부모 청크: 앞뒤 문맥 복원과 이슈 군집화
- raw 임베딩 문자열: 의회·위원회·회의록·안건·발언
- enriched 임베딩 문자열: 정책분야·핵심이슈·키워드 + raw 문자열
- 메타데이터 사용 여부와 메타데이터를 벡터에 포함할지는 별도 결정으로 관리

## 3. 임베딩 평가 표본

| 항목 | 값 |
|---|---:|
| 전체 청크 | 901,358 |
| 평가 표본 | 60,000 |
| source_key | 34 |
| 2024년 | 30,001 |
| 2025년 | 29,999 |
| high | 42,012 |
| medium | 12,002 |
| low | 5,986 |
| 무결성 검사 | PASS |

## 4. Arctic raw/enriched 비교

### 공통 모델

| 항목 | 값 |
|---|---|
| 모델 | `dragonkue/snowflake-arctic-embed-l-v2.0-ko` |
| revision | `55ec6e9358a56d56af759bc8372e970caf8c305f` |
| 차원 | 1,024 |
| 정규화 | L2 normalized |
| 저장 dtype | float16 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |

### raw 60k 임베딩 — 완료

| 항목 | 결과 |
|---|---:|
| 행 | 60,000 |
| 실행 시간 | 1,017.8초 (17분 7.7초) |
| 처리율 | 58.95행/초 |
| norm 평균 | 1.000935 |
| norm 최소 | 0.998047 |
| norm 최대 | 1.003791 |

### enriched 60k 임베딩

- 상태: 결과 수치 입력 대기

### Qwen 자동 관련도 판정

| 항목 | 설정 |
|---|---|
| 모델 | `qwen3:8b` |
| 실행 | Ollama localhost |
| thinking | false |
| 등급 | 0=무관, 1=간접 관련, 2=직접 관련 |
| 판정 | 2회 독립 판정 후 불일치 자동 재심 |
| 상태 | 실행 중/결과 대기 |

### raw/enriched 최종 비교

기록할 지표:

- nDCG@10
- Precision@10
- Direct Precision@10
- Pooled Recall@20
- MRR
- enriched - raw nDCG@10 차이
- query bootstrap 95% 신뢰구간
- 질의별 raw/enriched/tie 승수
- 최종 선택 및 이유

### 최종 결과 (2026-10-22)

| 표현 | nDCG@10 | P@10 | Pooled Recall@20 | MRR |
|---|---:|---:|---:|---:|
| raw | 0.8928 | 0.9750 | 0.5826 | 1.0000 |
| enriched | 0.8871 | 0.9667 | 0.5744 | 0.9896 |

- enriched - raw nDCG@10: `-0.0057`
- query bootstrap 95% CI: `[-0.0224, 0.0110]`
- 질의별 승수: raw 18, enriched 21, tie 9
- 통계 판단: 동률
- 최종 선택: `raw`

통계적으로 유의한 차이는 없지만 raw가 집계 지표 전체에서 근소하게 높다. 또한 LLM이
만든 요약·핵심이슈 문구가 벡터 의미를 과도하게 지배할 위험이 없고 파이프라인이 더
단순하므로 raw를 채택한다. LLM metadata는 버리지 않고 필터·이슈명·군집 검수·결과
표시·시계열 feature에 사용한다.

## 5. 임베딩 모델 2종 비교

raw/enriched 중 우승 텍스트를 고정한 뒤 아래 모델을 비교한다.

1. `dragonkue/snowflake-arctic-embed-l-v2.0-ko`
2. `nlpai-lab/KURE-v1`

`BAAI/bge-m3`는 dense·sparse·ColBERT를 함께 사용하는 심화 하이브리드 검색
후보로 보존하되, 제출 일정 내 dense 임베딩 선별 실험에서는 제외한다.

### 공정 비교 조건

- 동일한 60,000개 `chunk_id`
- 동일한 텍스트 표현
- 모델별 공식 query/document 인코딩 적용
- dense embedding 비교
- 전체 48개 질의 중 정책분야를 고르게 포함한 대표 질의 24개 사용
- 각 모델 top-10 합집합으로 blind pool 생성
- 모델명·순위·유사도 점수를 숨긴 상태로 Qwen3 단일 판정
- exact cosine 또는 FAISS IndexFlatIP 기준

시간 절약형 실험의 목적은 논문급 최종 순위가 아니라 전체 901,358개를
재임베딩하기 전 명백히 열세인 모델을 제외하는 것이다. top-10 밖의 pooled recall은
최종 판단 지표에서 제외하고, nDCG@10·Direct P@10·P@10·MRR을 중심으로 판단한다.

### 기록할 정확도 지표

| 모델 | nDCG@10 | P@10 | Direct P@10 | Pooled Recall@20 | MRR | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| Arctic | 대기 | 대기 | 대기 | 대기 | 대기 | 대기 |
| KURE | 대기 | 대기 | 대기 | 대기 | 대기 | 대기 |

### 기록할 운영 지표

| 모델 | 문서 처리율 | 질의 p50/p95 | 최대 VRAM | 모델 크기 | 벡터 차원 |
|---|---:|---:|---:|---:|---:|
| Arctic | 대기 | 대기 | 대기 | 대기 | 1,024 |
| KURE | 대기 | 대기 | 대기 | 대기 | 1,024 |

최종 결정 규칙:

1. nDCG@10이 2위보다 0.02 이상 높고 P@10이 악화되지 않으면 해당 모델
2. 차이가 0.02 미만이면 실질적 동률로 처리
3. 동률이면 처리 속도·VRAM·구조 단순성이 좋은 모델
4. 최종 우승 모델 하나로 전체 901,358개 임베딩
5. 최종 보고서에는 시간 절약형 선별 실험임을 명시

## 6. 이후 추가할 실험

- ChromaDB vs PostgreSQL+pgvector: Recall@k, p50/p95/p99, 적재시간, 인덱스 크기
- Dense vs BM25 vs Hybrid RRF vs Reranker
- 이슈 군집화: 군집 일관성, 중복 이슈, unassigned 비율, 안정성
- 최초 관측: 회의록명·날짜·원문 일치율
- 전파 추적: 의회별 최초 관측 순서 검수
- 예측: Logistic Regression vs LightGBM, F1, PR-AUC, Brier, MAE
- 근거 생성: 날짜·지역·회의록명·확률값 hallucination 검사

## 7. 최종 발표에서 설명할 핵심 의사결정

1. raw/enriched는 메타데이터 사용 여부가 아니라 메타데이터를 벡터에 포함할지 비교한 실험이다.
2. 메타데이터는 어떤 텍스트 표현을 선택해도 필터, 재순위, 군집명, 최초 관측, 시계열, 예측, 화면 표시에 사용한다.
3. 공개 벤치마크만 믿지 않고 지방의회 회의록 도메인 평가로 Arctic와 KURE를 재검증한다.
4. 임베딩 모델 성능과 벡터 DB 성능을 분리해 측정한다.
5. 예측값은 통계·머신러닝 모델이 계산하고 Qwen은 근거 기반 설명만 생성한다.

## 8. 미확정 항목

- enriched 60k 임베딩 시간 및 처리율
- 09번 Qwen 판정 결과와 등급 분포
- raw/enriched 비교 결과
- 세 임베딩 모델 비교 결과
- 전체 임베딩 우승 모델 및 텍스트 표현
- Chroma/pgvector 우승 구성
- 최종 군집화 알고리즘
- 최종 예측 모델

## 9. 전체 임베딩 이후 구현 완료 항목 (2026-10-22)

아래 코드는 최종 임베딩 모델과 text view가 확정되면 경로만 지정해 그대로 실행한다.

- `20_validate_embedding_artifact.py`: 행·차원·ID·필수 metadata·NaN/Inf·norm 검사
- `21_load_chroma_final.py`: 동일 float16 임베딩의 Chroma cosine 적재 및 재개
- `22_load_pgvector_final.py`: 동일 임베딩의 pgvector HNSW 적재 및 필터 인덱스
- `23_search_evidence.py`: 지역·연도·위원회·기간 필터와 부모 문맥을 포함한 근거 JSON
- `24_benchmark_vector_databases.py`: exact 기준 Recall@K/Top-1 및 p50/p95/p99 비교
- `api/app.py`: 근거 검색 API
- `demo/app.py`: 지도·필터·원문 근거·JSON 다운로드 화면
- `agent/tools.py`: 로컬 Agent가 호출할 `search_council_evidence` 도구

두 DB는 임베딩을 각각 생성하지 않고 `embeddings.f16.npy`와
`chunk_metadata.parquet`을 공동 입력으로 사용한다. DB 비교 결과는 임베딩 모델 비교와
분리하여 기록한다.


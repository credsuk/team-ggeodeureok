# 모델 구성 (model/)

본 디렉토리는 **의정이슈레이더 (Council Issue Radar)** 프로젝트에서 사용하는 핵심 모델(텍스트 임베딩, 교차 인코더 재순위, 로컬 설명 LLM, 시계열 통계 예측 모델, 벡터 데이터베이스 인덱스)의 메타데이터 명세와 아키텍처 가이드를 관리합니다.

저장소 경량화 및 실행 환경 재현성을 위해 수 GB 단위의 사전학습 모델 가중치와 런타임 학습 결과물은 Git 저장소에 직접 포함하지 않으며, 단일 진실 공급원(Single Source of Truth) 역할을 하는 기계 판독용 명세서([`MODEL_MANIFEST.json`](MODEL_MANIFEST.json))와 자동화 스크립트를 통해 온디맨드로 준비·생성되도록 설계되어 있습니다.

---

## 1. 디렉토리 구성 및 상태

현재 Git 저장소에는 명세 및 안내 파일만 포함되어 있으며, 가중치와 런타임 산출물은 실행 시 지정된 경로에 자동 생성 또는 캐시됩니다.

| 경로 | Git 관리 여부 | 내용 및 설명 |
|---|:---:|---|
| [`MODEL_MANIFEST.json`](MODEL_MANIFEST.json) | **포함** | 모델명, 고정 리비전(커밋 해시), 임베딩 차원, 검증 방식, 벡터 DB 인덱스 파라미터 등 프로젝트 전체 모델 메타데이터 명세서 |
| [`README.md`](README.md) | **포함** | 본 안내 문서 (모델 아키텍처, 역할 및 다운로드·재현 가이드) |
| `huggingface/` | *런타임 생성* | 사전학습 모델(임베딩·리랭커) 캐시 디렉토리 (`scripts/download_models.py` 실행 시 자동 다운로드) |
| `forecast/` | *파이프라인 생성* | 다음 달 언급량 예측 모델(`poisson_regression.joblib`) 및 지표 파일 위치 (`notebooks/07_council_issue_radar.ipynb` 파이프라인 실행 시 생성) |

---

## 2. 모델별 세부 역할 및 사양

[`MODEL_MANIFEST.json`](MODEL_MANIFEST.json)에 정의된 5가지 핵심 컴포넌트의 사양과 역할은 다음과 같습니다:

### 1) 한국어 Dense 임베딩
- **모델**: `dragonkue/snowflake-arctic-embed-l-v2.0-ko`
- **고정 리비전**: `55ec6e9358a56d56af759bc8372e970caf8c305f`
- **사양**: 1,024차원 정규화 벡터, 최대 2,048 토큰 컨텍스트
- **역할**: 계층형 부모-자식(Parent-Child) 청크와 사용자 질의를 고밀도 벡터로 임베딩하여 유사 의미 구간을 1차 검색합니다. 저장 시에는 `float16`, 질의 런타임 연산 시에는 `float32`를 적용합니다.

### 2) Cross-Encoder 재순위 (Reranker)
- **모델**: `BAAI/bge-reranker-v2-m3` (`XLMRobertaForSequenceClassification`)
- **역할**: Dense(임베딩)와 Sparse(BM25)의 RRF(Reciprocal Rank Fusion) 하이브리드 검색으로 도출된 후보 문서들을 질의와 직접 쌍(Pair)으로 정밀 비교하여 점수를 재산출하고 최적의 근거를 선별합니다.

### 3) 로컬 설명 Agent LLM
- **모델**: SK텔레콤 `A.X-4.0-Light 7B` (Ollama 식별자: `hf.co/mykor/A.X-4.0-Light-gguf:Q4_K_M`)
  - *대안 모델*: LG AI Research `exaone3.5:7.8b`
- **역할**: 검색·군집·시계열 통계 예측 결과를 종합하여, 인용 가능한 회의록 근거 번호(`[근거 N]`)와 함께 한국어 설명 리포트를 생성합니다.
- **원칙**: 회의록 원문을 외부 상용 API로 전송하지 않고 로컬 Ollama 환경(GPU)에서만 안전하게 실행합니다.

### 4) 시계열 통계 예측 모델 (Forecast)
- **논의 지속 여부 (분류)**: **Active Ratio Baseline** (최근 3개월 활동 비율 기준선 규칙 기반)
- **다음 달 언급량 (회귀)**: **Poisson Regression** (`scikit-learn` `PoissonRegressor` 파이프라인)
- **검증 체계**: 10-Fold 시간순 롤링 백테스트(Chronological Rolling Backtest)로 데이터 누수를 차단한 상태에서 일반화 성능을 검증하였습니다.

### 5) 벡터 데이터베이스 (Vector DB)
- **엔진**: PostgreSQL 17 + `pgvector` 0.8.7
- **테이블**: `council_arctic_raw_v1`
- **인덱스**: HNSW (`m=16`, `ef_construction=64`, 코사인 거리 메트릭)

---

## 3. 사전학습 모델 다운로드 (Hugging Face)

대용량 가중치(약 4.5GB)는 아래 스크립트를 실행하여 로컬 `model/huggingface/` 디렉토리에 내려받을 수 있습니다:

```bash
# 기본 필수 모델 다운로드 (임베딩 + 재순위 모델)
python scripts/download_models.py

# 비교 실험용 모델(KURE, BGE-M3)까지 다운로드할 경우
python scripts/download_models.py --comparison
```

다운로드 대상 모델 정보:
- `dragonkue/snowflake-arctic-embed-l-v2.0-ko` (Revision: `55ec6e9358a56d56af759bc8372e970caf8c305f`)
- `BAAI/bge-reranker-v2-m3` (Revision: `main`)

---

## 4. 로컬 LLM 환경 설정 (Ollama)

생성형 AI 모델은 로컬 Ollama(`http://127.0.0.1:11434`) 인스턴스를 통해 구동됩니다:

```bash
# 설명 Agent 주력 모델 설치 및 실행
ollama run hf.co/mykor/A.X-4.0-Light-gguf:Q4_K_M

# 대안 모델 (필요 시)
ollama pull exaone3.5:7.8b
```

> **참고**: `python scripts/download_models.py --ollama` 명령을 사용하면 프로젝트에 필요한 Ollama 모델들을 일괄 내려받을 수 있습니다.

---

## 5. 시계열 예측 모델 재현

시계열 예측 모델은 사전 학습된 정적 바이너리를 내려받을 필요 없이 파이프라인 내에서 즉시 학습 및 추론됩니다:

- **실행 위치**: [`notebooks/07_council_issue_radar.ipynb`](../notebooks/07_council_issue_radar.ipynb) (섹션 8. 미래 시점 이슈 예측)
- **학습 스크립트**: `src/32_train_multi_issue_forecast.py`
- 노트북을 순차 실행하면 과거 발언 시계열 데이터를 기반으로 포아송 회귀 모델이 학습되어 메모리 및 `model/forecast/`에 저장되며, 다음 달 예상 언급량과 논의 지속 확률이 도출됩니다.

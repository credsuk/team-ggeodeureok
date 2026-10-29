# 🏛️ 의정이슈레이더 (Local Council Issue Radar)

> **광역 지방의회 회의록 시맨틱·하이브리드 검색 및 정책 이슈 시계열 확산 분석 플랫폼**

**의정이슈레이더**는 전국 광역 지방의회(17개 시·도의회 등)의 대규모 회의록 데이터를 수집·정제·청킹하고, 고성능 한국어 임베딩(`Snowflake Arctic v2.0 ko`)과 **PostgreSQL/pgvector + BM25 하이브리드 검색**, **BGE Cross-Encoder 재순위**, **시계열 이슈 그래프 군집화**, 그리고 **로컬 LLM(Ollama)**을 결합하여 근거 중심의 정책 분석과 웹 대시보드를 제공하는 엔드투엔드(End-to-End) 로컬 AI 플랫폼입니다.

---

## 📑 목차

1. [시스템 아키텍처](#1-시스템-아키텍처)
2. [사전 준비사항 (Prerequisites)](#2-사전-준비사항-prerequisites)
3. [환경 설정 (`.env`)](#3-환경-설정-env)
4. [🚀 빠른 실행 가이드 (How to Run)](#4--빠른-실행-가이드-how-to-run)
   - [방법 A. Docker Compose로 실행 (가장 권장)](#방법-a-docker-compose로-실행-가장-권장)
   - [방법 B. 로컬 Python 환경에서 직접 실행 (개발 및 디버깅)](#방법-b-로컬-python-환경에서-직접-실행-개발-및-디버깅)
   - [방법 C. Streamlit 분석 데모 앱 실행](#방법-c-streamlit-분석-데모-앱-실행)
5. [데이터베이스(pgvector) 복원 및 준비](#5-데이터베이스pgvector-복원-및-준비)
6. [BM25 하이브리드 인덱스 구축](#6-bm25-하이브리드-인덱스-구축)
7. [주요 엔드포인트 및 API 명세](#7-주요-엔드포인트-및-api-명세)
8. [전체 데이터 파이프라인 (`src/`)](#8-전체-데이터-파이프라인-src)
9. [프로젝트 디렉토리 구조](#9-프로젝트-디렉토리-구조)
10. [자주 묻는 질문 및 트러블슈팅 (FAQ)](#10-자주-묻는-질문-및-트러블슈팅-faq)

---

## 1. 시스템 아키텍처

```text
[ 광역의회 회의록 원본 (HTML/JSON) ]
                │
                ▼ (정제 및 전처리)
[ 표준 Canonical 발언 코퍼스 (901,358 청크) ]
                │
                ├───────────────────────────────────────┐
                ▼ (Arctic v2.0 임베딩)                   ▼ (SQLite FTS5)
      [ pgvector Dense HNSW ]                     [ BM25 Sparse 검색 ]
                │                                       │
                └───────────────────┬───────────────────┘
                                    ▼
                     [ RRF (Reciprocal Rank Fusion) ]
                                    ▼ (Top-50 후보)
                     [ BGE Reranker-v2-m3 재순위 ]
                                    ▼ (Top-10 최종 근거)
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
[ 타임라인·전파 순서·시계열 예측 ]                  [ 로컬 LLM (Ollama Qwen) ]
         │                                                     │
         └──────────────────────────┬──────────────────────────┘
                                    ▼
         [ 웹 인터페이스 (FastAPI + SPA / Streamlit 데모) ]
```

- **임베딩 모델**: `dragonkue/snowflake-arctic-embed-l-v2.0-ko` (1,024차원, 행정/공공 문서 특화)
- **벡터 데이터베이스**: PostgreSQL 17 + `pgvector` 0.8.7 (HNSW 코사인 인덱스)
- **키워드 검색**: SQLite FTS5 BM25 + Reciprocal Rank Fusion(RRF)
- **재순위 모델**: `BAAI/bge-reranker-v2-m3` Cross-Encoder
- **로컬 생성 LLM**: Ollama 기반 `qwen3:8b` (외부 API 전송 없는 100% 로컬 보안 환경)

---

## 2. 사전 준비사항 (Prerequisites)

시스템을 구동하기 위해 아래 도구들이 필요합니다.

1. **Docker & Docker Compose**: PostgreSQL(pgvector) 컨테이너 및 웹 컨테이너 구동용 ([Docker Desktop 설치](https://www.docker.com/products/docker-desktop/))
2. **Python**: 3.11 또는 3.12 권장
3. **NVIDIA GPU 및 CUDA** (선택, 권장): 임베딩 및 재순위 모델 고속 연산을 위해 CUDA 지원 GPU 권장
4. **Ollama**: 로컬 LLM 구동용 ([Ollama 공식 사이트](https://ollama.ai/))
   ```bash
   # Ollama 설치 후 질의응답 모델 다운로드
   ollama pull qwen3:8b
   ```

---

## 3. 환경 설정 (`.env`)

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 아래 항목들을 설정합니다.

```ini
# ==========================================
# 1. 데이터베이스 설정 (PostgreSQL + pgvector)
# ==========================================
# [Docker 내부 실행 시]
# DATABASE_URL=postgresql://council:council-local-only@council-pgvector:5432/council_vectors

# [로컬(호스트 PC) 직접 실행 시 (도커 포트 5433 매핑 기준)]
DATABASE_URL=postgresql://council:council-local-only@localhost:5433/council_vectors

POSTGRES_USER=council
POSTGRES_PASSWORD=council-local-only
POSTGRES_DB=council_vectors
POSTGRES_PORT=5433
DB_TABLE=public.council_arctic_raw_v1

# ==========================================
# 2. 임베딩 모델 설정
# ==========================================
# Hugging Face 모델 허브 식별자 또는 다운로드된 로컬 디렉토리 경로
EMBEDDING_MODEL_PATH=dragonkue/snowflake-arctic-embed-l-v2.0-ko

# ==========================================
# 3. 로컬 LLM (Ollama) 설정
# ==========================================
# [Docker 내부 실행 시]
# OLLAMA_BASE_URL=http://host.docker.internal:11434

# [로컬 직접 실행 시]
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b

# ==========================================
# 4. 검색 엔진 모드 설정 (Dense / Hybrid)
# ==========================================
COUNCIL_RETRIEVAL_MODE=hybrid
COUNCIL_BM25_INDEX=data/bm25/arctic_raw_v1/bm25.sqlite3
COUNCIL_RRF_K=60
COUNCIL_DENSE_WEIGHT=1.0
COUNCIL_BM25_WEIGHT=1.0

# ==========================================
# 5. 프론트엔드 지도 API 키 (선택)
# ==========================================
CARTO_API_KEY=your_carto_api_key_here
```

---

## 4. 🚀 빠른 실행 가이드 (How to Run)

### 방법 A. Docker Compose로 실행 (가장 권장)

Docker를 사용하여 데이터베이스와 웹 애플리케이션을 간편하게 빌드하고 실행할 수 있습니다.

#### 1단계: pgvector 데이터베이스 컨테이너 구동
```bash
docker compose up -d pgvector
```
- 컨테이너 상태 확인: `docker compose ps`
- 기본적으로 호스트의 `5433` 포트가 컨테이너의 PostgreSQL `5432` 포트에 바인딩됩니다.

#### 2단계: 웹 서비스 컨테이너 구동
```bash
docker compose -f docker-compose.web.yml up -d --build
```
- Docker 내부에서 실행되므로 `.env`의 `DATABASE_URL`을 `postgresql://council:council-local-only@council-pgvector:5432/council_vectors`로 설정하거나 `docker-compose.web.yml` 환경변수를 따릅니다.

#### 3단계: 브라우저 접속
- **웹 대시보드 URL**: [http://localhost:8000](http://localhost:8000)
- 회의록 검색, 로컬 AI 요약 답변, 정책 이슈 레이더 시각화를 바로 이용할 수 있습니다.

---

### 방법 B. 로컬 Python 환경에서 직접 실행 (개발 및 디버깅)

로컬 머신(Windows / macOS / Linux)에서 소스코드를 직접 수정하며 디버깅할 때 사용합니다.

#### 1단계: pgvector DB 컨테이너 실행
```bash
docker compose up -d pgvector
```

#### 2단계: Python 가상환경 구성 및 패키지 설치
```bash
# 가상환경 생성 (Python 3.11 권장)
conda create -n council python=3.11 -y
conda activate council

# 필수 패키지 설치
pip install -r requirements.txt

# (권장) NVIDIA CUDA 가속 PyTorch 설치 (CUDA 12.1 기준)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 3단계: Ollama 백엔드 실행 확인
별도 터미널에서 Ollama가 정상 실행 중인지 확인합니다:
```bash
ollama run qwen3:8b
```

#### 4단계: FastAPI 백엔드 및 웹 프론트엔드 실행
```bash
# Uvicorn 개발 서버 실행
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
- 브라우저에서 [http://localhost:8000](http://localhost:8000)으로 접속합니다.
- `main.py`에 의해 `frontend/`의 단일 페이지 웹 애플리케이션(SPA)이 자동 서빙됩니다.

---

### 방법 C. Streamlit 분석 데모 앱 실행

의정이슈레이더의 대표 10대 이슈 타임라인, 지도 전파 시각화, 지자체 통계 지표, 로컬 에이전트 인터뷰 기능을 담은 Streamlit 데모 앱입니다.

```bash
# Streamlit 데모 실행 (포트 8501)
streamlit run demo/final_app.py --server.port 8501
```
- **접속 주소**: [http://localhost:8501](http://localhost:8501)
- **기능**:
  - 대표 이슈 10선 타임라인 분석 및 지도 전파 경로 시각화
  - KOSIS 국가통계포털 연계 지역 통계 지표 비교
  - 이슈 지속 여부(Classification) 및 언급량(Regression) 예측 결과 조회
  - Qwen 에이전트 정책 질의응답

---

## 5. 데이터베이스(pgvector) 복원 및 준비

공유받은 PostgreSQL 백업 덤프(`*.dump`) 파일이 있는 경우 아래 절차로 복원합니다.

### 1) DB 덤프 복원 스크립트 실행
```bash
# Linux / WSL 환경
bash scripts/restore_pgvector_from_dump.sh <다운로드한_덤프파일_경로>

# 기존 DB 테이블을 덮어써야 하는 경우 (--replace)
bash scripts/restore_pgvector_from_dump.sh <덤프파일_경로> --replace
```

### 2) 복원 상태 및 데이터 건수 확인
```bash
docker compose exec pgvector psql -U council -d council_vectors -c "SELECT count(*) FROM council_arctic_raw_v1;"
```
- 정상 복원 시 총 **901,358**개의 청크 행이 확인됩니다.

---

## 6. BM25 하이브리드 인덱스 구축

Dense 벡터 검색과 함께 어휘 기반 키워드 검색(BM25)을 결합하여 고유명사, 조례명, 법률 번호 등의 검색 정확도를 극대화합니다.

```bash
# 1. 10,000건 샘플로 색인 테스트
python src/35_build_bm25_index.py --limit 10000 --output data/bm25/arctic_raw_v1_test/bm25.sqlite3 --rebuild

# 2. 전체 901,358개 청크 인덱스 빌드
python src/35_build_bm25_index.py --rebuild
```
> 💡 BM25 인덱스 파일(`data/bm25/arctic_raw_v1/bm25.sqlite3`)이 존재하지 않아도, 시스템은 자동으로 pgvector Dense 단독 검색 모드로 안전하게 폴백(Fallback) 동작합니다.

---

## 7. 주요 엔드포인트 및 API 명세

FastAPI 서버(`main.py`)가 제공하는 핵심 REST API 엔드포인트입니다.

### 1) 헬스체크 (`GET /api/health`)
- **응답**: `{"status": "ok"}`

### 2) 통합 회의록 시맨틱 검색 (`POST /api/search`)
- **요청 Body**:
  ```json
  {
    "query": "청년 일자리 및 공공 임대주택 지원 조례",
    "limit": 5,
    "council": "경기도의회",
    "year": 2024
  }
  ```
- **응답 필드**:
  - `answer`: Ollama Qwen이 회의록 근거를 바탕으로 생성한 종합 답변
  - `results`: 검색된 상위 회의록 목록 (발언 일자, 의회명, 위원회, 안건, 유사도 스코어, 발언 본문)
  - `count`: 검색 결과 건수
  - `elapsed_ms`: 처리 소요 시간 (밀리초)

### 3) 동적 이슈 확산 분석 (`POST /api/analyze`)
- **요청 Body**:
  ```json
  {
    "query": "인공지능 첨단 모빌리티 혁신"
  }
  ```
- **동작**: 백그라운드에서 `src/33_analyze_free_query.py` 파이프라인을 실행하여 후보 검색 → 재순위 → 유사도 그래프 군집화 → 최초 관측 타임라인 및 시계열 확산 JSON을 실시간 반환합니다.

---

## 8. 전체 데이터 파이프라인 (`src/`)

프로젝트는 회의록 원본에서 웹 서비스까지 단계별 모듈로 체계화되어 있습니다.

| 번호 | 스크립트 | 설명 |
| :---: | :--- | :--- |
| **01~03** | `src/01_restart_inventory.py` ~ `03_validate_row_id_join.py` | 원본 회의록 TSV 및 메타데이터 무결성 검증 |
| **04** | `src/04_build_direct_canonical.py` | 17개 지자체 통합 Canonical 코퍼스 구축 (`corpus.parquet`) |
| **05** | `src/05_build_parent_child_chunks.py` | 발언 단위 부모 청크 및 검색용 자식 서브청킹 (총 901,358개) |
| **06~12** | `src/06_make_embedding_eval_sample.py` ~ `12_compare_embedding_models.py` | 임베딩 성능 평가(Arctic vs KURE 등) 및 검증 |
| **08** | `src/08_embed_full_corpus.py` | 확정 모델(`Snowflake Arctic v2.0 ko`) 전체 코퍼스 임베딩 |
| **16** | `src/16_load_pgvector.py` | 벡터 및 메타데이터 PostgreSQL/pgvector HNSW 색인 적재 |
| **17** | `src/17_benchmark_vector_databases.py` | pgvector vs ChromaDB 속도 및 재현율(Recall) 벤치마크 |
| **18** | `src/18_pgvector_retrieval.py` | Dense + BM25 + RRF 후보 검색 파이프라인 |
| **19** | `src/19_rerank_candidates.py` | BGE Cross-Encoder를 통한 후보 재순위화 |
| **20~21** | `src/20_expand_cluster_issue.py` ~ `21_build_issue_timeline.py` | 중심 벡터 확장, 코사인 유사도 그래프 군집화 및 타임라인 도출 |
| **22~23** | `src/22_build_forecast_features.py` ~ `23_forecast_issue.py` | 시계열 특징 추출 및 다음 달 이슈 지속·언급량 예측 |
| **32** | `src/32_train_multi_issue_forecast.py` | 다중 이슈 패널 기반 예측 평가 (Logistic / Poisson 모델 백테스트) |
| **33** | `src/33_analyze_free_query.py` | 임의 검색어에 대한 엔드투엔드 동적 이슈 분석 자동화 |
| **35** | `src/35_build_bm25_index.py` | SQLite FTS5 기반 고속 BM25 인덱스 생성 |

---

## 9. 프로젝트 디렉토리 구조

```plaintext
c:\study\Projects\01\branches\gwonwoo/
├── main.py                     # FastAPI 메인 웹/API 서버 (검색 & 동적 분석 서빙)
├── config.py                   # 프로젝트 전역 환경 변수 로더 및 설정 검증기
├── database.py                 # pgvector Dense 및 SQLite BM25 RRF 검색 모듈
├── embedding.py                # Arctic v2.0 로컬 임베딩 로드 및 인코더
├── llm.py                      # Ollama 연동 로컬 LLM 추론 및 답변 생성기
├── frontend.py                 # 정적 파일 서빙 및 Standalone 대시보드 빌더
│
├── frontend/                   # 웹 대시보드 정적 파일
│   ├── index.html              # SPA 대시보드 메인 마크업
│   ├── app.js                  # 대시보드 UI 이벤트, 지도 렌더링, API 호출 로직
│   └── dashboard_standalone.html # 오프라인 단독 구동용 통합 번들
│
├── agent/                      # 로컬 자율 분석 에이전트 패키지
│   ├── local_issue_agent.py    # 회의록 분석 및 근거 탐색 에이전트
│   ├── query_pipeline.py       # 하이브리드 검색 및 쿼리 파이프라인
│   └── followup_tools.py       # 후속 정책 분석 도구
│
├── demo/                       # 시연 및 데모 애플리케이션
│   ├── final_app.py            # Streamlit 기반 통합 의정이슈레이더 대시보드
│   └── issue_review_app.py     # 이슈 클러스터 검수용 관리자 앱
│
├── src/                        # 핵심 파이프라인 단계별 모듈 (01 ~ 35)
│   ├── bm25_runtime.py         # BM25 및 RRF 랭킹 알고리즘 구현체
│   ├── council_vector_runtime.py # 벡터 런타임 공통 모듈
│   └── 33_analyze_free_query.py # 동적 자유 질의 분석기
│
├── scripts/                    # 실행 및 자동화 쉘 스크립트
│   ├── 00_check_environment.py # OS, GPU, CUDA 환경 점검 도구
│   ├── export_pgvector_for_share.sh # DB 덤프 추출 스크립트
│   └── restore_pgvector_from_dump.sh # DB 덤프 복원 스크립트
│
├── config/                     # 정책 시드 목록, 데이터 규약, KOSIS 지표 매핑 YAML
├── docker-compose.yml          # PostgreSQL + pgvector 컨테이너 정의
├── docker-compose.web.yml      # FastAPI 웹 서비스 컨테이너 정의
├── Dockerfile                  # Web 서비스 컨테이너 빌드 파일
├── requirements.txt            # Python 필수 의존성 목록
└── README.md                   # 프로젝트 종합 설명서 (본 문서)
```

---

## 10. 자주 묻는 질문 및 트러블슈팅 (FAQ)

### Q1. `Docker DB council-pgvector:5432/council_vectors 연결이 필요합니다` 오류가 발생합니다.
- `main.py`의 `lifespan` 검증 로직은 Docker 내부 통신 규격(`council-pgvector:5432`)을 기본 대상으로 점검합니다.
- 호스트(로컬)에서 직접 `uvicorn main:app`을 실행하는 경우, `.env`의 `DATABASE_URL`을 로컬 포트(`localhost:5433`)에 맞추고 `main.py` 42~49행의 호스트네임 검사 부분을 로컬 환경에 맞게 조정하거나 Docker 컨테이너 환경(`docker compose -f docker-compose.web.yml up`)에서 실행하세요.

### Q2. `EMBEDDING_MODEL_PATH is missing` 오류가 발생합니다.
- `.env` 파일에 `EMBEDDING_MODEL_PATH`가 지정되어 있는지 확인하세요.
- 기본값: `EMBEDDING_MODEL_PATH=dragonkue/snowflake-arctic-embed-l-v2.0-ko`

### Q3. Ollama 연결 오류 또는 AI 답변이 실패합니다 (`Ollama 호출 실패`).
- Ollama 데몬이 실행 중인지 확인합니다:
  ```bash
  curl http://localhost:11434/api/tags
  ```
- 지정된 모델(`qwen3:8b` 등)이 로컬에 다운로드되어 있는지 확인하세요:
  ```bash
  ollama list
  ollama pull qwen3:8b
  ```
- Docker 컨테이너 내부에서 로컬 머신의 Ollama에 접근할 때는 `http://host.docker.internal:11434`를 사용해야 합니다.

### Q4. PyTorch에서 GPU를 인식하지 못하고 CPU로 동작합니다.
- `scripts/00_check_environment.py`를 실행하여 환경을 점검합니다:
  ```bash
  python scripts/00_check_environment.py
  ```
- CUDA 버전에 맞는 PyTorch를 다시 설치합니다:
  ```bash
  pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu121
  ```

### Q5. Windows PowerShell에서 한글 출력 깨짐 현상이 있습니다.
- 터미널 세션의 인코딩을 UTF-8로 지정합니다:
  ```powershell
  [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
  $OutputEncoding = [System.Text.Encoding]::UTF8
  ```

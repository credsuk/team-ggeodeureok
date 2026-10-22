# 🏛️ 지방의회 정책 확산 분석, 벡터 색인 및 미래 예측 시스템 (`team-ggeodeureok`)

지방의회(광역의회: 경기도, 충청남도, 충청북도, 강원특별자치도 등 34개 지자체) 회의록 데이터를 **수집·정제·하이브리드 조인**하고, 로컬 고성능 **PostgreSQL (`pgvector`)**에 벡터 색인하여 **정책 의제 시계열 확산 분석 및 미래 정책 예측**을 수행하는 엔드투엔드(End-to-End) 오픈소스 파이프라인입니다.

본 프로젝트는 상용 유료 API(OpenAI, Claude 등)를 사용하지 않고 **100% 로컬 환경(Ollama 오픈 LLM + Hugging Face 로컬 임베딩)**에서 독립적으로 구동됩니다.

---

## 📑 목차
1. [프로젝트 핵심 특징](#1-프로젝트-핵심-특징)
2. [Conda 기반 빠른 환경 구축 가이드](#2-conda-기반-빠른-환경-구축-가이드)
3. [디렉토리 구조](#3-디렉토리-구조)
4. [하이브리드 데이터 파이프라인 (`data/` + `change_text/`)](#4-하이브리드-데이터-파이프라인-data--change_text)
5. [PostgreSQL pgvector 데이터 적재 (`run_ingest.py`)](#5-postgresql-pgvector-데이터-적재-run_ingestpy)
6. [대회 제출 규격 정책 확산 및 미래 예측 모델 (`run_trend_analysis.py`)](#6-대회-제출-규격-정책-확산-및-미래-예측-모델-run_trend_analysispy)
7. [로컬 LLM 엔드투엔드 RAG 질의 (`rag_query.py`)](#7-로컬-llm-엔드투엔드-rag-질의-rag_querypy)
8. [시맨틱 검색 및 메타데이터 필터링 (`query_search.py`)](#8-시맨틱-검색-및-메타데이터-필터링-query_searchpy)
9. [로컬 LLM 정책 메타데이터 추출기 (`extract_llm_metadata.py`)](#9-로컬-llm-정책-메타데이터-추출기-extract_llm_metadatapy)
10. [임베딩 모델 손쉬운 변경 가이드](#10-임베딩-모델-손쉬운-변경-가이드)
11. [핵심 모듈 상세 안내 (`src/`)](#11-핵심-모듈-상세-안내-src)
12. [자주 묻는 질문 및 트러블슈팅](#12-자주-묻는-질문-및-트러블슈팅)
13. [작업 이력 (CHANGELOG)](#13-작업-이력-changelog)

---

## 1. 프로젝트 핵심 특징

- **PostgreSQL (`pgvector`) 기본 탑재**: 대규모 회의록 벡터 및 메타데이터를 안정적으로 관리하는 로컬 PostgreSQL 17 + `pgvector` 확장을 기본 벡터 DB로 채택 (`ChromaDB` 완벽 호환 지원).
- **하이브리드 데이터 융합 (Hybrid Loader)**: 형태소 기반 Kiwi 전처리 메타데이터(`data/*.jsonl`: 실질발언 판별, 카테고리, 쟁점, 키워드, 요약)와 회의록 원문(`change_text/*.txt`: 발언 본문, 공식 회의록명 `MINTS_NM`, 회의구분 `MTGNM`)을 `row_id` 기준 1:1 결합하여 검색 노이즈를 40% 이상 제거.
- **경진대회 주최 측 제출 규격 자동 생성**: 정책 이슈 키워드 입력 시 `최초 등장시점` $\rightarrow$ `타 지방의회로의 확산 흐름` $\rightarrow$ `향후 정책 예측` $\rightarrow$ `근거 회의록(`MINTS_NM`, 일자, 180자 `used_text_span` 발췌)` 표준 마크다운 표 및 JSON 즉시 산출.
- **최신 고성능 한국어 임베딩 기본 탑재**: 행정·공공문서 학습이 반영된 `dragonkue/snowflake-arctic-embed-l-v2.0-ko` 기본 지원 및 `nlpai-lab/KURE-v1` 등 다양한 프리셋 지원.
- **100% 로컬 오픈 LLM**: Ollama 기반 로컬 LLM(`gemma4:e2b`, `qwen2.5-coder:7b` 등)을 활용하여 비용 0원, 데이터 외부 유출 없이 동작.
- **`.env` 환경변수 자동 관리**: `python-dotenv`를 통해 데이터베이스 접속 정보 자동 주입.

---

## 2. Conda 기반 빠른 환경 구축 가이드

처음 프로젝트를 클론받아 설정하는 사용자를 위한 단계별 가이드입니다.

### Step 1. Conda 가상환경 생성 및 활성화
본 프로젝트는 **Python 3.11** 환경을 권장합니다.

```bash
# 가상환경 'project01' 생성 (Python 3.11)
conda create -n project01 python=3.11 -y

# 가상환경 활성화
conda activate project01
```

### Step 2. PyTorch 설치 (GPU 가속 권장)
NVIDIA 그래픽 카드가 장착된 환경에서는 임베딩 계산 속도를 극대화하기 위해 CUDA 지원 PyTorch를 설치합니다:

```bash
# [NVIDIA GPU 환경 - CUDA 12.1 권장]
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# [CPU 전용 환경]
pip install torch torchvision
```

### Step 3. 필수 의존성 패키지 설치
프로젝트 루트 디렉토리에서 패키지 목록을 설치합니다:

```bash
pip install -r requirements.txt
```

> **주요 설치 패키지**:
> - `pgvector>=0.2.5`, `psycopg2-binary>=2.9.9`: PostgreSQL pgvector 통신
> - `python-dotenv>=1.0.0`: `.env` 파일 환경변수 자동 로드
> - `langchain>=0.2.0`, `langchain-community>=0.2.0`, `langchain-text-splitters>=0.0.1`
> - `sentence-transformers<4.0.0`, `transformers<5.0.0`: Hugging Face 임베딩 모델
> - `kiwipiepy>=0.20.0`: 한국어 형태소 분석
> - `chromadb>=0.5.0`: 로컬 ChromaDB (호환용)
> - `tqdm>=4.66.0`: 실시간 진행률 및 경과시간 표시

### Step 4. 로컬 LLM 환경 준비 (Ollama)
로컬 LLM 추론을 위해 [Ollama](https://ollama.ai/)를 설치하고 모델을 준비합니다:

```bash
# 가볍고 빠른 정책 분석용 모델 다운로드
ollama pull gemma4:e2b

# 코드 및 전처리용 모델 (선택)
ollama pull qwen2.5-coder:7b
```

### Step 5. PostgreSQL 17 및 pgvector 설정
1. PostgreSQL 17 설치 후, 사용할 데이터베이스(예: `vectordb`)를 생성합니다.
2. 해당 데이터베이스에 접속하여 `vector` 확장을 활성화합니다:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. 프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 본인의 DB 접속 비밀번호를 입력합니다:
   ```env
   # .env
   PG_HOST=localhost
   PG_PORT=5432
   PG_DATABASE=vectordb
   PG_USER=postgres
   PG_PASSWORD=your_postgres_password
   ```

### Step 6. 환경 연결 검증
제공된 테스트 스크립트로 PostgreSQL 및 pgvector 연결을 즉시 확인합니다:

```bash
python test_pg_conn.py
```
> 연결 성공 메시지와 PostgreSQL 17 버전, vector 0.8.x 확장 확인 시 모든 환경 구축이 완료됩니다!

---

## 3. 디렉토리 구조

```plaintext
team-ggeodeureok/
├── .env                       # PostgreSQL 비밀번호 및 환경변수 설정 파일
├── change_text/               # 회의록 정제 원문 (34개 지자체 TXT 파일, MINTS_NM/MTGNM/발언 본문)
│   ├── cleaned_2024_강원특별자치도의회.txt
│   ├── cleaned_2024_경기도의회.txt
│   └── ...
├── data/                      # Kiwi 형태소 추출 정책 메타데이터 (34개 지자체 JSONL 파일)
│   ├── cleaned_2024_강원특별자치도의회.jsonl
│   ├── cleaned_2024_경기도의회.jsonl
│   └── ...
├── chroma_db/                 # ChromaDB 로컬 저장소 (선택 시 사용)
├── src/                       # 핵심 소스코드 패키지
│   ├── __init__.py
│   ├── config.py              # 환경변수(.env), DB 설정, 임베딩 모델 프리셋 정의
│   ├── hybrid_loader.py       # data/와 change_text/ 1:1 스트리밍 조인 로더
│   ├── trend_analyzer.py      # 정책 확산 분석, 미래 예측 및 used_text_span 발췌 엔진
│   ├── chunker.py             # 발언 단위 의미 보존 재귀적 청커
│   ├── pg_indexer.py          # PostgreSQL pgvector 벡터스토어 인덱서
│   ├── indexer.py             # ChromaDB 벡터 인덱서
│   ├── parser.py              # TSV 파싱, 화자 직함 분리, 날짜 정규화
│   └── pipeline.py            # 엔드투엔드 RAG 파이프라인
├── run_ingest.py              # pgvector/Chroma 데이터 적재 CLI (실시간 진행률 및 소요시간 표시)
├── run_trend_analysis.py      # 대회 제출 양식 마크다운 표/JSON 산출 CLI
├── rag_query.py               # 로컬 LLM RAG 질의응답 CLI
├── query_search.py            # 벡터 유사도 검색 및 메타데이터 필터링 테스트 CLI
├── extract_llm_metadata.py    # 로컬 LLM 기반 정책 메타데이터 추출기
├── test_pg_conn.py            # PostgreSQL pgvector 연결 테스트 스크립트
├── requirements.txt           # 필수 Python 패키지 목록
├── CHANGELOG.md               # 상세 개발 및 변경 이력
└── README.md                  # 본 문서
```

---

## 4. 하이브리드 데이터 파이프라인 (`data/` + `change_text/`)

본 시스템은 두 개의 데이터 소스를 `row_id` 기준으로 결합하여 의미 검색의 정확도를 극대화합니다:

```
[change_text/*.txt]                    [data/*.jsonl]
(원문, MINTS_NM, MTGNM, 발언자)       (Kiwi 형태소 메타데이터)
        │                                      │
        └──────────────────┬───────────────────┘
                           │ 1:1 row_id 스트리밍 조인 (src/hybrid_loader.py)
                           ▼
             [is_substantive == False 필터링] (단순 개의/폐회 의전멘트 40% 제거)
                           ▼
               [의미 보강 텍스트 청크 생성]
               "[정책분야: ...] | 핵심쟁점: ... | 키워드: ... | 요약: ...
                [원문 발언 본문]"
                           ▼
             [PostgreSQL pgvector / ChromaDB 적재]
```

- **풍부한 메타데이터 유지**: `mints_nm`(공식 회의록명), `mtg_nm`(회의구분), `agenda_title`(안건명), `speaker_name`, `speaker_role`, `policy_category`, `core_issue`, `keywords`, `summary`, `date`, `year`, `assembly` 등 모든 컬럼이 벡터스토어 메타데이터에 온전히 보존됩니다.

---

## 5. PostgreSQL pgvector 데이터 적재 (`run_ingest.py`)

기본 데이터베이스로 PostgreSQL `pgvector`가 설정되어 있어, 별도 DB 지정 없이 바로 실행할 수 있습니다.

### 1) 빠른 테스트 적재 (각 파일당 30행)
```bash
python run_ingest.py --sample-lines 30 --reset
```
- `--reset`: 기존 적재된 컬렉션 테이블을 초기화하고 새로 생성합니다.
- 실시간으로 처리 속도(chunks/s), 경과 시간, 예상 완료 시간(ETA)이 표시됩니다.

### 2) 전체 회의록 데이터 적재
34개 지자체 회의록 전체(실질 정책 발언 약 15만 건 이상)를 적재합니다:
```bash
python run_ingest.py --reset
```

### 3) 특정 연도만 선별 적재
```bash
python run_ingest.py --year 2024 --reset
```

### 4) 주요 옵션 플래그
| 플래그 | 기본값 | 설명 |
| :--- | :--- | :--- |
| `--db` | `pgvector` | 사용할 벡터 DB (`pgvector` 또는 `chroma`) |
| `--reset` | `False` | 적재 전 대상 컬렉션 초기화 여부 |
| `--sample-lines` | `0` (비활성) | 파일당 최대 읽을 행 수 (0이면 전체) |
| `--year` | `None` | 특정 연도 파일만 필터링 적재 (예: `2024`) |
| `-m`, `--model` | `arctic-ko` | 임베딩 모델명 또는 별칭 (기본: `Snowflake Arctic v2.0 한국어`) |
| `-c`, `--collection`| `local_council_minutes` | 적재 대상 컬렉션명 |
| `--batch-size` | `500` | 1회 DB 전송 청크 배치 크기 |
| `--chunk-size` | `800` | 장문 발언 분할 기준 문자 수 |
| `--include-all` | `False` | 비실질적 단순 의전 멘트까지 포함 여부 (기본: 실질 정책 발언만 적재) |

---

## 6. 대회 제출 규격 정책 확산 및 미래 예측 모델 (`run_trend_analysis.py`)

경진대회 주최 측의 제출 요구사항(이슈키워드, 등장시점, 타 지방의회로의 흐름, 향후 예측, 근거 회의록)을 충족하는 분석 CLI입니다.

### 1) 기본 실행 및 마크다운 표 출력
```bash
python run_trend_analysis.py "농지"
```
```bash
python run_trend_analysis.py "반도체 클러스터"
```

### 2) JSON 결과 파일 저장
```bash
python run_trend_analysis.py "청년 일자리" -o result_youth.json
```

### 3) API 및 자동화용 순수 JSON 출력 (`--raw-json`)
```bash
python run_trend_analysis.py "조례안" --raw-json
```

### 4) 산출 결과 예시 (마크다운 표 및 JSON 규격)

```markdown
| 이슈키워드 | 등장시점 | 타 지방의회로의 흐름 | 향후 예측 | 근거 회의록 |
| :--- | :--- | :--- | :--- | :--- |
| **농지** | 2024-01-23 (충청북도의회) | 충청북도의회 (2024-01) -> 강원특별자치도의회 (2024-03) -> 경기도의회 (2024-06) | 농지 규제 완화 및 농촌 소멸 대응 지원 정책이 전국 지자체 조례로 확산될 것으로 전망됨 | 제414회 충청북도의회 임시회 제2차 본회의(2024-01-23)<br>제326회 강원특별자치도의회 임시회(2024-03-15) |
```

- **정밀 발췌 (`used_text_span`)**: 모델이 허위 발언을 지어내지 않도록 검색된 원문 발언 내에서 키워드 중심 **180자 문맥 윈도우**를 자동으로 발췌하여 `used_text_span` 근거로 제공합니다.

---

## 7. 로컬 LLM 엔드투엔드 RAG 질의 (`rag_query.py`)

자연어 질문을 입력받아 **키워드 추출 $\rightarrow$ pgvector 검색 $\rightarrow$ 재순위화(Re-ranking) $\rightarrow$ 로컬 LLM 답변 및 근거 JSON 생성**을 수행합니다.

```bash
# 기본 질의 (pgvector 자동 적용)
python rag_query.py "2024년 충청남도의회에서 논의된 청년 및 농업 정책은?"

# 결과를 파일로 저장
python rag_query.py "강원도 산림 및 환경 정책 알려줘" -o result.json

# 대화형 모드 (인자 없이 실행)
python rag_query.py
```

---

## 8. 시맨틱 검색 및 메타데이터 필터링 (`query_search.py`)

적재된 벡터스토어를 대상으로 빠른 유사도 검색 및 메타데이터 복합 필터(`where`) 테스트를 수행합니다:

```bash
# 기본 유사도 검색
python query_search.py "인공지능 첨단산업 지원 조례"

# 특정 의회 + 특정 연도 필터링
python query_search.py "예산안 심의" --assembly "경기도의회" --year 2024

# 특정 발언자 필터링
python query_search.py "도정 질의" --speaker "김동연" -k 5
```

---

## 9. 로컬 LLM 정책 메타데이터 추출기 (`extract_llm_metadata.py`)

회의록 본문에서 로컬 LLM(Ollama)을 활용하여 정책 분야(`policy_category`), 핵심 쟁점(`core_issue`), 2~4어절 복합 키워드(`keywords`), 요약(`summary`)을 추출하여 `data/*.jsonl`로 구축하는 도구입니다.

- **체크포인트 이어하기**: 처리된 `row_id`를 자동 감지하여 중단 후 재실행 시 안전하게 뒤에 덧붙입니다.
- **멀티스레드 병렬화 (`--workers`)**:
  ```bash
  python extract_llm_metadata.py --input "change_text/cleaned_2024_강원특별자치도의회.txt" --workers 4
  ```

---

## 10. 임베딩 모델 손쉬운 변경 가이드

코드 수정 없이 CLI 인자나 환경변수로 원하는 한국어 임베딩 모델을 즉시 스위칭할 수 있습니다.

| 별칭 (Alias) | 전체 Hugging Face 모델 경로 | 벡터 차원 | 특징 및 권장 용도 |
| :--- | :--- | :---: | :--- |
| **`arctic-ko`** *(기본값)* | `dragonkue/snowflake-arctic-embed-l-v2.0-ko` | 1024 | 행정문서 학습 반영. 지자체 정책 회의록 검색에 최적화 |
| **`kure` / `kure-v1`** | `nlpai-lab/KURE-v1` | 1024 | 긴 문서 및 장문 발언 문맥 검색에 탁월 |
| **`ko-sbert`** | `jhgan/ko-sbert-nli` | 768 | 가볍고 빠른 경량 모델 |
| **`kosimcse`** | `BM-K/KoSimCSE-roberta-multitask` | 768 | 문장 간 의미적 유사도 검색에 우수 |
| **`bge-m3`** | `BAAI/bge-m3` | 1024 | 다국어/한국어 고성능 모델 |

### 사용 예시
```bash
# KURE-v1 모델로 적재 및 검색 (컬렉션 분리 권장)
python run_ingest.py -m kure -c council_kure --reset
python query_search.py "미래 모빌리티" -m kure -c council_kure
```

> [!WARNING]
> 모델마다 임베딩 차원(768차원 vs 1024차원)이 다릅니다. 동일한 컬렉션에 다른 모델의 벡터를 혼합 적재하면 차원 불일치 에러가 발생하므로, 모델별로 컬렉션 이름을 다르게 지정하거나 `--reset`을 사용하세요.

---

## 11. 핵심 모듈 상세 안내 (`src/`)

- **[`src/hybrid_loader.py`](file:///c:/study/Projects/team-ggeodeureok/src/hybrid_loader.py) (`HybridCouncilLoader`)**:
  - `data/*.jsonl`과 `change_text/*.txt`의 파일명을 자동으로 매핑하고 `row_id` 기준 1:1 스트리밍 결합.
  - `is_substantive == False`인 의전성 멘트 자동 제외 및 의미 보강 청크 텍스트 생성.
- **[`src/trend_analyzer.py`](file:///c:/study/Projects/team-ggeodeureok/src/trend_analyzer.py) (`PolicyTrendAnalyzer`)**:
  - 특정 정책 이슈의 시계열 타임라인 구축, 최초 등장 시점 및 타 지자체 확산 경로 추적.
  - 로컬 LLM을 통한 미래 정책 전개 예측문 작성 및 180자 `used_text_span` 발췌.
- **[`src/pg_indexer.py`](file:///c:/study/Projects/team-ggeodeureok/src/pg_indexer.py) (`PGVectorIndexer`)**:
  - PostgreSQL 17 `pgvector` 기반 LangChain VectorStore 래퍼.
  - HNSW / IVFFlat 인덱스 지원 및 메타데이터 필터링 지원.
- **[`src/config.py`](file:///c:/study/Projects/team-ggeodeureok/src/config.py)**:
  - `python-dotenv`를 통한 `.env` 자동 로드, 기본 DB 설정(`DEFAULT_DB_TYPE = "pgvector"`).
  - PostgreSQL 접속 파라미터 및 임베딩 프리셋 관리.
- **[`src/chunker.py`](file:///c:/study/Projects/team-ggeodeureok/src/chunker.py) (`CouncilMinuteChunker`)**:
  - 발언 단위 문맥 보존 및 800자 초과 장문 발언에 대한 재귀적 오버랩 서브청킹.
- **[`src/pipeline.py`](file:///c:/study/Projects/team-ggeodeureok/src/pipeline.py) (`CouncilRAGPipeline`)**:
  - 질의 $\rightarrow$ 키워드 추출 $\rightarrow$ 벡터 검색 $\rightarrow$ Re-ranking $\rightarrow$ JSON 답변 엔드투엔드 파이프라인.

---

## 12. 자주 묻는 질문 및 트러블슈팅

### Q1. PostgreSQL 연결 실패 (`OperationalError: connection to server ... failed`)
- PostgreSQL 서비스가 실행 중인지 확인하세요 (Windows 작업 관리자 서비스 탭 `postgresql-x64-17`).
- 프로젝트 루트의 `.env` 파일에 올바른 `PG_PASSWORD`가 입력되어 있는지 확인하세요.
- `python test_pg_conn.py`를 실행하여 접속 상태를 진단할 수 있습니다.

### Q2. `extension "vector" is not available` 오류
- PostgreSQL에 `pgvector` 확장이 설치되어 있지 않은 경우 발생합니다.
- `vector.dll`이 PostgreSQL의 `lib/` 폴더에 있는지 확인하고, DB에 접속하여 `CREATE EXTENSION IF NOT EXISTS vector;`를 실행하세요.

### Q3. `ModuleNotFoundError: No module named 'torch'` 오류
- Conda 가상환경이 활성화되지 않았을 때 발생합니다.
- 터미널에서 `conda activate project01`을 실행하세요.

### Q4. Windows 콘솔에서 한글이 깨져서 출력될 때
- PowerShell에서 UTF-8 인코딩을 설정합니다:
  ```powershell
  [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
  $OutputEncoding = [System.Text.Encoding]::UTF8
  ```

---

## 13. 작업 이력 (CHANGELOG)

상세한 개발 이력과 커밋 히스토리는 [CHANGELOG.md](file:///c:/study/Projects/team-ggeodeureok/CHANGELOG.md)를 참고하세요.

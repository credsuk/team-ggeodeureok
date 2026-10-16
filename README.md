# 🏛️ 지방의회 회의록 전처리, 벡터 색인 및 시맨틱 검색 파이프라인 (`team-ggeodeureok`)

지방의회(광역의회: 경기도, 충청남도, 충청북도, 강원특별자치도 등)의 대규모 회의록 데이터를 **수집·정제·청킹**하고, 풍부한 메타데이터(일시, 의회, 안건, 화자 직함 및 성명 등)를 추출하여 **LangChain & ChromaDB**에 벡터 색인한 뒤 고성능 시맨틱 검색 및 필터링을 수행하는 엔드투엔드(End-to-End) 시스템입니다.

---

## 📑 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [디렉토리 구조](#2-디렉토리-구조)
3. [개발 및 실행 환경 설정](#3-개발-및-실행-환경-설정)
4. [데이터 및 메타데이터 스키마](#4-데이터-및-메타데이터-스키마)
5. [데이터 전처리 파이프라인 (`scripts/`)](#5-데이터-전처리-파이프라인-scripts)
6. [ChromaDB 벡터 적재 파이프라인 (`run_ingest.py`)](#6-chromadb-벡터-적재-파이프라인-run_ingestpy)
7. [시맨틱 검색 및 메타데이터 필터링 (`query_search.py`)](#7-시맨틱-검색-및-메타데이터-필터링-query_searchpy)
8. [임베딩 모델 손쉬운 변경 가이드](#8-임베딩-모델-손쉬운-변경-가이드)
9. [모듈별 상세 구성 (`src/`)](#9-모듈별-상세-구성-src)
10. [자주 묻는 질문 및 트러블슈팅](#10-자주-묻는-질문-및-트러블슈팅)

---

## 1. 프로젝트 개요

- **대용량 지방의회 데이터 처리**: 지자체별로 상이한 회의록 HTML 구조를 로컬 LLM을 통해 지능적으로 분석·정제하여 표준 TSV 데이터셋으로 변환합니다.
- **발언 단위(Utterance) 의미 보존 청킹**: 의회 회의록 특성에 맞춰 발언 흐름과 문맥을 보존하며, 장문 발언에 한해 오버랩을 적용한 재귀적 서브청킹(Recursive Sub-chunking)을 수행합니다.
- **다차원 메타데이터 필터링**: 시간(연/월/일), 지역(의회/시도), 안건, 발언자(성명/직함/구분) 등 다차원 메타데이터를 벡터와 함께 보존하여 정밀한 하이브리드 필터 검색(`where`)을 지원합니다.
- **GPU 가속 및 유연한 임베딩 모델 교체**: 한국어 사전학습 임베딩 모델을 별칭(alias)이나 Hugging Face 경로를 통해 손쉽게 전환할 수 있으며, CUDA 가속을 기본 지원합니다.

---

## 2. 디렉토리 구조

```plaintext
team-ggeodeureok/
├── text/                      # 전처리 완료된 회의록 정제 텍스트(TSV 형식) 파일
│   ├── cleaned_2024_강원특별자치도의회.txt
│   ├── cleaned_2024_경기도의회.txt
│   ├── cleaned_2024_충청남도의회.txt
│   ├── cleaned_2024_충청북도의회.txt
│   ├── cleaned_2025_경기도의회.txt
│   ├── cleaned_2025_충청남도의회.txt
│   └── cleaned_2025_충청북도의회.txt
├── chroma_db/                 # LangChain Community Chroma 로컬 영구 벡터 저장소
├── src/                       # 핵심 처리 모듈 패키지
│   ├── __init__.py
│   ├── config.py              # 경로, 청킹 파라미터, 임베딩 프리셋 및 모델 설정
│   ├── parser.py              # TSV 파싱, 화자 직함 분리, 날짜 분해, 정제
│   ├── chunker.py             # 발언 단위 보존 및 장문 서브청킹 처리기
│   └── indexer.py             # LangChain HuggingFaceEmbeddings & Chroma 관리자
├── scripts/                   # 원본 HTML/JSON 전처리 및 LLM 기반 파싱 스크립트
│   ├── _constant.py           # 전처리 파이프라인 공통 상수 및 Ollama 클라이언트 설정
│   ├── 1_split_sample.py      # 원본 JSON에서 LLM 분석용 HTML 샘플 추출
│   ├── 2_rules_make.py        # 로컬 LLM을 통한 의회별 CSS Selector 파싱 규칙 생성
│   ├── 3_rules_summary.py     # 파싱 규칙 통일 및 엄격한 CSS 규칙 JSON 빌드
│   ├── 4_to_csv.py / 4_to_csv_2.py # CSS 규칙 기반 텍스트 추출 및 정제 (TSV 생성)
│   └── 5_create_master_csv.py # LLM 기반 정책분류/핵심이슈/키워드/요약 메타데이터 생성
├── run_ingest.py              # 데이터 적재 실행 CLI (ETA 실시간 진행률, 모델 지정 지원)
├── query_search.py            # Chroma 검색 및 메타데이터 필터링 테스트 CLI
├── requirements.txt           # 필수 Python 패키지 의존성 목록
├── forder.txt                 # 경진대회/제출물 디렉터리 규격 참조 문서
└── README.md                  # 프로젝트 사용법 및 상세 문서
```

---

## 3. 개발 및 실행 환경 설정

### 1) Conda 가상환경 활성화
본 프로젝트는 **Python 3.11** 기반의 `project01` 가상환경에서 동작합니다.

```bash
# 가상환경 활성화
conda activate project01
```

### 2) 필수 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

> **`requirements.txt` 주요 패키지**:
> - `chromadb>=0.5.0`
> - `langchain>=0.2.0`, `langchain-community>=0.2.0`, `langchain-text-splitters>=0.0.1`
> - `transformers<5.0.0`, `sentence-transformers<4.0.0`
> - `tqdm>=4.66.0`

### 3) GPU(CUDA) 가속 설정 (권장)
NVIDIA 그래픽 카드(예: RTX 4060 등)가 장착된 환경에서는 PyTorch CUDA 버전을 설치하면 임베딩 및 인덱싱 속도가 **10~30배 이상** 향상됩니다:

```bash
# CUDA 12.1 지원 PyTorch 설치 (NVIDIA GPU 환경)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4) 로컬 LLM 환경 (데이터 전처리 파이프라인 실행 시)
`scripts/` 내 전처리 스크립트를 실행하려면 [Ollama](https://ollama.ai/)가 구동 중이어야 합니다:
```bash
# Ollama 실행 및 필요한 모델 pull
ollama pull qwen2.5-coder:7b
ollama pull huihui_ai/exaone3.5-abliterated
```

---

## 4. 데이터 및 메타데이터 스키마

### 1) 입력 데이터 형식 (`text/*.txt`)
탭(Tab, `\t`)으로 구분된 UTF-8 TSV 형식이며, 상단 1행은 헤더입니다:

| 컬럼명 | 설명 | 예시 |
| :--- | :--- | :--- |
| `date` | 회의 일자 (YYYYMMDD) | `20240123` |
| `assembly` | 의회명 | `충청북도의회` |
| `agenda_title` | 안건명 | `제414회 충청북도의회 임시회 제2차 본회의` |
| `speaker_role` | 발언자 역할 | `위원장` |
| `speaker_name` | 발언자 성명 | `이상정` |
| `speaker_type` | 발언자 소속 구분 | `의회` / `집행부` / `미상` |
| `utterance` | 발언 내용 본문 | `성원이 되었으므로 제414회 ... 개의를 선포합니다.` |

### 2) ChromaDB 색인 메타데이터 스키마
ChromaDB의 빠른 `where` 연산(단일 및 `$and` 복합 조건)에 최적화된 스칼라 데이터 타입을 가집니다:

| 분류 | 필드명 | 데이터 타입 | 설명 및 예시 |
| :--- | :--- | :--- | :--- |
| **시간 축** | `date` | `str` | 정제된 일자 (`YYYY-MM-DD`, 예: `2024-01-23`) |
| | `year` | `int` | 회의 연도 (예: `2024`) |
| | `month` | `int` | 회의 월 (예: `1`) |
| | `day` | `int` | 회의 일 (예: `23`) |
| **지역/의회** | `assembly` | `str` | 전체 의회명 (예: `경기도의회`, `충청북도의회`) |
| | `region` | `str` | 광역 지자체 약칭 (`경기`, `충북`, `충남`, `강원`, `기타`) |
| **안건 축** | `agenda_title` | `str` | 정제된 안건 제목 (최대 250자) |
| **발언자 축** | `speaker_name` | `str` | 순수 성명 (예: `김꽃임`, `김동연`) |
| | `speaker_position` | `str` | 직함 (예: `위원장`, `도지사`, `의원`, `일반`) |
| | `speaker_role` | `str` | 원본 역할 표기 |
| | `speaker_type` | `str` | 소속 구분 (`의회`, `집행부`, `미상`) |
| **추적/청킹** | `source_file` | `str` | 원본 소스 파일명 (예: `cleaned_2024_경기도의회.txt`) |
| | `row_id` | `int` | 원본 TSV 행 번호 |
| | `chunk_index` | `int` | 서브청크 순번 (0부터 시작) |
| | `total_chunks` | `int` | 해당 발언의 총 서브청크 개수 |
| | `char_length` | `int` | 청크 본문 문자 수 |
| | `is_continuation` | `bool` | 앞선 청크에서 분할되어 이어지는지 여부 |

---

## 5. 데이터 전처리 파이프라인 (`scripts/`)

원본 지자체 회의록 JSON(HTML 본문 포함)에서 일관된 TSV를 구축하는 일련의 단계입니다.

```
[원본 JSON/HTML]
       │
       ▼
[1_split_sample.py]        --> 의회별 대표 HTML 본문 샘플 추출
       │
       ▼
[2_rules_make.py]          --> 로컬 LLM(Ollama)으로 안건/화자/발언 CSS Selector 추론
       │
       ▼
[3_rules_summary.py]       --> 엄격한 CSS 규칙 JSON으로 통합 및 검증
       │
       ▼
[4_to_csv_2.py]            --> 불필요한 의전 멘트 정리, 자간 오류 복원 후 TSV 변환 (text/ 저장)
       │
       ▼
[5_create_master_csv.py]   --> (선택) LLM을 활용한 정책분류/키워드/요약 메타데이터 생성
```

- **실행 방법**:
  ```bash
  # 1. 샘플 추출
  python scripts/1_split_sample.py

  # 2. LLM CSS 규칙 생성
  python scripts/2_rules_make.py

  # 3. 규칙 정리 및 JSON 생성
  python scripts/3_rules_summary.py

  # 4. TSV 정제 데이터 생성
  python scripts/4_to_csv_2.py
  ```

---

## 6. ChromaDB 벡터 적재 파이프라인 (`run_ingest.py`)

정제된 회의록 텍스트(`text/*.txt`)를 읽어 청킹 및 임베딩을 수행하고, 로컬 `chroma_db/`에 배치 스트리밍 방식으로 적재합니다.

### 1) 빠른 동작 검증 (테스트 샘플 모드)
전체 적재 전, 파일당 100행만 추출하여 적재 파이프라인이 정상 동작하는지 테스트합니다:
```bash
python run_ingest.py --sample-lines 100 --reset
```
- `--reset`: 대상 ChromaDB 컬렉션을 삭제하고 처음부터 새로 생성합니다.
- `--sample-lines 100`: 각 파일에서 최대 100행만 읽어옵니다.

### 2) 전체 데이터 적재 실행
약 26만 건 이상의 전체 회의록 데이터를 적재합니다:
```bash
python run_ingest.py --batch-size 500
```

### 3) 다른 임베딩 모델로 적재하기
```bash
# 예 1: 별칭(alias) 'kosimcse'를 사용하여 적재 (컬렉션 분리 권장)
python run_ingest.py -m kosimcse -c council_kosimcse --reset

# 예 2: 고성능 BGE-M3 모델로 적재
python run_ingest.py -m bge-m3 -c council_bge_m3 --reset
```

### 4) 주요 옵션 플래그
| 인자 | 기본값 | 설명 |
| :--- | :--- | :--- |
| `-m`, `--model` | `jhgan/ko-sbert-nli` | 사용할 임베딩 모델명 또는 별칭 (예: `ko-sbert`, `kosimcse`, `bge-m3`) |
| `-c`, `--collection` | `local_council_minutes` | 적재 대상 Chroma 컬렉션 이름 |
| `--batch-size` | `500` | ChromaDB 1회 add 호출 시 전송할 문서 배치 크기 |
| `--sample-lines` | `0` (비활성) | 파일당 최대 읽을 행 수 (테스트용, 0이면 전체) |
| `--chunk-size` | `800` | 장문 발언 서브청킹 기준 크기 (문자 수) |
| `--chunk-overlap` | `100` | 서브청킹 간 중첩 크기 (문자 수) |
| `--min-len` | `40` | 단순 진행/의전성 초단문 필터링 기준 (문자 수) |
| `--reset` | `False` | 적재 전 대상 컬렉션 초기화 여부 플래그 |

---

## 7. 시맨틱 검색 및 메타데이터 필터링 (`query_search.py`)

적재된 ChromaDB 벡터스토어를 대상으로 자연어 질의와 메타데이터 필터를 조합하여 유사도 기반 검색을 수행합니다.

### 1) 기본 자연어 시맨틱 검색
```bash
python query_search.py "청년 일자리 및 주거 지원 방안"
```

### 2) 특정 의회 + 특정 연도 필터링
```bash
python query_search.py "예산안 심사 및 심의" --assembly "경기도의회" --year 2024
```

### 3) 특정 발언자 필터링 및 출력 개수 지정
```bash
python query_search.py "도정 질의 및 공공기관 이전" --speaker "김동연" -k 5
```

### 4) 특정 임베딩 모델 및 컬렉션 지정 검색
인덱싱할 때 사용한 동일한 모델과 컬렉션을 지정하여 검색합니다:
```bash
python query_search.py "인공지능 첨단산업" -m kosimcse -c council_kosimcse
```

### 5) CLI 파라미터 안내
- `query` (필수): 검색할 자연어 질의어 (예: `"인공지능 산업 육성"`)
- `-m`, `--model`: 임베딩 모델명 또는 별칭 (기본값: `jhgan/ko-sbert-nli`)
- `-c`, `--collection`: 검색 대상 컬렉션명 (기본값: `local_council_minutes`)
- `--assembly`: 의회명 필터 (예: `"경기도의회"`, `"충청북도의회"`, `"강원특별자치도의회"`)
- `--year`: 연도 필터 (정수, 예: `2024`, `2025`)
- `--speaker`: 발언자 성명 필터 (예: `"김동연"`, `"이상정"`)
- `-k`, `--limit`: 출력할 유사 문서 수 (기본값: `3`)

---

## 8. 임베딩 모델 손쉬운 변경 가이드

본 프로젝트는 복잡한 코드 수정 없이 **프리셋 별칭**, **CLI 인자**, **환경변수**, **설정 파일**을 통해 임베딩 모델을 자유롭게 변경할 수 있도록 지원합니다.

### 1) 지원 프리셋 별칭 목록

| 별칭 (Alias) | 전체 Hugging Face 모델 식별자 | 벡터 차원 | 모델 특징 및 추천 용도 |
| :--- | :--- | :---: | :--- |
| **`ko-sbert`** *(기본)* | `jhgan/ko-sbert-nli` | 768 | **기본 추천.** 가볍고 빠르며 한국어 문장 의미 표현력이 우수함. |
| **`kosimcse`** | `BM-K/KoSimCSE-roberta-multitask` | 768 | 한국어 RoBERTa 기반 SimCSE. 단어·문장의 의미적 유사도 검색에 탁월. |
| **`bge-m3`** | `BAAI/bge-m3` | 1024 | 최신 다국어/한국어 고성능 모델. 긴 텍스트 임베딩에 우수하나 VRAM 요구량이 큼. |
| **`bge-m3-ko`** | `dragonkyle/bge-m3-ko` | 1024 | BGE-M3의 한국어 집중 파인튜닝 버전. |
| **`e5-base`** | `intfloat/multilingual-e5-base` | 768 | 다국어 검색 성능이 우수한 E5 시리즈 모델. |
| **`kr-sbert`** | `snunlp/KR-SBERT-V40K-klue-subword` | 768 | 서울대 한국어 연구팀의 사전학습 SBERT 모델. |

> 💡 **Hugging Face 직접 지정**: 별칭 외에도 Hugging Face에 존재하는 모든 한국어 임베딩 모델 경로(예: `"snunlp/KR-SBERT-V40K-klue-subword"`)를 그대로 입력하여 사용할 수 있습니다.

---

### 2) 변경 방법 3가지

#### 방법 A. CLI 명령행 인자로 즉시 변경 (가장 권장)
스크립트 실행 시 `-m` 또는 `--model` 옵션으로 간편하게 지정합니다:
```bash
# 적재 시 KoSimCSE 모델 사용
python run_ingest.py -m kosimcse -c council_kosimcse --reset

# 검색 시에도 동일한 모델 지정
python query_search.py "청년 일자리" -m kosimcse -c council_kosimcse
```

#### 방법 B. 환경변수로 전역 변경
터미널 세션 전체에서 기본 임베딩 모델을 변경합니다:
```powershell
# Windows PowerShell 환경
$env:EMBEDDING_MODEL_NAME = "bge-m3"

# 이후 옵션 없이 실행해도 bge-m3가 기본 적용됨
python query_search.py "미래 모빌리티 예산"
```
```bash
# Linux/macOS 환경
export EMBEDDING_MODEL_NAME="bge-m3"
```

#### 방법 C. `src/config.py` 코드 한 줄 수정으로 영구 변경
[`src/config.py`](file:///c:/study/Projects/team-ggeodeureok/src/config.py) 파일 상단의 기본값을 원하는 모델로 변경합니다:
```python
# src/config.py
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", EMBEDDING_PRESETS["kosimcse"])
```

---

### 3) ⚠️ 모델 변경 시 주의사항: 벡터 차원(Dimension) 불일치

> [!WARNING]
> 모델마다 벡터 임베딩 차원이 다릅니다. (예: `ko-sbert`=768차원, `bge-m3`=1024차원)
> 이미 다른 모델로 적재된 동일한 컬렉션에 새 모델로 데이터를 넣거나 검색하면 **차원 불일치(Dimension mismatch) 오류**가 발생합니다.

- **해결책 1 (컬렉션 분리 - 권장)**: 모델마다 다른 컬렉션 이름을 지정합니다 (`-c council_bge_m3`).
- **해결책 2 (컬렉션 초기화)**: 기존 데이터를 덮어쓰고 새로 적재할 때 `--reset` 옵션을 추가합니다.

---

## 9. 모듈별 상세 구성 (`src/`)

- **[`src/config.py`](file:///c:/study/Projects/team-ggeodeureok/src/config.py)**:
  - 경로 및 하이퍼파라미터 정의
  - `EMBEDDING_PRESETS` 딕셔너리 및 별칭 매핑 함수 `resolve_embedding_model()`
- **[`src/parser.py`](file:///c:/study/Projects/team-ggeodeureok/src/parser.py)**:
  - `clean_text()`: 비표준 공백(`\xa0` 등) 정규화
  - `parse_speaker()`: `위원장이상정` -> `('위원장', '이상정')` 직함/성명 자동 분리
  - `parse_date()`: `YYYYMMDD` 문자열을 정수형 `year`, `month`, `day`로 분해
  - `parse_tsv_row()`: 7개 컬럼 TSV 레코드 파싱
- **[`src/chunker.py`](file:///c:/study/Projects/team-ggeodeureok/src/chunker.py) (`CouncilMinuteChunker`)**:
  - 발언 단위 기본 유지 + 800자 초과 시 문맥 유실 방지를 위한 재귀적 청킹
  - 40자 미만 단순 의전 멘트 필터링
- **[`src/indexer.py`](file:///c:/study/Projects/team-ggeodeureok/src/indexer.py) (`ChromaIndexer`)**:
  - `HuggingFaceEmbeddings` 모델 로드 및 CUDA GPU 자동 감지
  - 배치 단위 `add_batch()` 및 유사도 필터 검색 `query()`
  - 차원 불일치 에러 시 친절한 원인/해결책 안내 예외 처리

---

## 10. 자주 묻는 질문 및 트러블슈팅

### Q1. `ModuleNotFoundError: No module named 'torch'` 오류가 발생합니다.
- Conda 가상환경이 활성화되지 않았을 때 발생합니다. `conda activate project01`을 실행하세요.

### Q2. Windows PowerShell에서 파일명이 깨져 보이거나 한글 출력 시 인코딩 오류가 발생합니다.
- PowerShell에서 아래 명령어로 터미널 인코딩을 UTF-8로 설정하세요:
  ```powershell
  [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
  $OutputEncoding = [System.Text.Encoding]::UTF8
  ```

### Q3. 임베딩 모델 변경 후 Dimension 오류가 발생합니다.
- 모델 간 차원(예: 768 vs 1024)이 달라서 발생하는 오류입니다.
- 새 모델로 적재 시 `-c <새_컬렉션명>`을 지정하거나, `--reset` 플래그로 기존 컬렉션을 초기화하세요.

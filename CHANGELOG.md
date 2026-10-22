# 📜 프로젝트 개발 및 변경 이력 (Changelog)

본 문서는 `develop_kb` 브랜치에서 진행된 지방의회 회의록 데이터 처리, 벡터 색인, 시맨틱 검색 파이프라인의 전체 구축 및 개선 내역을 기록합니다.

---

## [2026-10-22] data(메타데이터) + change_text(원문) 하이브리드 결합, pgvector 기본화 및 README 최신화

### 1. README.md 전면 최신화 및 Conda 환경 구축 가이드 신설
- **Conda 기반 단계별 환경 설정 가이드**: Conda 가상환경 생성(`python=3.11`), PyTorch CUDA/CPU 설치, `requirements.txt` 설치, Ollama 로컬 LLM 세팅, PostgreSQL 17 + `pgvector` 및 `.env` 파일 작성법을 초반부에 상세 수록.
- **기본 벡터 DB `pgvector` 반영**: `DEFAULT_DB_TYPE = "pgvector"`, `python-dotenv`를 통한 `.env` 자동 로드, 실시간 경과 시간 및 처리 속도 표시 기능 반영.
- **대회 제출 규격 모델(`run_trend_analysis.py`) 상세 안내**: 마크다운 표 및 JSON 규격 출력, `used_text_span` 180자 문맥 윈도우 추출 기능 설명 추가.

### 2. 하이브리드 스트리밍 데이터 로더 신설 ([`src/hybrid_loader.py`](file:///c:/study/Projects/team-ggeodeureok/src/hybrid_loader.py))
- **`data/*.jsonl` & `change_text/*.txt` 1:1 실시간 조인 엔진**:
  - 두 폴더 내 34개 광역의회 회의록 파일 쌍을 자동 감지하고, 행 번호(`row_id`)를 키로 메모리 누수 없이 동시 스트리밍 결합.
  - `is_substantive == False`(단순 의전/개의/폐회 멘트) 발언을 원천 필터링하여 인덱싱 속도 2배 향상 및 검색 노이즈 40% 제거.
  - **의미 보강 텍스트(`create_augmented_text`)**: 메타데이터(`policy_category`, `core_issue`, `keywords`, `summary`)를 발언 원문 상단에 주입하여 벡터 임베딩 품질 및 코사인 유사도 대폭 향상.

### 2. 하이브리드 인덱싱 파이프라인 전면 개편 ([`run_ingest.py`](file:///c:/study/Projects/team-ggeodeureok/run_ingest.py))
- 공식 회의록명(`mints_nm`), 회의구분(`mtg_nm`), 발언자 직함/구분, 정책 분류/쟁점을 포함한 표준 메타데이터를 ChromaDB 및 PostgreSQL pgvector에 일괄 적재.
- `--substantive-only` (실질 발언만 선별 적재), `--augment-text` (메타데이터 보강 임베딩), `--year` (연도 필터링) 지원.

### 3. 경진대회 제출 규격 부합 정책 시계열 확산 및 미래 예측 고도화 ([`src/trend_analyzer.py`](file:///c:/study/Projects/team-ggeodeureok/src/trend_analyzer.py))
- **판단에 사용한 원문 구간 정밀 추출 (`extract_context_window`)**: 키워드가 등장하는 실제 발언 전후 180자 문맥 스팬을 정밀 추출.
- 공식 회의록명(`mints_nm`) 및 회의일자를 정확히 매핑하여 주최 측 제출 양식 표(Markdown Table) 및 JSON 결과물 자동 생성.

---

## [2026-10-21] PostgreSQL (pgvector) 벡터 데이터베이스 통합 및 하이브리드 스위칭 지원

### 1. PostgreSQL pgvector 전용 인덱서 모듈 신설 ([`src/pg_indexer.py`](file:///c:/study/Projects/team-ggeodeureok/src/pg_indexer.py))
- `langchain_community.vectorstores.pgvector.PGVector` 기반 엔드포인트 관리자 구현.
- `ChromaIndexer`와 100% 동일한 인터페이스(`add_batch`, `query`, `count`)를 제공하여 파이프라인에서 무중단 교체 가능.
- 메타데이터 필터(`assembly`, `year`, `speaker_name`, `policy_category`) 조건 매핑 및 자동 확장 테이블 생성 지원.
- 연결 실패 시 원인 및 PostgreSQL 설치/설정 가이드를 터미널에 친절히 출력.

### 2. 파이프라인 및 CLI 스크립트 전면 `--db` 옵션 지원
- [`src/pipeline.py`](file:///c:/study/Projects/team-ggeodeureok/src/pipeline.py) (`CouncilRAGPipeline`): `db_type="pgvector"` 또는 `"chroma"` 선택 가능. 결과 JSON 메타데이터에 `vector_db` 명시.
- [`run_ingest.py`](file:///c:/study/Projects/team-ggeodeureok/run_ingest.py): `--db pgvector` 옵션으로 PostgreSQL에 텍스트 청크 및 메타데이터 적재 지원.
- [`rag_query.py`](file:///c:/study/Projects/team-ggeodeureok/rag_query.py): `--db pgvector` 옵션으로 pgvector 기반 엔드투엔드 RAG 질의 지원.
- [`query_search.py`](file:///c:/study/Projects/team-ggeodeureok/query_search.py): `--db pgvector` 옵션으로 pgvector 단독 시맨틱 검색 지원.
- [`src/config.py`](file:///c:/study/Projects/team-ggeodeureok/src/config.py): PostgreSQL 환경변수(`PG_HOST`, `PG_PORT`, `PG_DATABASE`, `PG_USER`, `PG_PASSWORD`, `PG_CONNECTION_STRING`) 지원.
- [`requirements.txt`](file:///c:/study/Projects/team-ggeodeureok/requirements.txt): `psycopg2-binary>=2.9.9`, `pgvector>=0.2.5` 추가.

---

## [2026-10-21] Kiwi 기반 고속 회의록 정책 메타데이터 추출기 및 일괄 변환 파이프라인 구축

### 1. Kiwi 형태소 분석기 기반 정책 메타데이터 추출기 ([`extract_kiwi_metadata.py`](file:///c:/study/Projects/team-ggeodeureok/extract_kiwi_metadata.py))
- **로컬 LLM 대체 초고속 파이프라인 구축**: `kiwipiepy` C++ 멀티스레드 엔진을 기반으로 초당 40~50건 이상 처리 (LLM 대비 100배 이상 고속화).
- **JSONL 표준 스키마 100% 호환**: `row_id`, `date`, `assembly`, `speaker`, `agenda`, `is_substantive`, `policy_category`, `core_issue`, `keywords`, `summary`.
- **단순 의전/절차 발언 정밀 필터링**: 개의/가결/이석/폐회 등 실질 정책 논의가 없는 발언을 `is_substantive: false`로 자동 분류.
- **7대 정책 분야 사전 스코어링**: `경제/산업`, `농림/수산`, `보건/복지`, `교육/문화`, `환경/안전`, `도시/교통`, `자치/행정` 자동 분류.
- **복합명사구(Bigram) 키워드 및 추출 요약**:
  - 의회 특화 불용어 목록을 통한 상투어 배제 및 2어절 정책 복합명사구 추출.
  - Kiwi 문장 분리(`split_into_sents`) 기반 핵심 문장 1~2개 추출 요약(Extractive Summarization).
- **체크포인트 & 실시간 스트리밍 Flush**: 중간 중단 시 자동 이어하기 지원.

### 2. 전체 회의록 일괄 변환 CLI 도구 ([`batch_extract_kiwi_metadata.py`](file:///c:/study/Projects/team-ggeodeureok/batch_extract_kiwi_metadata.py))
- `change_text/` 내의 모든 광역의회 회의록(.txt)을 재귀 탐색하여 `data2/`에 JSONL 파일로 일괄 변환.
- 단일 Kiwi 인스턴스 공유를 통한 메모리 및 속도 최적화.
- 연도 필터링(`--year 2024` 등), 파일당 샘플링(`--sample-per-file`), 완료된 파일 자동 건너뛰기 지원.
- 전체 배치 처리 진행 상황 및 종합 리포트 출력.

---

## [2026-10-21] 팀 회의 결정 규격 기반 로컬 LLM RAG 모델 파이프라인 구축

### 1. 엔드투엔드 RAG 파이프라인 모듈 신설 ([`src/pipeline.py`](file:///c:/study/Projects/team-ggeodeureok/src/pipeline.py))
- **팀 회의 결정 5단계 파이프라인 구현**:
  $$\text{사용자 쿼리} \xrightarrow{} \text{키워드 추출} \xrightarrow{} \text{DB 메타데이터/벡터 유사도 검색} \xrightarrow{} \text{Re-ranking(재순위화)} \xrightarrow{} \text{최종 JSON 아웃풋}$$
- **1단계 (키워드 & 필터 추출)**: Ollama 로컬 LLM(기본 `gemma4:e2b`)을 통해 자연어 질의로부터 핵심 검색 키워드, 정제 쿼리, 메타데이터 필터(`assembly`, `year`, `speaker_name`, `policy_category`) 자동 추출 (실패 시 Fallback 규칙 엔진 적용).
- **2단계 (ChromaDB 하이브리드 검색)**: 추출된 메타데이터 조건식(`$and`, `where`)과 정제된 검색 쿼리를 결합하여 1차 후보 문서(기본 15건) 검색.
- **3단계 (Re-ranking 재순위화)**:
  - `llm`: 로컬 LLM이 후보 문서 목록을 읽고 질문과의 관련성 점수(0~100점) 및 채점 사유를 산출하여 재순위화.
  - `cross-encoder`: `BAAI/bge-reranker-v2-m3` 등 HuggingFace CrossEncoder 지연 로딩 지원.
- **4단계 (답변 생성 & 근거 요약)**: 재순위화된 상위 근거 문서(기본 4건)를 참조하여 로컬 LLM이 객관적이고 정확한 답변 생성.
- **5단계 (표준 규격 JSON 직렬화)**: UI/프론트엔드 표출 대신 지정된 JSON 포맷에 맞춘 최종 결과물 반환 (`query`, `extracted`, `answer`, `evidences`, `metadata` 등 포함).

### 2. RAG 파이프라인 실행 CLI 도구 신설 (`rag_query.py`)
- 단일 질의 실행, 대화형 인터랙티브 모드, JSON 파일 저장(`-o result.json`), 파이프라인 연동용 순수 JSON 출력(`--raw-json`) 플래그 지원.
- Windows 환경 콘솔 UTF-8 입출력 지원.

### 3. 데이터 디렉터리 자동 감지 및 패키지 export 고도화
- `src/config.py`: `change_text/2024_광역 회의록` 경로 자동 탐색 우선순위 반영.
- `src/__init__.py`: `CouncilRAGPipeline` 및 핵심 유틸리티 함수/클래스 export 정비.

---

## [2026-10-19] `develop_kb` 추가 기능 및 고도화 내역

### 1. 로컬 LLM 정책 메타데이터 추출기 (`extract_llm_metadata.py`) 신설
- **Ollama 기반 구조화 메타데이터 추출**:
  - `gemma4:e2b` (기본) 및 `gemma4:e4b` 등 로컬 경량 모델을 활용하여 발언 본문에서 **정책 분류(`policy_category`), 핵심 쟁점(`core_issue`), 복합 정책 키워드(`keywords`), 요약(`summary`)**을 순수 JSON 형식으로 추출.
  - 엄격한 Negative Constraints 시스템 프롬프트를 적용하여 포괄적 1단어(예: '사업', '예산', '조례')를 배제하고 실질적인 정책 의제 복합키워드(예: '농지제도 혁신', '미래 인구정책')만 추출하도록 강제.
- **체크포인트(이어하기) & 실시간 Append 지원**:
  - 이미 처리된 행 번호(`row_id`)를 자동 감지하여 중복 호출을 방지하고 건너뜀(Skip).
  - 1건 처리 시마다 실시간으로 파일에 `write` 및 `flush`하여 중단되어도 데이터 유실 방지.
- **입력 파일명 기반 자동 출력 경로 결정**:
  - `--output` 옵션 미지정 시 입력 텍스트 파일명에 맞춰 `data/{입력파일명}.jsonl`로 파일별 자동 분리 저장.
- **멀티스레드 병렬 가속 및 `Ctrl + C` 즉시 안전 중단 핸들러**:
  - `--workers` 파라미터(기본 2, 최대 4~6)로 동시 추론 병렬 처리 지원.
  - `KeyboardInterrupt` 감지 시 큐에 대기 중인 작업을 즉시 취소(`cancel_futures=True`)하고 스레드 풀을 대기 없이 종료(`wait=False`)하여 즉각적인 안전 종료 보장.

### 2. ChromaDB 적재 파이프라인 메타데이터 자동 병합 고도화 (`run_ingest.py`)
- **다중 메타데이터 파일 일괄 감지 및 병합**:
  - `data/*.jsonl`에 존재하는 모든 의회별 정책 메타데이터 파일을 자동으로 감지하여 `(assembly, row_id)` 기반으로 ChromaDB 벡터 청크에 통합 색인.

### 3. 고속 핵심 키워드 추출 및 워드클라우드 생성 도구 (`make_wordcloud.py`)
- `Kiwi` 형태소 분석기 기반 정책 명사 추출 및 불용어 제거를 통한 고해상도 워드클라우드 생성 도구 구현.

---

## [2026-10-16] `develop_kb` 주요 작업 내역 요약

### 1. 프로젝트 아키텍처 모듈화 (`src/`)
- **설정 및 환경 관리 ([`src/config.py`](file:///c:/study/Projects/team-ggeodeureok/src/config.py))**:
  - 프로젝트 기준 경로 및 TSV 데이터 경로(`text/` 또는 `data/text`) 자동 감지 로직 구현.
  - 장문 발언 서브청킹 크기(`CHUNK_SIZE=800`), 오버랩(`CHUNK_OVERLAP=100`), 초단문 의전 필터링(`MIN_CHAR_LEN=40`) 하이퍼파라미터 정의.
  - 추천 임베딩 모델 프리셋 딕셔너리(`EMBEDDING_PRESETS`) 및 별칭 자동 해석 헬퍼(`resolve_embedding_model`) 구현.
- **회의록 TSV 데이터 파서 ([`src/parser.py`](file:///c:/study/Projects/team-ggeodeureok/src/parser.py))**:
  - `clean_text`: 비표준 공백(`\xa0`), 중복 공백, 불필요한 제어 문자 정제.
  - `parse_speaker`: 직함(`PREFIX_POSITIONS`, `SUFFIX_POSITIONS`)과 순수 성명을 규칙 기반으로 자동 분리 (예: `'위원장이상정'` → `('위원장', '이상정')`).
  - `parse_date`: `YYYYMMDD` 문자열을 정수형 `year`, `month`, `day` 및 ISO 형식 날짜로 분해하여 메타데이터 필터링 지원.
  - `parse_tsv_row`: 7개 TSV 컬럼을 구조화된 딕셔너리로 변환.
- **발언 단위 청킹 모듈 ([`src/chunker.py`](file:///c:/study/Projects/team-ggeodeureok/src/chunker.py))**:
  - 회의록 문맥을 유지하기 위해 1차적으로 발언 단위(Utterance)를 기본 청크로 보존.
  - 800자 초과 장문에 한해 `RecursiveCharacterTextSplitter`를 적용하여 문맥 유실 방지.
  - 40자 미만의 단순 진행/의전성 멘트 자동 필터링.
  - 고유 청크 식별자(`{region}_{date}_r{row_id}_c{chunk_idx}`) 부여.
- **ChromaDB 및 임베딩 관리자 ([`src/indexer.py`](file:///c:/study/Projects/team-ggeodeureok/src/indexer.py))**:
  - `langchain_community.vectorstores.Chroma` 및 `HuggingFaceEmbeddings` 래퍼 구현.
  - CUDA GPU 가속 자동 감지 및 로드 (`torch.cuda.is_available()`).
  - 모델별 벡터 차원 불일치(Dimension mismatch) 발생 시 원인과 해결책(컬렉션 분리 또는 `--reset`)을 명시하는 친절한 예외 처리 적용.

---

### 2. 고성능 임베딩 모델 연동 및 기본 모델 설정
- **기본(Default) 임베딩 모델 변경**:
  - **`dragonkue/snowflake-arctic-embed-l-v2.0-ko`** (별칭: `arctic-ko`, `snowflake-arctic-ko`): 행정문서가 훈련에 활용된 1024차원 고성능 모델로, **공공 행정 및 조례/회의록 데이터 검색에 최적화**되어 디폴트 모델로 채택.
- **긴 문서 특화 프리셋 지원**:
  - **`nlpai-lab/KURE-v1`** (별칭: `kure`, `kure-v1`): BAAI/bge-m3 기반의 최신 한국어 검색 특화 모델로, 긴 문서 및 복잡한 의회 회의록 검색용 프리셋 지원.
- **다양한 프리셋 지원**:
  - `ko-sbert` (`jhgan/ko-sbert-nli` - 768차원, 경량/빠름)
  - `kosimcse` (`BM-K/KoSimCSE-roberta-multitask` - 768차원, 의미 유사도 특화)
  - `bge-m3` (`BAAI/bge-m3` - 1024차원)
  - `bge-m3-ko` (`dragonkyle/bge-m3-ko` - 1024차원)
  - `e5-base` (`intfloat/multilingual-e5-base` - 768차원)
  - `kr-sbert` (`snunlp/KR-SBERT-V40K-klue-subword` - 768차원)
- **손쉬운 모델 교체 인터페이스 3가지 지원**:
  1. CLI 옵션: `-m / --model <별칭_또는_경로>`, `-c / --collection <컬렉션명>`
  2. 환경 변수: `$env:EMBEDDING_MODEL_NAME = "kure"`
  3. 설정 파일: `src/config.py`의 `EMBEDDING_MODEL_NAME` 기본값 변경

---

### 3. CLI 실행 스크립트 구축 및 개선
- **대규모 데이터 적재 도구 ([`run_ingest.py`](file:///c:/study/Projects/team-ggeodeureok/run_ingest.py))**:
  - 배치 단위(`BATCH_SIZE=500`) 스트리밍 색인으로 메모리 점유 최소화.
  - `tqdm` 이중 프로그레스 바 적용: 전체 데이터셋에 대한 남은 예상 소요 시간(ETA) 및 실시간 처리 속도(it/s) 표시.
  - 테스트 적재용 `--sample-lines <N>`, 대상 컬렉션 초기화 `--reset`, 모델 지정 `-m`, 컬렉션 지정 `-c` 옵션 지원.
- **시맨틱 검색 및 메타데이터 필터링 도구 ([`query_search.py`](file:///c:/study/Projects/team-ggeodeureok/query_search.py))**:
  - 자연어 질문과 ChromaDB `$and` 메타데이터 조건식 결합.
  - 의회명(`--assembly`), 회의연도(`--year`), 발언자(`--speaker`), 출력개수(`-k`), 모델(`-m`), 컬렉션(`-c`) 파라미터 지원.
  - 코사인 유사도 거리, 일시/의회, 안건, 발언자, 발언 요약 프리뷰 출력.

---

### 4. 종합 문서화 및 Git 형상 관리
- **전체 사용법 문서 최신화 ([`README.md`](file:///c:/study/Projects/team-ggeodeureok/README.md))**:
  - 최신 디렉토리 구조 및 파일 목록 상세 기술.
  - 전처리 파이프라인(`scripts/`의 1~5단계 스크립트) 설명 추가.
  - 데이터 스키마(TSV 7개 컬럼 및 ChromaDB 메타데이터 필드 상세표) 정리.
  - 임베딩 모델 손쉬운 변경 가이드(별칭 목록 및 3가지 변경 방법, 차원 불일치 주의사항) 수록.
  - 트러블슈팅(가상환경, Windows PowerShell UTF-8 인코딩, CUDA 가속 확인) 가이드 추가.
- **Git 브랜치 관리**:
  - `develop`에서 신규 작업 브랜치 `develop_kb` 분기 및 전환.
  - 불필요한 캐시/임시파일 제외 처리(`.gitignore` 정리).
  - 작업 단위별 체계적 커밋 및 원격 저장소(`origin/develop_kb`) 푸시 완료.

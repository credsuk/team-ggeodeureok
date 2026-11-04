# 의정이슈레이더 (Council Issue Radar) — 팀 꺼드럭's

2024~2025년 광역 지방의회 회의록에서 정책 이슈의 근거 발언을 검색하고, 의회별 최초 관측 시점과 후속 논의 흐름을 시간순으로 보여 주며, 다음 달 논의 지속 여부와 언급량을 예측하는 시스템입니다.

```
회의록 JSON → 정제 TSV → 정책 메타데이터 → 부모·자식 청킹 → 한국어 임베딩 → PostgreSQL+pgvector
           → 하이브리드 검색(Dense+BM25, RRF) → BGE 재순위 → 이슈 군집화 → 최초 관측·타임라인
           → 논의 지속·언급량 예측 → 근거 기반 설명 Agent(로컬 LLM) · 대시보드
```

- 회의록 원문은 **외부 AI API로 전송하지 않습니다.** 모든 모델은 로컬(GPU, Ollama)에서 실행합니다.
- 날짜·순서·예측 수치는 데이터와 통계 모델이 계산하고, LLM은 검색된 근거를 문장으로 설명하는 역할만 맡습니다.

---

## 1. 결과물 구성

```
├── README.md                  본 문서
├── requirements.txt           pip 고정 버전 (pip freeze 기반)
├── environment.yml            conda 환경 (conda env export + requirements.txt)
├── docker-compose.yml         PostgreSQL 17 + pgvector 0.8.7
├── .env.example               설정 예시 (.env로 복사해 사용)
├── data/                      데이터 (원본 회의록은 직접 배치 — data/README.md)
│   ├── db/                      구축 완료된 PostgreSQL+pgvector DB 덤프 (복원용)
│   ├── raw/                     대회 제공 원본 회의록 JSON 위치 (미포함)
│   ├── forecast/                대표 이슈 군집·예측 결과
│   └── llm_compare/             로컬 LLM 비교 실험 결과
├── model/                     모델 (model/README.md)
│   ├── forecast/                학습된 언급량 예측 모델(.joblib) · 평가 지표
│   └── huggingface/             사전학습 모델 캐시 (scripts/download_models.py로 내려받음)
├── notebooks/                 Jupyter Notebook (01~03 전처리·DB 구축, 04~06 평가, 07 서비스 실행)
├── src/                       소스코드
│   ├── 01~35_*.py               원본 전체 파이프라인 스크립트 (전처리·임베딩·적재·검색·군집·예측)
│   ├── council_vector_runtime.py, bm25_runtime.py, search_aliases.py, external_*.py, regional_stats.py  공용 모듈
│   ├── main.py                  API 서버 실행기 (Ollama 확인 → DB 검증 → FastAPI)
│   ├── api/                     FastAPI 검색·분석·채팅 API + 대시보드(api/frontend)
│   ├── agent/                   로컬 LLM Agent (질문 해석·도구 실행·답변 검증)
│   ├── demo/                    Streamlit 데모 화면
│   ├── dashboard/               노트북 07이 띄우는 대시보드 프런트엔드
│   └── config/                  대표 이슈·평가 질의·KOSIS 지표 등 설정
└── scripts/                   실행 스크립트 (모델 다운로드, 환경 점검, 파이프라인 실행, DB 덤프/복원)
```

---

## 2. 모델 실행 환경 (HW/SW)

### 2.1 검증한 실행 환경

| 구분 | 사양 / 버전 |
|---|---|
| 운영체제 | Windows 11 Pro (10.0.26200) |
| CPU / RAM | Intel Core i7-13620H / 32GB |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (VRAM 8GB) |
| NVIDIA Driver | 616.92 |
| CUDA | PyTorch CUDA 12.6 빌드 (`torch==2.14.1+cu126`) |
| Python | 3.11.16 (conda 26.7.1) |
| 컨테이너 | Docker 29.9.1 (Docker Desktop, WSL2 백엔드) |
| 벡터 DB | PostgreSQL 17.11 + pgvector 0.8.7 (`pgvector/pgvector:0.8.7-pg17` 이미지) |
| 로컬 LLM 런타임 | Ollama 0.40.13 |
| 디스크 | 원본 데이터 약 2.7GB + DB 약 12GB + 모델 약 4.5GB(HF) / 약 15GB(Ollama) |

> 전체 코퍼스 임베딩(901,358개, 약 4.1시간)과 원본 파이프라인 스크립트(`src/01~35`)는 Ubuntu(WSL2) 환경에서도 실행되었습니다. 코드는 OS 독립적이며, `scripts/*.sh`는 bash(WSL·Linux·Git Bash)에서 실행합니다.

### 2.2 주요 라이브러리

전체 고정 버전은 `requirements.txt`에 있습니다.

| 분야 | 라이브러리 (버전) |
|---|---|
| 딥러닝·임베딩 | torch 2.14.1+cu126, sentence-transformers 6.1.0, transformers 5.20.2, huggingface_hub 1.33.0 |
| DB | psycopg 3.3.6, pgvector 0.5.1 (Python), chromadb 1.5.9 (비교 실험) |
| 데이터 처리 | numpy 2.4.6, pandas 3.0.5, pyarrow 26.0.1, duckdb 2.0.0 |
| 예측 | scikit-learn 1.9.0, joblib 1.6.0 |
| 전처리 | beautifulsoup4 4.15.0, lxml 6.1.3, ijson 3.6.0, kiwipiepy 0.24.1 |
| 서비스·시각화 | fastapi 0.162.0, uvicorn 0.54.0, streamlit 1.66.1, pydeck 0.9.3, matplotlib 3.11.1 |
| LLM 클라이언트 | openai 3.44.0 (로컬 Ollama의 OpenAI 호환 엔드포인트 호출용), requests 2.34.2 |
| 노트북 | jupyterlab 4.6.5, ipykernel 7.4.0 |

### 2.3 사용 모델

| 모델 | 용도 | 실행 |
|---|---|---|
| `dragonkue/snowflake-arctic-embed-l-v2.0-ko` (rev `55ec6e9`) | 청크·질의 임베딩 | 로컬 GPU |
| `BAAI/bge-reranker-v2-m3` | 검색 후보 재순위 | 로컬 GPU |
| `hf.co/mykor/A.X-4.0-Light-gguf:Q4_K_M` | 근거 기반 설명 Agent | 로컬 Ollama |
| `gemma4:e2b` | 정책 메타데이터 추출 | 로컬 Ollama |
| `qwen2.5-coder:14b` | 의회별 HTML 파싱 규칙 생성 | 로컬 Ollama |
| Poisson Regression (scikit-learn) | 다음 달 언급량 예측 | 학습 결과 `model/forecast/` |

세부 목록과 비교 실험용 모델은 `model/README.md`를 참고하세요.

---

## 3. 환경 구성 및 실행 방법

> **노트북 07 하나로 서비스 실행 확인 (권장 경로)**
> conda 환경, Docker Desktop, Ollama만 설치되어 있으면 `notebooks/07_council_issue_radar.ipynb` 하나를 위에서부터 실행(Run All)하면 회의록 검색 웹사이트(대시보드)까지 뜹니다. 전처리·DB 구축(노트북 01~03)은 끝났다고 가정하며, 그 결과물이 제출 DB 덤프입니다.
> 노트북의 **1-1 셀**이 `.env` 생성, 임베딩 모델 다운로드(약 4.3GB), DB 컨테이너 시작과 제출 DB 복원(최초 1회 20~30분)을, **2번 셀**이 Ollama 채팅 모델 설치를 대신 처리합니다.
> 이후 셀에서 검색 → 이슈 군집 → 예측 특징 → **Poisson 모델 학습·다음 달 예측(8-4)** → 이슈 분석 → 로컬 LLM 채팅 → API·대시보드까지 실행됩니다. 성능 평가는 노트북 04(예측 모델 비교), 05(임베딩·벡터 DB 비교와 예측 평가 기록), 06(LLM 비교)에 있습니다.
>
> ```bash
> conda env create -f environment.yml
> conda activate council-radar
> jupyter lab notebooks        # 07_council_issue_radar.ipynb → Run All
> ```
>
> 아래 3.1은 같은 준비를 수동으로 하는 방법입니다(서버 `src/main.py`나 다른 노트북을 실행할 때).

### 3.1 설치

```bash
# 1) Python 환경 (둘 중 하나)
conda env create -f environment.yml
conda activate council-radar
#    또는: python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt

# 2) 설정 파일
copy .env.example .env          # Linux/macOS: cp .env.example .env
#    .env의 COUNCIL_MODEL_CACHE에 model/huggingface 폴더의 절대경로를 적습니다.

# 3) PostgreSQL + pgvector 실행 및 제출 DB 복원 (벡터 901,358개, 수십 분 소요)
docker compose up -d pgvector
powershell -ExecutionPolicy Bypass -File scripts\restore_db.ps1
#    Linux/WSL/Git Bash: bash scripts/restore_pgvector_from_dump.sh

# 4) 모델 다운로드 (Hugging Face → model/huggingface, Ollama 모델 설치)
python scripts/download_models.py --ollama

# 5) (전처리부터 다시 실행할 때만) 원본 회의록 배치:
#    대회 제공 JSON을 data/raw/2024_광역회의록, data/raw/2025_광역회의록 에 복사
```

설치 상태는 `python scripts/00_check_environment.py`로 점검할 수 있습니다.

### 3.1.1 제출 DB (PostgreSQL 덤프)

검색·분석·채팅·대시보드는 모두 PostgreSQL+pgvector 테이블 `council_arctic_raw_v1`을 사용합니다. 이 테이블을 처음부터 만들려면 전처리·LLM 메타데이터 추출·전체 임베딩(GPU 약 4시간)이 필요하므로, **구축이 끝난 DB를 덤프 파일로 함께 제출**했습니다.

| 파일 | 내용 |
|---|---|
| `data/db/council_vectors.dump` | `pg_dump` custom 형식 (압축). 테이블 1개, 901,358행, 1,024차원 벡터 |
| `data/db/council_vectors.dump.sha256` | 무결성 확인용 체크섬 |
| `data/db/council_vectors.dump.info.txt` | 생성 환경·행 수·pgvector 버전 |

- 복원 스크립트는 체크섬을 확인한 뒤 `pg_restore`로 데이터를 넣고 HNSW·B-tree 인덱스를 다시 만듭니다. (검증 환경에서 약 20~30분)
- 새 컨테이너에 복원한 DB와 원본 DB를 비교해 행 수(901,358)·벡터 값(오차 0)·검색 Top-10 결과가 같음을 확인했습니다.
- 복원 후 Docker 볼륨 `council_pgdata`에 약 12GB가 저장됩니다.
- DB는 호스트 포트 **15432**로 열립니다. PC에 PostgreSQL이 이미 설치되어 있으면 보통 5432를 쓰기 때문에 충돌을 피하려고 바꾼 값입니다.
- 복원만으로 노트북 05~07과 `src/main.py`를 바로 실행할 수 있습니다. 노트북 01~03은 DB를 처음부터 만드는 과정을 확인하기 위한 것이고, 노트북 04는 DB 없이 실행됩니다.

### 3.2 노트북 실행 순서 (학습·평가)

Jupyter는 `notebooks/` 폴더에서 실행합니다 (`jupyter lab notebooks`). 각 노트북은 셀 순서대로 실행하며, 단계별 설명이 Markdown으로 들어 있습니다.

| 순서 | 노트북 | 내용 | 비고 |
|---|---|---|---|
| 01 | `01_json_to_text_pipeline.ipynb` | 원본 JSON → HTML 샘플 → 로컬 LLM 파싱 규칙 → 정제 TSV | Ollama 필요 |
| 02 | `02_metadata_llm_pipeline.ipynb` | TSV 배치 분할 → 로컬 LLM 정책 메타데이터(JSONL) 추출 | 시연을 위해 기본 5건만 추출 |
| 03 | `03_pgvector_db_pipeline.ipynb` | 결합 검증 → canonical → 부모·자식 청킹 → 임베딩 → **pgvector DB 구축** → 검색 확인 | GPU 권장, 전체 임베딩 수 시간 |
| 04 | `04_experiments_forecast_comparison.ipynb` | 예측 모델 비교: 전월 유지 · 3개월 평균 · 3개월 가중평균 · 포아송 회귀를 **동일 패널·동일 시간순 Fold**에서 MAE·RMSE로 비교 | DB 불필요 (`data/forecast/` 사용), 수 초 |
| 05 | `05_experiments_embedding_model_DB_comparison.ipynb` | 검색 실험 3종(Raw/Enriched, Arctic/KURE, Chroma/pgvector)과 예측 Rolling Backtest 평가 기록·시각화 | 기본은 저장 결과 확인 모드 |
| 06 | `06_experiments_korean_llm_comparison.ipynb` | 한국어 로컬 LLM 6종 근거 답변 비교 (속도·인용·환각 검증) | 결과는 `data/llm_compare/` |
| 07 | `07_council_issue_radar.ipynb` | **서비스 실행**: 실행 전 자동 준비 → 하이브리드 검색 → 이슈 군집 → 예측 특징 → **Poisson 모델 학습·예측** → 이슈 분석 → 로컬 LLM 채팅 → API·대시보드 | **단독 실행 가능** (모델·DB 자동 준비) |

- 01~02 노트북은 기본으로 `TARGET_YEAR="2024"`, 02는 `강원특별자치도의회` 한 곳을 처리합니다. 셀 상단 설정값을 바꿔 다른 연도·의회를 처리합니다.
- 03 노트북은 01·02 결과를 읽어 `council_arctic_raw_v1` 테이블을 만듭니다. 이 테이블은 07 노트북과 API 서버가 그대로 사용합니다.
- 3.1에서 제출 DB를 복원했다면 01~03은 건너뛰고 04~07을 실행해도 됩니다. 03을 실행하면 같은 이름의 테이블에 다시 적재하므로, 03을 시험할 때는 셀 상단의 `PG_TABLE`을 다른 이름으로 바꾸세요.

### 3.3 서비스 실행 (추론)

```bash
python src/main.py
# → Ollama 확인·자동 시작 → PostgreSQL/임베딩 모델 검증 → http://127.0.0.1:8000 대시보드
```

| 기능 | 엔드포인트 |
|---|---|
| 대표 이슈 목록 / 상세 | `GET /api/v1/issues`, `GET /api/v1/issues/{key}` |
| 이슈 레이더 검색 | `POST /api/v1/radar/search` |
| 원문 검색 / 회의록·근거 조회 | `GET /api/v1/documents/search`, `GET /api/v1/documents/{document_id}`, `GET /api/v1/evidence/{chunk_id}` |
| 근거 기반 채팅 / LLM 상태 | `POST /api/v1/chat`, `GET /api/v1/chat/status` |

Streamlit 데모는 원본 파이프라인 스크립트(3.4)의 산출물(`outputs/`)이 있어야 실행됩니다. 제출물에는 포함되지 않았으므로 심사용 화면은 위 `src/main.py` 대시보드를 사용하세요. (`bash scripts/run_final_app.sh` → http://localhost:8501)

### 3.3.1 문제 해결

| 증상 | 원인 · 해결 |
|---|---|
| `password authentication failed for user "council"` | 다른 PostgreSQL에 접속한 경우입니다. `.env`의 `PGVECTOR_DSN` 포트가 `POSTGRES_PORT`(15432)와 같은지 확인하세요. |
| 서버 시작 시 포트 사용 중 | `.env`의 `PORT`를 다른 값(예: 8001)으로 바꿉니다. |
| 임베딩 모델을 찾을 수 없음 | `.env`의 `COUNCIL_MODEL_CACHE`를 `model/huggingface`의 **절대경로**로 지정했는지, `scripts/download_models.py`를 실행했는지 확인하세요. |
| 채팅이 근거 요약만 반환 | Ollama가 꺼져 있거나 `COUNCIL_AGENT_MODEL`이 설치되지 않은 경우입니다. `ollama pull hf.co/mykor/A.X-4.0-Light-gguf:Q4_K_M` |
| `restore_db.ps1` 실행 거부 | `powershell -ExecutionPolicy Bypass -File scripts\restore_db.ps1` 형식으로 실행합니다. |

### 3.3.2 재현 검증 기록

제출 전, 심사 환경과 같은 절차로 처음부터 다시 실행해 확인했습니다. (Windows 11, RTX 4060 Laptop)

| 단계 | 결과 |
|---|---|
| `conda env create -f environment.yml` | 설치 성공, `pip check` 이상 없음, CUDA 사용 가능 |
| `scripts/download_models.py` | 임베딩·재순위 모델 4.3GB 다운로드 |
| `docker compose up` + `scripts/restore_db.ps1` | 901,358행 복원 (약 29분) |
| `python src/main.py` | DB·임베딩 모델 검증 후 서버 기동, 모든 API 엔드포인트 200 응답, 로컬 LLM 채팅 `[근거 N]` 인용 답변 |
| 노트북 07 (당시 04) | 25개 코드 셀 전체 실행 성공 (`backtest` 모드). 재학습한 예측 모델·결과가 제출본과 동일 |
| 노트북 07 **단독 실행** (당시 04) | `.env`·모델·DB가 하나도 없는 상태에서 노트북만 Run All → 1-1 셀이 모델 다운로드·DB 컨테이너 시작·덤프 복원(20분)까지 처리, 전체 코드 셀 오류 없이 완료 (총 43분, 당시 평가 셀 포함). 이후 평가 셀(8-5)은 평가 노트북 담당으로 정리해 제거 |
| 노트북 04 (예측 모델 비교) | 전체 실행 성공, DB 없이 `data/forecast/` 캐시로 패널 232행 구성 → 10 Fold · 모델별 100건 비교 |
| 노트북 05, 06 | 전체 실행 성공, 저장 결과 파일 변화 없음 |

노트북 01~03(전처리부터 DB 구축)은 원본 회의록과 수 시간의 처리가 필요해 이번 검증에서는 실행하지 않았고, 03은 합성 데이터로 전 단계 동작을 확인했습니다.

### 3.4 원본 파이프라인 스크립트 (선택)

노트북은 심사용으로 단계를 정리한 버전이고, 실제 최종 결과(2,057,214 발화, 901,358 청크, 대표 이슈 10개, 예측 평가)는 `src/01~35` 스크립트로 생성했습니다. 스크립트는 `data/input/`(정제 TSV·메타데이터)을 입력으로 하며 산출물은 `data/`, `outputs/`에 저장됩니다.

```bash
python src/01_restart_inventory.py            # 입력 검사
python src/04_build_direct_canonical.py       # canonical 생성
python src/05_build_parent_child_chunks.py    # 청킹
python src/08_embed_full_corpus.py            # 전체 임베딩
python src/16_load_pgvector.py                # pgvector 적재
bash scripts/build_bm25_hybrid.sh             # BM25 색인
bash scripts/run_representative_issues.sh candidates   # 대표 이슈 후보
bash scripts/run_forecast_evaluation.sh       # 예측 학습·평가
```

---

## 4. 주요 결과

| 항목 | 결과 |
|---|---|
| 임베딩 입력 (Raw vs Enriched) | Raw 선택 — nDCG@10 0.8928 vs 0.8871 |
| 임베딩 모델 (Arctic vs KURE) | Arctic 선택 — nDCG@10 0.8969 vs 0.8657, 58.95 vs 56.25 행/초 |
| 벡터 DB (Chroma vs pgvector) | pgvector 선택 — 지역+연도 필터 p50 3.97ms vs 586.29ms |
| 논의 지속 예측 F1 | 0.8814 (직전 달 기준선 대비 +0.0621) |
| 언급량 MAE / RMSE | 7.12건 / 10.04건 (기준선 대비 29.8% / 24.3% 개선) |
| 방향성 정확도 | 0.76 |
| 설명 Agent LLM | A.X-4.0-Light — 평균 7.7초, 8/8 문항 인용, 미끼 질문에 '근거 부족' 응답 |

평가 상세: `notebooks/04`, `notebooks/05`, `notebooks/06`, `model/forecast/forecast_metrics.json`, `data/llm_compare/report.md`

---

## 5. 외부 AI 서비스 및 API

**외부 AI 서비스(유료 LLM API 등)는 사용하지 않습니다.** 생성형 모델은 모두 로컬 Ollama, 임베딩·재순위 모델은 로컬 GPU에서 실행합니다. 인터넷은 최초 설치 시 패키지·모델 파일·Docker 이미지를 내려받을 때만 필요합니다.

선택 기능으로 다음 **공공데이터 API**를 참고 정보 조회에만 사용합니다. 예측 모델의 입력에는 쓰지 않으며, 인증값이 없으면 해당 출처를 건너뛰고 나머지 기능은 정상 동작합니다.

| 서비스 | 용도 | 설정 키 | 외부로 전송하는 값 |
|---|---|---|---|
| 법제처 자치법규 Open API | 광역·기초 조례 제정 여부·날짜 확인 | `LAW_OC` | 검증된 짧은 정책어 |
| 열린국회정보 Open API | 관련 국회 의안 동향 | `ASSEMBLY_API_KEY` | 검증된 짧은 정책어 |
| KOSIS 국가통계포털 Open API | 지역별 정책 수요 지표 | `KOSIS_API_KEY` | 사전 등록된 통계표 ID |

- 회의록 원문·질문 원문·발화자 정보는 전송하지 않습니다.
- 인증정보는 제출물에 포함하지 않았습니다. **심사용 임시 인증정보는 별도로 전달**하며(조회 전용·사용기간 제한), `.env`의 해당 키에 입력하면 됩니다.

---

## 6. 데모영상

- 파일/링크: `TODO — 데모영상 경로 또는 URL 기입`
- 내용: 설치 확인 → `python src/main.py` 실행 → 대표 이슈 선택 → 지도 타임라인·근거 확인 → 예측 카드 → 로컬 LLM 채팅

---

## 7. 외부 자료 이용 및 이용조건

| 자료 | 종류 | 출처 | 이용조건 확인 |
|---|---|---|---|
| 광역의회 회의록 2024~2025 | 데이터 | 경진대회 제공 | 대회 규정에 따름, 재배포하지 않음 |
| 법제처·열린국회정보·KOSIS | 공공데이터 API | 각 기관 Open API | 공공누리·각 API 이용약관 (출처 표시) |
| snowflake-arctic-embed-l-v2.0-ko | 사전학습 모델 | Hugging Face | ☐ 모델 카드 라이선스 확인 |
| bge-reranker-v2-m3 · bge-m3 | 사전학습 모델 | Hugging Face (BAAI) | ☐ 모델 카드 라이선스 확인 |
| KURE-v1 | 사전학습 모델 (비교 실험) | Hugging Face (nlpai-lab) | ☐ 모델 카드 라이선스 확인 |
| A.X-4.0-Light, EXAONE 3.5/4.0, Kanana 1.5, Tri-7B, HyperCLOVAX-SEED, Gemma, Qwen2.5-Coder | 로컬 LLM | Ollama / Hugging Face GGUF | ☐ 각 모델 라이선스 확인 (일부 비상업 조건 포함 가능) |
| PostgreSQL, pgvector | DB | 공식 Docker 이미지 | PostgreSQL License |
| Python 라이브러리 | 코드 | PyPI | `requirements.txt` 참고 (MIT·BSD·Apache 계열) |

모델은 가중치를 수정·재배포하지 않고 원 출처에서 내려받아 추론에만 사용합니다.

---

## 8. 민감정보 점검

- 제출물에 `.env`, API Key, 비밀번호, 개인 인증정보를 포함하지 않았습니다. (`.env.example`은 빈 값과 로컬 전용 기본값만 포함)
- `docker-compose.yml`·`.env.example`의 DB 비밀번호(`council-local-only`)는 로컬 개발용 기본값이며 외부에 노출되는 서비스가 아닙니다.
- 원본 회의록과 모델 가중치는 포함하지 않았습니다. DB 덤프(`data/db/`)는 대회 데이터에서 파생된 회의록 본문을 포함하므로 심사용으로만 제출하며 공개 저장소에 올리지 않습니다.
- 회의록에 포함된 발언자 성명은 공개 회의록의 공무상 발언 정보이며, 외부 API로 전송하지 않습니다.

---

## 9. AI 활용 범위 및 검토 과정

| 활용 범위 | 사용 도구 | 검토 과정 |
|---|---|---|
| 전처리·DB 구축 스크립트를 단계별 설명이 있는 노트북(03)으로 재구성, 노트북 01·02·06 데이터 경로 통일 | Claude Code | 합성 데이터로 01~09단계 전 과정 실행, 실제 pgvector 컨테이너에 임시 테이블 적재·자기검색 검증 후 삭제 |
| 제출 폴더 구성, 경로 수정, DB 덤프·복원 스크립트, README·환경 설정 파일 작성 | Claude Code | API 단위테스트 24개 통과, 새 conda 환경·새 DB 컨테이너에서 README 절차대로 재현 실행(3.3.2), 팀원 검토 |
| `TODO — 팀원별 기타 AI 활용 내역 기입` | | |

시스템 내부에서 사용하는 LLM(메타데이터 추출, 파싱 규칙 생성, 설명 Agent)은 모두 로컬 모델이며, 생성 결과는 다음과 같이 검토합니다.

- LLM 메타데이터·관련도 라벨은 초안으로만 쓰고, 대표 이슈의 핵심 근거는 사람이 검수한 뒤 `issue_id`를 고정했습니다.
- 설명 Agent 답변은 숫자·날짜·의회명을 검색 근거와 자동 대조하고, 근거에 없는 질문에는 "근거 부족"으로 답하도록 했습니다.

---

## 10. 제출 전 확인사항

- [x] README.md에 결과물 구성 및 실행방법 설명
- [x] 실행환경·라이브러리 버전 명시 (`requirements.txt`, `environment.yml`)
- [x] 학습·평가(추론) 과정 노트북 (`notebooks/01~07`)
- [ ] 데모영상 포함 (6장 TODO)
- [ ] 외부 데이터·코드·모델 이용조건 확인 (7장 체크박스)
- [x] 개인정보·인증정보 미포함
- [ ] AI 활용 범위 및 검토 과정 작성 (9장 팀원 내역 TODO)

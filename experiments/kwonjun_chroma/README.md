# 의정이슈레이더

광역 지방의회 회의록에서 정책 이슈의 근거를 검색하고, 최초 관측 시점과
의회별 시간 흐름을 추적하며, 다음 달 논의 지속 여부와 언급량을 예측하는
로컬 AI 분석 프로젝트입니다.

## 핵심 구성

```text
원본 회의록·메타데이터
→ canonical 통합
→ 부모·자식 청킹
→ Arctic 한국어 임베딩
→ pgvector Dense 검색 + BM25
→ RRF 결합
→ BGE 재순위
→ 이슈 확장·그래프 군집화
→ 최초 관측·의회별 관측 순서
→ 시간순 예측
→ Streamlit 지도·로컬 Qwen Agent
```

## 확정 모델 및 데이터베이스

| 구분 | 선택 |
|---|---|
| 임베딩 | `dragonkue/snowflake-arctic-embed-l-v2.0-ko` |
| 임베딩 차원 | 1,024 |
| 검색 청크 | 901,358개 |
| 벡터 DB | PostgreSQL + pgvector HNSW |
| 키워드 검색 | SQLite FTS5 BM25 |
| 결합 | Reciprocal Rank Fusion |
| 재순위 | `BAAI/bge-reranker-v2-m3` |
| 로컬 Agent | Ollama `qwen3:8b` |
| 웹 | Streamlit + PyDeck |

## 주요 실험 결과

### 임베딩 입력 및 모델

| 비교 | nDCG@10 |
|---|---:|
| Raw | 0.8928 |
| Enriched | 0.8871 |
| Arctic | 0.8969 |
| KURE | 0.8657 |

Raw 입력과 Arctic 모델을 최종 선택했습니다. 메타데이터는 임베딩 텍스트에
강제로 합치지 않고 필터링, 결과 표시, 이슈 분석에 사용합니다.

### 예측

| 항목 | 결과 |
|---|---:|
| 논의 지속 F1 | 0.8814 |
| 언급량 MAE | 7.1224 |
| 언급량 RMSE | 10.0425 |
| 방향성 정확도 | 0.7600 |
| 전월 유지 대비 MAE 개선 | 29.83% |
| 전월 유지 대비 RMSE 개선 | 24.31% |

무작위 분할 없이 월 단위 expanding-window rolling backtest를 사용했습니다.

## 실행 환경

- Ubuntu 24.04 / WSL2
- Python 3.12
- NVIDIA GPU 및 CUDA 지원 PyTorch
- Docker Desktop + PostgreSQL/pgvector
- Ollama

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-base.txt
pip install -r requirements-enrichment.txt
pip install -r requirements-vector.txt
pip install -r requirements-forecast.txt
pip install -r requirements-app.txt
```

PyTorch는 사용하는 CUDA 환경에 맞춰 공식 설치 명령으로 별도 설치합니다.

## 실행

`.env.example`을 복사한 뒤 로컬 비밀번호와 경로를 설정합니다.

```bash
cp .env.example .env
docker compose up -d pgvector
bash scripts/run_final_app.sh
```

기본 접속 주소는 `http://127.0.0.1:8501`입니다.

## 주요 코드

- `src/18_pgvector_retrieval.py`: Dense·BM25·RRF 후보 검색
- `src/19_rerank_candidates.py`: BGE Cross-Encoder 재순위
- `src/20_expand_cluster_issue.py`: 중심 벡터 확장과 유사도 그래프 군집화
- `src/21_build_issue_timeline.py`: 최초 관측 및 의회별 시간 순서
- `src/32_train_multi_issue_forecast.py`: 시간순 예측 평가
- `demo/final_app.py`: 최종 지도·검색·Agent 화면

전체 처리 순서는 `docs/ACTIVE_PIPELINE.md`를 참고합니다.

## 데이터 이용과 보안

- 회의록 원문은 외부 상용 LLM API로 전송하지 않습니다.
- 임베딩, 검색, 재순위, 군집화와 Agent는 로컬에서 실행합니다.
- 외부 상용 LLM API를 사용하지 않습니다.
- GitHub 저장소에는 원본 데이터, 임베딩, 벡터 DB, 원문 검색 결과를 포함하지 않습니다.
- 예측은 제공된 2024~2025년 회의록 범위만 사용합니다.
- 지도 연결선은 의회별 최초 관측의 시간 순서이며 인과적 정책 전달을 의미하지 않습니다.

## 대용량 파일

다음 파일은 GitHub에 포함되지 않습니다.

```text
data/input/
data/canonical_v3/
data/embeddings/
data/bm25/
data/chroma/
data/postgres/
models/huggingface/
outputs/
```

필요한 모델명과 로컬 배치 방법은 `models/README.md`, 데이터 구조는
`data/README.md`를 참고합니다.


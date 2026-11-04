# 모델 구성 (model/)

본 디렉토리에는 **의정이슈레이더 (Council Issue Radar)** 프로젝트에서 사용하는 모델(임베딩, 재순위, 로컬 설명 LLM, 시계열 통계 예측 모델)의 가중치, 평가 지표 및 메타데이터 명세를 관리합니다.

대용량 사전학습 가중치(합계 약 4.5GB)와 로컬 Ollama 모델 저장소는 Git 저장소 용량 및 배포 규정 준수를 위해 직접 포함하지 않으며, 정확한 모델 식별 정보와 재현 방법을 명시하여 스크립트로 내려받도록 구성되어 있습니다.

---

## 1. 디렉토리 구성

| 경로 | 내용 |
|---|---|
| `MODEL_MANIFEST.json` | 모델 명칭·버전·차원·파라미터 등 기계 판독용 전체 모델 명세서 |
| `forecast/poisson_regression.joblib` | 다음 달 언급량 예측 모델 (scikit-learn `PoissonRegressor` 파이프라인). 노트북 07 · 8-4에서 학습·저장 |
| `forecast/forecast_metrics.json` | 원본 파이프라인(`src/32`)의 10-Fold 시간순 Rolling Backtest 평가 결과 (F1, MAE, RMSE, 방향성 정확도) — 최종 보고서 지표 |
| `huggingface/` | 사전학습 모델 캐시 위치 (비어 있음 — `scripts/download_models.py`로 다운로드) |

> **참고**: 다음 달 논의 지속 여부 분류는 학습 파라미터가 없는 규칙 모델(최근 3개월 활동률 기준선)을 적용하므로 별도 가중치 파일이 없습니다.

---

## 2. 모델별 세부 역할 및 파이프라인

1. **Arctic 한국어 임베딩 (`dragonkue/snowflake-arctic-embed-l-v2.0-ko`)**
   - 회의록 청크(부모-자식 계층 분할)와 사용자 질의를 1,024차원 정규화 벡터로 변환
   - PostgreSQL + pgvector(HNSW 인덱스, Cosine 거리) 기반의 Dense 검색 수행

2. **BGE Reranker (`BAAI/bge-reranker-v2-m3`)**
   - Dense(임베딩)와 Sparse(BM25)의 하이브리드 검색(RRF)으로 수집된 상위 후보를 질의와 함께 Cross-Encoder로 정밀 재순위화

3. **로컬 설명 Agent LLM (`hf.co/mykor/A.X-4.0-Light-gguf:Q4_K_M` / `SK Telecom`)**
   - 검색·군집·시계열 예측 결과를 종합하여 회의록 근거 번호(`[근거 N]`)를 인용한 한국어 설명 답변 생성
   - 로컬 Ollama 환경에서 실행하며 회의록 원문을 외부 상용 API로 전송하지 않음

4. **최근 활동률 기준선 (Active Ratio Baseline)**
   - 최근 관측 월의 활동률을 이용하여 다음 달 논의 지속 여부 이진 분류

5. **Poisson Regression (`scikit-learn`)**
   - 0 이상의 정수형 언급량 분포에 최적화된 포아송 회귀 모델
   - 과거 발언 이력 및 시간순 Rolling Backtest(10-Fold)로 검증 및 선택

정확한 모델 식별 정보와 파라미터는 [`MODEL_MANIFEST.json`](MODEL_MANIFEST.json)을 확인하세요.

---

## 3. 사전학습 모델 (파인튜닝 없이 사용)

용량(합계 약 4.5GB) 때문에 가중치는 Git에 직접 포함하지 않았습니다. 다음 명령으로 `model/huggingface/`에 내려받습니다:

```bash
python scripts/download_models.py              # 필수: 임베딩 + 재순위
python scripts/download_models.py --comparison # 비교 실험 모델 포함
```

| 모델 | revision | 용도 | 출처 |
|---|---|---|---|
| `dragonkue/snowflake-arctic-embed-l-v2.0-ko` | `55ec6e9358a56d56af759bc8372e970caf8c305f` | 청크·질의 임베딩 (1,024차원) | https://huggingface.co/dragonkue/snowflake-arctic-embed-l-v2.0-ko |
| `BAAI/bge-reranker-v2-m3` | main | 검색 후보 재순위 (Cross-Encoder) | https://huggingface.co/BAAI/bge-reranker-v2-m3 |
| `nlpai-lab/KURE-v1` | main | 임베딩 비교 실험 | https://huggingface.co/nlpai-lab/KURE-v1 |
| `BAAI/bge-m3` | main | 임베딩 비교 실험 | https://huggingface.co/BAAI/bge-m3 |

---

## 4. 로컬 LLM (Ollama)

모든 생성형 모델은 로컬 Ollama(`http://127.0.0.1:11434`)에서 실행되며 외부 API를 쓰지 않습니다.

| Ollama 모델 | 용도 | 비고 |
|---|---|---|
| `hf.co/mykor/A.X-4.0-Light-gguf:Q4_K_M` | 근거 기반 설명 Agent (기본) | 로컬 구동 주력 모델 |
| `exaone3.5:7.8b` | 설명 Agent 대안, LLM 비교 | LG AI Research |
| `gemma4:e2b` | 정책 메타데이터 추출 (노트북 02) | 경량 메타데이터 처리 |
| `qwen2.5-coder:14b` | 의회별 HTML 파싱 규칙 생성 (노트북 01) | 파싱 자동화 |
| Kanana 1.5 8B, Tri-7B, HyperCLOVAX-SEED 1.5B, EXAONE 4.0 1.2B | LLM 비교 실험만 (노트북 06) | 비교 평가용 |

설치 명령은 `python scripts/download_models.py`가 출력합니다 (`--ollama`를 붙이면 바로 설치).

# model/

## 포함된 파일

| 경로 | 내용 |
|---|---|
| `forecast/poisson_regression.joblib` | 다음 달 언급량 예측 모델 (scikit-learn `PoissonRegressor` 파이프라인). 노트북 07 · 8-4에서 학습·저장 |
| `forecast/forecast_metrics.json` | 원본 파이프라인(`src/32`)의 시간순 Rolling Backtest 평가 결과 (F1, MAE, RMSE, 방향성 정확도) — 보고서 수치 |
| `huggingface/` | 사전학습 모델 캐시 위치 (비어 있음 — 아래 명령으로 내려받음) |

논의 지속 여부 분류는 학습 파라미터가 없는 규칙 모델(최근 3개월 활동률 기준선)이라 별도 파일이 없습니다.

## 사전학습 모델 (파인튜닝 없이 사용)

용량(합계 약 4.5GB) 때문에 가중치는 제출물에 포함하지 않았습니다. 다음 명령으로 `model/huggingface/`에 내려받습니다.

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

## 로컬 LLM (Ollama)

모든 생성형 모델은 로컬 Ollama(`http://127.0.0.1:11434`)에서 실행되며 외부 API를 쓰지 않습니다.

| Ollama 모델 | 용도 |
|---|---|
| `hf.co/mykor/A.X-4.0-Light-gguf:Q4_K_M` | 근거 기반 설명 Agent (기본) |
| `exaone3.5:7.8b` | 설명 Agent 대안, LLM 비교 |
| `gemma4:e2b` | 정책 메타데이터 추출 (노트북 02) |
| `qwen2.5-coder:14b` | 의회별 HTML 파싱 규칙 생성 (노트북 01) |
| Kanana 1.5 8B, Tri-7B, HyperCLOVAX-SEED 1.5B, EXAONE 4.0 1.2B | LLM 비교 실험만 (노트북 06) |

설치 명령은 `python scripts/download_models.py`가 출력합니다 (`--ollama`를 붙이면 바로 설치).

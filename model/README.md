# model/ — 검색 앙상블 구현과 모델 안내

`model/ensemble/`에는 검색과 재순위화 단계에서 사용하는 순위 결합 구현이 있습니다. 이 코드는 모델 추론·검색 결과를 결합하는 구현이며, 프로젝트가 직접 학습한 신경망 가중치가 아닙니다.

## 코드 위치

- `model/ensemble/rank_fusion.py`: Chroma Dense 검색과 SQLite FTS5 BM25 결과의 weighted Reciprocal Rank Fusion (RRF)
- `model/ensemble/cross_encoder.py`: 하나 이상의 Cross-Encoder 추론 및 모델별 순위를 가중 RRF로 결합
- `src/search/retrieval.py`: 검색 실행 흐름과 Chroma/BM25/원문 데이터 연결
- `src/search/ranking.py`, `src/search/reranking.py`: 기존 `src.*` import를 유지하는 호환 경로. 실제 앙상블 구현은 `model/ensemble/`에 있습니다.

## 모델 가중치와 실행 설정

현재 로컬 환경은 `BAAI/bge-reranker-v2-m3`와 `dragonkue/bge-reranker-v2-m3-ko`를 1:1로 사용하도록 설정되어 있습니다. 해당 외부 체크포인트는 저장소 안이 아니라 Windows Hugging Face 캐시에 있으며, 기본 설정은 자동 다운로드를 허용하지 않습니다. 임베딩 모델과 Ollama LLM도 별도의 외부 사전학습 모델입니다.

Dense/BM25 상대 순위 기여는 `DENSE_RRF_WEIGHT`와 `BM25_RRF_WEIGHT`로 조정하고, Cross-Encoder 목록과 순위 기여는 `RERANKER_MODELS`와 `RERANKER_WEIGHTS`로 설정합니다. 이 값들은 신경망 내부 가중치를 학습하거나 바꾸는 값이 아닙니다. 이 프로젝트에서 특정 가중치 조합이 더 정확하다는 평가는 아직 완료되지 않았습니다.

## 학습 여부와 제출 시 주의

현재 프로젝트는 임베딩·재순위화·LLM 가중치를 직접 fine-tune하지 않았으며, `model/` 안에 프로젝트 자체의 학습 체크포인트는 없습니다. 검색·앙상블 코드를 두었다고 해서 모델을 직접 학습했다고 표시하지 마세요.

대회/제출 규칙이 실제 체크포인트 파일을 요구하는지 먼저 확인하세요. 외부 사전학습 모델 파일을 저장소에 포함하려면 각 모델의 라이선스, 재배포 허용 여부, 파일 크기와 실행 환경을 확인한 뒤 별도 승인을 받아야 합니다. 직접 학습한 모델을 추가할 때는 체크포인트와 함께 기반 모델, 학습 데이터·라이선스, 학습 설정, 평가 결과, 재현 방법을 기록하세요.

# scripts/ — 데이터 처리 및 검색 인덱스 스크립트

현재 스크립트는 회의록 전처리, 메타데이터 처리, Chroma 벡터 인덱스 및 SQLite FTS5 BM25 인덱스 구축, 사람 검토 평가 준비·채점에 사용됩니다. 별도의 모델 파라미터 학습 스크립트는 확인되지 않았습니다.

## 주요 스크립트

- `json_to_text/1_split_sample.py`: 원본 회의록에서 규칙 분석용 HTML 샘플을 추출
- `json_to_text/2_rules_make.py`: 로컬 Ollama 호출로 HTML 발언 구조에 맞는 추출 규칙을 생성·검토
- `json_to_text/3_rules_summary.py`, `4_to_txt.py`: 회의록 발언·안건을 추출하고 정리해 텍스트로 저장
- `json_to_text/5_jsonl_to_org.py`: 메타데이터와 발언 TXT를 안정 행 ID에 맞춰 결합
- `make_chroma_db/1_metadata_to_chroma.py`: 메타데이터 JSONL의 요약을 임베딩해 ChromaDB에 적재. 실제 적재는 CUDA GPU를 요구하며 dry-run은 입력 검증용
- `make_chroma_db/2_build_bm25_index.py`: 결합 JSONL 발언으로 SQLite FTS5 BM25 인덱스를 생성
- `metadata_llm/split_council_tsv.py`: TSV를 메타데이터 검토용 작은 묶음으로 분할
- `evaluation/reranker_eval.py`: 현재 Chroma/BM25 검색 후보의 blinded review CSV를 만들고, 사람 relevance 0/1/2 라벨 입력 후 BAAI-only·Korean-only·1:1 앙상블을 같은 후보에서 NDCG@10/MRR@10으로 비교. 색인 재생성이나 라벨 자동 생성은 하지 않음

## 사용 전에

1. 프로젝트 루트에서 경로와 각 스크립트의 `--help`를 확인합니다.
2. 입력 데이터·출력 경로·모델·장치 설정 및 필요한 패키지를 확인합니다.
3. 재생성은 기존 인덱스나 파일을 바꿀 수 있으므로 원본을 백업하고, 사용자가 의도한 경우에만 실행합니다.
4. Chroma 적재와 FTS5 인덱스 생성 후 산출 경로·레코드 범위·개수를 확인합니다.
5. 제출물에는 실행 명령, 입력/출력 설명, 필요한 환경을 기록합니다. 성공하지 않은 실행을 완료로 보고하지 않습니다.

기본 앱 실행 및 색인 명령은 루트 `README.md`에 있습니다. 전처리 과정에서 로컬 Ollama 설정이 사용되며, 관련 설정과 데이터 처리 범위를 먼저 확인하세요. 평가용 질문·후보 파일과 실행 명령은 `.skill/evaluation-process.md` 및 `C:/hermes/mini_solo_reranker_eval/README.md`를 참고하세요. 후보 생성은 기존 색인을 읽기만 하며, 모델 품질 비교는 사람이 0/1/2 관련도 라벨을 입력한 뒤에만 실행합니다.

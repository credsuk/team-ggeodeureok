# 지방의회 회의록 기반 정책 이슈 분석

지방의회 회의록을 자연어로 검색하고, 관련 발언의 근거·쟁점·시간 흐름을 정리해 보여주는 로컬 실행형 Streamlit 분석 앱입니다. ChromaDB 벡터 검색과 SQLite FTS5 키워드 검색을 함께 사용하며, 검색된 결과는 로컬 AI 모델로 요약하고 지도와 표로 시각화합니다.

## 결과물 구성

```text
mini_solo/
├── README.md                 # 프로젝트, 데이터·모델, 추론·평가, 설치·실행 안내
├── requirements.txt          # 앱 및 데이터 처리 스크립트의 Python 의존성
├── data/                     # 회의록 데이터·인덱스 (README만 Git 포함, 데이터 파일은 제외)
│   └── README.md             # 데이터 범위·제출 시 확인할 점
├── model/                    # 학습된 모델 파일 (현재 없음)
│   └── README.md             # 모델 현황·추가 준비 안내
├── src/                      # Streamlit 앱, 검색·분석 모듈, 단위 테스트
├── scripts/                  # 전처리 및 검색 인덱스 스크립트
│   └── README.md             # 스크립트 역할·실행 전 확인
├── notebooks/                # 현재 노트북 없음
│   └── README.md             # 추가할 노트북 작성 안내
├── demo/                     # 현재 시연 영상 없음
│   └── README.md             # 녹화 순서·개인정보 점검
├── figures/                  # 현재 그림 파일 없음
│   └── README.md             # 결과 그림 준비 안내
├── .streamlit/               # Streamlit 설정
└── .skill/                   # 프로젝트 구현·평가·테스트 기록
```

`src/README.md`에는 추가 실행 전제와 화면 동작이 자세히 정리되어 있습니다. `notebooks/`, `demo/`, `figures/`에는 현재 안내문만 있으며 실제 노트북·시연 영상·그림은 아직 추가되지 않았습니다.

## 데이터

주요 데이터 경로와 역할은 다음과 같습니다.

- `data/org/`: 원본 지방의회 회의록 JSON 입력
- `data/to_text/`: 발언 단위로 정리된 TXT/TSV 파일. 분석 시 원문 발언을 다시 읽는 기준 자료
- `data/metadata_jsonl_python/`: 발언에 연결할 메타데이터와 요약을 담은 JSONL
- `data/org_jsonl/`: 메타데이터와 원문 발언을 `source_file:row_id` 식별자로 결합한 검색용 JSONL
- `data/chroma_db/`: 벡터 검색 데이터베이스. 기본 컬렉션은 `assembly_metadata`
- `data/bm25_index/assembly_bm25.sqlite3`: 발언 텍스트의 SQLite FTS5 BM25 키워드 검색 인덱스
- `data/source_metadata.sqlite3`: 날짜·의회·회의·안건과 원본 TXT의 행 위치를 연결하는 SQLite 인덱스. 원문 발언 전체를 복사해 저장하지 않음
- `data/answer_cache.sqlite3`: 구조화된 분석 결과와 질문 해시 캐시를 저장하는 SQLite DB

로컬 작업 폴더에는 2024·2025년 광역의회 자료가 있습니다. 이는 디렉터리에서 확인한 자료 범위이며, 모든 의회·기간의 데이터가 완전하게 수록되었거나 모든 인덱스가 같은 범위를 포함한다고 보증하지 않습니다. 전처리 규칙과 검색 색인은 실제 입력 데이터에 따라 다시 생성해야 합니다. 원본 자료 및 생성 데이터는 `.gitignore` 대상이므로 저장소만 복사하면 함께 제공되지 않을 수 있습니다. 데이터의 사용·재배포 권한은 별도로 확인하세요.

## 모델과 사용 목적

현재 `src/.env`를 불러온 실행 설정에서 확인한 모델은 다음과 같습니다. 로컬 환경에 따라 `.env` 설정은 달라질 수 있으며, 비밀값은 이 문서에 기록하지 않습니다.

| 모델 | 프로젝트 내 역할 |
|---|---|
| `dragonkue/snowflake-arctic-embed-l-v2.0-ko` | 한국어 질문/메타데이터 임베딩 및 Chroma 벡터 검색 |
| `BAAI/bge-reranker-v2-m3` | 질문-발언 후보 쌍의 관련성을 다시 점수화해 순위 조정 |
| `dragonkue/bge-reranker-v2-m3-ko` | 한국어 후보의 순위 평가에 추가 사용. 현재 로컬 설정은 BAAI 모델과 1:1 순위 RRF 앙상블 |
| `huihui_ai/exaone3.5-abliterated` | Ollama에서 로컬 실행. 질문 조건 구조화, 근거 요약, 핵심 사실·전망 생성 |

임베딩 모델은 의미상 비슷한 검색 후보를 찾고, 재순위화 모델은 그 후보의 순서를 다시 평가합니다. 현재 로컬 `src/.env`는 두 Cross-Encoder 순위를 1:1 RRF로 합치도록 설정되어 있지만, 회의록 품질이 더 좋아졌는지는 아직 사람 검토 평가 전이라 확인되지 않았습니다. 두 재순위 모델 가중치는 저장소가 아닌 로컬 Hugging Face 캐시에 있어 실행 환경에 미리 준비되어야 하며, 기본 설정은 모델 자동 다운로드를 허용하지 않습니다. LLM은 별도로 학습된 프로젝트 전용 모델이 아니라 설정된 사전학습 모델의 추론입니다. `model/`에는 현재 학습된 가중치 파일이 없습니다.

## 데이터 처리·검색·추론 흐름

1. `scripts/json_to_text/` 스크립트가 회의록 HTML에서 발언자·안건·발언을 추출하고 불필요한 요소, 일부 진행 문구와 깨진 띄어쓰기를 정리합니다. 짧은 발언은 제외하고 같은 화자의 연속 발언을 병합하는 규칙도 포함됩니다.
2. `scripts/json_to_text/5_jsonl_to_org.py`가 메타데이터 JSONL과 발언 TXT를 행 ID 순서로 결합해 `data/org_jsonl/`을 만듭니다.
3. `scripts/make_chroma_db/1_metadata_to_chroma.py`가 메타데이터 JSONL의 `summary`를 임베딩해 ChromaDB에 적재합니다. 이 스크립트의 적재 경로는 CUDA GPU를 요구하며, CPU로 자동 전환하지 않습니다.
4. `scripts/make_chroma_db/2_build_bm25_index.py`가 결합 JSONL의 발언 텍스트로 SQLite FTS5 BM25 인덱스를 만듭니다.
5. 사용자의 질문에서 LLM이 쟁점·의회·날짜 조건을 추출합니다. 동일 문구 결과가 있는지 SQLite 캐시를 먼저 확인하고, 캐시 미스면 Chroma 유사 질문 캐시도 검토합니다.
6. Chroma 벡터 검색과 FTS5 BM25 검색을 수행하고, 순위 기반 RRF로 결과를 합칩니다. 기본은 양쪽 가중치 1.0이며 `DENSE_RRF_WEIGHT`와 `BM25_RRF_WEIGHT`로 상대 비중을 조정할 수 있습니다. Cross-Encoder가 후보를 재평가하고, 선택된 근거의 원문 발언은 TXT에서 읽습니다. `RERANKER_MODELS`에 여러 재순위 모델을 설정하면 점수가 아니라 각 모델의 순위를 다시 가중 결합합니다.
7. LLM이 요약·핵심 사실 및 근거에 연결된 조건부 전망을 생성합니다. 결과에는 출처 ID와 회의 메타데이터가 포함되며, 질문에 특정 의회가 있으면 타 의회 후보도 제한적으로 비교할 수 있습니다.
8. Streamlit 화면에서 근거, 쟁점 묶음, 날짜순 논의 흐름, 의회별 대표 위치·순서를 지도와 표로 표시합니다.

SQLite 캐시는 정확한 동일 질문 결과의 기준 저장소입니다. Chroma의 `analysis_question_cache`는 표현이 다른 유사 질문을 찾는 보조 수단이며, 유사도만으로 결과를 재사용하지 않습니다. 자료 버전·프롬프트 버전은 물론 LLM/임베딩/재순위 모델 ID, 검색·재순위 가중치 및 주요 검색 한도가 캐시 identity에 포함됩니다. 따라서 설정을 바꾸면 이전 결과가 새 설정 결과인 것처럼 재사용되지 않습니다.

## 학습, 평가 및 추론

- 모델 파라미터를 학습(fine-tuning)하는 학습 파이프라인은 현재 코드에 없습니다. 모델 사용은 사전학습된 임베딩·재순위화·로컬 LLM의 추론이며, Chroma/BM25 스크립트는 검색 인덱스를 구축합니다. 인덱스 구축을 모델 학습으로 부르지 않습니다.
- `scripts/json_to_text/2_rules_make.py`는 샘플 HTML 구조를 바탕으로 전처리용 선택 규칙을 생성하는 보조 절차입니다. 설정상 로컬 Ollama API를 호출하는 방식이며, 모델 가중치를 학습하는 과정은 아닙니다.
- `src/tests/`에는 설정, 질문 구조화, 검색 결과 결합, 캐시, 원문 조회, 재순위화, 분석 파이프라인, 지도·UI 동작을 검사하는 단위 테스트가 있습니다.
- 단위 테스트는 고정된 입력·대체 객체·임시 파일을 이용한 코드 동작 확인입니다. 실제 전체 데이터에서의 검색 정확도, 예측 정확도, 지연시간 또는 모델 성능을 입증하지 않습니다.
- 2026-11-03 기준 WSL Python에는 PyDeck 등 앱 의존성이 부족해 지도 렌더링 테스트가 실패하지만, 프로젝트의 Windows `mini-gpu` Python 환경에서는 전체 단위 테스트 89개가 통과했고 `python -m compileall -q src scripts`도 통과했습니다. 앞선 실제 Chroma·BM25·임베딩·단일 Cross-Encoder smoke는 5개 후보의 결과를 확인하는 데 모델 로딩 포함 약 213초가 걸렸고, 후보 300개 전체 재순위 시도는 420초 안에 끝나지 않아 중단했습니다. 별도로 사람 검토 팩 준비를 위해 9개 질문에서 실제 하이브리드 검색 후보를 생성했지만, 이 단계는 재순위 모델의 품질을 비교하지 않았습니다. 이는 전체 앱, 답변 품질 또는 실사용 지연시간의 검증이 아닙니다.
- 사람 검토 평가를 위해 `C:/hermes/mini_solo_reranker_eval/`에 9개 초안 질문과 실제 검색 후보 270건(질문당 상위 30개)을 준비했습니다. 사람이 0/1/2 관련도 라벨을 채우면 BAAI 단독, 한국어 reranker 단독, 1:1 순위 앙상블의 NDCG@10·MRR@10을 같은 후보에서 비교합니다. 아직 라벨과 실제 품질 결과는 없으며, 현재 파일럿은 재순위 단계만 측정하고 전체 검색 재현율이나 답변 품질은 측정하지 않습니다. 절차는 `.skill/evaluation-process.md`와 평가 폴더의 README에 기록했습니다.

분석 결과의 정확성 한계:

- “최초 관측”은 검색·재순위화되어 채택된 후보 중 가장 이른 근거이지, 전체 회의록의 절대 최초를 보증하지 않습니다.
- 의회 간 비슷한 논의의 관측만으로 정책 전파나 인과관계를 주장하지 않습니다. 확산 확률을 계산하지 않습니다.
- 전망 신뢰도는 정성적 표기이며 통계적으로 보정된 확률이 아닙니다.
- 지도 좌표 일부는 의회 청사의 정확한 주소가 아니라 행정 중심점 근사입니다.

## 실행 환경 및 실행 방법

- Python: 프로젝트 의존성을 설치할 수 있는 Python 환경 필요
- 로컬 LLM: Ollama 서버와 `src/.env`에 지정한 모델 필요
- 검색 데이터: 사전 준비된 Chroma 컬렉션, BM25 SQLite 인덱스, 원본 TXT 필요
- 질문 임베딩과 Cross-Encoder 재순위화는 기본 CPU 설정입니다. 모델 가중치가 로컬 캐시에 있어야 하며 기본 설정에서는 자동 다운로드하지 않습니다.
- 지도 배경은 외부 지도 타일 연결이 필요한 환경일 수 있습니다.

프로젝트 루트에서 실행합니다.

```bash
python -m pip install -r requirements.txt
# src/.env.example을 참고하여 src/.env를 준비하고 로컬 Ollama/model 설정
python src/app.py
```

기존 방식으로도 실행할 수 있습니다.

```bash
streamlit run src/app.py
```

앱은 Chroma 임베딩 인덱스와 BM25 인덱스를 자동 생성하지 않습니다. 별도 생성이 필요한 경우 입력 데이터와 CUDA/모델 사전조건을 확인한 뒤 실행합니다.

```bash
python scripts/make_chroma_db/1_metadata_to_chroma.py --dry-run
python scripts/make_chroma_db/1_metadata_to_chroma.py
python scripts/make_chroma_db/2_build_bm25_index.py
```

TXT 메타데이터 SQLite가 없으면 앱이 생성할 수 있습니다. 기존 DB가 유효하지 않은 경우 자동으로 덮어쓰지 않습니다. 강제 재생성은 의도와 데이터 백업을 확인한 뒤에만 사용하세요.

## 테스트

프로젝트 루트에서 단위 테스트를 실행합니다.

```bash
python -m unittest discover -s src/tests -v
python -m compileall -q src
```

이 명령들은 코드 및 테스트 픽스처를 검사하며, 실제 Ollama·Chroma·대규모 모델 가중치·전체 데이터 통합 실행과는 구분됩니다.

## 주요 경로

- Streamlit 진입점: `src/app.py`
- 핵심 설정·모델·캐시·LLM·파이프라인: `src/core/`
- 하이브리드 검색·순위 융합·재순위화: `src/search/`
- 클러스터링·의회 간 비교·타임라인·발췌: `src/analysis/`
- 원문 인덱스 구현: `src/data/source_index.py` (기존 실행 명령 호환용 `src/source_index.py`도 유지)
- 지도·시각화: `src/visualization/`
- 테스트: `src/tests/`
- 전처리·색인 및 평가 스크립트: `scripts/` (`scripts/evaluation/reranker_eval.py`는 사람 라벨 기반 3설정 재순위 비교)
- 상세 앱 안내: `src/README.md`

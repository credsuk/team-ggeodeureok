# 지방의회 회의록 기반 정책 이슈 분석

지방의회 회의록을 자연어로 검색하고, 관련 발언의 근거·쟁점·시간 흐름을 정리해 보여주는 로컬 실행형 Streamlit 분석 앱입니다. ChromaDB 벡터 검색과 SQLite FTS5 키워드 검색을 함께 사용하며, 검색된 결과는 로컬 AI 모델로 요약하고 지도와 표로 시각화합니다.

## 결과물 구성

```text
mini_solo/
├── README.md                 # 프로젝트, 데이터·모델, 추론·평가, 설치·실행 안내
├── requirements.txt          # 앱 및 데이터 처리 스크립트의 Python 의존성
├── data/                     # 회의록 데이터·인덱스 (README만 Git 포함, 데이터 파일은 제외)
│   └── README.md             # 데이터 범위·제출 시 확인할 점
├── model/                    # 검색 앙상블·재순위화 구현과 모델 안내
│   ├── ensemble/             # Dense/BM25 및 Cross-Encoder 순위 결합 코드
│   └── README.md             # 코드·외부 가중치·학습 여부 안내
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

임베딩 모델은 앱의 검색 초기화 단계에서, BAAI와 한국어 Cross-Encoder는 첫 캐시 미스 질문의 재순위화 단계에서 필요해집니다. 각 모델이 Hugging Face 캐시에 없으면 `ALLOW_MODEL_DOWNLOAD=1`(현재 기본값)에서 해당 시점에 자동 다운로드합니다. 이 설정은 BAAI뿐 아니라 임베딩·한국어 재순위 모델 등 Hugging Face 모델 전반에 적용되므로 네트워크와 디스크 공간이 필요할 수 있습니다. 오프라인 실행은 `ALLOW_MODEL_DOWNLOAD=0`으로 설정하세요. 내려받은 가중치는 저장소가 아니라 Hugging Face 캐시에 남습니다. 회의록 품질이 더 좋아졌는지는 아직 사람 검토 평가 전이라 확인되지 않았습니다. LLM은 별도로 학습된 프로젝트 전용 모델이 아니라 설정된 사전학습 모델의 추론입니다. `model/ensemble/`에는 검색·재순위 앙상블 구현이 있고, 프로젝트가 직접 학습한 가중치 파일은 없습니다.

## 데이터 처리·검색·추론 흐름

1. `scripts/json_to_text/` 스크립트가 회의록 HTML에서 발언자·안건·발언을 추출하고 불필요한 요소, 일부 진행 문구와 깨진 띄어쓰기를 정리합니다. 짧은 발언은 제외하고 같은 화자의 연속 발언을 병합하는 규칙도 포함됩니다.
2. `scripts/json_to_text/5_jsonl_to_org.py`가 메타데이터 JSONL과 발언 TXT를 행 ID 순서로 결합해 `data/org_jsonl/`을 만듭니다.
3. `scripts/make_chroma_db/1_metadata_to_chroma.py`가 메타데이터 JSONL의 `summary`를 임베딩해 ChromaDB에 적재합니다. 이 스크립트의 적재 경로는 CUDA GPU를 요구하며, CPU로 자동 전환하지 않습니다.
4. `scripts/make_chroma_db/2_build_bm25_index.py`가 결합 JSONL의 발언 텍스트로 SQLite FTS5 BM25 인덱스를 만듭니다.
5. 사용자의 질문에서 LLM이 쟁점·의회·날짜 조건을 추출합니다. 동일 문구 결과가 있는지 SQLite 캐시를 먼저 확인하고, 캐시 미스면 Chroma 유사 질문 캐시도 검토합니다.
6. Chroma 벡터 검색과 FTS5 BM25 검색을 수행하고, `model/ensemble/rank_fusion.py`의 순위 기반 RRF로 결과를 합칩니다. 기본은 양쪽 가중치 1.0이며 `DENSE_RRF_WEIGHT`와 `BM25_RRF_WEIGHT`로 상대 비중을 조정할 수 있습니다. `model/ensemble/cross_encoder.py`가 후보를 재평가하고, 선택된 근거의 원문 발언은 TXT에서 읽습니다. `RERANKER_MODELS`에 여러 재순위 모델을 설정하면 점수가 아니라 각 모델의 순위를 다시 가중 결합합니다. `src/search/`는 이 기능을 호출하는 검색 흐름과 기존 import 호환 경로를 담당합니다.
7. LLM이 요약·핵심 사실 및 근거에 연결된 조건부 전망을 생성합니다. 결과에는 출처 ID와 회의 메타데이터가 포함되며, 질문에 특정 의회가 있으면 타 의회 후보도 제한적으로 비교할 수 있습니다.
8. Streamlit 화면에서 근거, 쟁점 묶음, 날짜순 논의 흐름, 의회별 대표 위치·순서를 지도와 표로 표시합니다.

SQLite 캐시는 정확한 동일 질문 결과의 기준 저장소입니다. Chroma의 `analysis_question_cache`는 표현이 다른 유사 질문을 찾는 보조 수단이며, 유사도만으로 결과를 재사용하지 않습니다. 자료 버전·프롬프트 버전은 물론 LLM/임베딩/재순위 모델 ID, 검색·재순위 가중치 및 주요 검색 한도가 캐시 identity에 포함됩니다. 따라서 설정을 바꾸면 이전 결과가 새 설정 결과인 것처럼 재사용되지 않습니다.

## 학습·추론·평가 과정

### 학습 여부

이 프로젝트는 임베딩 모델, Cross-Encoder, Ollama LLM의 가중치를 직접 학습하거나 fine-tune하지 않았습니다. `model/ensemble/`에는 사전학습 모델의 추론 결과와 검색 순위를 결합하는 코드가 있고, 프로젝트 자체 학습 체크포인트는 없습니다. Chroma/BM25 색인 구축은 검색용 데이터 준비이며 모델 학습이 아닙니다. `scripts/json_to_text/2_rules_make.py`도 로컬 Ollama로 HTML 추출 규칙을 생성·검토하는 절차이지 가중치를 학습하는 절차가 아닙니다.

### 추론 절차

1. 동일 문구의 저장 결과가 있으면 캐시에서 반환합니다. 캐시가 없으면 Ollama가 질문의 쟁점·의회·날짜 조건을 구조화하고, 정확한 의도 및 조건이 맞는 유사 질문 캐시도 확인합니다.
2. 캐시 미스이면 Chroma 벡터 검색과 SQLite FTS5 BM25 검색을 수행합니다. `model/ensemble/rank_fusion.py`의 가중 RRF로 순위를 합치고, Cross-Encoder가 후보 발언을 다시 순위화합니다. 특정 의회 질문에는 별도 비교 검색을 수행해 타 의회 후보를 최대 8건까지 확인합니다.
3. 선택된 근거의 원문 발언은 요약문이 아니라 `data/to_text/`의 원본 TXT에서 읽습니다. 근거를 쟁점별·날짜순으로 정리한 뒤, Ollama의 첫 호출이 발언 발췌와 메타데이터를 바탕으로 답변 요약·핵심 사실·출처별 요약을 만듭니다.
4. 두 번째 Ollama 호출은 원문 발언 대신 출처별 요약·메타데이터·검색 결과 통계를 받아 조건부 전망과 향후 논의 주제를 작성합니다. 인용 ID는 검색된 근거와 대조하고, 설정·자료·프롬프트 버전이 같은 결과만 캐시에서 재사용합니다.
5. Streamlit 화면은 요약, 근거, 최초 관측, 시간 흐름, 의회 간 비교와 지도를 표시합니다. 캐시 적중 시에는 저장 결과를 반환해 이후 검색·재순위화·생성 단계를 건너뜁니다.

### 평가 방법과 현재 상태

- 코드 동작 검증: Windows Python 3.11.16 환경에서 단위 테스트 94개와 `python -m compileall -q src model scripts`가 통과했습니다. 단위 테스트는 로직·인터페이스를 검증할 뿐 검색 정확도, 전망 정확도 또는 답변 품질을 측정하지 않습니다.
- 재순위 모델 비교: 사람 검토용 9개 초안 질문에서 질문당 30개씩, 총 270개 하이브리드 검색 후보를 준비했습니다. 검토자는 각 후보에 0(무관), 1(관련은 있으나 직접 근거는 아님), 2(직접 관련 근거)로 라벨을 매깁니다. 같은 후보를 BAAI 단독, 한국어 Cross-Encoder 단독, 두 모델의 1:1 순위 앙상블로 비교하고 NDCG@10·MRR@10을 계산하도록 `scripts/evaluation/reranker_eval.py`가 구현되어 있습니다. 양성 라벨이 하나 이상 있는 질문에 대해 평균 지표를 내며, 무근거 대조 질문은 별도로 집계합니다. 결과 파일은 `summary.csv`, `per_query.csv`, `rankings.csv`입니다.
- 현재 제출 상태: 기록상 사람 라벨링과 채점은 보류되어 있으며 실제 비교 점수와 우승 모델은 없습니다. 점수를 임의로 채우거나 검색 순위를 정답 라벨로 간주하지 않습니다.
- 평가 범위의 한계: 후보가 이미 검색된 상위 30개로 제한되므로 이 평가는 재순위 단계만 비교합니다. 전체 corpus 검색 재현율, 답변·전망 품질, 정책 확산 판단의 정확도는 측정하지 않습니다. 전망에 대한 rolling backtest도 아직 완료되지 않았습니다.
- 별도 smoke 결과는 통합 가능성 확인으로만 해석합니다. 합성 한국어 근거 2건으로 두 Cross-Encoder의 오프라인 호환성 smoke를 수행해 유한한 순위 점수를 확인했으며, cold load 포함 약 38.8초였습니다. 읽기 전용 5개 후보 검색·단일 재순위 실행은 모델 로딩 포함 약 213초가 걸렸고, 300개 후보 실행은 420초 안에 끝나지 않아 중단했습니다. 이는 품질 우위나 대표 지연시간의 증거가 아닙니다. 전체 앱을 대상으로 한 신선한 검색부터 화면 표시까지의 end-to-end 검증도 완료되지 않았습니다.

평가 절차와 한계는 `.skill/evaluation-process.md`, 실행 이력은 `.skill/test-history.md`에 기록되어 있습니다. “최초 관측”은 재순위된 후보 중 가장 이른 기록이며 전체 회의록의 절대 최초를 보증하지 않습니다. 의회 간 유사한 논의만으로 정책 전파나 인과관계를 단정하지 않습니다.

## 실행 환경 및 실행 방법

- 운영체제/Python: 확인된 기준 환경은 Windows, Python 3.11.16입니다. 이는 현재 검증에 사용한 기준이지, 프로젝트가 선언한 최소 Python 버전은 아닙니다. 다른 OS/Python 조합은 전체적으로 검증되지 않았습니다.
- Python 라이브러리 버전 범위는 루트 `requirements.txt`가 기준입니다. 이 파일은 재현 가능한 고정 버전(lock)이 아니라 허용 범위입니다.

| 기능 | 패키지 | requirements.txt 버전 범위 |
|---|---|---|
| 앱 화면 | `streamlit` | `>=1.66,<2` |
| 지도 | `pydeck` | `>=0.8.0b4,<1` |
| 벡터 검색 | `chromadb` | `>=1.0,<2` |
| 임베딩·재순위화 | `sentence-transformers` | `>=4.0,<6` |
| 임베딩·재순위화 백엔드 | `torch` | `>=2.14,<3` |
| TorchVision 연동 | `torchvision` | `>=0.29,<0.30` |
| 데이터 처리 | `pandas` | `>=2.2` |
| HTML 전처리 스크립트 | `beautifulsoup4` | `>=4.12,<5` |
| Ollama 호환 전처리 호출 | `openai` | `>=1.0,<4` |

현재 확인한 Windows 기준 환경의 설치 버전은 Python 3.11.16, Streamlit 1.66.1, PyDeck 0.9.3, ChromaDB 1.5.9, sentence-transformers 5.7.0, PyTorch 2.14.1+cu130, TorchVision 0.29.2, pandas 3.1.0, OpenAI 3.37.0입니다. 이는 해당 PC의 버전 스냅샷이지, 모든 버전 조합의 호환성 보증은 아닙니다. `beautifulsoup4`는 확인 시 설치되어 있지 않았으므로 HTML 전처리 스크립트를 실행하기 전에 아래 설치 명령으로 의존성을 준비해야 합니다.

- 앱의 질문 임베딩과 Cross-Encoder 재순위화는 기본 CPU 설정으로 동작합니다. Chroma 임베딩 색인을 새로 구축할 때만 `scripts/make_chroma_db/1_metadata_to_chroma.py`가 NVIDIA GPU와 CUDA 지원 PyTorch를 요구하며, GPU를 사용할 수 없으면 중단합니다. 필요한 최소 VRAM은 측정·명시되지 않았습니다. CPU/CUDA용 PyTorch·TorchVision 휠은 서로 맞는 조합이어야 하며, 확인된 설치 예시는 `src/README.md`를 참고하세요.
- BM25 인덱스는 SQLite FTS5를 사용합니다. 실행 Python의 `sqlite3`가 FTS5를 지원해야 하며, 현재 기준 환경에서 확인했습니다.
- Ollama는 pip 패키지가 아닌 별도 로컬 서버입니다. `src/.env`에 지정한 모델과 서버를 준비해야 합니다. `src/.env`가 없으면 `src/.env.example`을 사용합니다. Hugging Face 모델 가중치(임베딩·재순위화)는 캐시에 없을 경우 기본적으로 첫 사용 시 자동 다운로드합니다. 오프라인 환경에서는 `ALLOW_MODEL_DOWNLOAD=0`으로 설정하고 필요한 가중치를 미리 준비해야 합니다.
- 검색 데이터로 준비된 Chroma 컬렉션, BM25 SQLite 인덱스, 원본 TXT가 필요합니다. 이 데이터·색인은 저장소에 포함되지 않을 수 있으며 앱이 Chroma/BM25 인덱스를 자동 생성하지 않습니다.
- 지도 배경은 외부 지도 타일 연결이 필요한 환경일 수 있습니다.

프로젝트 루트에서 실행합니다.

```bash
python -m pip install -r requirements.txt
# src/.env가 없으면 src/.env.example을 자동으로 사용합니다. 개인 설정 변경 시에만 복사해 수정하세요.
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
python -m compileall -q src model scripts
```

이 명령들은 코드 및 테스트 픽스처를 검사하며, 실제 Ollama·Chroma·대규모 모델 가중치·전체 데이터 통합 실행과는 구분됩니다.

## 주요 경로

- Streamlit 진입점: `src/app.py`
- 핵심 설정·모델·캐시·LLM·파이프라인: `src/core/`
- 검색 실행 흐름·후보 수집: `src/search/retrieval.py`
- Dense/BM25 및 Cross-Encoder 앙상블 구현: `model/ensemble/`
- 기존 import 호환 및 인용 검증: `src/search/ranking.py`, `src/search/reranking.py`
- 클러스터링·의회 간 비교·타임라인·발췌: `src/analysis/`
- 원문 인덱스 구현: `src/data/source_index.py` (기존 실행 명령 호환용 `src/source_index.py`도 유지)
- 지도·시각화: `src/visualization/`
- 테스트: `src/tests/`
- 전처리·색인 및 평가 스크립트: `scripts/` (`scripts/evaluation/reranker_eval.py`는 사람 라벨 기반 3설정 재순위 비교)
- 상세 앱 안내: `src/README.md`

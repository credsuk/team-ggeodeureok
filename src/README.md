# 지방의회 회의록 분석 챗봇

## 구성

`src/app.py`가 Streamlit 채팅 화면과 지도를 제공합니다. 기능별 코드는 `src/core/`(설정·모델·캐시·LLM·파이프라인), `src/search/`(검색 실행 흐름·인용 검증·기존 import 호환), `model/ensemble/`(Dense/BM25 RRF와 Cross-Encoder 앙상블 구현), `src/analysis/`(클러스터링·의회 비교·타임라인·발췌), `src/data/`(원문 인덱스), `src/visualization/`(지도)에 나뉘어 있습니다. 기존 명령과의 호환을 위해 `src/source_index.py`는 얇은 실행 래퍼로 남겨 두었고, 실제 구현은 `src/data/source_index.py`에 있습니다. 질문마다 아래 순서로 처리합니다.

1. 질문 전체를 정규화해 원문 대신 해시로 SQLite에 저장한 직접 캐시를 먼저 확인합니다. 같은 표현(공백·대소문자·문장부호 차이 포함)이면 로컬 LLM 호출 전에 결과를 재사용합니다. 자료 버전·프롬프트 버전·모델·전망 기준일이 다르면 재사용하지 않습니다.
2. 직접 캐시가 없으면 Ollama 로컬 모델이 쟁점, 의회, 날짜 조건을 JSON으로 추출합니다. 그 후 SQLite의 정확한 분석 캐시를 확인하고, 없으면 Chroma의 별도 `analysis_question_cache` 컬렉션에서 유사 질문을 찾습니다. 유사도만으로 재사용하지 않고 자료 버전·프롬프트 버전·검색 범위·쟁점도 확인합니다.
3. 메타데이터 Chroma 검색과 기존 SQLite FTS5 BM25 검색 결과를 `source_file:row_id`로 합칩니다.
4. Cross-Encoder가 후보 발언을 다시 순위화하고, `core_issue` 기준으로 묶어 날짜순 흐름을 만듭니다.
5. 선택된 행의 원문 발언만 `data/to_text` 및 하위 폴더의 TXT에서 읽습니다. `source_index.py`가 만드는 SQLite에는 원문 텍스트가 아니라 회의명·날짜·의회·행의 바이트 오프셋만 들어갑니다.
6. Ollama에 두 번 나눠 요청합니다. 첫 요청은 선택된 발언의 최대 1,200자 발췌와 메타데이터로 답변 요약·핵심 사실·출처별 320자 이내 요약을 만듭니다. 요약을 만들지 못한 출처는 원문 발췌로 대신하지 않고 메타데이터만 표시하며, 발언 내용의 인용 후보에서 제외합니다. 질문에 특정 의회가 있으면 별도로 의회 필터를 푼 검색도 한 번 수행하고, Cross-Encoder 상위 32개 중 다른 의회 후보를 최대 8개, 의회 다양성 우선으로 선택합니다. 기존 근거와 합쳐도 요약 입력은 최대 24건입니다. 두 번째 예측 요청에는 원문 발언을 넣지 않고 요약·메타데이터·통계·구조화된 `regional_spread` 판단을 전달합니다. 통계는 주 재순위화 근거와 최대 8개 비교 후보 기준이며 전체 corpus 통계가 아닙니다. 여러 의회에서 관련 안건이 관측되어도 정책 전파를 단정하지 않고, 확산 확률은 산출하지 않습니다. 타 의회 비교 근거가 부족하면 판단 불가로 표시합니다. UI는 해당 의회·날짜·출처를 별도 영역에서 보여줍니다. 전망 인용(`forecast_citations`)은 실제 요약 source ID와 대조하고, 신뢰도는 별도의 정성 평가로 표시합니다.

정책 전파는 날짜가 앞선다는 이유만으로 단정하지 않습니다. UI의 “최초 관측”은 Cross-Encoder가 최종 채택한 후보 중 가장 이른 기록이며, 전체 발언의 절대 최초를 보증하지 않습니다.

## 최초 실행 전

프로젝트 루트 `C:\project\mini_solo`에서 실행합니다. Windows PowerShell 또는 해당 프로젝트의 Python 환경을 사용하세요.

1. 현재 Chroma 임베딩 작업이 끝날 때까지 기다립니다. 앱은 Chroma 전체 색인을 자동 생성하지 않습니다. 검색 인덱스가 부분 상태이면 분석도 부분 결과가 됩니다.
2. `data/source_metadata.sqlite3`가 없으면 앱이 첫 시작 때 `data/to_text` 아래의 모든 `*.txt` 파일(하위 폴더 포함)을 이용해 자동 생성합니다. 기존 DB가 유효하면 다시 만들지 않고 재사용합니다. 파일은 있지만 손상되었거나 스키마가 맞지 않으면 자동 덮어쓰지 않고 오류를 표시합니다. 의도적으로 다시 만들 때만 `python src/source_index.py --force`를 사용하세요. 이 DB에는 원문을 복사하지 않습니다.
3. 기존 org_jsonl에서 FTS5/BM25 검색 인덱스를 만듭니다.

   `python scripts/make_chroma_db/2_build_bm25_index.py`

   BM25 검색에는 검색용 텍스트 인덱스가 필요합니다. 답변에 보여주는 원문 인용은 BM25 복사본이 아니라 원본 TXT에서 다시 읽습니다.

4. `src/.env.example`을 `src/.env`로 복사하고 필요한 설정을 조정합니다. shell 환경변수가 `.env`보다 우선합니다. 비밀값은 이 파일에 넣지 마세요.
5. 앱 의존성을 설치합니다. 검색 질문 임베딩과 Cross-Encoder는 기본 CPU 실행으로 두어 Ollama 로컬 모델과 GPU 메모리를 나눠 쓰지 않도록 했습니다. 여유 VRAM이 충분하고 Chroma 적재가 끝난 경우에만 `EMBEDDING_DEVICE=cuda`로 바꿔 보세요. CUDA를 요청했는데 사용할 수 없으면 CPU로 몰래 바꾸지 않고 중단합니다.

   `python -m pip install -r requirements.txt`

   이 루트 파일에는 앱·지도와 Python 전처리/인덱싱 코드가 직접 사용하는 패키지가 포함됩니다. `src/requirements-app.txt`도 같은 목록을 포함하는 호환용 설치 경로입니다.

   로컬 임베딩/재순위 모델 가중치는 기본적으로 자동 다운로드하지 않습니다(`ALLOW_MODEL_DOWNLOAD=0`). 필요한 모델이 Hugging Face 캐시에 없으면 먼저 모델을 준비하거나, 명시적으로 `ALLOW_MODEL_DOWNLOAD=1`을 설정하세요. Cross-Encoder는 기본 CPU 실행으로 두어 Ollama·임베딩 모델과 GPU 메모리를 과도하게 공유하지 않도록 했습니다.

   `CrossEncoder` 첫 검색에서 `ModuleNotFoundError: No module named 'torchvision'`가 나오면 설치된 PyTorch와 같은 릴리스 계열의 TorchVision을 설치해야 합니다. 현재 확인된 `mini-gpu` 환경(`torch 2.14.1+cu130`, Python 3.11.16)의 대응 버전은 `torchvision 0.29.1`입니다. PyTorch 공식 버전 표의 CUDA 13.0 조합과 동일합니다 (https://pytorch.org/get-started/previous-versions/). PowerShell에서 아래 명령을 사용하세요.

   `& C:\Users\Admin\miniconda3\envs\mini-gpu\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130`

6. Ollama에서 설정한 모델을 사용할 수 있는지 확인한 뒤 앱을 실행합니다. `python src/app.py`로 직접 실행하면 Streamlit 서버를 자동 시작하며, 기존 방식 `streamlit run src/app.py`도 계속 사용할 수 있습니다.

   `python src/app.py`

질문 분석이 진행되는 동안 채팅 입력은 비활성화됩니다. 화면 상단의 `캐시 관리` 버튼은 저장된 답변과 유사 질문 캐시만 비우며, 채팅 기록과 검색 인덱스는 유지합니다.

## 주요 설정

- `CHROMA_DIR`, `CHROMA_COLLECTION`, `EMBEDDING_MODEL`: 기존 메타데이터 Chroma 위치·컬렉션·임베딩 모델
- `BM25_DB`: FTS5 인덱스 위치
- `SOURCE_METADATA_DB`: TXT 행과 회의 메타데이터/바이트 오프셋 인덱스
- `OLLAMA_URL`, `OLLAMA_MODEL`, `OLLAMA_KEEP_ALIVE`: 로컬 Ollama 서버·모델·모델 메모리 유지 시간
- `RERANKER_MODEL`, `RERANKER_DEVICE`: 기본 Cross-Encoder와 실행 장치
- `RERANKER_MODELS`, `RERANKER_WEIGHTS`: 선택적 복수 재순위 모델과 같은 순서의 가중치. 기본은 단일 `RERANKER_MODEL`; 복수 모델은 점수 평균 대신 가중 RRF 사용
- `DENSE_RRF_WEIGHT`, `BM25_RRF_WEIGHT`: Chroma 벡터 검색과 SQLite FTS5 BM25 순위의 상대 가중치. 기본값은 각각 `1.0`
- `DATA_VERSION`, `PROMPT_VERSION`: 데이터/프롬프트 의미가 바뀔 때 올리는 캐시 버전 (`PROMPT_VERSION` 기본값은 `local-rag-v8`)
- `CACHE_SIMILARITY`: Chroma 질문 캐시 유사도 문턱값(확률이 아님)

검색/재순위 모델 ID, 해당 가중치, 검색 한도, LLM·임베딩 모델은 캐시 identity에 들어가므로 설정이 달라지면 이전 결과를 새 결과처럼 재사용하지 않습니다. 가중치를 바꾸기 전에 결과가 확인된 평가 질문 세트에서 기존값과 비교하세요. 원시 회의록이나 검색 인덱스를 다시 만들면 `DATA_VERSION`을 올리고, 프롬프트·응답 규칙을 바꾸면 `PROMPT_VERSION`을 올리세요.

## 결과 화면

향후 전망과 핵심 사실을 상단의 테두리 카드에 먼저 보여주고, 전망 아래에는 근거 요약이 직접 뒷받침하는 ‘향후 논의 가능 주제’를 출처 ID와 함께 표시합니다. 주제를 뒷받침할 요약 근거가 없으면 추측으로 채우지 않고 확인할 수 없다고 표시합니다. 신뢰도는 전망 본문보다 먼저 표시하고, 직접 연결된 근거는 `전망 근거 보기`에서 확인할 수 있습니다. 질문별 결과는 질문 제목을 눌러 펼치며, 결과 카드 안에서도 위치·논의순서·출처·인용 등 기존 카테고리를 각각 접고 펼칠 수 있습니다. 최초 관측에서는 질문과 관련된 발췌를 바로 표시하고, `원문 전체 보기`를 눌러 전체 발언을 펼칩니다. 지도와 의회별 논의 순서, 쟁점별 근거 묶음, 일반 출처, 주의사항은 필요할 때 펼쳐 볼 수 있습니다. 지도에는 의회별 원형 표식 하나만 표시하며 숫자는 의회의 첫 발언 날짜순 연속 번호(1, 2, 3…)입니다. 표에는 모든 발언의 전체 순번과 첫 발언 행의 지도 번호를 구분해 보여주고, 마우스 오버 설명은 없습니다. 화면 제목 옆에는 작은 캐시 초기화 버튼이 있습니다. 우측 상단 메뉴는 숨기되, 사이드바를 접어도 다시 펼칠 수 있는 Streamlit 버튼은 유지합니다.

## 주의사항

- `data/`와 모델 가중치는 로컬 산출물이며 Git에 올리지 않습니다.
- Chroma 적재 중에는 앱을 실행하지 마세요. 질문 캐시가 같은 Chroma DB에 별도 컬렉션을 추가하므로, 적재 완료 뒤 앱을 켜는 것이 안전합니다.
- 지도는 의회별 대표 좌표에 점을 찍으며 일부 좌표는 시·도 중심점 근사입니다. Streamlit의 기본 지도 배경은 외부 지도 타일을 사용하므로 완전 오프라인 환경에서는 배경 지도가 표시되지 않을 수 있습니다. 회의록 본문은 외부 LLM/API로 전송하지 않습니다.
- 예측은 근거가 적어도 출력하되, 근거 부족 시 낮은 신뢰도로 표시합니다. 이는 통계적으로 검증된 확률이 아닙니다.
- 배포/실행 환경에는 프로젝트 루트의 `requirements.txt` 의존성, Ollama 서버/모델, 준비된 Chroma 회의록 컬렉션, BM25 FTS5 인덱스가 필요합니다. Streamlit/Chroma/PyTorch/Ollama 실데이터 통합은 단위 테스트로 대체되지 않습니다.

## 테스트

프로젝트 루트에서:

`python -m unittest discover -s src/tests -v`

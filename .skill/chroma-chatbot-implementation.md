# Chroma 기반 지방의회 챗봇 구현

## 현재 소스 경로

- 화면/실행 진입점: `src/app.py`
- 설정: `src/config.py`, `src/.env.example`
- 질문 구조화 및 로컬 Ollama 통신: `src/intent.py`, `src/llm.py`
- 검색·원문 조회: `src/retrieval.py`, `src/source_index.py`
- 순위화·쟁점 묶기·시간순 정리: `src/ranking.py`, `src/reranking.py`, `src/clustering.py`, `src/timeline.py`
- 결과 캐시/흐름: `src/cache.py`, `src/pipeline.py`, `src/models.py`
- 의회별 지도 좌표: `src/map_view.py`

## 데이터 흐름과 원문 보존

1. 메타데이터 임베딩은 기존 `assembly_metadata` Chroma 컬렉션에서 찾습니다. 컬렉션의 `embedding_model`이 설정과 다르면 실행을 중단합니다.
2. 기존 `data/bm25_index/assembly_bm25.sqlite3` FTS5 인덱스로 어휘 검색을 하고, `source_metadata.sqlite3`와 조인해 의회·날짜 조건 및 안정 ID `source_file:row_id`를 유지합니다.
3. `source_index.py`는 `data/to_text` 아래의 TXT 행(하위 폴더 포함)에 대해 메타데이터와 UTF-8 바이트 오프셋을 기록합니다. ID는 기존 형식 `source_file:row_id`를 유지하고, `SourceTextReader`는 먼저 바로 아래 경로를 확인한 뒤 필요하면 하위 폴더에서 같은 stem의 원문을 찾습니다. 원문 발언은 SQLite에 복사하지 않습니다. Cross-Encoder 후보에 대해서만 오프셋으로 원본 TXT 행을 읽습니다.
4. BM25는 검색을 위해 기존 FTS5 테이블의 텍스트 복사본을 사용하지만, 화면에 보이는 근거 발언은 원본 TXT에서 읽은 텍스트입니다.
5. SQLite가 전체 답변·근거 결과의 원본 캐시입니다. 앱은 먼저 같은 질문 문구를 Unicode/공백/문장부호 기준으로 정규화한 뒤 해시만 SQLite에 저장해 일치 여부를 확인합니다. 원문 질문을 별도 저장하지 않으며, 답변 재사용은 자료 버전·프롬프트 버전·모델·전망 기준일이 모두 일치해야 합니다. 신규 분석은 이 직접 캐시와 일치하는 질문 해시를 함께 저장합니다. 이후 캐시 매핑이 없던 레거시 intent/semantic hit도 원 질문과 intent를 매핑해 다음 재질문부터 LLM intent 추출 전에 바로 hit합니다. 캐시 초기화는 결과와 질문 해시 매핑 모두를 삭제합니다. `analysis_question_cache` Chroma 컬렉션은 표현이 다른 유사 질문 후보를 찾는 용도이고, 유사도만으로 재사용하지 않으며 자료/프롬프트/모델/검색 범위/쟁점이 다르면 재사용하지 않습니다.
6. 날짜 범위는 Chroma 메타데이터 `date`가 문자열로 저장된 기존 인덱스와 호환되도록 BM25의 SQLite 메타데이터에서 적용합니다. Chroma에는 의회명 등 equality 필터만 전달합니다.
7. `AnalysisWriter`가 구조화한 `core_facts`를 `AnalysisPipeline`이 `AnalysisResult`로 전달하고, `to_dict`/`from_dict`가 SQLite 캐시 왕복에도 보존합니다. 구형 캐시 JSON은 필드 없이도 읽을 수 있습니다. 전망은 `forecast_citations`에 직접 뒷받침하는 source ID를 별도로 보존하고, 이 필드도 캐시 직렬화에 포함합니다. 요약/예측 분리와 타 의회 비교 변경 후 기본값 및 `src/.env.example`, 현재 `src/.env`를 당시 `local-rag-v7`로 갱신했으며, 주제별 근거 ID를 출력하는 형식 변경 후 `local-rag-v8`로 올렸습니다.
8. `AnalysisWriter`는 두 Ollama 호출로 나눕니다. 요약 호출에는 선택 발언의 최대 1,200자 발췌와 메타데이터를 보내고, 답변 요약·핵심 사실·출처별 최대 320자 요약을 생성합니다. 요약 생성에 실패한 출처는 원문 발췌 대신 `metadata_only` 메타데이터를 사용하며 발언 내용 인용 후보에서 제외합니다. 질문에 특정 의회가 있으면 pipeline은 해당 의회 필터를 푼 별도 검색을 실행하고, Cross-Encoder 상위 32개 중 다른 의회 후보를 최대 8개, 의회 다양성 우선으로 선택합니다. 이를 주 근거와 합쳐도 요약 단계의 입력은 최대 24건입니다. 예측 호출에는 원문이나 `excerpt`를 넣지 않고 요약·메타데이터·통계·구조화된 `regional_spread` 판단을 전달합니다. 통계와 확산 판단은 주 재순위화 근거 및 최대 8개 비교 후보의 범위이며 전체 corpus 통계가 아닙니다. 다른 의회에서 관련 논의가 관측돼도 정책 전파/인과관계를 주장하지 않으며, 자료가 부족하면 판단 불가라고 표시하고 확산 확률을 만들지 않습니다. 화면에는 기준 의회와 비교 의회를 구분해 의회·날짜·출처 발언을 보여줍니다. `forecast_citations`와 확산 source ID는 retrieved evidence와 대조합니다. `likely_discussion_topics`에는 근거 요약으로 뒷받침되고 검증된 source ID가 붙은 최대 3개 추가 논의 주제만 저장하며, 빈 근거를 추측으로 보완하지 않습니다. 신뢰도는 정성적 범위 제한이지 백테스트로 보정된 확률이 아닙니다. UI에서는 `신뢰도: 낮음/보통/높음` 한 줄을 전망 본문보다 먼저 표시하고, forecast prompt는 전망을 1~3문장으로 작성하며 반복적인 불확실성 경고를 생략하도록 합니다. 요약/예측 및 cross-council search, 간결한 전망 문구 변경은 캐시를 분리하도록 `PROMPT_VERSION=local-rag-v8`로 주제별 근거 ID를 포함하는 새 전망 응답을 구형 캐시와 분리합니다. 의회 지도는 의회마다 첫 발언 하나를 표시합니다. 숫자는 각 의회의 첫 발언 날짜 순위라서 지도에는 1, 2, 3…으로 이어집니다. 아래 표의 ‘발언 순서’는 모든 발언의 전체 시간순 번호이고, ‘지도 번호’는 해당 의회의 첫 발언 행에만 연결됩니다. Hover는 비활성화합니다. 사이드바의 ‘처음 실행 전’ 안내는 제거했고 프로젝트 `.streamlit/config.toml`의 `client.toolbarMode="minimal"`과 빈 `menu_items`로 우측 메뉴만 숨깁니다. Streamlit의 사이드바 펼치기 버튼도 toolbar 영역에 있으므로 `stToolbar` 전체를 CSS로 숨기지 않습니다.

## 실행 전제

1. Chroma 임베딩 작업이 완료된 상태에서 시작합니다. 앱은 Chroma 전체 색인이나 BM25 색인을 자동으로 만들지 않습니다.
2. 프로젝트 루트에서 BM25 색인이 아직 없을 때 `python scripts/make_chroma_db/2_build_bm25_index.py`를 실행합니다.
3. `data/source_metadata.sqlite3`가 없는 경우에만 앱이 `data/to_text`를 사용해 자동 생성합니다. 기존 DB는 필요한 테이블/열과 레코드를 검사해 재사용하며, 손상되거나 스키마가 맞지 않는 기존 파일은 자동 덮어쓰지 않고 중단합니다. CLI도 기본적으로 같은 방식으로 건너뛰며, 의도적 재생성은 `python src/source_index.py --force`로만 요청합니다.
4. 앱 의존성과 필요한 로컬 모델 가중치를 준비하고 `python src/app.py`를 실행합니다. 이 명령은 Streamlit 1.66+의 `st.App(...).run()`으로 서버를 자동 시작합니다. 기존 `streamlit run src/app.py` 방식도 사용할 수 있습니다. 생성 중에는 채팅 입력을 잠그며, 제목 옆 작은 캐시 초기화 버튼은 답변 SQLite 캐시·정규화 질문 해시 매핑·유사 질문 Chroma 캐시만 삭제합니다. 질문별 기록은 쟁점/질문 제목의 접이식 패널로 묶고, 열었을 때 결과 안의 기존 카테고리도 각각 접고 펼칠 수 있도록 중첩 expander로 표시합니다. `torchvision` 누락 시에는 설치한 PyTorch/CUDA와 맞는 TorchVision을 설치해야 합니다.

앱은 Chroma와 BM25의 대량 색인을 자동 생성하지 않습니다. 단, 원본 TXT 메타데이터 sidecar가 없을 때만 첫 앱 시작 시 한 번 생성합니다. `ALLOW_MODEL_DOWNLOAD=0`이 기본값이므로 허깅페이스 모델은 캐시된 가중치만 사용합니다. 질문 임베딩과 재순위 모델은 기본 CPU 실행입니다. 이유는 프로젝트의 로컬 Ollama 모델과 GPU VRAM을 경쟁하지 않게 하기 위해서입니다. `EMBEDDING_DEVICE=cuda`는 사용자가 설정한 경우에만 사용하며, CUDA가 없으면 명시적으로 실패합니다.

BM25 빌더는 임시 SQLite 파일을 만든 뒤 최종 경로로 원자 교체합니다. SQLite 연결은 교체 전에 명시적으로 닫습니다. `AnalysisCache`도 전용 `_connect()` context manager에서 트랜잭션 종료 후 연결을 닫아 Windows 임시 DB 정리 시 WinError 32를 막습니다. 교체에서 Windows WinError 32가 나면 stderr에 실패 단계, 임시/최종 경로, PID, 파일 존재/크기와 Windows Restart Manager가 찾은 사용 프로세스를 기록합니다. 프로세스 조회는 best-effort이며, WSL/Linux에서는 Windows 잠금 소유자를 조회할 수 없습니다. Source indexer도 SQLite 연결을 명시적으로 닫은 다음 임시 파일을 교체하거나 정리합니다. 실패 시 원래 예외와 실패 단계/임시 경로를 stderr에 남기고, 정리 자체가 실패해도 원래 예외를 보존합니다.

## 정확성 경계

- `first_seen`은 Cross-Encoder가 채택한 검색 후보 중 가장 이른 기록입니다. 전체 corpus의 절대 최초를 보장하지 않습니다.
- 의회 간 흐름은 시점 순서와 함께 제시하되, 회의록에 명시적 참조가 없으면 정책 전파/인과관계로 단정하지 않습니다.
- `core_issue`가 같은 메타데이터 행을 우선 묶는 결정적 그룹화입니다. 아직 별도 임베딩 군집 모델을 평가한 것은 아닙니다.
- 전망의 신뢰도는 정성적 설명이며 통계적 확률이나 검증된 백테스트 점수가 아닙니다.
- 지도의 일부 점은 정확한 의회 건물 좌표가 아니라 행정 중심 좌표 근사입니다. Streamlit의 기본 배경 타일은 외부 지도 제공자에서 오므로 회의록 본문과 별개로 지도 렌더링 시 네트워크가 필요할 수 있습니다.

## 주요 검증 명령

프로젝트 루트에서 `python -m unittest discover -s src/tests -v`를 실행합니다. 테스트는 SQLite FTS5 및 임시 TXT 파일을 실제로 사용하지만, Chroma·Streamlit·PyTorch·Ollama의 실데이터 통합 실행을 대신하지 않습니다.

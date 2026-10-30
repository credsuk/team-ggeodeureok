# FastAPI 백엔드

`src/main.py`가 실행 런처이고, `src/api/main.py`는 FastAPI 앱을 정의합니다. 프로젝트 루트에서 다음 명령으로 시작합니다.

```bash
python src/main.py
```

After startup completes, open the local dashboard at `http://127.0.0.1:8000/`. The page and `/api/v1/...` requests are served from the same FastAPI origin; no separate npm/Vite server or CORS setup is needed. The frontend files are in `src/api/frontend/` (`index.html`, `app.js`, `api-client.js`). Startup loads the issue catalog but does not automatically analyze the first issue or run its database search. The top search and chatbot input submit on Enter; every non-empty top search switches to the radar tab before requesting results and remains there on errors. The top search button is anchored with local CSS so it stays inside the input even if the utility stylesheet does not load. Font-size controls mark the selected size visibly and expose it with `aria-pressed`. Map-point selection is independent of playback; it scrolls to the matching council card and loads/opens that council's matching full evidence. Monthly document counts use a horizontally scrollable SVG line chart with always-visible month and count labels. The chat send button keeps its compact fixed dimensions while busy; a small CSS spinner appears without replacing the send icon or wrapping a long loading label. Radar analysis shows a screen-level spinner and disables its action button without changing its label; the loading overlay stacks above the Leaflet map while the search header remains above the overlay. The dashboard uses the API for the fixed issue list/detail, free-form radar search, filtered source search, evidence/document detail, and local Ollama chat. The observation map keeps its Leaflet markers and paths and uses OpenFreeMap's Positron style through MapLibre. Tailwind, icons, fonts, map libraries, and map tiles are loaded from external CDNs/services, so those assets require internet access; the API itself remains local.

## 최근 UI 동작

- 챗봇은 답변 생성 중 입력창을 잠그고, 응답 성공이나 오류 후 잠금을 해제합니다.
- 월별 매칭 문서 그래프의 세로축 상한은 관측 최대 문서 수에 50건을 더한 값입니다.
- 상위 근거의 상세 보기와 같은 회의록 청크 목록은 폭·높이를 제한하고 긴 텍스트를 줄바꿈합니다. 청크 요청 중에는 안내 문장 대신 접근성 레이블이 있는 움직이는 로딩 바가 나타납니다.
- 타임라인 재생은 관측 의회 카드 선택·근거 열기와 분리되어 있습니다. 카드와 지도 지점 선택은 재생 위치를 바꾸지 않고, 카드에는 날짜순 번호 배지와 순서별 색상을 표시합니다.
- 글자 크기 컨트롤은 헤더에서 줄바꿈되지 않도록 고정하며, 화면 너비가 1023px 이하이면 `A−`, `A`, `A+` 기호로 바뀝니다. 각 버튼의 접근성 이름과 툴팁은 유지됩니다.

런처는 `src/.env`를 읽고(이미 셸에 있는 변수는 덮어쓰지 않음), 기본 설정에서는 로컬 Ollama를 준비한 뒤 PostgreSQL 연결·테이블 스키마·질의 임베딩 차원을 확인합니다. 확인에 실패하면 오류와 traceback을 콘솔에 출력하고 API 서버를 시작하지 않습니다. 대표 이슈 결과 JSON, 웹 번들, 로컬 parents.parquet, 임베딩 `.npy`, SQLite BM25 인덱스는 이 API 시작 요건이 아닙니다. PostgreSQL은 자동으로 시작하지 않고 모델도 자동 다운로드하지 않습니다.

## 검색 방식과 예측 범위

- Dense 검색은 PostgreSQL `pgvector`의 `embedding`을 사용합니다.
- 키워드 검색은 같은 테이블의 `document`에 PostgreSQL `to_tsvector`/`ts_rank_cd`를 적용합니다. 두 검색 순위를 RRF로 합칩니다.
- 이 구성은 PostgreSQL 내 하이브리드 검색이지만, 현재는 엄밀한 BM25가 아닙니다. 테이블 생성 코드에는 BM25 점수/인덱스 컬럼이 없으므로, 응답의 텍스트 검색 부분은 PostgreSQL 기본 전문검색 점수입니다.
- 월별 예측은 PostgreSQL 전문검색에 매칭된 원문에서 `document_id`의 월별 개수를 세고, 최근 3개월 평균을 다음 달 기준선으로 반환합니다. 결과 `basis`는 `postgresql_full_text_matches`, `count_unit`은 `distinct_document_id`입니다. 이는 가벼운 기준선이지 기존 학습 모델의 예측이나 확률이 아닙니다. 응답의 `forecast.method`에 방식을 표시합니다.
- 이슈 키·이름·검색어는 `config/representative_issue_keys_v1.txt`와 `config/issue_seed_catalog_v1.csv`의 작은 설정 목록에서 읽습니다. 분석 결과는 매 요청 시 PostgreSQL에서 계산합니다.

## 엔드포인트

- `GET /api/v1/issues`: 설정된 고정 키워드 10개 목록
- `GET /api/v1/issues/{issue_key}`: 고정 이슈의 PostgreSQL 하이브리드 검색 근거, 월별 집계, 다음 달 기준선
- `POST /api/v1/radar/search` — `{"query":"저출생 정책"}`: 고정 이슈 매칭이면 해당 DB 질의를 사용하고, 아니면 입력 문장을 그대로 동적 검색합니다. `match_type`으로 경로를 표시합니다.
- `GET /api/v1/documents/search?q=...&council=...&year=...&committee=...&date_from=YYYY-MM-DD&date_to=YYYY-MM-DD&top_k=10`: 필터를 적용한 하이브리드 검색
- `GET /api/v1/documents/{document_id}?limit=100`: DB에서 해당 원문의 청크를 가져옴
- `GET /api/v1/evidence/{chunk_id}`: DB에서 특정 검색 청크를 조회
- `POST /api/v1/chat` — `{"message":"...","history":[],"top_k":6}`: PostgreSQL에서 근거를 검색한 뒤 localhost Ollama로 한국어 답변을 생성하고 근거 목록을 반환
- `GET /api/v1/chat/status`: Ollama 연결과 모델 상태 확인

대화 이력은 요청에 포함된 `history`로만 전달되며 서버 메모리에 보관하지 않습니다. 회의록 본문은 외부 LLM API로 보내지 않습니다.

## 필수 환경 설정

`src/.env.example`을 참고하되 비밀 값은 채팅이나 로그에 붙이지 마세요.

- `PGVECTOR_DSN`: PostgreSQL 연결 문자열
- `COUNCIL_PG_TABLE`: 기본 `council_arctic_raw_v1`
- `COUNCIL_PG_EF_SEARCH`: HNSW 검색 깊이, 기본 `200` (후보 개수 이상으로 적용)
- `COUNCIL_MODEL_CACHE`: 로컬 SentenceTransformers 모델 캐시 폴더
- `COUNCIL_ALLOW_DOWNLOAD`: 기본 `0`은 로컬 캐시만 사용
- `COUNCIL_DEVICE`: `auto`, `cpu`, 또는 `cuda`
- `OLLAMA_URL`, `COUNCIL_AGENT_MODEL`: 기본 `http://127.0.0.1:11434`, `qwen3:8b`
- `HOST`, `PORT`: 기본 `127.0.0.1`, `8000`

PostgreSQL 테이블은 프로젝트의 적재 스크립트가 정의한 컬럼을 가져야 합니다: `chunk_id`, `parent_chunk_id`, `document_id`, `council`, `meeting_date`, `year`, `committee`, `meeting_name`, `agenda_title`, `priority`, `document`, `metadata`, `embedding`. `requirements-app.txt`에는 API에 필요한 FastAPI/Uvicorn 및 벡터 검색 의존성이 포함됩니다.

성공적으로 바인딩되면 Uvicorn이 서버 주소를 출력합니다. 문서는 `http://127.0.0.1:8000/docs`에서 볼 수 있습니다.

## 검증

테스트 실행 위치는 `src/`입니다.

```bash
python -m unittest discover -s api/tests -v
```

현재 자동 검증은 PostgreSQL 연결을 모의한 테스트입니다. 실제 데이터베이스·모델·Ollama 연결은 실행 환경의 `.env`와 서비스가 있어야 별도로 확인할 수 있습니다.

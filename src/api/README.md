# FastAPI API

FastAPI HTTP layer over the existing council issue radar code. It lives beside `agent/`, `demo/`, and `src/` and does not modify the Streamlit demo or data preparation scripts.

## 실행

프로젝트 루트에서 `python src/main.py`만 실행하면 런처가 `src/`로 이동하고, `src/.env`를 읽은 뒤, 필요하면 로컬 Ollama 서버를 시작하고, 입력 결과 파일이 준비돼 있으면 대표 이슈 웹 번들을 생성한 다음 FastAPI를 실행한다. PostgreSQL/pgvector는 자동 시작하지 않는다. Ollama 모델 자체도 자동 다운로드하지 않는다.

```bash
python src/main.py
```

`src/` 폴더 안에서 실행한다면 다음과 같다.

```bash
python main.py
```

Interactive docs: `http://127.0.0.1:8000/docs`. FastAPI and Uvicorn are included by `requirements-app.txt`. Existing shell environment variables take precedence over values in `.env`.

## Endpoints

- `GET /api/v1/issues`: safe, compact list of curated issue keys and summary fields.
- `GET /api/v1/issues/{issue_key}`: full precomputed issue analysis, evidence, forecast and any precomputed `external_followup`.
- `POST /api/v1/radar/search` with `{"query":"저출생"}`: reuses a curated issue payload when the existing string matcher finds a match (score threshold 0.5); otherwise runs `src/33_analyze_free_query.py`. `match_type`, `matched_issue_id`, and `routing_score` identify the path. Dynamic results are provisional. The external follow-up result is included when the existing follow-up tools can calculate it; otherwise `external_followup_status` explains availability.
- `GET /api/v1/documents/search?q=...&council=...&year=...&committee=...&date_from=YYYY-MM-DD&date_to=YYYY-MM-DD&top_k=10`: existing dense/hybrid search and parent context.
- `GET /api/v1/documents/{document_id}?limit=100`: up to 500 matching stored chunks for a document, not a rendered HTML transcript.
- `GET /api/v1/evidence/{chunk_id}`: exact chunk and parent context.
- `POST /api/v1/chat` with `{"message":"...","conversation_id":null,"history":[],"top_k":6}`: existing local Agent response, citations, execution plan and policy fields. The client sends prior turns in `history`; the API does not keep server-side conversation state. It returns a conversation ID for client-side continuity.
- `GET /api/v1/chat/status`: checks local Ollama and the configured model once when the client is about to use chat. No periodic polling is performed.

## Operational notes

- `main.py` prints startup failures and their tracebacks/logs to the console, then exits without binding the API port. It also checks the issue bundle, initializes the search assets/model, and verifies the configured pgvector table before Uvicorn starts; DB errors are therefore caught before the server is exposed.
- Uvicorn's `Uvicorn running on http://...` console line is emitted only after the server successfully binds. Ollama startup output is written to `logs/ollama.log`; if Ollama does not become reachable, its last log lines are also printed and API startup stops.
- Search and chat use the same local assets/configuration as the demo. The required embedding files, BM25 index (in hybrid mode), parent metadata, PostgreSQL table, and representative issue bundle must exist before starting.
- Free-query analysis can take a long time; `COUNCIL_DYNAMIC_TIMEOUT_SECONDS` controls its timeout (default 1800 seconds). Existing result files are reused.
- The chat status endpoint reports Ollama model availability; the API launcher starts the Ollama server but does not automatically download a missing model.
- External follow-up is supplementary/reference data and may be unavailable without credentials or source data. Meeting text is not sent to external services.
- `date_from` and `date_to` use ISO `YYYY-MM-DD` format. Search `top_k` is limited to 1–50; document chunk retrieval is limited to 1–500.
- No summary-comparison, standalone regional-statistics, review-queue, or service-health endpoints are included, per design.

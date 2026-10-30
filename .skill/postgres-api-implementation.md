# PostgreSQL-backed FastAPI implementation

## Current design

The launcher is `src/main.py`; the ASGI app is `src/api/main.py`. Startup loads `src/.env`, prepares local Ollama when enabled, then checks `PGVECTOR_DSN`, verifies the configured table and its vector dimension, and loads a local query encoder to compare its output dimension with PostgreSQL. Failure prints an error and traceback and exits before Uvicorn is invoked. It no longer requires representative-issue result JSON, a web bundle, `parents.parquet`, local embedding arrays, or a SQLite BM25 file.

Search reads the `council_arctic_raw_v1`-style table written by `src/src/16_load_pgvector.py`. Dense retrieval uses `embedding <=> query_vector`; the transaction-local HNSW `ef_search` value comes from `COUNCIL_PG_EF_SEARCH` and is raised to at least the candidate count. Keyword retrieval uses PostgreSQL `to_tsvector` and `ts_rank_cd`; reciprocal-rank fusion combines the two rankings. The table creation code does not define a BM25 index or score column, so the lexical path is PostgreSQL full-text ranking, not BM25. It computes `to_tsvector` from `document` during queries; a GIN index or a BM25 extension may be needed for scale, but the API does not modify the user's database schema.

The fixed-keyword list is read from `src/config/representative_issue_keys_v1.txt` and `src/config/issue_seed_catalog_v1.csv`. Issue details, dynamic radar queries, evidence, and original-document chunks are read from PostgreSQL. Monthly counts use distinct `document_id`; next-month output is a labeled trailing-three-month-mean baseline, not the prior trained forecasting model and not a probability.

Chat retrieves PostgreSQL evidence and sends that context only to the configured localhost Ollama service. It does not send meeting text to an external LLM service. Conversation history is client-supplied and is not stored server-side.

## Local dashboard

FastAPI serves the browser UI from `src/api/frontend/` at `/` and serves the API from `/api/v1/` on the same origin. `index.html` loads `api-client.js` and `app.js`; the client uses relative `/api/v1` URLs, so no CORS or separate frontend server is required. The interface reads its issue list and analysis, radar searches, document searches, evidence/document details, and chat replies from the API. The UI uses actual API results and labels the forecast only as the existing three-month mean baseline; the old downloaded demo contained hard-coded sample issue/forecast values, which are not used for live output.

The frontend is plain HTML/JavaScript (no npm build step). Tailwind, icons, fonts, Leaflet, and CARTO tiles come from external CDNs; the page layout and local API remain available without a database-backed frontend build, but map tiles and CDN styling require internet access.

## Routes

- `GET /api/v1/issues`
- `GET /api/v1/issues/{issue_key}`
- `POST /api/v1/radar/search`
- `GET /api/v1/documents/search`
- `GET /api/v1/documents/{document_id}`
- `GET /api/v1/evidence/{chunk_id}`
- `POST /api/v1/chat`
- `GET /api/v1/chat/status`

Detailed run instructions and request behavior are in `src/api/README.md`.

## Operational limits

- A real PostgreSQL connection and Windows/Conda runtime were not exercised in this WSL session. The user's actual `.env` values were not read out or disclosed.
- PostgreSQL native full-text ranking is not a BM25 implementation. Verify the desired lexical method and indexes against the user's database before claiming BM25 parity.
- The forecast is an explicitly simple baseline and needs a time-based holdout evaluation before being used as a decision signal.

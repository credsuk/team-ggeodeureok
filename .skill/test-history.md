# Test history

## 2026-10-29 — PostgreSQL-backed API changes

Command:

```text
cd /mnt/c/project/mini/src
python -m unittest discover -s api/tests -v
```

Result: 29 tests passed after the final implementation edits. Coverage includes RRF ranking, full-text query formatting, HNSW search-depth configuration, three-month baseline arithmetic, service routing and propagation dates against an injected fake PostgreSQL repository, API contracts, and launcher fail-fast/success-call behavior.

## 2026-10-30 — Local frontend/API integration

- `node --check src/api/frontend/app.js` / `api-client.js` and `node --test src/api/tests/frontend_client.test.cjs src/api/tests/frontend_ui.test.cjs`: 7 tests passed; the suite covers same-origin API calls, Enter submission in both inputs, and screen/button loading indicators.
- `python -m unittest api.tests.test_frontend_serving -v`: 3 tests passed (dashboard HTML, `app.js`, and `api-client.js` are served by FastAPI); full `python -m unittest discover -s api/tests -v`: 32 tests passed.
- Temporary Uvicorn HTTP smoke after the UI update: `/` and `/app.js` returned 200 and included the search/loader hooks; the temporary server was stopped.
- Visual browser smoke could not run because the installed Chromium cannot start (`libnspr4.so` is missing). These checks do not exercise database search, model embeddings, or Ollama chat.

## 2026-10-30 — OpenFreeMap map and startup/loading behavior

- Replaced the CARTO raster basemap with OpenFreeMap Positron rendered by MapLibre GL through the Leaflet adapter; existing Leaflet observation markers and route layers are retained.
- Startup loads the issue catalog without automatically requesting the first issue's analysis. The screen-loading overlay uses z-index 1200 above Leaflet panes/controls; the sticky search header is at z-index 1300. Radar analysis buttons disable without replacing their labels. Local CSS anchors the top search button inside the input, and every non-empty top search switches to the radar tab before the request.
- Font-size controls now update the selected option state (`aria-pressed` plus visible active styling). Map-marker clicks no longer change the playback slider/player; they select and scroll to the matching council card, then fetch and expand its matching full evidence when a chunk ID exists. Monthly match counts render as a responsive horizontal SVG line graph with visible count and month labels. The chat composer uses fixed input/send-button sizing and keeps the send icon while a CSS spinner indicates a pending request.
- `node --test src/api/tests/frontend_client.test.cjs src/api/tests/frontend_ui.test.cjs`: 16 tests passed, covering API calls, Enter handlers, font-size selection, search-button layout/navigation, loader cleanup/stacking, stable analysis/chat button layouts, map point/player independence and evidence expansion, line-chart labels, startup behavior, and map layer/style wiring.
- Full `python -m unittest discover -s api/tests -v`: 32 tests passed. Temporary Uvicorn smoke returned 200 for `/` and `/app.js`; external style and pinned module assets also returned 200.
- Visual browser inspection was blocked because Chromium cannot start in this environment (`libnspr4.so` missing); no pixel-level rendering claim is made.

Additional checks: `python -m py_compile` on the launcher, API service/backend, and routers passed; the fixed-keyword configuration loaded as 10 entries; path-limited `git diff --check` passed for the API files. The WSL interpreter lacks `psycopg`, `pgvector`, `sentence_transformers`, and `torch`, so no live PostgreSQL/model/Ollama integration was run here. These tests do not establish database compatibility, production performance, Ollama answer quality, or forecast accuracy.

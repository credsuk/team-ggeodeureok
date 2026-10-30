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

## 2026-10-30 — UI playback, loading, chart, and responsive controls

- Separated the playback timeline from council-card selection and evidence expansion: advancing or scrubbing the map timeline no longer highlights/opens the matching council card, and selecting a council card or map point does not move playback.
- Added visible ordinal badges with distinct sequential hues to council first-observation cards. Font-size controls now stay on one row with fixed button dimensions; at viewport widths up to 1023px, labels compact to `A−`, `A`, and `A+` while accessible names and titles remain.
- Chat input is disabled through answer generation and restored on both success and error. The monthly chart axis maximum is exactly the observed maximum plus 50.
- Same-meeting chunk loading uses an accessible animated bar instead of visible loading copy. Evidence and document expansion panels now constrain height/width and wrap long text to avoid breaking the layout.
- `node --test src/api/tests/*.test.cjs`: 21 tests passed, including exact chart maximum, pending/success/error chat states, timeline/list independence, order indicators, responsive control hooks, loading bar, and long-text panel layout.
- From `src/`, `python -m unittest discover -s api/tests -v`: 32 tests passed. Temporary Uvicorn HTTP smoke returned 200 for `/`, `/app.js`, and `/api-client.js`; the server was stopped after the check.
- Pixel-level browser inspection remains blocked: Chromium exits because `libnspr4.so` is missing. Tests verify DOM behavior and CSS hooks, but the changed appearance was not visually rendered here. No live PostgreSQL/model/Ollama query was run.

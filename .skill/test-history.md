# Test history

## 2026-10-29 — PostgreSQL-backed API changes

Command:

```text
cd /mnt/c/project/mini/src
python -m unittest discover -s api/tests -v
```

Result: 29 tests passed after the final implementation edits. Coverage includes RRF ranking, full-text query formatting, HNSW search-depth configuration, three-month baseline arithmetic, service routing and propagation dates against an injected fake PostgreSQL repository, API contracts, and launcher fail-fast/success-call behavior.

Additional checks: `python -m py_compile` on the launcher, API service/backend, and routers passed; the fixed-keyword configuration loaded as 10 entries; path-limited `git diff --check` passed for the API files. The WSL interpreter lacks `psycopg`, `pgvector`, `sentence_transformers`, and `torch`, so no live PostgreSQL/model/Ollama integration was run here. These tests do not establish database compatibility, production performance, Ollama answer quality, or forecast accuracy.

## 2026-10-30 — Local Chroma/Streamlit chatbot implementation

From `/mnt/c/project/mini_solo`:

```text
python -m unittest discover -s src/tests -v
python -m compileall -q src
python src/app.py
python src/source_index.py --help
```

Result: all 19 unit tests passed; `compileall` passed. Tests cover config/env loading, stable intent keys, citation filtering, RRF, year-date normalization, local Ollama request payload shape, exact and semantic cache hits, SQLite FTS5-to-source joins, source-row offsets, hybrid-evidence loading contract, reranking, timeline, and map ordering. `source_index.py --help` verified only its CLI; it did not index the corpus.

Runtime limits verified on this host: `streamlit`, `chromadb`, `sentence_transformers`, `torch`, and `pandas` are not installed. `data/chroma_db` and the 34 TXT / 34 org_jsonl inputs exist, but `data/bm25_index/assembly_bm25.sqlite3` and `data/source_metadata.sqlite3` do not. Running `python src/app.py` therefore only printed the missing-Streamlit instruction; no Streamlit server, real Chroma query, Ollama request, source indexing, or BM25 build was run. Git status is unavailable because `/mnt/c/project/mini_solo` is not inside a Git worktree. These checks establish unit-level behavior only, not full retrieval quality, index coverage, local-model performance, or forecast accuracy.

## 2026-10-31 — BM25 WinError 32 diagnostics

Updated `scripts/make_chroma_db/2_build_bm25_index.py` so SQLite connections close explicitly before the temporary database is atomically renamed. Added stage/path/PID/file-state diagnostics and a best-effort Windows Restart Manager query for processes using the temporary or target database if WinError 32 recurs. Added three tests for connection closure, diagnostic fields, and simulated atomic-replace sharing violations.

Verified: `python -m unittest discover -s src/tests -v` passed all 22 tests; `python -m py_compile scripts/make_chroma_db/2_build_bm25_index.py src/tests/test_bm25_builder.py` passed; a one-record temporary JSONL fixture built and returned one BM25 search hit. The Windows Restart Manager branch and the user's actual locked-file scenario were not executable in WSL.

## 2026-10-31 — Source index WinError 32 cleanup diagnostics

The source indexer used `with sqlite3.connect(...)` and then attempted to unlink the temporary database in its exception handler. That transaction context does not close the SQLite connection, so cleanup could hit WinError 32 and mask the exception that originally caused the build to fail. Updated `src/source_index.py` to close SQLite connections explicitly, report the failing stage and original exception, and preserve that original exception even if temporary-file cleanup also fails.

Verified: all 24 unit tests pass; `python -m compileall -q src scripts/make_chroma_db/2_build_bm25_index.py` passes; `python src/source_index.py --help` passes without building an index. Fixture tests verify connection closure and that a simulated cleanup lock does not hide the original schema error. No full corpus indexing or native Windows lock reproduction was run.

## 2026-10-31 — Direct app.py Streamlit launch

Added a direct entry point in `src/app.py`: plain Python uses Streamlit `st.App(...).run()`, while a script already running in Streamlit calls the UI function without starting a nested server. Updated the runtime requirement to Streamlit 1.66+ and documented that this only launches the app; it does not perform bulk indexing. The user's `mini-gpu` interpreter reports Streamlit 1.66.1 and has chromadb, sentence_transformers, torch, and pandas importable.

Verified: all 26 unit tests pass; `python -m compileall -q src` passes. The launcher branch is tested with a fake Streamlit runtime. The actual UI was not started, so Chroma/Ollama/model/data integration remains unverified.

## 2026-10-31 — Chat input lock, cache reset, and source-index auto setup

`src/app.py` now queues a submitted prompt, reruns with chat input disabled during analysis, and re-enables it after success or error. The sidebar cache button clears SQLite answer rows and Chroma semantic-question IDs in bounded batches while preserving chat history and the main Chroma/BM25 indexes. `ensure_source_index` builds the TXT metadata sidecar only when absent; existing valid databases are reused, and existing invalid ones stop with an error rather than being overwritten. The standalone source-index CLI follows the same skip-by-default behavior; `--force` is explicit opt-in. App startup performs this check after confirming Chroma and BM25 prerequisites.

Verified: `python -m unittest discover -s src/tests -v` passes all 38 tests; `python -m compileall -q src` and `python src/source_index.py --help` pass. Fixture tests verify nested TXT resolution, missing-index creation, valid-index reuse, invalid-index no-overwrite, CLI skip behavior, cache deletion, and input-lock state transitions. The Windows `mini-gpu` environment reports Streamlit 1.66.1, torch 2.14.1+cu130, transformers 5.20.2, sentence-transformers 5.7.0, chromadb 1.5.9, and no torchvision. Its source metadata DB exists and has the expected schema with at least one row. A pip dry-run selected `torchvision 0.29.1+cu130` for Python 3.11 and reported only that package would be installed; no package was installed and no live UI/search was run.

## 2026-11-01 — Resolve nested source TXT paths

The Windows source sidecar contains IDs whose `source_file` is a filename stem, while the matching TXT files are stored in nested year/meeting folders under `data/to_text`. Index construction already scans recursively, but `SourceTextReader` previously checked only the top-level directory. Added a cached recursive fallback by exact TXT basename while preserving `source_file:row_id` IDs and avoiding a large-index rebuild. Added a regression fixture with a nested year directory.

Verified: the regression test passes. Using the actual Windows `mini-gpu` interpreter and existing DB, `SourceTextReader.read` successfully loaded `cleaned_2025_세종특별자치시의회:1` from its nested folder (242 characters); no index rebuild or data write occurred. Full suite passes all 38 tests; `compileall` and the source-index help smoke check pass. A full chat query/UI session remains unverified because `torchvision` is still absent.

## 2026-11-01 — Structured core facts and cache propagation

`AnalysisResult` now carries `core_facts` through JSON serialization and the SQLite analysis cache. `AnalysisPipeline` copies the writer's structured facts into the result, and the renderer displays them under a dedicated “핵심 사실” heading. The prompt/cache version was `local-rag-v2` in code and the active `.env`, preventing old v1 analysis rows from being reused; old cached JSON without the field remains readable. The first-observation excerpt/expand renderer remains covered by its UI-level unit test.

Verified: 8 targeted tests pass for writer/schema output, cache serialization/backward reading, pipeline exact/semantic cache paths, facts rendering, and the first-observation excerpt/expand UI. `python -m unittest discover -s src/tests -v` passes all 46 tests, `python -m compileall -q src` passes, and the Windows `mini-gpu` settings resolve `prompt_version=local-rag-v2`. No live Streamlit browser or local-model end-to-end query was run.

## 2026-11-01 — Forecast visibility and expandable result details

The answer writer now replaces an empty forecast with a clearly labeled conservative note and forces qualitative confidence to `낮음`; with no usable evidence, it says a direction cannot be supported. The result UI puts the forecast first in a bordered summary card, keeps core facts visible, leaves the first-observation excerpt visible with full text in an expander, and collapses the map/timeline, clusters, citations, and caveats into separate detail sections. Streamlit `use_container_width` arguments in `src/app.py` were migrated to `width="stretch"`.

Verified: 10 focused tests pass for empty-forecast fallback, no-evidence handling, cache reuse, visible forecast ordering, empty-forecast warning, collapsed detail sections, and first-observation expansion. `python -m unittest discover -s src/tests -v` passes all 50 tests; `python -m compileall -q src` passes. The Windows `mini-gpu` environment reports Streamlit 1.66.1, resolves `prompt_version=local-rag-v3`, and exposes the tested `container(border=...)`, expander, and `width` API parameters. Repository search finds no remaining `use_container_width` calls. No live browser or local-model end-to-end run was performed.

## 2026-11-01 — Numbered map markers and app chrome

The “처음 실행 전” sidebar block was removed. The collapsed council flow now uses PyDeck circles plus a TextLayer to display date-order numbers directly on map markers; the table retains a numeric “순서” column. The top-right Streamlit toolbar menu is hidden using an empty page menu, project-level `client.toolbarMode="minimal"`, and the toolbar CSS selector.

Verified: the focused rendering/map/config set passes 8 tests, including serialized map layers containing order labels `1` and `2`; `python -m compileall -q src` passes. The project’s Streamlit config resolves to `client.toolbarMode=minimal` on Windows Streamlit 1.66.1, and the installed frontend bundle contains the `stToolbar` identifier. A full 52-test Windows run had 9 `PermissionError [WinError 32]` failures during temporary SQLite-file cleanup in existing cache/BM25/retrieval/source-index tests; it did not finish green. No live browser screenshot was taken.

## 2026-11-01 — Marker scale and forecast provenance

The numbered map keeps the same council coordinates, reduces the circle radius from 14 to 6 pixels, sets the pixel unit as a literal PyDeck value, and reduces the number label to 9px. The visible forecast card initially explained that no separate future-outcome dataset or statistical forecast model existed.

A read-only check of the latest v3 answer-cache row found a nonempty 411-character forecast, 120 timeline records, three core facts, and one general citation. It was a recommendation-style narrative, not a quantitative projection, and the schema had no forecast-specific citation field. This confirmed the forecast text existed but did not have a dedicated predictive data series/provenance link.

Verified at that stage: 11 focused rendering/map/writer tests passed, including PyDeck serialization with 6px circles, `radiusUnits="pixels"`, 9px labels, preserved sequence numbers, and the forecast limitation caption. `python -m compileall -q src` passed. No browser screenshot or live full-corpus run was performed.

## 2026-11-01 — Evidence-linked historical forecasts

`AnalysisWriter` now requests a conditional next-step forecast based on repeated discussion, dates, and ordinance/budget/implementation signals in retrieved minutes, rather than allowing a recommendation list to stand in for a prediction. `AnalysisResult.forecast_citations` carries the forecast-specific source IDs through SQLite cache serialization. The pipeline accepts only IDs present in retrieved evidence and does not invent a source by falling back to the first general citation. If the model returns an empty forecast or no valid direct source IDs, the writer replaces it with a conditional outlook derived from the evidence date span and links the dated records it used. Confidence is kept qualitative: fewer than two cited IDs or fewer than two distinct dates forces `낮음`; a model-requested `높음` is capped at `보통` unless at least four cited records span at least three dates. The UI displays the forecast basis in `전망 근거 보기`. `PROMPT_VERSION` is now `local-rag-v4` in the active env file, default, and example so old v3 answers are not reused.

A Windows `mini-gpu` live Ollama writer smoke test used 24 evenly sampled records from the latest v3 cached timeline. The model response was recommendation-style and supplied no valid forecast citations; the new validation therefore emitted a dated conditional fallback with two valid source IDs and low confidence. The sample spanned 23 distinct meeting dates from 20240124 to 20251209. This exercised the writer with real cached minutes, but was not a fresh Chroma/BM25 retrieval or a full Streamlit end-to-end run.

The same full-suite run exposed Windows file-lock errors because SQLite's `with connection` commits but does not close the handle. `AnalysisCache._connect` now closes explicitly, `BM25Repository` uses the existing managed SQLite context, and test fixtures use managed connections to avoid leaving handles open. Verified: `python -m unittest discover -s src/tests -v` passes all 56 tests; `python -m compileall -q src` passes. No live browser rendering was performed.

## 2026-11-02 — Summary-only forecast input

`AnalysisWriter` now uses separate summary and forecast Ollama calls. The summary call receives metadata plus up to 1,200 characters per selected utterance and creates answer/core facts and per-source summaries capped at 320 characters. If a per-source summary is missing, the forecast payload uses metadata-only text instead of copying an excerpt; such IDs are excluded from model citation candidates. The forecast call receives summaries, meeting metadata, and aggregates over the retrieved/reranked candidate set (up to 120), not the full corpus, and contains no `excerpt` field. Aggregates include date coverage and counts by year/month/council/agenda plus weak keyword-signal counts. The UI states this input policy. `PROMPT_VERSION=local-rag-v5` prevents reuse of older forecast cache entries.

Verified on Windows `mini-gpu`: all 58 tests pass with `python -m unittest discover -s src/tests -v`; `python -m compileall -q src` passes and settings resolve to `local-rag-v5`. A live Ollama writer smoke test on 24 evenly sampled records from an older cached timeline made two calls, returned a nonempty forecast with two citations and a valid confidence label, and produced three per-source summaries plus 21 metadata-only records. The captured forecast payload had no `excerpt` field, no matching 100-character source prefixes, and no full source strings of at least 100 characters. This verifies the writer's two-stage payload path, not a fresh Chroma/BM25 retrieval or full Streamlit UI session.

## 2026-11-02 — Larger map markers and concise forecast card

The PyDeck council circles were increased from a 6px radius to 18px, and discussion-order labels from 9px to 27px; council coordinates and sequence logic are unchanged. The forecast card now shows a single `신뢰도: ...` label before the forecast text. Redundant forecast captions about qualitative confidence, future-result limitations, and payload composition were removed. The forecast prompt requests a direct 1–3 sentence outlook and suppresses repeated uncertainty disclaimers; evidence-grounded/no-evidence behavior remains. Deterministic fallback forecasts no longer repeat confidence or phrases such as “cannot conclude.” `PROMPT_VERSION=local-rag-v7` separates cached answers created under the previous forecast presentation/prompt.

Verified on Windows `mini-gpu`: all 68 tests pass with `python -m unittest discover -s src/tests -v`; `python -m compileall -q src` passes; active settings resolve to `local-rag-v7`. Focused tests assert 18px/27px PyDeck serialization, confidence label preceding forecast text, concise fallback language, and prompt instructions. No live browser screenshot or local Ollama run was performed.

## 2026-11-02 — Number-only map labels

Removed the circle-marker ScatterplotLayers after the user asked for numbers without the graphic underneath. The map now renders one TextLayer only; each 27px number uses blue for the first observed record and orange for later discussions. Coordinates, numeric order, and the table remain unchanged.

Verified: the map test checks there is exactly one TextLayer, no ScatterplotLayer/radius properties, explicit pixel text sizing, and the order labels. The full suite passes all 68 tests and `python -m compileall -q src` passes. No live browser screenshot was taken.

## 2026-11-02 — One circle per council; disable hover

Restored the circle marker and white number layers, but deduplicated mapped evidence so each council has one marker labeled with that council's earliest chronological result number. The table retains every meeting and sequence number. Both layers have `pickable=False`, and the Deck tooltip was removed so hovering displays nothing.

Verified on Windows `mini-gpu`: all 69 tests pass with `python -m unittest discover -s src/tests -v`; `python -m compileall -q src` passes. Focused tests confirm one point per council, earliest source/order labels, circle and text layers, and disabled picking/no tooltip. No live browser screenshot was taken.

## 2026-11-02 — Exact-question cache, grouped history, visible reset

Cache diagnosis: the current app uses `DATA_VERSION=2024-2025-v1`, `PROMPT_VERSION=local-rag-v7`, and model `huihui_ai/exaone3.5-abliterated`. The existing SQLite answer store had 9 rows, but none were v7: 2 v1, 2 v2, 1 v3, and 4 v6. The Chroma question-cache collection had the same version distribution and no v7 rows. Those older entries are correctly ineligible after prompt changes; cache-version gating is intentional. Separately, prior to this change, raw questions were not cache keys: the LLM first extracted an intent using conversation history, then exact/semantic lookup ran. This could vary intent and add a local-LLM call for an identical repeated question.

Added a SQLite mapping from normalized question hashes (not raw question text) to structured cache entries, gated by corpus version, prompt version, model, and forecast as-of date. The fast path runs before intent extraction; older exact/semantic hits also gain a mapping for subsequent repeats. Reset deletes these mappings along with answer rows. Grouped each user/assistant turn in a collapsed expander titled with the extracted issue and question; opening it shows the full result without nested expanders. Moved the reset button from the sidebar to the main page immediately under the title.

Verified on Windows `mini-gpu`: focused cache/UI tests pass; all 71 tests pass and `python -m compileall -q src` passes. The renderer tests verify the question expander is collapsed by default, full results stay within one expander, and the main-page cache button is rendered. Live browser confirmation remains outstanding: Streamlit reported port 8501 unavailable and the WSL browser automation could not start Chromium because `libnspr4.so` is missing.

## 2026-11-02 — Header cache action, nested result categories, and grounded future topics

Moved cache reset into a compact `width="content"` button beside the page title using a 6:1 column layout. Question history stays inside a collapsed question-level expander; the result's existing category panels (council order, timeline, clusters, sources, and citations) are now expandable inside it. Added `likely_discussion_topics` to the analysis result and cache serialization. The forecast prompt requests up to three specific follow-up discussion topics, each linked to one or more summarized source IDs; the writer and pipeline discard topics with IDs outside the forecastable/retrieved evidence. The UI shows those topics and their source IDs, or says when summaries do not support a specific topic. `PROMPT_VERSION` was raised to `local-rag-v8` so v7 forecasts without this field are not reused.

Verified on Windows `mini-gpu`: all 71 tests pass with `python -m unittest discover -s src/tests -v`; `python -m compileall -q src` passes; active settings and both `.env` files resolve to `local-rag-v8`. Streamlit 1.66.1 `AppTest` executed the real header and result renderer with zero exceptions and found the cache button, collapsed question/category expanders, and future-topic text. No interactive browser screenshot or live Ollama analysis was run. Streamlit 1.46+ permits nested expanders; current documentation advises keeping nested layouts limited, so this implementation uses only question → category levels.

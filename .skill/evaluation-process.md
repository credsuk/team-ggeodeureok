# Evaluation process: PostgreSQL API

## Retrieval

1. Confirm the PostgreSQL table and schema from the actual configured runtime without logging the DSN.
2. Create a human-reviewed query set that reflects the 10 curated issue keywords, free-form queries, names/ordinances, and queries with no matching evidence.
3. For each query, record relevance labels and relevant `chunk_id` values; evaluate Dense-only, PostgreSQL full-text-only, and RRF using Recall@k, Precision@k, and MRR. Keep the same filters and corpus for all methods.
4. Review false positives and near-duplicate chunks. The current API does not use a reranker or local parent-chunk context.
5. Run actual DB smoke tests separately from mocked unit/contract tests. Do not infer real DB performance or retrieval quality from mocks.

## Forecast

The API currently predicts next month's count of matching distinct `document_id` values using the mean of the last three observed months. Evaluate with rolling-origin backtests over the available 2024–2025 data; compare with last-month and seasonal-naive baselines using MAE and MAE by issue. Report the number of evaluated months and the exact forecast method. Do not label the predicted count as a probability or as the previous trained model's output.

## Current verified scope

Automated tests cover the RRF helper, full-text query formatting, monthly-baseline arithmetic, service routing against an injected fake repository, API contracts, and launcher fail-fast behavior. They do not establish database compatibility, query latency at production corpus size, Ollama answer quality, or forecast accuracy.

## Local Chroma/Streamlit chatbot

1. Create a fixed, human-reviewed query set over the intended 2024–2025 corpus. Include common issues, spelling/alias variants, explicit assembly/date filters, no-evidence questions, and questions whose earliest mention is known from a corpus-wide audit.
2. Record relevant stable IDs (`source_file:row_id`) and, where practical, the relevant assembly/date/meeting. Compare Dense-only, FTS5 BM25-only, RRF, and Cross-Encoder reranking with identical filters and candidate budgets. Report Recall@k, Precision@k, MRR, query count, and missing-source examples; distinguish result count from relevance.
3. Evaluate “first observed” separately: compare the earliest returned evidence against the reviewed earliest corpus record. A top-k or reranked list cannot prove the corpus-wide first occurrence. Keep chronological order separate from evidence of explicit cross-council reference; have reviewers label actual references before evaluating policy-transfer claims.
4. Evaluate the cache with paraphrase pairs and hard negatives. A hit is correct only if issue, corpus version, model/prompt version, task, assembly scope, dates, and forecast cutoff match. Measure correct-reuse rate and false-hit rate; tune `CACHE_SIMILARITY` on this set rather than treating Chroma distance as a probability.
5. Backtest forecasts by rolling the observation cutoff forward within the available years, never using later meetings as input to an earlier forecast. Compare with simple last-observation and seasonal-naive baselines; report MAE/MASE (or an appropriate count metric), evaluated issue-month count, and uncertainty. The app’s qualitative confidence label is not calibrated probability until separately calibrated and tested.
6. Validate representative map pins against actual council office addresses. Track approximate administrative-center coordinates separately, and review any inferred cross-council flow visually and against source citations.
7. Run live smoke tests only after Chroma ingestion is complete, the FTS5 index exists, dependencies and local models are installed, and Ollama is available. The app can create the source-offset sidecar only when missing and reuses a valid existing DB; Chroma/BM25 are still prerequisites. Mocked/unit tests do not prove data coverage, answer quality, or latency.

## Current implementation verification

The local Chroma/Streamlit app has unit coverage for intent normalization, cache version gates and clearing, RRF fusion, original-TXT evidence loading including nested source folders, reranking, timeline/map ordering, chat-state transitions, source-index reuse, and direct Streamlit launch. The checked suite currently has 38 passing tests. This is unit/fixture coverage plus a read-only Windows source-row smoke check: no live browser session or end-to-end local-model search has been verified. On the checked Windows `mini-gpu` environment, PyTorch is `2.14.1+cu130` but TorchVision is absent; install the matching package before retrying Cross-Encoder search.

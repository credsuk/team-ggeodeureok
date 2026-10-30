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

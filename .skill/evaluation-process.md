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

### Search quality experiment

1. Create a fixed, human-reviewed query set over the intended 2024–2025 corpus. Include common issues, spelling/alias variants, explicit assembly/date filters, no-evidence questions, and questions whose earliest mention is known from a corpus-wide audit. Record relevant stable IDs (`source_file:row_id`) and relevant assembly/date/meeting where possible.
2. Keep the corpus, query filters, and candidate budgets identical. Compare Dense-only, FTS5 BM25-only, equal RRF (both weights 1.0), and weighted RRF using `DENSE_RRF_WEIGHT` / `BM25_RRF_WEIGHT`. Report Recall@k, Precision@k, MRR, query count, missing-source examples, and latency. Search weights are rank-contribution controls, not trained neural parameters or probabilities.
3. For model ensembling, compare the default single `RERANKER_MODEL` against `RERANKER_MODELS` + matching `RERANKER_WEIGHTS`. The current implementation fuses each Cross-Encoder's rank list with weighted RRF; it does not average raw scores. Record GPU/CPU memory and total latency as well as relevance. Multiple models are optional and require local checkpoint availability; no second checkpoint is bundled.
4. Review false positives and near-duplicate utterances. Evaluate “first observed” separately by comparing the earliest returned evidence against the reviewed earliest corpus record. A top-k or reranked list cannot prove the corpus-wide first occurrence. Label actual cross-council references before evaluating policy-transfer claims.
5. Evaluate the cache with paraphrase pairs and hard negatives. A hit is correct only if issue, corpus version, model/prompt version, task, assembly scope, dates, and forecast cutoff match. Measure correct-reuse and false-hit rates; tune `CACHE_SIMILARITY` on this set rather than treating Chroma distance as probability. Current cache identity also includes LLM/embedding/reranker identifiers, fusion weights, key search limits, and execution devices.
6. Backtest forecasts by rolling the observation cutoff forward within the available years, never using later meetings as input to an earlier forecast. Compare with simple last-observation and seasonal-naive baselines; report MAE/MASE (or an appropriate count metric), evaluated issue-month count, and uncertainty. The app’s qualitative confidence label is not calibrated probability until separately calibrated and tested.
7. Validate representative map pins against actual council office addresses. Track approximate administrative-center coordinates separately, and review any inferred cross-council flow visually and against source citations.
8. Run live smoke tests only after Chroma ingestion is complete, the FTS5 index exists, dependencies and local models are installed, and Ollama is available. The app can create the source-offset sidecar only when missing and reuses a valid existing DB; Chroma/BM25 are still prerequisites. Mocked/unit tests do not prove data coverage, answer quality, or latency.

### Current implementation verification (2026-11-03)

Verified on Windows `mini-gpu`: all 78 tests pass and `compileall -q src scripts` passes. A read-only 5-candidate Chroma/BM25/embedding/single-Cross-Encoder smoke returned five finite reranked scores in about 213 seconds including initialization. The 300-candidate smoke exceeded 420 seconds and was stopped. No live Ollama answer or Streamlit browser flow was run. No human-judged retrieval set is available, so quality gains remain unmeasured.

After source package reorganization, all 81 tests passed and `compileall -q src scripts` passed. `import src.app`, `src/source_index.py --help`, and `python -m src.data.source_index --help` passed. The Korean-tuned `dragonkue/bge-reranker-v2-m3-ko` checkpoint is now cached alongside the existing BAAI reranker; an offline two-model CPU smoke on two synthetic Korean evidence items loaded both and returned finite fused scores in 38.8 seconds. This is a compatibility/cold-load smoke only, not representative full-corpus latency or relevance evaluation. Continue to compare Dense-only, BM25-only, equal/weighted RRF, and single/ensemble reranking on the same human-reviewed query set before claiming improvements.

### 2026-11-03 — Human-reviewed reranker pilot pack

Created `scripts/evaluation/reranker_eval.py` and a local review pack at `C:/hermes/mini_solo_reranker_eval/`. The query file has 9 draft questions (8 issue questions and 1 suspected no-evidence control), each scoped to 2024–2025. A real read-only retrieval run against the existing `assembly_metadata` Chroma collection and BM25/source metadata produced 30 candidates per query (270 total). It did not rebuild or overwrite Chroma, BM25, source metadata, or answer caches. Candidate rows are deterministically hash-shuffled after taking the top 30 hybrid candidates, so their retrieval/model ranks are not shown in the annotation CSV. `review_candidates.meta.json` records the corpus version, retrieval limits and weights, collection count, model IDs, and cached revisions.

The three comparison settings are BAAI-only, Korean-only, and equal-weight (1.0/1.0) rank-RRF ensemble on exactly the same candidates. The human label rubric is 0 = irrelevant, 1 = related but not direct evidence, 2 = directly relevant. After labels are complete, the runner emits macro NDCG@10 and MRR@10 over queries with at least one positive label in the reviewed pool, plus per-query results and candidate rankings. Raw `ranker_score` values from single models and RRF ensemble are on different scales; compare ranks and ranking metrics, not score magnitudes. Queries with no positive labels in the pool are excluded from those ranking averages; the no-evidence control's top-1 graded relevance is reported separately. This pilot measures reranking conditional on the first-stage hybrid candidate pool; it does not measure corpus-wide retrieval recall, answer-generation quality, or establish a model winner.

At this point the 270 labels have not been supplied, so no actual quality comparison has been run. The query wording is a draft and should be reviewed before annotation. `README.md` inside the review-pack folder has the exact Windows commands and rubric.

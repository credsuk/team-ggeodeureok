#!/usr/bin/env bash
set -euo pipefail

python src/17_enrich_agenda_metadata.py \
  --input data/index/arctic_ko_semantic_test_v2/chunk_metadata.parquet \
  --output data/index/arctic_ko_semantic_test_v2/chunk_metadata_enriched.parquet \
  --report-dir outputs/agenda_metadata \
  --update-chroma \
  --chroma-dir data/chroma/arctic_policy_v1 \
  --collection council_policy_arctic_v1

python src/09_build_issue_timeline.py \
  --assignments data/index/arctic_ko_semantic_test_v2/clustering_k160/cluster_assignments.parquet \
  --labels outputs/llm_labels_semantic_test_v2_k160_strict/llm_cluster_labels.csv \
  --evidence-metadata data/index/arctic_ko_semantic_test_v2/chunk_metadata_enriched.parquet \
  --output-dir outputs/issue_timeline_final \
  --include-status accepted_candidate \
  --min-similarity 0.55 \
  --evidence-per-council 3

python src/11_forecast_issue_flow.py \
  --timelines outputs/issue_timeline_final/issue_timelines.json \
  --output-dir outputs/issue_forecast_final \
  --analysis-cutoff 2025-12-31 \
  --validation-months 4 \
  --min-validation-f1 0.25 \
  --min-balanced-accuracy 0.55

python scripts/verify_final_outputs.py


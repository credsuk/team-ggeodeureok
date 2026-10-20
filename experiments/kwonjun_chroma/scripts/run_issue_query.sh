#!/usr/bin/env bash
set -euo pipefail

query="${1:-저출생 출산율 인구감소}"

python src/18_chroma_issue_query.py \
  --query "$query" \
  --chroma-dir data/chroma/arctic_policy_v1 \
  --collection council_policy_arctic_v1 \
  --assignments data/index/arctic_ko_semantic_test_v2/clustering_k160/cluster_assignments.parquet \
  --timelines outputs/issue_timeline_final/issue_timelines.json \
  --forecasts outputs/issue_forecast_final/issue_forecasts.json \
  --output-dir outputs/chroma_issue_search \
  --local-files-only


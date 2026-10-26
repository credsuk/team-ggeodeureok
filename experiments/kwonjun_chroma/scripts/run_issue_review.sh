#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

QUEUE="${COUNCIL_ISSUE_REVIEW_QUEUE:-$PROJECT_ROOT/outputs/issues/candidates_v1/issue_review_queue.csv}"
if [[ ! -f "$QUEUE" ]]; then
  echo "검수 대기열이 없습니다: $QUEUE" >&2
  echo "COUNCIL_ISSUE_REVIEW_QUEUE에 실제 CSV 경로를 지정하세요." >&2
  exit 1
fi

export COUNCIL_ISSUE_REVIEW_QUEUE="$QUEUE"
exec python -m streamlit run demo/issue_review_app.py \
  --server.port "${PORT:-8502}" \
  --server.address "${HOST:-127.0.0.1}"

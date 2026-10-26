#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PAYLOAD="${COUNCIL_ISSUE_PAYLOAD:-$PROJECT_ROOT/outputs/web/birth_rate_issue.json}"
if [[ ! -f "$PAYLOAD" ]]; then
  echo "웹 결과 JSON이 없습니다: $PAYLOAD" >&2
  exit 1
fi
export COUNCIL_ISSUE_PAYLOAD="$PAYLOAD"
exec python -m streamlit run demo/issue_flow_app.py \
  --server.port "${PORT:-8501}" \
  --server.address "${HOST:-127.0.0.1}"

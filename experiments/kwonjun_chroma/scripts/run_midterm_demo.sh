#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PAYLOAD="${COUNCIL_ISSUE_PAYLOAD:-$PROJECT_ROOT/outputs/web/birth_rate_issue.json}"
if [[ ! -f "$PAYLOAD" ]]; then
  echo "웹 결과 JSON이 없습니다: $PAYLOAD" >&2
  echo "먼저 src/24_build_web_payload.py를 실행하세요." >&2
  exit 1
fi

if command -v docker >/dev/null 2>&1; then
  docker compose up -d pgvector >/dev/null
else
  echo "주의: Docker를 찾지 못했습니다. 지도·고정 이슈 화면은 실행되지만 pgvector 검색은 사용할 수 없습니다." >&2
fi

if ! curl -fsS "${OLLAMA_URL:-http://127.0.0.1:11434}/api/tags" >/dev/null 2>&1; then
  echo "안내: Ollama가 실행 중이 아닙니다. Agent는 구조화 결과 요약 모드로 동작합니다." >&2
fi

export COUNCIL_ISSUE_PAYLOAD="$PAYLOAD"
exec python -m streamlit run demo/midterm_app.py \
  --server.port "${PORT:-8501}" \
  --server.address "${HOST:-127.0.0.1}"

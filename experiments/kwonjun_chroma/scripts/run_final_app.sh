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

if command -v docker >/dev/null 2>&1; then
  docker compose up -d pgvector >/dev/null
fi

python -u src/34_build_final_web_bundle.py

if ! curl -fsS "${OLLAMA_URL:-http://127.0.0.1:11434}/api/tags" >/dev/null 2>&1; then
  echo "안내: Ollama가 실행 중이 아닙니다. Agent는 구조화 요약으로 대체됩니다." >&2
fi

exec python -m streamlit run demo/final_app.py \
  --server.port "${PORT:-8501}" \
  --server.address "${HOST:-127.0.0.1}"

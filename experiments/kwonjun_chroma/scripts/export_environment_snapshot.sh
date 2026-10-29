#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$PROJECT_ROOT/environment}"
mkdir -p "$OUTPUT_DIR"

python --version >"$OUTPUT_DIR/python-version.txt" 2>&1
python -m pip freeze --all | LC_ALL=C sort >"$OUTPUT_DIR/requirements-lock.txt"

{
  uname -a
  echo
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  else
    echo "NVIDIA GPU: not detected"
  fi
} >"$OUTPUT_DIR/system-info.txt" 2>&1

if command -v docker >/dev/null 2>&1; then
  {
    docker --version
    docker compose version
    docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}'
  } >"$OUTPUT_DIR/docker-info.txt" 2>&1 || true
fi

if command -v ollama >/dev/null 2>&1; then
  ollama list >"$OUTPUT_DIR/ollama-models.txt" 2>&1 || true
fi

echo "환경 스냅샷 생성 완료: $OUTPUT_DIR"


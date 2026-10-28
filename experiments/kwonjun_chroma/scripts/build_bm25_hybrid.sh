#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python -u src/35_build_bm25_index.py "$@"

echo
echo "BM25 인덱스가 준비됐습니다. 하이브리드 검색 예시:"
echo "python -u src/18_pgvector_retrieval.py --query '출산율 저하 대응 정책' --output outputs/retrieval/hybrid_smoke.json"

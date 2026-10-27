#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "사용법: bash scripts/restore_pgvector_from_dump.sh <dump 파일> [--replace]" >&2
  exit 2
fi
DUMP_PATH="$(readlink -f "$1")"
REPLACE="${2:-}"
if [[ ! -f "$DUMP_PATH" ]]; then
  echo "dump 파일이 없습니다: $DUMP_PATH" >&2
  exit 1
fi
if [[ -n "$REPLACE" && "$REPLACE" != "--replace" ]]; then
  echo "두 번째 인자는 --replace만 허용됩니다." >&2
  exit 2
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

SERVICE="${PGVECTOR_SERVICE:-pgvector}"
DB_USER="${POSTGRES_USER:-council}"
DB_NAME="${POSTGRES_DB:-council_vectors}"
TABLE="${COUNCIL_PG_TABLE:-council_arctic_raw_v1}"

if [[ ! "$TABLE" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
  echo "잘못된 테이블명: $TABLE" >&2
  exit 1
fi

CHECKSUM="${DUMP_PATH}.sha256"
if [[ -f "$CHECKSUM" ]]; then
  (
    cd "$(dirname "$DUMP_PATH")"
    sha256sum -c "$(basename "$CHECKSUM")"
  )
else
  echo "주의: checksum 파일이 없어 무결성 검사를 생략합니다: $CHECKSUM" >&2
fi

docker compose up -d "$SERVICE" >/dev/null
for _ in $(seq 1 30); do
  if docker compose exec -T "$SERVICE" \
    pg_isready -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
docker compose exec -T "$SERVICE" \
  pg_isready -U "$DB_USER" -d "$DB_NAME" >/dev/null

EXISTING="$(docker compose exec -T "$SERVICE" \
  psql -At -U "$DB_USER" -d "$DB_NAME" \
  -c "SELECT COALESCE(to_regclass('public.$TABLE')::text, '');")"
if [[ -n "$EXISTING" && "$REPLACE" != "--replace" ]]; then
  echo "이미 $TABLE 테이블이 있습니다." >&2
  echo "기존 DB를 교체하려는 경우에만 --replace를 추가하세요." >&2
  exit 1
fi

RESTORE_ARGS=(--no-owner --no-privileges --exit-on-error)
if [[ "$REPLACE" == "--replace" ]]; then
  RESTORE_ARGS+=(--clean --if-exists)
fi

echo "pgvector DB 복원 중입니다. HNSW 인덱스 재생성 때문에 시간이 걸릴 수 있습니다."
docker compose exec -T "$SERVICE" \
  pg_restore -U "$DB_USER" -d "$DB_NAME" "${RESTORE_ARGS[@]}" \
  < "$DUMP_PATH"

ROW_COUNT="$(docker compose exec -T "$SERVICE" \
  psql -At -U "$DB_USER" -d "$DB_NAME" \
  -c "SELECT count(*) FROM public.$TABLE;")"
PGVECTOR_VERSION="$(docker compose exec -T "$SERVICE" \
  psql -At -U "$DB_USER" -d "$DB_NAME" \
  -c "SELECT extversion FROM pg_extension WHERE extname='vector';")"

echo
echo "=== pgvector DB 복원 완료 ==="
echo "테이블: $TABLE | 행: $ROW_COUNT | pgvector: $PGVECTOR_VERSION"
echo "이제 bash scripts/run_final_app.sh 를 실행하세요."

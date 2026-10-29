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

SERVICE="${PGVECTOR_SERVICE:-pgvector}"
DB_USER="${POSTGRES_USER:-council}"
DB_NAME="${POSTGRES_DB:-council_vectors}"
TABLE="${COUNCIL_PG_TABLE:-council_arctic_raw_v1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${1:-$PROJECT_ROOT/exports/${DB_NAME}_${STAMP}.dump}"

if [[ ! "$TABLE" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
  echo "잘못된 테이블명: $TABLE" >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "docker 명령을 찾을 수 없습니다." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
OUTPUT="$(cd "$(dirname "$OUTPUT")" && pwd)/$(basename "$OUTPUT")"
TEMP="${OUTPUT}.partial"
trap 'rm -f "$TEMP"' EXIT

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

TABLE_EXISTS="$(docker compose exec -T "$SERVICE" \
  psql -At -U "$DB_USER" -d "$DB_NAME" \
  -c "SELECT COALESCE(to_regclass('public.$TABLE')::text, '');")"
if [[ -z "$TABLE_EXISTS" ]]; then
  echo "공유할 테이블이 없습니다: $TABLE" >&2
  exit 1
fi

ROW_COUNT="$(docker compose exec -T "$SERVICE" \
  psql -At -U "$DB_USER" -d "$DB_NAME" \
  -c "SELECT count(*) FROM public.$TABLE;")"
DB_SIZE="$(docker compose exec -T "$SERVICE" \
  psql -At -U "$DB_USER" -d "$DB_NAME" \
  -c "SELECT pg_size_pretty(pg_database_size(current_database()));")"
PGVECTOR_VERSION="$(docker compose exec -T "$SERVICE" \
  psql -At -U "$DB_USER" -d "$DB_NAME" \
  -c "SELECT COALESCE((SELECT extversion FROM pg_extension WHERE extname='vector'), 'not-installed');")"

echo "PostgreSQL custom dump 생성 중: $OUTPUT"
echo "DB 크기: $DB_SIZE | $TABLE 행: $ROW_COUNT | pgvector: $PGVECTOR_VERSION"
docker compose exec -T "$SERVICE" \
  pg_dump -U "$DB_USER" -d "$DB_NAME" \
  --format=custom --compress=6 --no-owner --no-privileges \
  > "$TEMP"

if [[ ! -s "$TEMP" ]]; then
  echo "pg_dump 결과가 0바이트입니다. 백업을 중단합니다." >&2
  exit 1
fi
if ! docker compose exec -T "$SERVICE" pg_restore --list < "$TEMP" >/dev/null; then
  echo "생성된 파일이 올바른 PostgreSQL custom dump가 아닙니다." >&2
  exit 1
fi

mv "$TEMP" "$OUTPUT"
chmod 600 "$OUTPUT"
OUTPUT_DIR="$(dirname "$OUTPUT")"
OUTPUT_NAME="$(basename "$OUTPUT")"
(
  cd "$OUTPUT_DIR"
  sha256sum "$OUTPUT_NAME" > "${OUTPUT_NAME}.sha256"
)
cat > "${OUTPUT}.info.txt" <<EOF
database=$DB_NAME
table=$TABLE
row_count=$ROW_COUNT
database_size=$DB_SIZE
pgvector_version=$PGVECTOR_VERSION
docker_image=pgvector/pgvector:0.8.7-pg17
created_at=$(date --iso-8601=seconds)
dump_format=PostgreSQL custom archive
contains_contest_derived_text=true
sharing_scope=competition_team_only
EOF

echo
echo "=== 공유용 pgvector dump 생성 완료 ==="
ls -lh "$OUTPUT" "${OUTPUT}.sha256" "${OUTPUT}.info.txt"
echo "Google Drive에는 위 3개 파일을 팀원 계정에만 제한 공유하세요."

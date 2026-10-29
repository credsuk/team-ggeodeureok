#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODE="${1:-candidates}"
KEY_FILE="${COUNCIL_REPRESENTATIVE_KEYS:-config/representative_issue_keys_v1.txt}"
CANDIDATE_DIR="${COUNCIL_CANDIDATE_DIR:-outputs/issues/candidates_v1}"
FINAL_DIR="${COUNCIL_FINAL_DIR:-outputs/issues/final_v1}"
WEB_DIR="${COUNCIL_WEB_DIR:-outputs/web}"
CLUSTER_SIMILARITY="${COUNCIL_CLUSTER_SIMILARITY:-0.68}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ ! -f "$KEY_FILE" ]]; then
  echo "대표 이슈 key 파일이 없습니다: $KEY_FILE" >&2
  exit 1
fi

mapfile -t ISSUE_KEYS < <(sed -e 's/#.*$//' -e '/^[[:space:]]*$/d' "$KEY_FILE")
if [[ "${#ISSUE_KEYS[@]}" -ne 10 ]]; then
  echo "대표 이슈는 정확히 10개여야 합니다. 현재: ${#ISSUE_KEYS[@]}" >&2
  exit 1
fi

case "$MODE" in
  candidates)
    if [[ -z "${PGVECTOR_DSN:-}" ]]; then
      echo "PGVECTOR_DSN이 없습니다. .env를 확인하세요." >&2
      exit 1
    fi
    python -u src/25_build_issue_candidates.py \
      --keys "${ISSUE_KEYS[@]}" \
      --output-dir "$CANDIDATE_DIR" \
      --device "${COUNCIL_DEVICE:-cuda}" \
      --candidate-k 50 \
      --rerank-top-k 10 \
      --expansion-k 500 \
      --ef-search 500 \
      --min-centroid-similarity 0.45 \
      --cluster-similarity "$CLUSTER_SIMILARITY" \
      --skip-existing \
      --continue-on-error
    echo
    echo "후보 생성 완료. 다음 명령으로 10개 근거를 검수하세요."
    echo "COUNCIL_ISSUE_REVIEW_QUEUE=$CANDIDATE_DIR/issue_review_queue.csv bash scripts/run_issue_review.sh"
    ;;
  finalize)
    if [[ -z "${PGVECTOR_DSN:-}" ]]; then
      echo "PGVECTOR_DSN이 없습니다. .env를 확인하세요." >&2
      exit 1
    fi
    python -u src/30_finalize_reviewed_issues.py \
      --review-queue "$CANDIDATE_DIR/issue_review_queue.csv" \
      --output-dir "$FINAL_DIR" \
      --web-output-dir "$WEB_DIR" \
      --refresh-clusters \
      --cluster-similarity "$CLUSTER_SIMILARITY" \
      --continue-on-error
    ;;
  panel)
    python -u src/31_build_multi_issue_panel.py \
      --input-dir "$FINAL_DIR" \
      --output-dir outputs/forecast/multi_issue_v1
    python -u src/32_train_multi_issue_forecast.py \
      --panel outputs/forecast/multi_issue_v1/multi_issue_panel.csv \
      --forecast-input outputs/forecast/multi_issue_v1/latest_forecast_inputs.json \
      --output-dir outputs/forecast/multi_issue_v1/model
    ;;
  *)
    echo "사용법: bash scripts/run_representative_issues.sh {candidates|finalize|panel}" >&2
    exit 2
    ;;
esac

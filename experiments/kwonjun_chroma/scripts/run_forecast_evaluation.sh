#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

FINAL_DIR="${COUNCIL_FINAL_DIR:-outputs/issues/final_v1}"
EVAL_DIR="${COUNCIL_FORECAST_DIR:-outputs/forecast/multi_issue_v1}"
MODEL_DIR="$EVAL_DIR/model"

python -u src/31_build_multi_issue_panel.py \
  --input-dir "$FINAL_DIR" \
  --output-dir "$EVAL_DIR"

python -u src/32_train_multi_issue_forecast.py \
  --panel "$EVAL_DIR/multi_issue_panel.csv" \
  --forecast-input "$EVAL_DIR/latest_forecast_inputs.json" \
  --output-dir "$MODEL_DIR"

python - "$MODEL_DIR/multi_issue_forecast.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
evaluation = report["required_evaluation"]
comparison = evaluation["last_month_persistence_comparison"]

print("\n=== 최종 예측 평가 요약 ===")
print(f"Rolling folds: {report['rolling_backtest']['fold_count']}")
print(f"논의 지속 F1: {evaluation['discussion_active_f1']:.4f}")
print(f"언급량 MAE: {evaluation['mention_count_mae']:.4f}")
print(f"언급량 RMSE: {evaluation['mention_count_rmse']:.4f}")
print(f"방향성 정확도: {evaluation['directional_accuracy']:.4f}")
print(f"전월 유지 대비 F1 Δ: {comparison['classification_f1_delta']:+.4f}")
print(
    "전월 유지 대비 MAE 개선율: "
    f"{comparison['count_mae_improvement_percent']}%"
)
print(
    "전월 유지 대비 RMSE 개선율: "
    f"{comparison['count_rmse_improvement_percent']}%"
)
print(
    "전월 유지 대비 방향성 정확도 Δ: "
    f"{comparison['directional_accuracy_delta']:+.4f}"
)
print(f"JSON: {path}")
print(f"Rolling CSV: {path.parent / 'rolling_backtest_metrics.csv'}")
PY


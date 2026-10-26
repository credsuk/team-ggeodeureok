#!/usr/bin/env python3
"""다중 이슈 패널로 다음 달 논의 지속과 언급량을 시간순 평가한다.

무작위 분할을 사용하지 않는다. 월 단위 expanding-window 검증으로 기준선과
Logistic/Poisson 모델을 비교하고, 선택 모델은 전체 관측기간으로 다시 학습한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTIFIERS = {"issue_id", "issue_name", "feature_month", "target_month"}
TARGETS = {
    "next_month_mention_count", "target_discussion_active",
    "target_discussion_growth", "target_new_council_spread",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--forecast-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-train-months", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve().open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"CSV가 비었습니다: {path}")
    return rows


def numeric_matrix(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    result = []
    for row in rows:
        current = []
        for column in columns:
            try:
                value = float(row.get(column, ""))
                current.append(value if math.isfinite(value) else np.nan)
            except (TypeError, ValueError):
                current.append(np.nan)
        result.append(current)
    return np.asarray(result, dtype=np.float64)


def classification_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced", max_iter=3000, random_state=42,
        )),
    ])


def regression_pipeline() -> TransformedTargetRegressor:
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", PoissonRegressor(alpha=0.5, max_iter=3000)),
    ])
    return TransformedTargetRegressor(
        regressor=pipeline, func=np.log1p, inverse_func=np.expm1,
    )


def safe_auc(metric: Any, y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(metric(y_true, scores))


def main() -> None:
    args = parse_args()
    if args.minimum_train_months < 3:
        raise ValueError("minimum-train-months는 3 이상이어야 합니다")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold는 0과 1 사이여야 합니다")

    all_rows = read_csv(args.panel)
    required = IDENTIFIERS | TARGETS
    missing = required - set(all_rows[0])
    if missing:
        raise ValueError(f"패널 필수 열 누락: {sorted(missing)}")
    rows = [
        row for row in all_rows
        if all(str(row.get(name, "")).strip() != "" for name in TARGETS)
    ]
    feature_columns = [
        name for name in all_rows[0] if name not in IDENTIFIERS | TARGETS
    ]
    months = sorted({row["target_month"] for row in rows})
    if len(months) <= args.minimum_train_months:
        raise ValueError("시간순 검증에 필요한 월이 부족합니다")

    predictions: list[dict[str, Any]] = []
    for test_index in range(args.minimum_train_months, len(months)):
        train_months = set(months[:test_index])
        test_month = months[test_index]
        train = [row for row in rows if row["target_month"] in train_months]
        test = [row for row in rows if row["target_month"] == test_month]
        if not train or not test:
            continue
        y_active_train = np.asarray(
            [int(row["target_discussion_active"]) for row in train], dtype=np.int64
        )
        if len(np.unique(y_active_train)) < 2:
            continue
        x_train = numeric_matrix(train, feature_columns)
        x_test = numeric_matrix(test, feature_columns)
        classifier = classification_pipeline().fit(x_train, y_active_train)
        regressor = regression_pipeline().fit(
            x_train,
            np.asarray([float(row["next_month_mention_count"]) for row in train]),
        )
        active_probability = classifier.predict_proba(x_test)[:, 1]
        predicted_count = np.maximum(0.0, regressor.predict(x_test))
        for row, probability, count in zip(test, active_probability, predicted_count):
            baseline_probability = min(
                0.98, max(0.02, float(row["active_month_ratio_3m"]))
            )
            baseline_count = max(0.0, float(row["mention_mean_3m"]))
            predictions.append({
                "issue_id": row["issue_id"],
                "issue_name": row["issue_name"],
                "target_month": test_month,
                "actual_active": int(row["target_discussion_active"]),
                "logistic_probability": float(probability),
                "baseline_probability": baseline_probability,
                "actual_count": float(row["next_month_mention_count"]),
                "poisson_count": float(count),
                "baseline_count": baseline_count,
            })

    if not predictions:
        raise ValueError("시간순 검증 예측이 생성되지 않았습니다")
    y_active = np.asarray([row["actual_active"] for row in predictions])
    logistic_scores = np.asarray([row["logistic_probability"] for row in predictions])
    baseline_scores = np.asarray([row["baseline_probability"] for row in predictions])
    y_count = np.asarray([row["actual_count"] for row in predictions])
    poisson_count = np.asarray([row["poisson_count"] for row in predictions])
    baseline_count = np.asarray([row["baseline_count"] for row in predictions])

    def class_metrics(scores: np.ndarray) -> dict[str, Any]:
        labels = (scores >= args.threshold).astype(np.int64)
        return {
            "f1": float(f1_score(y_active, labels, zero_division=0)),
            "precision": float(precision_score(y_active, labels, zero_division=0)),
            "recall": float(recall_score(y_active, labels, zero_division=0)),
            "roc_auc": safe_auc(roc_auc_score, y_active, scores),
            "pr_auc": safe_auc(average_precision_score, y_active, scores),
        }

    def count_metrics(values: np.ndarray) -> dict[str, float]:
        return {
            "mae": float(mean_absolute_error(y_count, values)),
            "rmse": float(mean_squared_error(y_count, values) ** 0.5),
        }

    metrics = {
        "classification": {
            "logistic_regression": class_metrics(logistic_scores),
            "active_ratio_baseline": class_metrics(baseline_scores),
        },
        "count_regression": {
            "poisson_regression": count_metrics(poisson_count),
            "mean_3m_baseline": count_metrics(baseline_count),
        },
    }
    selected_classifier = max(
        metrics["classification"],
        key=lambda name: metrics["classification"][name]["f1"],
    )
    selected_regressor = min(
        metrics["count_regression"],
        key=lambda name: metrics["count_regression"][name]["mae"],
    )

    x_all = numeric_matrix(rows, feature_columns)
    classifier = classification_pipeline().fit(
        x_all, np.asarray([int(row["target_discussion_active"]) for row in rows])
    )
    regressor = regression_pipeline().fit(
        x_all, np.asarray([float(row["next_month_mention_count"]) for row in rows])
    )
    latest_payload = json.loads(
        args.forecast_input.expanduser().resolve().read_text(encoding="utf-8")
    )
    latest_rows = list(latest_payload.get("rows") or [])
    x_latest = numeric_matrix(latest_rows, feature_columns)
    latest_probability = classifier.predict_proba(x_latest)[:, 1]
    latest_count = np.maximum(0.0, regressor.predict(x_latest))
    forecasts = []
    for row, probability, count in zip(latest_rows, latest_probability, latest_count):
        if selected_classifier == "active_ratio_baseline":
            probability = min(0.98, max(0.02, float(row["active_month_ratio_3m"])))
        if selected_regressor == "mean_3m_baseline":
            count = max(0.0, float(row["mention_mean_3m"]))
        forecasts.append({
            "issue_id": row["issue_id"],
            "issue_name": row["issue_name"],
            "target_month": row["target_month"],
            "discussion_active_probability": round(float(probability), 6),
            "predicted_mention_count": int(round(float(count))),
            "selected_classifier": selected_classifier,
            "selected_count_model": selected_regressor,
        })

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, output_dir / "logistic_regression.joblib")
    joblib.dump(regressor, output_dir / "poisson_regression.joblib")
    with (output_dir / "time_split_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    report = {
        "status": "complete",
        "method": "expanding_window_by_target_month",
        "leakage_control": "무작위 분할 금지; 각 검증 월보다 이전 월만 학습",
        "issue_count": len({row["issue_id"] for row in rows}),
        "training_rows": len(rows),
        "validation_predictions": len(predictions),
        "feature_columns": feature_columns,
        "metrics": metrics,
        "selection": {
            "classifier": selected_classifier,
            "count_model": selected_regressor,
        },
        "forecasts": forecasts,
        "limitations": [
            "대표 이슈 수가 적어 확률 보정 전의 실험 모델임",
            "예측은 제공된 2024~2025 회의록 이력만 사용함",
            "이슈 정의와 군집 품질 변화에 민감하므로 사람 검수가 선행되어야 함",
        ],
    }
    atomic_json(output_dir / "multi_issue_forecast.json", report)

    print("\n=== 다중 이슈 예측 모델 시간순 평가 완료 ===")
    print(
        f"이슈: {report['issue_count']} | 학습 행: {len(rows):,} | "
        f"검증 예측: {len(predictions):,}"
    )
    print(
        f"선택 분류: {selected_classifier} | "
        f"F1={metrics['classification'][selected_classifier]['f1']:.4f}"
    )
    print(
        f"선택 언급량: {selected_regressor} | "
        f"MAE={metrics['count_regression'][selected_regressor]['mae']:.4f}"
    )
    print(f"result: {output_dir / 'multi_issue_forecast.json'}")


if __name__ == "__main__":
    main()

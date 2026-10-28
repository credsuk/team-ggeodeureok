# 예측 평가 기준

## 검증 방식

무작위 분할을 사용하지 않고 월 단위 expanding-window rolling backtest를 사용한다.
검증 대상 월보다 이전 시점의 행만 학습하므로 미래 정보가 학습 데이터로 섞이지 않는다.

## 필수 지표

- `MAE`: 실제 다음 달 언급량과 예측 언급량의 절대 오차 평균
- `RMSE`: 큰 예측 오차에 더 큰 패널티를 주는 제곱근 평균제곱오차
- `directional_accuracy`: 현재 월 대비 다음 달 언급량의 증가·지속·감소 방향 일치율
- `discussion_active_f1`: 다음 달 논의 지속 여부의 F1 점수
- `rolling_backtest`: 각 검증 월 이전 자료만 사용한 시간순 평가

## 전월 유지 기준선

가장 단순한 비교 대상으로 다음 달이 현재 달과 같다고 예측한다.

- 언급량: `next_count = current_count`
- 논의 지속: 현재 달 언급이 있으면 다음 달도 지속한다고 예측
- 방향: 항상 `persistence`

최종 모델은 이 기준선과 MAE, RMSE, 방향성 정확도, 논의 지속 F1을 같은 검증 구간에서 비교한다.

## 생성 파일

- `multi_issue_forecast.json`: 전체 지표, 선택 모델, 기준선 비교, 최종 예측
- `time_split_predictions.csv`: 이슈×검증 월별 실제값과 예측값
- `rolling_backtest_metrics.csv`: 검증 월별 모델 및 기준선 성능
- `logistic_regression.joblib`: 학습된 논의 지속 분류 모델
- `poisson_regression.joblib`: 학습된 언급량 회귀 모델

## 실행

```bash
bash scripts/run_forecast_evaluation.sh
```

대표 10개 이슈를 하이브리드 검색기로 다시 확정한 다음 이 명령을 실행해야 최종 제출 결과와 평가 대상이 일치한다.


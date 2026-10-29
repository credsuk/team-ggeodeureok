# 브랜치 통합 검토 기록

검토일: 2026-10-29

## 결론

공식 분석·제출 기준은 `feature/radar-hybrid-external`의
`experiments/kwonjun_chroma/`로 유지한다. 이 브랜치가
`feature/chroma-kwonjun`의 전처리·임베딩·검색·군집·예측을 포함하면서
외부 참고 기능과 지역명 정규화 수정까지 반영한 최신 후속 브랜치다.

## 브랜치 상태

| 브랜치 | 확인 결과 | 최종 활용 |
|---|---|---|
| `feature/chroma-kwonjun` | 실제 모델 파이프라인의 기준 | 모델·실험 근거 |
| `feature/radar-hybrid-external` | 위 브랜치보다 9커밋 앞선 통합본 | 최종 제출 기준 |
| `gwonwoo` | `main`과 같은 과거 커밋이며 모델 통합 전 상태 | 최종본으로 사용하지 않음 |
| `develop_kb` | 별도 pgvector 파이프라인과 정적 HTML 프런트 포함 | 디자인 참고만 활용 |

## 정적 HTML을 그대로 채택하지 않은 이유

`develop_kb/frontend/app.js`의 `MASTER_ISSUES`에는 대표 이슈, 발언자,
회의록, 예측값과 모델명이 JavaScript 상수로 작성되어 있다. 이 값은 최종
pgvector 검색·군집·Poisson 예측 산출물과 동적으로 연결되지 않고, 실제 최종
구성과 다른 `XGBoost Classifier`, `Ridge ElasticNet` 등의 표기도 포함한다.

따라서 디자인은 참고할 수 있지만 공식 결과 화면으로 제출하려면 반드시
`outputs/web/` JSON 또는 백엔드 API와 연결하고 하드코딩 결과를 제거해야 한다.
현재 제출 후보는 실제 결과 JSON과 DB를 읽는 Streamlit 화면을 사용한다.

## 규정 대응 수정

- 공식 모드에서 외부 공공데이터 자동 호출을 기본 비활성화했다.
- 2026년 행정구역 통합 규칙과 기준일 이후 KOSIS 카탈로그는 공식 release에
  복사하지 않는다.
- 외부 확장 소스는 재현성 확인용으로 유지하지만 수치 예측 입력과 분리한다.
- 전처리·AI 활용·데모·제출 체크 문서를 추가했다.


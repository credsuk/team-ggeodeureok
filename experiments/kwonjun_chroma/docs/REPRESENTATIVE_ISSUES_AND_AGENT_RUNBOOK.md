# 대표 이슈 10개 → 다중 예측 → 자유검색 Agent 실행서

## 목표

대표 이슈는 자유검색 가능한 단어 목록이 아니다. 검수 가능한 기준 세트를 먼저
확정해 검색·군집·최초 관측·전파·예측의 품질을 측정하고, 이후 같은 파이프라인을
임의 검색어에 적용한다.

대표 10개는 다음과 같다.

1. 저출생·출산율 대응 정책
2. 인구감소·지역소멸 대응
3. 청년 일자리와 지역정착 지원
4. 전세사기 피해 예방과 지원
5. 지역의료 공백과 필수의료 대응
6. 돌봄 공백과 통합돌봄 정책
7. 학교폭력 예방과 피해자 보호
8. 집중호우·침수 재난 대응
9. 기후위기와 탄소중립 전환
10. 소상공인 경영 안정 지원

## 1. 설치 및 환경 확인

```bash
cd ~/projects/council-vector-benchmark
source .venv/bin/activate
pip install -r requirements-forecast.txt

docker compose up -d pgvector
set -a
source .env
set +a
```

## 2. 대표 이슈 후보 생성

```bash
bash scripts/run_representative_issues.sh candidates
```

기존 후보가 있으면 재사용하고 없는 이슈만 실행한다. 각 이슈는 다음 순서를 거친다.

```text
pgvector 50개 후보 → BGE 재순위 10개 → 500개 확장
→ 동일 부모·동일 원문 제거 → cosine graph 군집(0.68)
→ 사람 검수 대기열
```

## 3. 화면 검수

```bash
COUNCIL_ISSUE_REVIEW_QUEUE=outputs/issues/candidates_v1/issue_review_queue.csv \
  bash scripts/run_issue_review.sh
```

브라우저에서 `http://localhost:8502`를 연다. 이슈별 대표 근거와 경계 근거를 확인한
뒤 `approve`, `revise`, `reject` 중 하나를 기록한다.

검수 기준은 다음 네 가지다.

- 상위 근거 10개 중 다른 주제 발언이 과도하게 섞이지 않았는가?
- 최소 2개 이상의 의회와 15개 이상의 고유 근거가 있는가?
- 한 의회가 군집의 80% 이상을 독점하지 않는가?
- 최초 관측 원문이 실제 이슈의 본격적 정책 논의인가?

## 4. 승인 이슈 최종화

```bash
bash scripts/run_representative_issues.sh finalize
```

승인된 이슈만 최신 중복 제거 코드로 다시 군집화한 뒤 다음 결과를 만든다.

- 고정 `issue_id`
- 최초 관측 날짜와 의회
- 다른 의회의 시간적 후속 관측 순서
- 월별 언급량과 확산 특징
- 단일 이슈 시간순 기준선 예측
- 지도 웹 JSON

`전파`는 회의록에 나타난 시간적 순서이며 인과관계를 뜻하지 않는다.

## 5. 다중 이슈 예측 심화

승인·최종화 이슈가 5개 이상, 권장 10개 모이면 실행한다.

```bash
bash scripts/run_representative_issues.sh panel
```

이 단계는 모든 이슈의 월별 특징을 하나의 패널로 합치고 다음을 비교한다.

- 다음 달 논의 지속: 최근 활동 비율 기준선 vs Logistic Regression
- 다음 달 언급량: 최근 3개월 평균 기준선 vs Poisson Regression
- 평가 방식: 무작위 분할이 아닌 월별 expanding-window 검증
- 선택 기준: 분류 F1, 언급량 MAE

모델이 기준선보다 낮으면 모델을 억지로 채택하지 않고 기준선을 유지한다.

## 6. 임의 검색어 전체 분석

```bash
python -u src/33_analyze_free_query.py \
  --query "전기차 지하주차장 화재 안전" \
  --device cuda
```

결과는 `outputs/dynamic_issues/query_*/05_dynamic_result.json`에 저장된다. 고정 이슈가
아니므로 registry를 변경하지 않고 `provisional_until_review` 상태로 표시한다.
달력월 6개월, 관측월 4개월, 의회 2개 미만이면 예측을 생략하고 근거 부족을 반환한다.

## 7. Agent 원칙

- 검수 완료 이슈와 질문이 일치하면 고정 결과를 재사용한다.
- 일치하지 않으면 전체 pgvector에서 원문 근거를 검색한다.
- 자유검색 전체 분석은 `33_analyze_free_query.py`를 실행한 결과만 사용한다.
- 수치와 날짜는 코드가 계산하고, 로컬 Qwen은 원문 근거를 인용해 설명만 한다.
- 회의록은 localhost Ollama 밖으로 전송하지 않는다.
- 제공 기간 이후 외부정보는 이슈 탐지와 예측에 사용하지 않는다.

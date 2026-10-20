# 18단계: Chroma 검색과 이슈·전파·예측 연결

## 구조

```text
사용자 질의
  → Arctic 질의 임베딩
  → Chroma 정책 청크 Top 50
  → chunk_id로 군집 배정 연결
  → 검수 완료 issue_id별 점수 집계
  → 최초 등장·타 의회 전파 타임라인
  → 다음 달 논의·신규 의회 확산 예측
```

기존 10번은 질의와 군집 중심 벡터를 직접 비교했습니다. 18번은 실제 관련 발언을 먼저 검색한 뒤 그 발언의 고정 `chunk_id`를 군집에 연결합니다. 따라서 최종 결과에 “왜 이 이슈가 선택됐는지”를 보여주는 검색 근거 문장이 포함됩니다.

## 1. 보정 메타데이터로 타임라인 다시 생성

임베딩과 군집화는 다시 하지 않습니다. 09번만 다시 실행해 타임라인 근거의 안건 표시를 보정합니다.

```bash
cd ~/projects/council-minutes
source .venv/bin/activate

python src/09_build_issue_timeline.py \
  --evidence-metadata data/index/arctic_ko_semantic_test_v2/chunk_metadata_enriched.parquet
```

정상 완료 기준:

```text
완료: 이슈 N개, 관측 N건
타임라인: outputs/issue_timeline_final/issue_timelines.json
```

이 단계에서는 이슈별 월별 관측 개수가 변하지 않으므로 11번 예측을 다시 학습할 필요는 없습니다. `issue_id`가 달라졌거나 09번의 포함 기준을 바꾼 경우에만 11번을 다시 실행합니다.

## 2. 통합 검색

```bash
python src/18_chroma_issue_query.py \
  --query "출산율 저하 대응 정책" \
  --candidate-chunks 50 \
  --top-k-issues 3 \
  --device cuda
```

또는 실행 스크립트를 사용합니다.

```bash
bash scripts/run_issue_query.sh "출산율 저하 대응 정책"
```

결과 파일:

```text
outputs/chroma_issue_search/출산율_저하_대응_정책.json
```

## 점수 계산

각 검색 청크의 기여도는 다음 요소를 사용합니다.

```text
검색 유사도 × 군집 소속 유사도 × 검색 순위 할인
```

같은 이슈의 청크가 여러 개 검색되면 최대 5개 근거의 평균과 지지 청크 수 보너스를 반영합니다. 단순히 한 문장이 우연히 유사한 이슈보다 서로 다른 관련 발언이 반복 검색된 이슈가 더 높은 점수를 받습니다.

다음 후보는 최종 결과에서 제외합니다.

- 군집 배정과 연결되지 않은 청크
- `cluster_similarity < 0.55`
- 사람이 검수한 타임라인에 포함되지 않은 군집

## 출력에서 확인할 항목

- `matched_reviewed_chunk_count`: 검색 청크 중 검수 이슈에 연결된 수
- `matched_issue_count`: 연결된 서로 다른 이슈 수
- 1위 이슈가 출산·저출생·인구감소 계열인지
- `query_evidence`: 실제 검색된 회의록 근거
- `first_observed_date`, `origin_councils`: 최초 관측
- `propagation`: 타 의회 관측 순서
- `forecast`: 다음 달 언급 방향과 신규 의회 확산 가능성

전파 순서는 제공 데이터에서 관측된 시간 순서이며 실제 정책의 인과관계를 단정하지 않습니다.

## 필터 검색

경기도의회 발언으로 이슈를 찾아 전체 전파 타임라인을 확인할 수 있습니다.

```bash
python src/18_chroma_issue_query.py \
  --query "출산율 저하 대응 정책" \
  --council "경기도의회" \
  --year 2024 \
  --device cuda
```

필터는 이슈를 찾는 초기 Chroma 후보에만 적용됩니다. 선택된 이슈의 전파 타임라인은 모든 의회를 보여줍니다.


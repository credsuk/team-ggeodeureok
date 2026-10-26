# 의정이슈레이더 중간제출 결과

팀 꺼드럭's는 2024~2025년 광역 지방의회 회의록에서 정책 이슈의 최초 관측,
타 의회 후속 관측 순서, 월별 변화와 다음 달 전망을 근거 원문과 함께 제공하는
`의정이슈레이더`를 개발한다.

## 1. 현재 완성 범위

| 단계 | 현재 결과 |
|---|---|
| Canonical 데이터 | 2,057,214개 발화와 LLM 메타데이터를 `row_id`로 결합 |
| 부모·자식 청킹 | 검색 청크 901,358개, 원문 위치와 부모 문맥 보존 |
| 검색 임베딩 | `dragonkue/snowflake-arctic-embed-l-v2.0-ko`, 1024차원, 전체 검증 PASS |
| 벡터 검색 | PostgreSQL 17 + pgvector 0.8.7 + HNSW cosine |
| 이슈 분석 | 검색·재순위·확장·군집화·이슈 ID 고정 |
| 흐름 분석 | 최초 관측일, 의회별 후속 관측 순서, 월별 추이 |
| 예측 | 시간순 백테스트 기반 다음 달 언급량·방향·경험적 구간 |
| 웹 | 지도, 타임라인, 예측, 원문 근거, 전체 검색, 로컬 Agent |
| 외부지표 | 기준일 검증 후 참고 화면에만 표시, 공식 예측에서는 제외 |

## 2. 검증된 주요 선택

- Raw 대 Enriched 검색: 통계적으로 우열을 확정할 수 없어 더 단순하고 빠른 Raw를 채택했다.
  - Raw nDCG@10 0.8928, P@10 0.9750, Recall@20 0.5826
  - Enriched nDCG@10 0.8871, P@10 0.9667, Recall@20 0.5744
- Arctic 대 KURE 빠른 비교에서 Arctic을 채택했다.
  - Arctic nDCG@10 0.8969, 문서 임베딩 58.95행/초
  - KURE nDCG@10 0.8657, 문서 임베딩 56.25행/초
- 전체 임베딩은 901,358행 × 1024차원이며 NaN/Inf 0, ID·행 순서 검증을 통과했다.
- 전체 DB 비교에서 필터 검색 성능과 정확도를 종합해 pgvector를 서비스 DB로 채택했다.
  - pgvector 필터 검색 p50 3.97ms, p95 8.18ms, Recall@10 1.000
  - Chroma 필터 검색 p50 586.29ms, p95 861.08ms, Recall@10 1.000
  - Chroma는 재현용 비교 기준선으로 보존한다.

## 3. 중간본 신규 기능

### 통합 대시보드

`demo/midterm_app.py`는 다음 다섯 화면을 제공한다.

1. 이슈 레이더: 대한민국 지도, 의회별 관측 순서, 월별 변화, 다음 달 전망
2. 원문 근거 탐색: 전체 pgvector에서 필터 검색하고 부모 문맥 복원
3. 로컬 AI Agent: 검색 근거에 제한된 질의응답과 `[근거 n]` 인용
4. 외부지표 참고: 출처·기준일 검증을 통과한 공개지표의 분리 표시
5. 방법·재현성: 처리 단계, 한계, 결과 JSON 다운로드

### 로컬 Agent 보안 규칙

- 기본 모델은 Ollama의 `qwen3:8b`이다.
- localhost 이외의 Ollama 주소는 코드가 거부한다.
- 회의록은 상용 LLM API로 전송하지 않는다.
- 검색 근거에 없는 질문은 추측하지 않고 근거 부족으로 답한다.
- 최초 관측 순서는 인과적 전파로 표현하지 않는다.
- Ollama가 꺼져 있어도 구조화 결과 요약 모드로 데모가 중단되지 않는다.

### 외부지표 사용 규칙

- `config/external_context_template.csv`에 공개지표와 출처를 작성한다.
- `src/29_validate_external_context.py`가 2025-12-31 이후 관측과 기준일을 차단한다.
- `usage_scope=reference_only`만 허용한다.
- 외부지표는 이슈 탐지, 군집 구성, 최초 관측 계산, 2026-01 예측에 사용하지 않는다.
- 최종 제출 전에 주최 측에 외부 공개지표 사용 가능 범위를 다시 확인한다.

## 4. 실행

```bash
cd ~/projects/council-vector-benchmark
source .venv/bin/activate

pip install -r requirements-midterm.txt
cp -n .env.example .env

docker compose up -d pgvector
ollama serve
ollama pull qwen3:8b

python scripts/check_midterm_readiness.py
bash scripts/run_midterm_demo.sh
```

브라우저에서 `http://localhost:8501`을 연다. WSL이 브라우저를 자동으로 열지
못해도 서버가 실패한 것은 아니다.

외부 참고지표를 넣을 때만 다음을 추가로 실행한다.

```bash
mkdir -p data/external
cp config/external_context_template.csv data/external/context.csv
# context.csv에 출처가 확인된 2024~2025 지표를 작성
python src/29_validate_external_context.py \
  --input data/external/context.csv \
  --output outputs/external/external_context.json
```

## 5. 이슈 결과를 갱신할 때

`src/20_expand_cluster_issue.py`는 이제 부모 중복뿐 아니라 공백만 다른 동일 원문도
제거한다. 기존 출산율 결과 JSON에는 수정 전 중복이 남아 있을 수 있으므로 최종
제출 전 20~24번을 다시 실행한다. 고정 이슈 구성원이 달라지면 기존 registry를
억지로 덮어쓰지 말고 새 이슈 ID로 검수한다.

## 6. 중간제출 Git 범위

Git에는 코드, 설정, 문서, 작은 결과 예시만 올린다. 다음 대용량 파일은 올리지 않는다.

- 원본·정제 회의록
- Parquet canonical·청크
- 901,358개 임베딩 NPY
- Chroma 저장소
- PostgreSQL 데이터 디렉터리
- 모델 캐시

실제 결과 JSON은 비식별·저용량 샘플만 `examples/`에 복사하고, 전체 실행 산출물은
`.gitignore`의 `outputs/` 규칙으로 제외한다.

## 7. 다음 심화 순서

1. 대표 이슈 10개 이상을 같은 임계값 0.68로 생성하고 경계 사례를 검수한다.
2. 다중 이슈 월별 패널을 만들어 단일 이슈 예측을 로지스틱 회귀·LightGBM과 비교한다.
3. 검색 후보에 한국어 cross-encoder 재순위를 추가하고 nDCG·Recall 변화를 측정한다.
4. Agent 답변의 인용 정확도, 근거 충실도, 무근거 응답률을 별도 평가한다.
5. 팀원 LLM 메타데이터 결과는 공통 골드셋으로 비교하되 현재 파이프라인은 동결한다.

중간본의 핵심은 모델 이름이 아니라, 모든 수치와 답변을 원문 근거까지 역추적할 수
있는 재현 가능한 분석 흐름이다.

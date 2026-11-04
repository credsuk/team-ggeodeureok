# data/

## 포함된 파일

| 경로 | 내용 | 생성 위치 |
|---|---|---|
| `db/council_vectors.dump` (+ `.sha256`, `.info.txt`) | 구축 완료된 PostgreSQL+pgvector DB 덤프 — `scripts/restore_db.ps1`로 복원 | 노트북 03 / `src/16_load_pgvector.py` 결과 |
| `forecast/members_*.json` | 대표 이슈 10개의 구성 청크(이슈 군집 결과) | 노트북 07 · 8-1 (노트북 04 비교 패널 입력) |
| `forecast/multi_issue_forecast.json` | 다중 이슈 다음 달 예측 결과 | 노트북 07 · 8-4 |
| `llm_compare/evidence_top6.json` | LLM 비교용 질문별 회의록 근거 6건 | 노트북 06 · 4 |
| `llm_compare/results.jsonl` | 모델 6종 × 질문 8개 답변·자동 채점 결과 | 노트북 06 · 6 |
| `llm_compare/report.md` | 모델별 요약 표 | 노트북 06 · 7 |
| `llm_compare/answers.md` | 모델별 답변 원문 | 노트북 06 · 8 |

## 포함되지 않은 파일 (직접 배치)

**원본 회의록**은 경진대회 제공 데이터이고 용량이 커서(약 2.7GB) 제출물에 넣지 않았습니다. 다음 위치에 그대로 복사합니다.

```
data/raw/2024_광역회의록/2024_강원특별자치도의회.json ... (18개)
data/raw/2025_광역회의록/2025_강원특별자치도의회.json ... (18개)
```

> `*_전남광주통합특별시의회.json`은 제공 시점에 0바이트 파일이며 처리 과정에서 자동으로 건너뜁니다.

## 실행 중 생성되는 폴더

| 경로 | 내용 | 만드는 곳 |
|---|---|---|
| `sample/`, `llm_rules_outputs/` | HTML 샘플, 의회별 파싱 규칙 | 노트북 01 |
| `to_text/연도_광역 회의록/` | 정제 회의록 TSV | 노트북 01 |
| `split/`, `metadata_jsonl/` | 배치 분할 TSV, 정책 메타데이터 JSONL | 노트북 02 |
| `canonical/`, `embeddings/`, `reports/` | 발화·청크 parquet, 임베딩 벡터, 검증 보고서 | 노트북 03 |
| `external/cache/` | 외부 공공데이터 응답 캐시 | 노트북 07, `src/main.py` |
| `canonical_v3/`, `bm25/`, `input/` | 원본 파이프라인 스크립트(`src/01~35`) 산출물 | `src/*.py` |

PostgreSQL 데이터는 Docker 볼륨 `council_pgdata`에 저장됩니다. (복원 후 약 12GB)

> **공개 저장소 안내**: 이 GitHub 브랜치에는 회의록 원문 발췌가 들어 있는 `llm_compare/evidence_top6.json`·`answers.md`·`results.jsonl`과
> 노트북 실행 출력, DB 덤프(`db/council_vectors.dump`)를 올리지 않았습니다. 이 파일들은 심사 제출본에만 포함됩니다.

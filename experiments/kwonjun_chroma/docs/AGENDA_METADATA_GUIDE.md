# 17단계: 안건 메타데이터 보정

## 보정 이유

Chroma 검색 자체는 정상입니다. 다만 다음과 같이 정책 발언에 절차 안건이 표시되는 사례가 발견됐습니다.

```text
원문 표시 안건: 3. 휴회의 건(의장 제의)
실제 검색 본문: 인구 감소·출산율 대응 정책
```

이는 임베딩 오류가 아니라 청킹 과정에서 유지된 `agenda_title`과 semantic 자식 청크의 정책 내용이 맞지 않는 메타데이터 문제입니다. 잘못된 안건명을 근거 제목으로 제시하면 심사의 “최초 등장 및 전파 경로 추적 정밀도”에 불리하므로 다음 필드를 분리합니다.

| 필드 | 의미 |
|---|---|
| `agenda_title_raw` | 기존 파서가 추출한 원본 안건명 |
| `agenda_title_effective` | 실제 화면과 분석에 사용할 보정 안건명 |
| `agenda_confidence` | `high`, `medium`, `low`, `none` |
| `agenda_reason` | 보정 판단 사유 |
| `evidence_title` | 유효 안건이 없으면 회의록명으로 대체한 근거 제목 |

임베딩 벡터와 `semantic_text`, `chunk_id`, 행 순서는 바꾸지 않습니다. 따라서 임베딩과 군집화를 다시 수행할 필요가 없습니다.

## 1. 버전 경고 해결

```bash
cd ~/projects/council-minutes
source .venv/bin/activate
pip install --upgrade "sentence-transformers==3.4.1"
```

## 2. Parquet 보정 및 기존 Chroma 갱신

```bash
python src/17_enrich_agenda_metadata.py \
  --input data/index/arctic_ko_semantic_test_v2/chunk_metadata.parquet \
  --output data/index/arctic_ko_semantic_test_v2/chunk_metadata_enriched.parquet \
  --update-chroma
```

`--update-chroma`는 기존 25,340개 벡터를 재구축하지 않고 메타데이터만 갱신합니다.

생성 파일:

```text
data/index/arctic_ko_semantic_test_v2/chunk_metadata_enriched.parquet
outputs/agenda_metadata/agenda_metadata_summary.json
outputs/agenda_metadata/agenda_metadata_samples.csv
```

샘플 검수:

```bash
python - <<'PY'
import pandas as pd

p = "outputs/agenda_metadata/agenda_metadata_samples.csv"
d = pd.read_csv(p, encoding="utf-8-sig")
print(d[[
    "agenda_title_raw", "agenda_title_effective",
    "agenda_confidence", "agenda_reason", "semantic_preview"
]].head(20).to_string(index=False))
PY
```

## 3. 출산율 검색 재확인

```bash
python src/16_chroma_search.py \
  --query "출산율 저하 대응 정책" \
  --top-k 10 \
  --device cuda \
  --compare-faiss \
  --metadata data/index/arctic_ko_semantic_test_v2/chunk_metadata_enriched.parquet \
  --output outputs/chroma_search/출산율_보정.json
```

첫 번째 검색 결과가 기존과 같은 정책 문장을 유지하면서 화면에는 `휴회의 건` 대신 회의록명 또는 신뢰 가능한 정책 안건명이 표시되어야 합니다.

## 보정 규칙

- semantic 단계에서 검증된 `semantic_topic_hint`가 있으면 `high`
- 조례안·정책·사업·지원·대책 등 명확한 정책 안건은 `high`
- 부서 소관 업무보고처럼 넓은 안건은 `medium`
- 여러 안건이 한 청크에 섞이면 `low`로 표시하고 화면에서는 회의록명 사용
- 휴회·회기·서명의원·의사일정 등 절차 안건은 `none`으로 제거
- 유효 안건이 없으면 `meeting_name`을 `evidence_title`로 사용

최종 산출물에는 `agenda_title_raw`를 숨기지 않고 보존하므로 보정 과정을 재현하고 감사할 수 있습니다.


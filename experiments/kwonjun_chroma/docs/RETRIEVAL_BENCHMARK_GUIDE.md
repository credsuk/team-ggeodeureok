# 임베딩 검색 성능향상 실행 가이드

## 목적

기존 Arctic-ko 임베딩을 기준선으로 보존하고, 동일한 정책 의미 청크에서
KURE-v1과 BGE-M3를 비교합니다. 모델마다 검색한 후보의 합집합을 사람이
라벨링하므로 특정 모델이 만든 군집을 정답으로 재사용하지 않습니다.

## 1. 평가 질의 생성

```bash
cd ~/projects/council-minutes
source .venv/bin/activate

python src/12_make_retrieval_evalset.py \
  --output-dir outputs/retrieval_eval
```

기본 질의는 64개이며 `dev 48개 + test 16개`입니다. dev는 모델 선택에,
test는 최종 확인에 사용합니다.

## 2. 세 모델 임베딩·후보 합집합 생성

```bash
python src/13_embedding_benchmark.py \
  --mode prepare \
  --metadata data/index/arctic_ko_semantic_test_v2/chunk_metadata.parquet \
  --reuse-arctic-embedding data/index/arctic_ko_semantic_test_v2/embeddings.f16.npy \
  --models arctic,kure,bge-m3 \
  --pool-k 8 \
  --eval-k 5 \
  --batch-size 8 \
  --device cuda
```

- Arctic 벡터는 기존 파일을 그대로 사용합니다.
- KURE와 BGE-M3만 새로 임베딩하며 중단 시 같은 명령으로 재개됩니다.
- 첫 실행은 Hugging Face에서 모델 파일만 다운로드합니다.
- 회의록 텍스트는 외부 API로 전송하지 않습니다.

## 3. 관련도 검수

다음 파일을 엽니다.

```text
outputs/retrieval_eval/retrieval_judgments_union.csv
```

`label` 열에 다음 숫자를 입력합니다.

- `2`: 질의 정책을 직접 논의
- `1`: 같은 분야이지만 부분적으로만 관련
- `0`: 실질적으로 무관

모델 편향을 줄이려면 검수 중 `source_models`, `source_ranks`,
`similarity_scores` 열은 숨깁니다. 파일은 UTF-8 CSV로 저장합니다.

## 4. 최종 성능 계산

```bash
python src/13_embedding_benchmark.py \
  --mode evaluate \
  --metadata data/index/arctic_ko_semantic_test_v2/chunk_metadata.parquet \
  --reuse-arctic-embedding data/index/arctic_ko_semantic_test_v2/embeddings.f16.npy \
  --models arctic,kure,bge-m3 \
  --pool-k 8 \
  --eval-k 5 \
  --batch-size 8 \
  --device cuda
```

주요 결과:

```text
outputs/retrieval_eval/embedding_benchmark_summary.json
outputs/retrieval_eval/embedding_benchmark_summary.csv
outputs/retrieval_eval/embedding_benchmark_per_query.csv
```

1차 비교는 라벨링 양을 줄이기 위해 `nDCG@5`를 1순위로 사용하고, 동률이면
`Recall@5`, `MRR@5` 순서로 선택합니다. 상위 두 모델만 남긴 뒤 필요하면
`--pool-k 15 --eval-k 10 --overwrite-pool`로 후보를 확장하고 새로 추가된 행만
라벨링합니다. 기존 label은 `query_id + chunk_id` 기준으로 보존됩니다. test 점수는
최종 설정을 확정한 뒤 한 번만 해석합니다.

## 다음 단계

우승 임베딩을 결정한 뒤에만 다음을 진행합니다.

1. BM25와 Dense 검색 결과를 RRF로 결합
2. Top 50을 `BAAI/bge-reranker-v2-m3`로 재순위화
3. 동일 평가셋에서 임베딩 단독 대비 성능 향상 확인
4. 성능이 실제로 향상된 구성만 전체 438,708개 청크에 적용
5. 선택 벡터를 ChromaDB에 적재하고 FAISS 정확 검색을 검증 기준선으로 유지

# Dense + BM25 + RRF + BGE 실행서

이 변경은 canonical, 청킹, Arctic 전체 임베딩과 pgvector를 그대로 재사용한다.
새로 만드는 것은 `chunk_metadata.parquet` 기반의 BM25 문자 인덱스뿐이다.

## 최종 검색 순서

```text
Arctic Dense Top-100 (pgvector HNSW)
+ SQLite FTS5 BM25 Top-100
→ RRF Top-50
→ BGE Cross-Encoder Top-10
→ 군집화·타임라인·예측
```

## 1. 코드 교체 후 환경 확인

```bash
cd ~/projects/council-vector-benchmark
source .venv/bin/activate

python -c "import sqlite3; print(sqlite3.sqlite_version)"
python -c "import pyarrow; print(pyarrow.__version__)"
```

별도 BM25 패키지는 필요 없다. Python 표준 SQLite의 FTS5 `bm25()`를 사용한다.

## 2. 작은 색인 시험

```bash
python -u src/35_build_bm25_index.py \
  --limit 10000 \
  --output data/bm25/arctic_raw_v1_test/bm25.sqlite3 \
  --rebuild
```

## 3. 전체 901,358개 색인

```bash
bash scripts/build_bm25_hybrid.sh
```

기존 인덱스를 교체할 때만:

```bash
bash scripts/build_bm25_hybrid.sh --rebuild
```

## 4. 하이브리드 검색 시험

```bash
python -u src/18_pgvector_retrieval.py \
  --query "출산율 저하 대응 정책" \
  --candidate-k 50 \
  --ef-search 200 \
  --device cuda \
  --output outputs/retrieval/hybrid_birth_rate.json
```

JSON의 `retrieval`에 Dense/BM25 후보 수, BM25 토큰, RRF 설정이 기록된다.

전남 별칭과 위원회 개편 전후 필터 시험:

```bash
python -u src/18_pgvector_retrieval.py \
  --query "전남 행정자치위원회 청년 정책" \
  --council "전라남도" \
  --committee "행정자치위원회" \
  --output outputs/retrieval/jeonnam_alias_test.json
```

이 질의에서 `전남·전라남도·전라남도의회` 표기를 함께 검색하고,
`행정자치위원회·기획행정위원회` 필터를 함께 참조한다.

## 5. BGE 재순위

```bash
python -u src/19_rerank_candidates.py \
  --input outputs/retrieval/hybrid_birth_rate.json \
  --output outputs/retrieval/hybrid_birth_rate_reranked.json \
  --device cuda
```

## 6. 웹 실행

`.env`에 아래 값을 추가한다.

```dotenv
COUNCIL_RETRIEVAL_MODE=hybrid
COUNCIL_BM25_INDEX=data/bm25/arctic_raw_v1/bm25.sqlite3
COUNCIL_RRF_K=60
COUNCIL_DENSE_WEIGHT=1.0
COUNCIL_BM25_WEIGHT=1.0
```

```bash
bash scripts/run_final_app.sh
```

## Dense 기준선 재현

```bash
python -u src/18_pgvector_retrieval.py \
  --retrieval-mode dense \
  --query "출산율 저하 대응 정책" \
  --output outputs/retrieval/dense_birth_rate.json
```

Dense 기준선과 하이브리드 결과를 동일 질의·동일 0/1/2 관련도 라벨로 비교한다.
우선 비교 지표는 Recall@50, nDCG@10, Precision@10 및 p95 지연시간이다.


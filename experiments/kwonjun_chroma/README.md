# 광역 지방의회 벡터 DB 비교 프로젝트

동일한 회의록 청크, metadata, 임베딩을 사용하여 **ChromaDB**와 **PostgreSQL+pgvector**의 검색 정확도·속도·운영성을 비교한다. 기존 FAISS 파이프라인은 삭제하지 않고 exact-search 기준선으로 보존한다.

## 이번 재시작의 핵심

1. 팀 공용 데이터의 열과 품질을 먼저 검사한다.
2. 고정 `chunk_id`를 가진 canonical corpus를 한 번만 만든다.
3. 임베딩도 한 번만 생성해 두 DB가 그대로 공유한다.
4. Chroma와 pgvector에 동일 데이터를 적재한다.
5. 동일 질의로 정확도, 지연시간, 필터 및 자원 사용량을 비교한다.

## 권장 환경

- Ubuntu 24.04 또는 WSL2 Ubuntu 24.04
- Python 3.11
- NVIDIA GPU 및 NVMe SSD
- WSL 사용 시 프로젝트와 대용량 데이터는 `/mnt/c`가 아닌 `~/projects`에 저장
- 임베딩은 가장 빠른 GPU 컴퓨터에서 1회만 수행
- DB 비교는 Chroma와 pgvector를 반드시 같은 컴퓨터에서 수행

## 폴더 구성

```text
config/                     데이터 계약
data/input/                  팀 공용 입력 데이터(Git 제외)
data/canonical/              통일 corpus와 임베딩(Git 제외)
docs/                        기준선과 비교 규칙
outputs/                     검사·벤치마크 결과(Git 제외)
scripts/                     환경 점검
src/                         데이터 및 DB 처리 코드
```

## 1단계: 새 환경 만들기

```bash
cd ~/projects/council-vector-benchmark
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-base.txt
python scripts/00_check_environment.py
```

PyTorch는 사용 컴퓨터의 CUDA에 맞는 명령을 PyTorch 공식 설치 선택기에서 확인해 별도로 설치한다. 설치 뒤 아래 값이 `True`인지 확인한다.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2단계: 팀 공용 데이터 배치

팀에서 통일할 Parquet/CSV/JSONL 파일을 `data/input/`에 복사한다. 원문 데이터와 임베딩은 GitHub에 올리지 않는다.

파일명이 `team_chunks.parquet`라고 가정하면 다음처럼 실행한다.

```bash
python src/01_inspect_unified_data.py \
  --input data/input/team_chunks.parquet \
  --output-dir outputs/inventory
```

`outputs/inventory/summary.json`과 `schema_quality.csv`에는 원문 문장이 포함되지 않는다. 두 보고서를 비교하여 열 매핑을 확정한 후 canonical corpus를 생성한다.

## 주의

- PostgreSQL만으로 비교하는 것이 아니라 벡터 확장인 **pgvector**를 사용한다.
- Chroma용과 pgvector용 임베딩을 따로 만들지 않는다.
- 팀원의 LLM metadata도 `chunk_id`가 일치하는 것만 결합한다.
- 회의록 원문이나 샘플 문장은 외부 LLM/API로 전송하지 않는다.


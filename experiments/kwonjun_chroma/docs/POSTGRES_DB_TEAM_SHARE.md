# PostgreSQL + pgvector 팀 공유 방법

## 왜 `data/postgres`를 그대로 압축하지 않나

`data/postgres`는 실행 중인 PostgreSQL의 내부 파일과 소유권을 포함합니다. 이를 복사하면 PostgreSQL 버전, Linux UID, 파일 권한, 비정상 종료 상태 때문에 다른 컴퓨터에서 열리지 않을 수 있습니다. 프로젝트에서는 PostgreSQL 표준 백업인 `pg_dump` custom archive를 공유합니다.

## 1. 공유자가 dump 만들기

프로젝트 루트에서 실행합니다.

```bash
cd ~/projects/council-vector-benchmark
source .venv/bin/activate
bash scripts/export_pgvector_for_share.sh
```

`exports/`에 세 파일이 생성됩니다.

```text
council_vectors_YYYYMMDD_HHMMSS.dump
council_vectors_YYYYMMDD_HHMMSS.dump.sha256
council_vectors_YYYYMMDD_HHMMSS.dump.info.txt
```

덤프에는 벡터, 청크 metadata, 검색 테이블과 인덱스 정의가 포함됩니다. 실제 HNSW 인덱스 파일을 복사하는 대신 팀원 컴퓨터에서 복원하면서 인덱스를 다시 만듭니다.

## 2. Google Drive 공유 설정

1. Google Drive에 팀 전용 폴더를 만듭니다.
2. 위의 세 파일을 함께 업로드합니다.
3. 공유 범위는 `제한됨`으로 유지합니다.
4. 대회 참가 팀원의 Google 계정만 추가합니다.
5. `링크가 있는 모든 사용자` 또는 공개 공유는 사용하지 않습니다.

DB에는 대회 회의록에서 파생된 본문과 metadata가 들어 있으므로 경진대회 팀 외부에 공유하면 안 됩니다. `.env`, API 키, PostgreSQL 비밀번호는 업로드하지 않습니다.

## 3. 팀원이 복원하기

세 파일을 같은 폴더에 다운로드한 뒤 실행합니다.

```bash
cd ~/projects/council-vector-benchmark
cp .env.example .env  # 기존 .env가 없을 때만
nano .env

bash scripts/restore_pgvector_from_dump.sh \
  ~/Downloads/council_vectors_YYYYMMDD_HHMMSS.dump
```

이미 같은 테이블이 있고 해당 DB를 교체하려는 경우에만 다음을 실행합니다.

```bash
bash scripts/restore_pgvector_from_dump.sh \
  ~/Downloads/council_vectors_YYYYMMDD_HHMMSS.dump \
  --replace
```

`--replace`는 기존 DB 객체를 제거하고 dump 내용으로 교체하므로 필요한 경우 기존 DB를 먼저 백업합니다.

## 4. 복원 확인

```bash
docker compose exec pgvector \
  psql -U council -d council_vectors \
  -c "SELECT count(*) FROM council_arctic_raw_v1;"

docker compose exec pgvector \
  psql -U council -d council_vectors \
  -c "SELECT extversion FROM pg_extension WHERE extname='vector';"
```

기대 행 수는 `901,358`개입니다. 이후 다음 명령으로 웹을 실행합니다.

```bash
bash scripts/run_final_app.sh
```

## 5. GitHub에 포함하지 않는 파일

```text
.env
exports/
data/postgres/
*.dump
*.dump.sha256
*.dump.info.txt
```

GitHub에는 코드, `.env.example`, 실행 문서만 올립니다. DB dump는 제한된 Google Drive 폴더로만 전달합니다.

# 최종 코드를 팀 GitHub 브랜치에 올리는 방법

## 1. Git에 올릴 깨끗한 release 생성

원본 프로젝트에서 실행합니다. 기존 release는 timestamp가 붙은 백업 폴더로 이동합니다.

```bash
cd ~/projects/council-vector-benchmark
source .venv/bin/activate

python scripts/41_prepare_release.py \
  --output ~/projects/council-issue-radar-release \
  --replace
```

release에는 코드·문서·비민감 평가 요약만 들어갑니다. 원문, 벡터, PostgreSQL 파일, `.env`, API 키는 제외됩니다.

## 2. 팀 저장소와 브랜치 확인

```bash
cd ~/projects/team-ggeodeureok
git fetch --all --prune
git switch feature/chroma-kwonjun

git branch --show-current
git remote -v
git status --short
```

작업 중인 변경이 표시되면 복사 전에 커밋하거나 별도 폴더로 백업합니다.

## 3. 내 실험 폴더만 최종 release로 교체

다음 `--delete`는 팀 저장소 전체가 아니라 `experiments/kwonjun_chroma/` 내부만 release와 같게 맞춥니다.

```bash
mkdir -p experiments/kwonjun_chroma

rsync -av --delete \
  ~/projects/council-issue-radar-release/ \
  experiments/kwonjun_chroma/
```

## 4. 민감정보와 대용량 파일 검사

```bash
git status --short
git diff --check

find experiments/kwonjun_chroma -type f -size +25M -print

find experiments/kwonjun_chroma -type f \
  \( -name '.env' -o -name '*.dump' -o -name '*.npy' \
     -o -name '*.parquet' -o -name '*.sqlite3' -o -name '*.faiss' \) \
  -print
```

두 `find` 명령에서 아무것도 나오지 않아야 합니다.

## 5. 커밋

```bash
git add experiments/kwonjun_chroma
git diff --cached --check
git diff --cached --stat

git commit -m "feat: finalize council issue radar pipeline"
```

## 6. 팀 GitHub 브랜치로 push

`origin`이 개인 fork이고 `upstream`이 팀 저장소라면 팀원들이 바로 볼 수 있게 `upstream`으로 push합니다.

```bash
git push -u upstream feature/chroma-kwonjun
```

권한 오류가 나면 개인 fork에 push한 뒤 팀 저장소로 Pull Request를 만듭니다.

```bash
git push -u origin feature/chroma-kwonjun
```

중요한 점은 `main`에 직접 push하는 것이 아니라 `feature/chroma-kwonjun` 브랜치에 올리는 것입니다.

## 7. 업로드 확인

```bash
git status
git log -1 --oneline
git ls-remote --heads upstream feature/chroma-kwonjun
```

마지막 명령에 커밋 해시와 브랜치가 표시되면 팀 저장소 업로드가 완료된 것입니다.

# 팀 GitHub 반영 가이드

대상 저장소: <https://github.com/credsuk/team-ggeodeureok>  
권장 브랜치: `feature/chroma-retrieval-kwonjun`

팀 저장소의 기존 `src/`, `scripts/`를 건드리지 않도록 이번 작업은
`experiments/kwonjun_chroma/` 아래에 독립적으로 넣는다. 팀 검토 후 필요한 코드만 공용
파이프라인으로 옮기면 된다.

## 1. 쓰기 권한 확인

현재 연결된 계정에서는 팀 저장소 읽기 권한만 확인되었다. 아래 둘 중 하나를 선택한다.

- 팀장에게 GitHub 사용자 `kwonjun99`를 collaborator로 추가해 달라고 요청
- 팀 저장소를 개인 계정으로 Fork한 뒤 Pull Request 생성

권한을 받기 전에는 `main`에 직접 push하거나 강제 push하지 않는다.

## 2-A. 팀 저장소 쓰기 권한을 받은 경우

```bash
cd ~/projects
git clone https://github.com/credsuk/team-ggeodeureok.git
cd team-ggeodeureok

git switch main
git pull --ff-only origin main
git switch -c feature/chroma-retrieval-kwonjun

# 전달받은 ZIP 경로에 맞게 수정
unzip ~/Downloads/council_team_github_patch_20261020.zip -d .
```

## 2-B. 쓰기 권한이 없는 경우: Fork + PR

1. 팀 저장소 페이지에서 **Fork**를 눌러 개인 계정에 복제한다.
2. 개인 Fork를 clone하고 팀 저장소를 `upstream`으로 등록한다.

```bash
cd ~/projects
git clone https://github.com/kwonjun99/team-ggeodeureok.git
cd team-ggeodeureok

git remote add upstream https://github.com/credsuk/team-ggeodeureok.git
git fetch upstream
git switch -c feature/chroma-retrieval-kwonjun upstream/main

unzip ~/Downloads/council_team_github_patch_20261020.zip -d .
```

## 3. 데이터 유출 검사

```bash
git status --short

# 이번 브랜치에는 아래 폴더만 새로 보여야 한다.
git status --short experiments/kwonjun_chroma

# 원본·벡터·인덱스가 추적 대상이면 중단한다.
git ls-files | rg '^(data|models|outputs)/|\.(parquet|npy|faiss|joblib|gguf)$' || true
find experiments/kwonjun_chroma -type f \
  \( -name '*.parquet' -o -name '*.npy' -o -name '*.faiss' -o -name '*.gguf' \)
```

마지막 두 명령은 아무 결과도 없어야 한다. ZIP에는 코드, 문서, 합성 예시, 지도 이미지
외에는 포함하지 않는다.

## 4. 로컬 실행 확인

```bash
cd experiments/kwonjun_chroma
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m py_compile demo/app.py src/*.py
bash scripts/run_demo.sh
```

실제 인덱스가 없으면 합성 예시 화면까지만 확인한다. 실제 회의록·인덱스는 GitHub에 올리지
않고 팀의 승인된 로컬 저장 경로에서만 연결한다.

## 5. Commit과 Push

저장소 루트로 돌아간 뒤 이번 폴더만 stage한다.

```bash
cd ~/projects/team-ggeodeureok
git add experiments/kwonjun_chroma
git diff --cached --stat
git diff --cached --name-only

git commit -m "feat: add evaluated Arctic-Chroma retrieval prototype"
git push -u origin feature/chroma-retrieval-kwonjun
```

## 6. Pull Request

- Base repository: `credsuk/team-ggeodeureok`
- Base branch: `main`
- Compare branch: `feature/chroma-retrieval-kwonjun`
- 제목: `feat: Arctic-Chroma 정책 이슈 검색 프로토타입 추가`
- 본문: `PULL_REQUEST_BODY.md` 내용을 붙여 넣는다.

## 7. 개인 GitHub에도 보관하기

Fork 방식이면 같은 브랜치가 이미 개인 GitHub에 올라간다. 별도의 개인 프로젝트 저장소에도
올리고 싶다면 데이터 없는 실험 폴더만 복사한다.

```bash
git remote -v

# 별도 개인 저장소를 만들었을 때만 추가
git remote add personal https://github.com/kwonjun99/council-minutes-ai.git
git push -u personal feature/chroma-retrieval-kwonjun
```

팀 저장소와 개인 저장소 모두 `main` 직접 push보다 기능 브랜치와 PR을 권장한다.


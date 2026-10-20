# GitHub 공개·협업 원칙

## 저장소 구분

- 팀 통합 저장소: <https://github.com/credsuk/team-ggeodeureok>
- 개인 작업 저장소: 코드 백업과 포트폴리오용
- 로컬 전용: 회의록 원문, 정제 데이터, 임베딩, 인덱스, 실제 분석 결과

팀 저장소에는 우선 `experiments/kwonjun_chroma/`로 올려 기존 팀원 코드와 충돌을 피한다.
병합 방법과 명령은 [TEAM_GITHUB_GUIDE.md](TEAM_GITHUB_GUIDE.md)를 따른다.

## 공개 가능

- 소스 코드와 실행 스크립트
- 직접 만든 합성 예시
- 집계 성능과 기술 의사결정
- 원문을 포함하지 않는 구조도·UI 자산

## 공개 금지

- `data/raw/`, `data/processed/`, `data/index/`, `data/chroma/`
- 실제 회의록 문장과 실제 검색·타임라인 JSON
- `.parquet`, `.npy`, `.faiss`, `.joblib`, `.gguf`, 모델 가중치
- Hugging Face·Ollama 캐시
- `.env`, 토큰, API 키, 개인 절대 경로

## Push 전 검사

```bash
git status --short
bash scripts/check_public_repo.sh
git ls-files | rg '^(data|models|outputs)/|\.(parquet|npy|faiss|joblib|gguf)$'
```

세 번째 명령은 결과가 없어야 한다. 이미 추적된 민감 파일이 나오면 push하지 말고 원격 공개
상태와 커밋 기록부터 확인한다. 과거 커밋에 원문이 들어갔다면 단순 삭제로 해결되지 않으므로
팀과 상의해 기록 정리 또는 새 저장소 이관을 진행한다.

## 권장 Git 흐름

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/chroma-retrieval-kwonjun

git add experiments/kwonjun_chroma
git diff --cached --stat
git commit -m "feat: add evaluated Arctic-Chroma retrieval prototype"
git push -u origin feature/chroma-retrieval-kwonjun
```

`main`에 직접 push하거나 `--force`를 사용하지 않는다. 팀 저장소 쓰기 권한이 없으면 개인
Fork에 push하고 팀 저장소로 Pull Request를 연다.


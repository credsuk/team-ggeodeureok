# GitHub 공개 전 체크리스트

## 코드

- [ ] 모든 Python 파일 AST 검사 통과
- [ ] 모든 Bash 파일 `bash -n` 통과
- [ ] `.env.example`만 포함하고 `.env` 제외
- [ ] README 실행 명령과 실제 경로 일치
- [ ] 최종 웹이 `demo/final_app.py`인지 확인

## 데이터·보안

- [ ] 원본 JSON/TXT/JSONL 없음
- [ ] Parquet/Numpy/SQLite/FAISS 파일 없음
- [ ] PostgreSQL 볼륨 없음
- [ ] Hugging Face 모델 가중치 없음
- [ ] 회의록 원문이 포함된 결과 JSON 없음
- [ ] API Key·비밀번호·개인정보 없음
- [ ] 25MB 초과 파일 없음

## 결과

- [ ] `results/forecast_metrics.json` 존재
- [ ] 실험 원장과 README 수치 일치
- [ ] 지도 화살표가 시간적 관측 순서임을 명시
- [ ] 예측 기준일과 제공 데이터 범위를 명시
- [ ] 외부 상용 LLM API 미사용 명시

## Git

- [ ] 올바른 저장소와 브랜치 확인
- [ ] `git status --short` 검토
- [ ] `git diff --cached --stat` 검토
- [ ] 민감 파일 추적 여부 검사
- [ ] 커밋 후 원격 브랜치에 push


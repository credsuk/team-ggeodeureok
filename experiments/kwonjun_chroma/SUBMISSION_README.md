# 의정이슈레이더 최종 제출 안내

공식 제출 기준 코드는 이 폴더다. 모델 파이프라인은
`feature/chroma-kwonjun`을 기반으로 하고, 민지 작업의 지역명 정규화와 외부
참고 기능을 통합했다. `gwonwoo`는 이전 커밋이며 최종 실행 기준이 아니다.

## 빠른 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-all.txt
docker compose up -d pgvector
bash scripts/run_final_app.sh
```

공식 제출 모드에서는 외부 공공데이터 자동 호출이 비활성화된다. 핵심 검색,
군집화, 예측은 제공된 2024~2025년 회의록만 사용한다.

## 필수 문서

- `README.md`: 전체 기술 해설과 실행 방법
- `docs/PREPROCESSING.md`: 데이터 정제·canonical·부모자식 청킹
- `docs/AI_USAGE.md`: 로컬 모델 및 생성형 AI 활용·검증 범위
- `docs/FORECAST_EVALUATION.md`: 예측 모델과 평가 지표
- `docs/DEMO_VIDEO_GUIDE.md`: 4분 데모 영상 구성
- `docs/SUBMISSION_CHECKLIST.md`: 제출 전 최종 점검
- `examples/`: 공개 가능한 예시 입력·출력
- `notebooks/01_end_to_end_demo.ipynb`: 실행 흐름 확인 Notebook

## 제출 패키지 생성

```bash
python scripts/41_prepare_release.py --output ../council-issue-radar-release
```

패키지 생성기는 원본 데이터, 임베딩, DB, 비밀정보, 기준일 이후 외부 카탈로그를
제외하고 필수 파일·크기·체크섬을 검증한다.


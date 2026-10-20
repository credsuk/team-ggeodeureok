# 패키지 구성

`council_team_github_patch_20261020.zip`을 팀 저장소 루트에 풀면 다음 경로가 생성됩니다.

```text
experiments/kwonjun_chroma/
├── README.md
├── PROJECT_PROGRESS.md
├── TEAM_GITHUB_GUIDE.md
├── PULL_REQUEST_BODY.md
├── requirements.txt
├── .gitignore
├── src/                 # 01~18 전체 로컬 분석 파이프라인
├── demo/
│   ├── app.py
│   └── assets/          # 오프라인 대한민국 지도
├── scripts/             # 데모·통합 검색·검증·공개 안전 검사
├── examples/            # 합성 예시만 포함
└── docs/                # 오늘 기록·의사결정·단계별 가이드
```

포함하지 않는 것: 회의록 원문, 실제 결과, Parquet, NPY, FAISS/Chroma 인덱스,
모델 가중치, LLM 로그, 인증정보.


# Council Minutes Policy Flow AI

2024–2025년 광역 지방의회 회의록에서 정책 이슈를 찾고, 제공 데이터 범위 안의
최초 관측 시점, 다른 의회의 후속 관측 순서, 월별 변화와 다음 달 흐름을 원문 근거와
함께 보여주는 **완전 로컬 AI 분석 파이프라인**입니다.

> 이 공개 저장소에는 경진대회 원문, 정제 데이터, 임베딩, 실제 검색 결과가 없습니다.
> 회의록은 외부 LLM API로 전송하지 않으며 모델 다운로드 이후 모든 처리를 로컬에서 수행합니다.

## 무엇을 보여주나

사용자가 `저출생`, `교통약자`, `수돗물 안전` 같은 정책어를 입력하면 다음 결과를 한 화면에 제공합니다.

1. 검색어와 직접 관련된 회의록 근거
2. 검수 완료 이슈와의 연결
3. 제공 데이터 범위 내 최초 관측 의회·날짜
4. 다른 광역의회의 최초 후속 관측 순서
5. 월별 언급량 변화와 다음 달 보조 예측
6. 회의록명·회의일자·안건·발화자·원문 구간

## 파이프라인

```text
JSON/HTML
  → 정제·중복 제거
  → 부모 청킹
  → 정책 후보 필터
  → 의미 자식 청킹
  → Arctic 한국어 임베딩
  → FAISS 정확 검색으로 모델 비교·검증
  → Chroma 검색·메타데이터 필터
  → KMeans 군집화
  → 로컬 LLM 이슈명 제안
  → 사람 검수 및 issue_id 고정
  → 최초 관측·후속 관측·월별 변화
  → 성능 게이트를 통과한 예측 또는 보수적 규칙
  → Streamlit 지도·타임라인·원문 근거
```

## 핵심 기술

| 영역 | 선택 | 역할 |
|---|---|---|
| 임베딩 | `dragonkue/snowflake-arctic-embed-l-v2.0-ko` | 한국어 정책 의미 검색 |
| 검증 검색 | FAISS `IndexFlatIP` | 정확 코사인 검색 기준선 |
| 서비스 검색 | Chroma | 영속 저장, 메타데이터 필터, 데모 질의 |
| 군집화 | MiniBatchKMeans | 유사 정책 발언을 이슈 후보로 묶기 |
| 로컬 LLM | Ollama `qwen3:8b` | 군집 이름·요약 제안, 구조화 출력 |
| 예측 | Logistic Regression, LightGBM, trend fallback | 다음 달 증가·확산 보조 예측 |
| 데모 | Streamlit, Plotly | 대한민국 지도·시계열·원문 근거 |

## 2026-10-20 검색기 비교 결과

동일한 검수 질의와 정책 청크에서 비교한 결과 Arctic 단일 검색기가 가장 안정적이었습니다.

| 구성 | nDCG@5 | Recall@5 | MRR |
|---|---:|---:|---:|
| Arctic | **0.9908** | **0.2851** | **1.0000** |
| Dense RRF | 0.9899 | 0.2851 | 1.0000 |
| KURE | 0.9813 | 0.2841 | 1.0000 |
| Hybrid RRF | 0.9684 | 0.2803 | 1.0000 |
| Hybrid rerank | 0.9642 | 0.2813 | 0.9896 |
| BM25 | 0.8568 | 0.2675 | 0.9896 |

따라서 현재 버전은 **Arctic + Chroma**를 데모 검색기로 사용하고, FAISS 정확 검색은
회귀 테스트와 결과 일치 검증용으로 보존합니다. Chroma에 적재한 25,340개 정책 청크는
FAISS와 `overlap@10=1.000`, Top-1 일치를 확인했습니다.

## 데모 실행

Ubuntu 22.04 또는 WSL2, Python 3.12 환경을 권장합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

bash scripts/run_demo.sh
```

브라우저에서 `http://127.0.0.1:8501`을 엽니다. 실제 결과 파일이 없으면
`examples/sample_issue_timelines.json`과 `examples/sample_issue_forecasts.json`을 사용합니다.

실제 로컬 인덱스가 준비된 경우 통합 질의도 실행할 수 있습니다.

```bash
bash scripts/run_issue_query.sh "출산율 저하 대응 정책"
```

`app.py`는 일반 Python 파일처럼 실행하지 말고 반드시 `streamlit run`으로 실행합니다.

## 저장소 구조

```text
src/        데이터 처리·평가·검색·군집·예측 코드
demo/       Streamlit 앱과 오프라인 대한민국 지도 자산
scripts/    실행·검증·공개 안전 검사
docs/       설계, 의사결정, 일일 작업 기록
examples/   실제 회의록이 아닌 합성 예시
```

## 결과 해석 원칙

- `최초 관측`은 전국 최초가 아니라 **제공 데이터 범위 내 최초 관측**입니다.
- `전파 순서`는 의회별 최초 관측의 시간 순서이며 인과적 전달을 증명하지 않습니다.
- 검색 대표 근거와 최초 관측 근거는 서로 다를 수 있어 화면에서 분리합니다.
- LLM은 군집을 만들지 않고 이슈명과 설명을 제안하는 보조 역할만 수행합니다.
- 예측 모델은 시간순 검증 기준을 통과한 경우에만 채택합니다.
- 분석 기준일 이후 정보와 외부 데이터는 예측에 사용하지 않습니다.

## 공개 금지 항목

- 경진대회 제공 원문과 정제 데이터
- 실제 회의록 문장이 포함된 CSV, JSON, Parquet
- 임베딩 NPY, FAISS/Chroma 인덱스, 모델 가중치
- 실제 LLM 프롬프트 로그와 검색 결과
- API 키, 토큰, `.env`, 개인 경로

업로드 전에는 반드시 다음을 실행합니다.

```bash
bash scripts/check_public_repo.sh
git status --short
git ls-files | rg '^(data|models|outputs)/|\.(parquet|npy|faiss|joblib|gguf)$'
```

## 문서

- [프로젝트 진행 현황](PROJECT_PROGRESS.md)
- [오늘 작업 기록](docs/DAILY_LOG_2026-10-20.md)
- [주요 의사결정](docs/DECISION_LOG.md)
- [팀 GitHub 반영 가이드](TEAM_GITHUB_GUIDE.md)


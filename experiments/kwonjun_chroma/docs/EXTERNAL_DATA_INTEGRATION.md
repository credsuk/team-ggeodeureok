# 외부 공공데이터 보강 기능

이 문서는 의정이슈레이더 최신 UI에 법제처 자치법규, 열린국회정보, KOSIS 참고 정보를 결합한 범위와 실행 방법을 설명합니다.

## 적용 원칙

- 회의록 원문, 사용자 질문 원문, 발화자 정보는 외부로 전송하지 않습니다.
- 외부 API에는 검증된 짧은 정책어 또는 사전 등록된 KOSIS 통계표 ID만 보냅니다.
- 외부 데이터는 이슈 탐지, 군집화, 공식 예측 모델의 입력으로 사용하지 않습니다.
- 대회 기본 모드는 `backtest`이며 `2025-12-31` 이후 정보는 공식 예측과 분리합니다.
- `live`는 별도의 사후 탐색 데모에서 사용자가 최신·현재를 명시한 경우에만 사용합니다.

## 추가된 파일

| 파일 | 역할 |
|---|---|
| `src/external_policy_sources.py` | 허용된 공공 API 클라이언트, 정책어 검증, 캐시, 접근 기록 |
| `src/external_followup.py` | 조례화 단계, 후속 후보 점수, 사후 점검 계산 |
| `src/32a_external_followup_flow.py` | 대표 이슈 10개의 외부 후속 흐름 일괄 생성 |
| `src/regional_stats.py` | KOSIS 시도별 지표 조회 |
| `config/kosis_indicators.json` | 허용한 KOSIS 지표 목록 |
| `agent/query_pipeline.py` | 질문 해석, 검증, 도구 실행, 답변 검증 |
| `agent/followup_tools.py` | Agent용 외부 후속 흐름·통계·국회 도구 |

## 설정

`.env.example`을 복사해 `.env`를 만들고 필요한 값만 입력합니다. 실제 키는 Git에 올리지 않습니다.

```bash
cp .env.example .env
```

```dotenv
COUNCIL_FOLLOWUP_MODE=backtest
COUNCIL_MINUTES_CUTOFF=2025-12-31
COUNCIL_EXTERNAL_CACHE=data/external/cache
LAW_OC=
ASSEMBLY_API_KEY=
KOSIS_API_KEY=
```

API 키가 없어도 기존 회의록 검색, 지도, 예측, 로컬 Agent는 동작합니다. 이미 생성된 외부 결과가 있으면 인터넷 없이 웹 번들에 다시 결합할 수 있습니다.

## 실행

정책어와 연결 지표만 먼저 확인합니다.

```bash
python -u src/32a_external_followup_flow.py --mode backtest --dry-run
```

공공데이터 참고 결과를 생성하고 최종 웹 JSON에 결합합니다.

```bash
python -u src/32a_external_followup_flow.py --mode backtest
python -u src/34_build_final_web_bundle.py
bash scripts/run_final_app.sh
```

`run_final_app.sh`는 API 키가 하나라도 설정돼 있으면 32a를 먼저 실행합니다. 조회가 실패해도 기존 회의록 결과로 앱을 계속 시작합니다.

## 화면과 Agent

- 이슈 레이더에는 의회별 회의록 첫 논의, 광역 조례 공포, 소요일, 기초 조례 수, 후속 후보가 표시됩니다.
- 후속 후보 점수는 규칙 기반 정렬값이며 확률이나 인과적 전파 판정이 아닙니다.
- 챗봇은 질문을 구조화하고 정책어·지역을 검증한 뒤 필요한 도구만 실행합니다.
- 답변 아래에서 질문 해석, 실행 단계, 외부 전송 값, 숫자·의회명 검증 결과를 확인할 수 있습니다.

## 결과 위치

```text
data/external/cache/               외부 API 응답 캐시·접근 기록
outputs/external/followup_v1/      이슈별 후속 흐름 결과
outputs/web_final/                 외부 참고 결과가 결합된 최종 웹 JSON
```

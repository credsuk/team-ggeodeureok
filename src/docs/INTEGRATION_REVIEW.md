# main 통합 검토 기록

## 브랜치 상태

- 기준: `main`
- 기능 출처: `feature/radar-hybrid-external`
- 두 브랜치는 공통 조상 이후 각각 9개 커밋이 있어 기능 브랜치를 main에 그대로 덮어쓰지 않는다.
- main의 FastAPI/프론트엔드 구조를 유지하고 기능 브랜치의 외부 공공데이터, KOSIS 주제 카탈로그, 의회 통합 규칙, 자유검색 개선을 실제 실행 경로인 `src/`로 이식한다.

## LLM 변경

운영 기본값을 `qwen3:8b`/Gemma 계열에서 공식 LG AI Research EXAONE `exaone3.5:7.8b`로 변경한다. 임베딩(Arctic), pgvector/BM25 검색, 군집화, 예측 모델은 바꾸지 않으므로 벡터 재생성은 필요 없다. 단, EXAONE의 JSON 구조화 출력·근거 인용·한국어 품질 회귀시험은 필수다.

초기 관련성 자동 판정에 Qwen을 사용한 실험 기록은 재현성을 위해 `config/experiment_registry.json`에 그대로 남긴다.

## 병합 전 필수 검증

1. `ollama pull exaone3.5:7.8b` 후 대표 질문 30개로 구조화 출력 성공률, 근거 인용, 무응답률 검사
2. Streamlit과 FastAPI에서 고정 이슈·자유검색·Agent 모두 실행
3. 외부 API는 회의록 원문을 전송하지 않고 정책어·지역·기간만 전송하는지 접근 로그 확인
4. 외부 데이터 `backtest` 모드에서 분석 기준일 이후 자료가 차단되는지 확인
5. 광주·전남 통합 규칙의 시행일 전/후 표시와 KOSIS 지역 코드 확인
6. 오프라인 실행 시 OpenFreeMap 온라인 타일 의존성을 제거하거나 오프라인 지도 대체 화면 제공

## 반드시 구분할 예측 경로

- 검수 완료 대표 이슈: 기존 시간순 다중 이슈 평가 결과(활동률 기준선/Poisson, rolling backtest)를 사용
- 자유검색 동적 이슈: 현재 FastAPI 경로는 검색 결과 월별 집계와 최근 3개월 평균 기준선을 사용하므로 `검수 전 조건부 기준선`으로 표시

두 결과를 같은 모델 성능으로 설명하면 안 된다. 특히 기존 F1/MAE 수치는 검수 완료 다중 이슈 평가에만 적용한다.

## 검색 구현 차이

- Streamlit 최종 파이프라인: Arctic Dense + SQLite FTS5 BM25 + RRF + 선택적 reranker
- FastAPI main 경로: pgvector Dense + PostgreSQL full-text + RRF

둘은 같은 하이브리드 검색 개념이지만 희소 검색 엔진과 평가 조건이 다르다. 최종 제출 문서에서 실행 화면별 구현을 구분한다.

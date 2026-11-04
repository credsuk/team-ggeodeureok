# figure/ — 시스템 및 실험 시각화 자료

본 디렉토리에는 **의정이슈레이더 (Council Issue Radar)** 프로젝트의 전체 아키텍처, 데이터 파이프라인, 청킹 전략 및 모델 평가 결과를 정리한 시각화 자료가 포함되어 있습니다.

---

## 자료 목록

| 파일명 | 제목 | 설명 | 관련 코드 / 노트북 |
|---|---|---|---|
| [`01_전체_아키텍처.png`](01_전체_아키텍처.png) | 전체 시스템 아키텍처 | 회의록 원문 수집부터 정제, 부모-자식 청킹, PostgreSQL+pgvector 벡터 DB 구축, Dense+BM25 RRF 하이브리드 검색, Cross-Encoder 재순위화, 이슈 군집화, 시계열 예측 및 로컬 설명 Agent·대시보드 UI까지의 End-to-End 전체 시스템 흐름 | `README.md`, `notebooks/07_council_issue_radar.ipynb` |
| [`02_부모_자식_청킹.png`](02_부모_자식_청킹.png) | 부모-자식 청킹 전략 | 긴 회의록 발언 단위의 문맥(Context)을 보존하는 상위 부모 청크와 고정밀 벡터 검색을 위한 하위 자식 청크의 계층 분할 및 메타데이터 연결 구조 | `src/05_build_parent_child_chunks.py`, `notebooks/03_pgvector_db_pipeline.ipynb` |
| [`03_검색_군집화.png`](03_검색_군집화.png) | 하이브리드 검색 및 이슈 군집화 | Arctic 한국어 임베딩 기반 Dense 검색과 Kiwi 형태소 기반 BM25 Sparse 검색의 RRF 결합, BGE Cross-Encoder 재순위화 및 시공간 이슈 확산 군집화 파이프라인 | `src/18_pgvector_retrieval.py`, `src/19_rerank_candidates.py`, `src/20_expand_cluster_issue.py` |
| [`04_예측_평가.png`](04_예측_평가.png) | 시계열 예측 및 백테스트 평가 | 10-Fold 시간순 Rolling Backtest(Expanding Window) 기반의 논의 지속 여부 분류(F1-score 0.881) 및 Poisson 회귀 기반 다음 달 언급량 예측 성능(MAE 7.12, RMSE 10.04) 비교 평가 결과 | `src/32_train_multi_issue_forecast.py`, `notebooks/04_experiments_forecast_comparison.ipynb` |

# 의정이슈레이더 자유검색 지도 수정본

## 변경 사항

- `📍 이슈 레이더` 탭에 자유검색 입력창 추가
- 고정 10대 이슈와 일치하면 검수 완료 레이더를 즉시 표시
- 그 외 검색어는 전체 청크 검색 → BGE 재순위 → 군집화 → 최초 관측 계산
- 자유검색 결과에도 대한민국 지도 노드와 시간순 연결선 생성
- 월별 논의량, 조건부 전망, 대표 원문 근거를 같은 레이더 화면에 표시
- 생성된 자유검색 결과를 자유분석 탭과 Agent 탭에서 재사용

## 적용

실행 중인 Streamlit을 `Ctrl+C`로 종료하고 프로젝트 루트에서 실행합니다.

```bash
cd ~/projects/council-vector-benchmark
LATEST_ZIP=$(ls -t /mnt/c/Users/Admin/Downloads/radar_free_search_fix_20261026*.zip | head -1)
unzip -o "$LATEST_ZIP" -d .
source .venv/bin/activate
bash scripts/run_final_app.sh
```

웹의 `📍 이슈 레이더` 탭에서 `자전거` 또는 `자전거 안전 정책`을 검색합니다.
첫 실행은 검색·재순위·군집화 때문에 시간이 걸릴 수 있고 같은 질의는 저장된 결과를 재사용합니다.

재임베딩과 pgvector 재적재는 필요하지 않습니다.

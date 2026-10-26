# Agent UI 연결 수정본

## 수정 내용

- 답변 직후 `st.rerun()` 때문에 Agent 탭이 사라져 보이던 문제 수정
- 현재 Agent 탭에서 질문과 답변을 즉시 표시
- Ollama 서버 및 `qwen3:8b` 설치 상태를 웹 화면에 표시
- Ollama 미연결/모델 미설치 시 오래 대기하지 않고 근거 요약으로 전환
- Agent 처리 중 예외가 생겨도 Streamlit 전체가 종료되지 않도록 처리
- Qwen3 추론 모드를 끄고 답변 길이를 제한하여 응답 시간 단축
- psycopg에서 `SET hnsw.ef_search = $1` 오류가 나지 않도록 `set_config()` 적용

## 적용

프로젝트 루트에서 ZIP을 그대로 덮어씁니다.

```bash
cd ~/projects/council-vector-benchmark
unzip -o /mnt/c/Users/Admin/Downloads/agent_ui_connection_fix_20261026.zip -d .
```

`.env`에 다음 값이 있는지 확인합니다.

```dotenv
OLLAMA_URL=http://127.0.0.1:11434
COUNCIL_AGENT_MODEL=qwen3:8b
COUNCIL_AGENT_TIMEOUT=90
```

Ollama와 모델을 확인합니다.

```bash
curl -s http://127.0.0.1:11434/api/tags
ollama list
```

목록에 `qwen3:8b`가 없을 때만 실행합니다.

```bash
ollama pull qwen3:8b
```

앱을 재시작합니다.

```bash
source .venv/bin/activate
bash scripts/run_final_app.sh
```

Agent 탭 위쪽에 `로컬 Agent 연결됨`이 표시되면 Qwen 응답이 사용됩니다.
미연결이어도 검색된 회의록을 이용한 구조화 요약은 즉시 출력됩니다.

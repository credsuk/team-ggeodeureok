# 로컬 모델

모델 가중치는 GitHub에 포함하지 않습니다.

| 용도 | 모델 |
|---|---|
| 임베딩 | `dragonkue/snowflake-arctic-embed-l-v2.0-ko` |
| 재순위 | `BAAI/bge-reranker-v2-m3` |
| Agent | Ollama `qwen3:8b` |

Hugging Face 모델은 최초 1회 다운로드한 뒤 `models/huggingface/`에 보관하고,
제출·오프라인 실행 시에는 `local_files_only=True`를 사용합니다.

```bash
ollama pull qwen3:8b
ollama list
```

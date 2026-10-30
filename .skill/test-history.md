# Test history

## 2026-10-29 — PostgreSQL-backed API changes

Command:

```text
cd /mnt/c/project/mini/src
python -m unittest discover -s api/tests -v
```

Result: 29 tests passed after the final implementation edits. Coverage includes RRF ranking, full-text query formatting, HNSW search-depth configuration, three-month baseline arithmetic, service routing and propagation dates against an injected fake PostgreSQL repository, API contracts, and launcher fail-fast/success-call behavior.

Additional checks: `python -m py_compile` on the launcher, API service/backend, and routers passed; the fixed-keyword configuration loaded as 10 entries; path-limited `git diff --check` passed for the API files. The WSL interpreter lacks `psycopg`, `pgvector`, `sentence_transformers`, and `torch`, so no live PostgreSQL/model/Ollama integration was run here. These tests do not establish database compatibility, production performance, Ollama answer quality, or forecast accuracy.

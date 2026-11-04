# FastAPI 백엔드 및 웹 대시보드

`src/main.py`가 전체 서비스 실행 런처이며, `src/api/main.py`는 FastAPI 백엔드 애플리케이션을 구동합니다. 프로젝트 루트에서 다음 명령으로 시작합니다.

```bash
python src/main.py
```

서버 실행 후 브라우저에서 `http://127.0.0.1:8000/`으로 접속하면 통합 웹 대시보드를 바로 이용할 수 있습니다. 프런트엔드 정적 파일(`src/api/frontend/`)과 `/api/v1/...` API 엔드포인트가 동일한 FastAPI 인스턴스에서 통합 제공되므로 별도의 빌드 도구(npm/Vite)나 추가 설정 없이 단일 실행됩니다.

런처는 `.env` 설정을 읽어 로컬 Ollama 환경, PostgreSQL 연결, 테이블 스키마 및 임베딩 모델 무결성을 사전에 검증한 뒤 안전하게 서버를 기동합니다.

## 검색 방식 및 예측 기능

- **하이브리드 검색 (Hybrid Retrieval)**: PostgreSQL `pgvector`의 Dense 임베딩 검색과 전문검색(`to_tsvector` / `ts_rank_cd`) 결과를 상호 순위 융합(RRF, Reciprocal Rank Fusion) 방식으로 결합하여 높은 재현율과 검색 정밀도를 제공합니다.
- **월별 시계열 집계 및 예측 기준선**: 일치 원문의 월별 언급 추이를 집계하고, 최근 3개월 추세를 기반으로 다음 달 예측 기준선을 실시간으로 계산하여 반환합니다.
- **대표 이슈 및 동적 레이더 검색**: 사전 구성된 대표 정책 이슈 카탈로그 조회 및 임의의 정책 질의에 대한 동적 레이더 분석을 지원합니다.
- **근거 기반 로컬 LLM 질의응답**: 검색된 회의록 근거를 바탕으로 로컬 Ollama를 통해 사실에 입각한 한국어 답변 및 근거 번호(`[근거 N]`) 인용을 제공합니다.

## 엔드포인트

- `GET /api/v1/issues`: 대표 정책 이슈 목록
- `GET /api/v1/issues/{issue_key}`: 특정 이슈의 하이브리드 검색 근거, 월별 집계, 다음 달 기준선
- `POST /api/v1/radar/search`: 자유 질의 기반 동적 이슈 레이더 검색 (`{"query": "저출생 정책"}`)
- `GET /api/v1/documents/search`: 필터(의회, 연도, 위원회, 기간)를 적용한 원문 하이브리드 검색
- `GET /api/v1/documents/{document_id}`: 특정 문서의 전체 청크 조회
- `GET /api/v1/evidence/{chunk_id}`: 특정 검색 청크 상세 조회
- `POST /api/v1/chat`: 회의록 근거 기반 로컬 LLM 질의응답 (`{"message": "...", "history": [], "top_k": 6}`)
- `GET /api/v1/chat/status`: Ollama 연결 및 LLM 상태 확인

대화 이력은 요청에 포함된 `history`로만 전달되며 서버 메모리에 잔존하지 않습니다. 모든 회의록 본문과 질의는 외부 상용 API로 전송되지 않고 로컬 환경 내에서 안전하게 처리됩니다.

## 필수 환경 설정

설정은 `.env` 파일을 통해 관리됩니다 (`.env.example` 참조).

- `PGVECTOR_DSN`: PostgreSQL 연결 문자열
- `COUNCIL_PG_TABLE`: 기본 `council_arctic_raw_v1`
- `COUNCIL_PG_EF_SEARCH`: HNSW 탐색 깊이 (기본값: `200`)
- `COUNCIL_MODEL_CACHE`: 로컬 임베딩 모델 캐시 경로
- `COUNCIL_ALLOW_DOWNLOAD`: 기본값 `0` (로컬 캐시 우선 사용)
- `COUNCIL_DEVICE`: `auto`, `cpu`, 또는 `cuda`
- `OLLAMA_URL`, `COUNCIL_AGENT_MODEL`: 기본값 `http://127.0.0.1:11434`, `hf.co/mykor/A.X-4.0-Light-gguf:Q4_K_M`
- `HOST`, `PORT`: 기본값 `127.0.0.1`, `8000`

대화형 API 문서는 서버 기동 후 `http://127.0.0.1:8000/docs`에서 확인할 수 있습니다.

## 검증

`src/` 디렉토리에서 단위 테스트를 실행하여 API 엔드포인트 동작과 응답 규격을 검증할 수 있습니다.

```bash
python -m unittest discover -s api/tests -v
```

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent.followup_tools import FollowupTools
from agent.local_issue_agent import LocalCouncilAgent, route_fixed_issue
from agent.query_pipeline import QuestionPipeline

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
INTERNAL_SRC = SRC_ROOT / "src"
if str(INTERNAL_SRC) not in sys.path:
    sys.path.insert(0, str(INTERNAL_SRC))



def _json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 최상위 값이 object가 아닙니다: {path}")
    return value


class RadarApiService:
    """Existing demo/data pipeline adapter. Heavy services are initialized lazily."""

    def __init__(self) -> None:
        self.web_dir = Path(os.environ.get("COUNCIL_FINAL_WEB_DIR", SRC_ROOT / "outputs" / "web_final")).expanduser().resolve()
        self.dynamic_dir = Path(os.environ.get("COUNCIL_DYNAMIC_OUTPUT_DIR", SRC_ROOT / "outputs" / "dynamic_issues")).expanduser().resolve()
        self._catalog: dict[str, Any] | None = None
        self._payloads: list[dict[str, Any]] | None = None
        self._search: Any | None = None
        self._followup: FollowupTools | None = None
        self._lock = threading.RLock()

    def _load_catalog(self) -> dict[str, Any]:
        if self._catalog is None:
            path = self.web_dir / "issue_catalog.json"
            if not path.is_file():
                raise FileNotFoundError(f"대표 이슈 카탈로그가 없습니다: {path}")
            catalog = _json_file(path)
            if catalog.get("schema_version") != "council_issue_catalog_v1":
                raise ValueError("지원하지 않는 대표 이슈 카탈로그 schema_version")
            self._catalog = catalog
        return self._catalog

    def _load_payloads(self) -> list[dict[str, Any]]:
        with self._lock:
            if self._payloads is None:
                loaded = []
                for entry in self._load_catalog().get("issues", []):
                    value = entry.get("payload")
                    if not value:
                        continue
                    path = Path(value)
                    if not path.is_absolute():
                        path = self.web_dir / path
                    path = path.resolve()
                    if not path.is_file():
                        continue
                    payload = _json_file(path)
                    if payload.get("schema_version") == "council_issue_web_v2":
                        loaded.append(payload)
                self._payloads = loaded
            return self._payloads

    def list_issues(self) -> list[dict[str, Any]]:
        catalog = self._load_catalog()
        public_fields = {
            "issue_key", "issue_id", "issue_name", "aliases",
            "first_observation_date", "first_observation_councils",
            "council_count", "evidence_unit_count", "target_month",
            "discussion_active_probability", "predicted_mention_count",
            "selected_classifier", "selected_count_model", "single_issue_direction",
            "external_followup_candidates",
        }
        return [{key: value for key, value in row.items() if key in public_fields}
                for row in (catalog.get("issues") or [])]

    def get_issue(self, issue_key: str) -> dict[str, Any] | None:
        key = issue_key.casefold()
        entry = next((row for row in self.list_issues()
                      if key in {str(row.get("issue_key", "")).casefold(),
                                 str(row.get("issue_id", "")).casefold()}), None)
        if entry is None:
            return None
        payload = next((item for item in self._load_payloads()
                        if str((item.get("issue") or {}).get("issue_id")) == str(entry.get("issue_id"))), None)
        if payload is None:
            raise FileNotFoundError(f"대표 이슈 payload가 없습니다: {issue_key}")
        return {key: value for key, value in payload.items() if key != "source_files"}

    def _new_search_service(self):
        from council_vector_runtime import CouncilSearchService

        return CouncilSearchService(
            backend="pgvector",
            embedding_dir=Path(os.environ.get("COUNCIL_EMBEDDING_DIR", SRC_ROOT / "data" / "embeddings" / "arctic_raw_v1")).expanduser().resolve(),
            parent_path=Path(os.environ.get("COUNCIL_PARENT_PATH", SRC_ROOT / "data" / "canonical_v3" / "chunks_v1" / "parents.parquet")).expanduser().resolve(),
            pg_dsn=os.environ.get("PGVECTOR_DSN"),
            pg_table=os.environ.get("COUNCIL_PG_TABLE", "council_arctic_raw_v1"),
            pg_ef_search=int(os.environ.get("COUNCIL_PG_EF_SEARCH", "200")),
            retrieval_mode=os.environ.get("COUNCIL_RETRIEVAL_MODE", "hybrid"),
            bm25_index=Path(os.environ.get("COUNCIL_BM25_INDEX", SRC_ROOT / "data" / "bm25" / "arctic_raw_v1" / "bm25.sqlite3")).expanduser().resolve(),
            rrf_k=int(os.environ.get("COUNCIL_RRF_K", "60")),
            dense_weight=float(os.environ.get("COUNCIL_DENSE_WEIGHT", "1.0")),
            bm25_weight=float(os.environ.get("COUNCIL_BM25_WEIGHT", "1.0")),
            device=os.environ.get("COUNCIL_DEVICE", "auto"),
            model_cache=Path(os.environ.get("COUNCIL_MODEL_CACHE", SRC_ROOT / "models" / "huggingface")).expanduser().resolve(),
            local_files_only=os.environ.get("COUNCIL_ALLOW_DOWNLOAD", "0") != "1",
        )

    def search_service(self, required: bool = True) -> Any | None:
        with self._lock:
            if self._search is None:
                try:
                    self._search = self._new_search_service()
                except Exception:
                    if required:
                        raise
                    return None
            return self._search

    def _followup_tools(self, search: Any = None) -> FollowupTools:
        with self._lock:
            if self._followup is None:
                self._followup = FollowupTools(issue_payloads=self._load_payloads(), search_service=search)
            elif search is not None:
                self._followup.search_service = search
            return self._followup

    def _attach_fixed_followup(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = dict(payload)
        existing = payload.get("external_followup")
        response["external_followup_status"] = "available" if existing else "not_precomputed"
        return response

    def radar_search(self, query: str) -> dict[str, Any]:
        payload, score = route_fixed_issue(query, self._load_payloads())
        if payload is not None:
            result = self._attach_fixed_followup(payload)
            return {
                **result,
                "match_type": "fixed_issue",
                "matched_issue_id": (payload.get("issue") or {}).get("issue_id"),
                "routing_score": score,
                "query": query,
            }
        output_path = self._dynamic_result_path(query)
        if not output_path.is_file():
            command = [sys.executable, "-u", "src/33_analyze_free_query.py", "--query", query,
                       "--output-dir", str(self.dynamic_dir),
                       "--device", os.environ.get("COUNCIL_DEVICE", "auto"), "--reuse"]
            if os.environ.get("COUNCIL_ALLOW_DOWNLOAD", "0") == "1":
                command.append("--allow-download")
            timeout = int(os.environ.get("COUNCIL_DYNAMIC_TIMEOUT_SECONDS", "1800"))
            subprocess.run(command, cwd=SRC_ROOT, check=True, capture_output=True,
                           text=True, timeout=timeout)
        if not output_path.is_file():
            raise FileNotFoundError(f"자유검색 결과 파일이 생성되지 않았습니다: {output_path}")
        loaded = _json_file(output_path)
        result = {key: value for key, value in loaded.items() if key != "source_files"}
        followup_status = "unavailable"
        try:
            search = self.search_service(required=False)
            followup = self._followup_tools(search).topic_followup(query, [query],
                        os.environ.get("COUNCIL_FOLLOWUP_MODE", "backtest"), [])
            result["external_followup"] = followup.get("result")
            followup_status = "available" if followup.get("result") else "unavailable"
        except Exception as exc:
            result["external_followup_error"] = f"{type(exc).__name__}: {exc}"
        result.update({"match_type": "dynamic", "matched_issue_id": None,
                       "routing_score": score, "query": query,
                       "external_followup_status": followup_status})
        return result

    def _dynamic_result_path(self, query: str) -> Path:
        digest = hashlib.sha256(" ".join(query.split()).encode("utf-8")).hexdigest()[:12]
        return self.dynamic_dir / f"query_{digest}" / "05_dynamic_result.json"

    def document_search(self, query: str, *, council: str | None = None,
                        year: int | None = None, committee: str | None = None,
                        date_from: str | None = None, date_to: str | None = None,
                        top_k: int = 10) -> dict[str, Any]:
        from council_vector_runtime import SearchFilters
        filters = SearchFilters(council=council, year=year, committee=committee,
                                date_from=date_from, date_to=date_to)
        service = self.search_service()
        result = service.search(query=query, top_k=top_k,
                                candidate_k=max(50, top_k * 10), filters=filters,
                                include_parent=True)
        return {"query": query, "result_count": result.get("result_count", len(result.get("results") or [])),
                "results": result.get("results") or [], "search_ms": result.get("search_ms"),
                "query_embedding_ms": result.get("query_embedding_ms")}

    def evidence(self, chunk_id: str) -> dict[str, Any] | None:
        service = self.search_service()
        from council_vector_runtime import rows_by_chunk_ids, parent_contexts
        row = rows_by_chunk_ids(service.artifact.paths.metadata, [chunk_id]).get(chunk_id)
        if not row:
            return None
        parent_id = row.get("parent_chunk_id")
        parent = parent_contexts(service.parent_path, [parent_id]).get(str(parent_id)) if parent_id else None
        return {"chunk_id": chunk_id, "parent_chunk_id": parent_id,
                "text": row.get("text"), "metadata": {k: v for k, v in row.items()
                if k not in {"text", "embedding_text"}}, "parent_context": parent}

    def document(self, document_id: str, limit: int = 100) -> dict[str, Any] | None:
        service = self.search_service()
        import duckdb
        con = duckdb.connect()
        try:
            table = con.execute("SELECT * FROM read_parquet(?) WHERE CAST(document_id AS VARCHAR)=? LIMIT ?",
                                [str(service.artifact.paths.metadata), document_id, limit]).fetch_arrow_table()
        finally:
            con.close()
        rows = table.to_pylist()
        if not rows:
            return None
        return {"document_id": document_id, "result_count": len(rows),
                "items": [{"chunk_id": r.get("chunk_id"), "text": r.get("text"),
                           "metadata": {k: v for k, v in r.items() if k not in {"text", "embedding_text"}}}
                          for r in rows]}

    def chat(self, request: Any) -> dict[str, Any]:
        payloads = self._load_payloads()
        if not payloads:
            raise FileNotFoundError("챗봇에 필요한 대표 이슈 payload가 없습니다")
        search = self.search_service(required=False)
        tools = self._followup_tools(search)
        pipeline = QuestionPipeline(issue_payloads=payloads, tools=tools, search_service=search)
        agent = LocalCouncilAgent(issue_payload={}, search_service=search,
            ollama_url=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
            model=os.environ.get("COUNCIL_AGENT_MODEL", "qwen3:8b"),
            timeout_seconds=int(os.environ.get("COUNCIL_AGENT_TIMEOUT", "90")),
            pipeline=pipeline, parser_model=os.environ.get("COUNCIL_PARSER_MODEL") or None,
            parser_timeout_seconds=int(os.environ.get("COUNCIL_PARSER_TIMEOUT", "120")))
        response = agent.ask(request.message,
            history=[item.model_dump() for item in request.history], top_k=request.top_k)
        response.pop("issue_payload", None)
        return {**response, "conversation_id": request.conversation_id or hashlib.sha256(os.urandom(24)).hexdigest()[:24]}

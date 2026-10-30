from __future__ import annotations

import csv
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from agent.local_issue_agent import route_fixed_issue
from api.postgres_backend import PostgresHybridSearch, forecast_monthly_counts

SRC_ROOT = Path(__file__).resolve().parents[1]


def load_issue_definitions() -> list[dict[str, str]]:
    """Read the small, version-controlled keyword catalog, not generated result JSON."""
    keys_path = SRC_ROOT / "config" / "representative_issue_keys_v1.txt"
    catalog_path = SRC_ROOT / "config" / "issue_seed_catalog_v1.csv"
    keys = {
        line.split("#", 1)[0].strip()
        for line in keys_path.read_text(encoding="utf-8-sig").splitlines()
        if line.split("#", 1)[0].strip()
    }
    with catalog_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = []
    for row in rows:
        if row.get("issue_key") not in keys or str(row.get("enabled", "1")).lower() not in {"1", "true", "yes"}:
            continue
        selected.append({
            "issue_key": row["issue_key"],
            "issue_name": row["issue_name"],
            "query": row["query"],
            "aliases": row.get("aliases", ""),
            "category": row.get("category", ""),
        })
    if len(selected) != 10:
        raise RuntimeError(f"대표 이슈 설정은 10개여야 합니다. 현재 {len(selected)}개입니다")
    return selected


class RadarApiService:
    """FastAPI service backed by the configured PostgreSQL/pgvector table."""

    def __init__(
        self,
        repository: Any | None = None,
        issue_definitions: list[dict[str, str]] | None = None,
    ) -> None:
        self._repository = repository
        self._issues = issue_definitions

    @property
    def issue_definitions(self) -> list[dict[str, str]]:
        if self._issues is None:
            self._issues = load_issue_definitions()
        return self._issues

    def search_service(self) -> Any:
        if self._repository is None:
            self._repository = PostgresHybridSearch()
        return self._repository

    def validate_database(self) -> dict[str, Any]:
        service = self.search_service()
        database = service.validate()
        vector = service.embed_query("의정 이슈 검색 시작 점검")
        if len(vector) != database["vector_dimensions"]:
            raise RuntimeError(
                "질의 임베딩과 PostgreSQL 벡터 차원이 다릅니다: "
                f"{len(vector)} != {database['vector_dimensions']}"
            )
        return database

    def list_issues(self) -> list[dict[str, Any]]:
        return [
            {
                "issue_key": row["issue_key"],
                "issue_name": row["issue_name"],
                "query": row["query"],
                "aliases": [part for part in row["aliases"].split("|") if part],
                "category": row["category"],
            }
            for row in self.issue_definitions
        ]

    def get_issue(self, issue_key: str) -> dict[str, Any] | None:
        definition = next(
            (row for row in self.issue_definitions if row["issue_key"].casefold() == issue_key.casefold()),
            None,
        )
        if definition is None:
            return None
        return self._build_analysis(definition["query"], definition)

    def _build_analysis(
        self,
        query: str,
        definition: dict[str, str] | None = None,
        *,
        top_k: int = 10,
    ) -> dict[str, Any]:
        service = self.search_service()
        search = service.search(query, top_k=top_k, candidate_k=max(50, top_k * 5))
        monthly = service.monthly_counts(query)
        council_observations = service.council_first_observations(query)
        forecast = forecast_monthly_counts(monthly)
        forecast["basis"] = "postgresql_full_text_matches"
        forecast["count_unit"] = "distinct_document_id"
        councils = sorted({str(row.get("council")) for row in council_observations if row.get("council")})
        first = monthly[0] if monthly else {}
        issue_key = definition["issue_key"] if definition else None
        issue_name = definition["issue_name"] if definition else query
        aliases = (
            [part for part in definition["aliases"].split("|") if part]
            if definition else []
        )
        total_documents = sum(int(row.get("document_count") or 0) for row in monthly)
        propagation = council_observations
        return {
            "schema_version": "council_issue_db_v1",
            "status": "complete" if monthly else "no_matching_history",
            "issue": {
                "issue_id": issue_key or hashlib.sha256(query.encode("utf-8")).hexdigest()[:12],
                "issue_key": issue_key,
                "issue_name": issue_name,
                "aliases": aliases,
                "status": "fixed" if definition else "provisional",
            },
            "query": query,
            "summary": {
                "first_observation_date": str(first.get("first_observation_date") or first.get("month") or "")[:10] or None,
                "first_observation_councils": first.get("councils") or [],
                "council_count": len(councils),
                "evidence_unit_count": total_documents,
                "analysis_scope": "postgresql_minutes_table",
            },
            "monthly_counts": monthly,
            "forecast": forecast,
            "evidence": search.get("results") or [],
            "retrieval": {
                "backend": search.get("backend"),
                "mode": search.get("retrieval_mode"),
                "result_count": search.get("result_count", 0),
                "search_ms": search.get("search_ms"),
            },
            "propagation": propagation,
            "external_followup": None,
            "external_followup_status": "not_configured",
        }

    def _attach_fixed_followup(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = dict(payload)
        response.setdefault(
            "external_followup_status",
            "available" if response.get("external_followup") else "not_configured",
        )
        return response

    def _route_profiles(self) -> list[dict[str, Any]]:
        return [
            {"issue": {
                "issue_id": row["issue_key"],
                "issue_name": row["issue_name"],
                "aliases": [part for part in row["aliases"].split("|") if part],
            }}
            for row in self.issue_definitions
        ]

    def radar_search(self, query: str) -> dict[str, Any]:
        profile, score = route_fixed_issue(query, self._route_profiles())
        if profile is not None:
            key = str((profile.get("issue") or {}).get("issue_id"))
            definition = next(row for row in self.issue_definitions if row["issue_key"] == key)
            result = self._build_analysis(definition["query"], definition)
            result.update({
                "match_type": "fixed_issue",
                "matched_issue_id": key,
                "routing_score": score,
                "query": query,
            })
            return result
        result = self._build_analysis(query)
        result.update({
            "match_type": "dynamic",
            "matched_issue_id": None,
            "routing_score": score,
            "query": query,
            "status": "provisional_until_review" if result["status"] == "complete" else result["status"],
        })
        return result

    def document_search(
        self,
        query: str,
        *,
        council: str | None = None,
        year: int | None = None,
        committee: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        filters = {
            "council": council,
            "year": year,
            "committee": committee,
            "date_from": date_from,
            "date_to": date_to,
        }
        return self.search_service().search(
            query, top_k=top_k, candidate_k=max(50, top_k * 5), filters=filters,
        )

    def evidence(self, chunk_id: str) -> dict[str, Any] | None:
        return self.search_service().get_evidence(chunk_id)

    def document(self, document_id: str, limit: int = 100) -> dict[str, Any] | None:
        rows = self.search_service().get_document(document_id, limit=limit)
        if not rows:
            return None
        return {"document_id": document_id, "result_count": len(rows), "items": rows}

    def chat(self, request: Any) -> dict[str, Any]:
        service = self.search_service()
        found = service.search(
            request.message,
            top_k=request.top_k,
            candidate_k=max(50, request.top_k * 5),
        )
        evidence = found.get("results") or []
        if not evidence:
            raise RuntimeError("PostgreSQL 검색 결과가 없어 답변 근거를 찾지 못했습니다")
        host = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        if (urlparse(host).hostname or "").lower() not in {"localhost", "127.0.0.1", "::1"}:
            raise RuntimeError("보안 정책상 localhost Ollama만 사용할 수 있습니다")
        model = os.environ.get("COUNCIL_AGENT_MODEL", "qwen3:8b")
        citations = []
        evidence_blocks = []
        for index, row in enumerate(evidence, start=1):
            metadata = row.get("metadata") or {}
            citations.append({
                "citation": index,
                "chunk_id": row.get("chunk_id"),
                "council": metadata.get("council"),
                "meeting_date": metadata.get("meeting_date"),
                "meeting_name": metadata.get("meeting_name"),
                "agenda_title": metadata.get("agenda_title"),
                "text": row.get("text"),
            })
            evidence_blocks.append(
                f"[근거 {index}] {metadata.get('council', '미상')} | "
                f"{metadata.get('meeting_date', '날짜 미상')}\n{str(row.get('text') or '')[:1800]}"
            )
        messages = [{
            "role": "system",
            "content": "지방의회 회의록 근거를 사용해 한국어로 답하세요. 근거에 없는 사실이나 숫자는 만들지 말고, 각 사실 뒤에 [근거 N]을 표시하세요. 근거가 부족하면 그렇게 말하세요.",
        }]
        for item in request.history:
            messages.append({"role": item.role, "content": item.content})
        messages.append({
            "role": "user",
            "content": f"질문: {request.message}\n\nPostgreSQL 검색 근거:\n" + "\n\n".join(evidence_blocks),
        })
        body = json.dumps({"model": model, "stream": False, "messages": messages}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            host + "/api/chat", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=int(os.environ.get("COUNCIL_AGENT_TIMEOUT", "90"))) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"로컬 Ollama 답변 생성 실패: {type(exc).__name__}: {exc}") from exc
        answer = str((payload.get("message") or {}).get("content") or "").strip()
        if not answer:
            raise RuntimeError("Ollama가 빈 답변을 반환했습니다")
        return {
            "answer": answer,
            "citations": citations,
            "conversation_id": request.conversation_id or uuid4().hex[:24],
            "retrieval_mode": found.get("retrieval_mode"),
            "generation_mode": f"ollama:{model}",
            "warning": None,
            "plan": {
                "backend": found.get("backend"),
                "result_count": found.get("result_count", len(evidence)),
                "query_embedding_ms": found.get("query_embedding_ms"),
            },
            "policy": {
                "external_llm_api_used": False,
                "meeting_text_sent_externally": False,
                "answer_grounded_in_retrieved_evidence": True,
            },
            "search_ms": found.get("search_ms"),
        }

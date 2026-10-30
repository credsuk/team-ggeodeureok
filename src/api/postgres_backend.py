from __future__ import annotations

import os
import re
import threading
import time
from datetime import date, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable


DEFAULT_EMBEDDING_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
DEFAULT_EMBEDDING_REVISION = "55ec6e9358a56d56af759bc8372e970caf8c305f"
_TABLE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def reciprocal_rank_fusion(
    dense_rows: list[dict[str, Any]],
    text_rows: list[dict[str, Any]],
    *,
    limit: int,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    text_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Combine PostgreSQL vector and full-text rankings without mixing score scales."""
    fused: dict[str, dict[str, Any]] = {}
    for source, rows, weight in (
        ("dense", dense_rows, dense_weight),
        ("text", text_rows, text_weight),
    ):
        for rank, row in enumerate(rows, start=1):
            chunk_id = str(row.get("chunk_id") or "")
            if not chunk_id:
                continue
            current = fused.get(chunk_id)
            if current is None:
                current = dict(row)
                current["retrieval_score"] = 0.0
                current["retrieval_sources"] = []
                current["dense_rank"] = None
                current["text_rank"] = None
                fused[chunk_id] = current
            current["retrieval_score"] += weight / (rrf_k + rank)
            current["retrieval_sources"].append(source)
            current[f"{source}_rank"] = rank
            if source == "text" and row.get("text_rank_score") is not None:
                current["text_rank_score"] = row["text_rank_score"]
            if source == "dense" and row.get("similarity") is not None:
                current["similarity"] = row["similarity"]
    ordered = sorted(
        fused.values(),
        key=lambda row: (-float(row["retrieval_score"]), str(row["chunk_id"])),
    )[:max(0, limit)]
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
    return ordered


def build_full_text_query(query: str) -> str:
    """Build safe websearch syntax so one missing keyword does not erase all hits."""
    terms = list(dict.fromkeys(re.findall(r"[0-9A-Za-z가-힣]+", query)))
    return " OR ".join(f'"{term}"' for term in terms)


def _as_month(value: Any) -> date:
    if isinstance(value, datetime):
        return date(value.year, value.month, 1)
    if isinstance(value, date):
        return date(value.year, value.month, 1)
    parsed = date.fromisoformat(str(value)[:10])
    return date(parsed.year, parsed.month, 1)


def forecast_monthly_counts(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """A transparent trailing-three-month baseline over DB-matched document chunks."""
    counts: dict[date, int] = {}
    for row in rows:
        month_value = row.get("month")
        count_value = row.get("document_count")
        if month_value is None or count_value is None:
            continue
        counts[_as_month(month_value)] = max(0, int(count_value))
    if not counts:
        return {
            "status": "insufficient_history",
            "method": "trailing_3_month_mean",
            "target_month": None,
            "predicted_document_count": None,
            "direction": "unknown",
            "history_months": 0,
        }

    first, last = min(counts), max(counts)
    monthly: list[int] = []
    cursor = first
    while cursor <= last:
        monthly.append(counts.get(cursor, 0))
        cursor = date(cursor.year + (cursor.month == 12), 1 if cursor.month == 12 else cursor.month + 1, 1)
    target = date(last.year + (last.month == 12), 1 if last.month == 12 else last.month + 1, 1)
    target_month = target.strftime("%Y-%m")
    if len(monthly) < 3:
        return {
            "status": "insufficient_history",
            "method": "trailing_3_month_mean",
            "target_month": target_month,
            "predicted_document_count": None,
            "direction": "unknown",
            "history_months": len(monthly),
        }

    recent = mean(monthly[-3:])
    previous = mean(monthly[-6:-3]) if len(monthly) >= 6 else None
    if previous is None or abs(recent - previous) < 1e-9:
        direction = "stable"
    else:
        direction = "increase" if recent > previous else "decrease"
    return {
        "status": "complete",
        "method": "trailing_3_month_mean",
        "target_month": target_month,
        "predicted_document_count": max(0, int(round(recent))),
        "direction": direction,
        "history_months": len(monthly),
        "recent_3_month_mean": round(recent, 3),
        "previous_3_month_mean": round(previous, 3) if previous is not None else None,
    }


class PostgresHybridSearch:
    """Query the PostgreSQL pgvector table directly; no local corpus/BM25 files."""

    def __init__(
        self,
        dsn: str | None = None,
        table: str | None = None,
        *,
        connect_factory: Callable[..., Any] | None = None,
        query_encoder: Any | None = None,
    ) -> None:
        self.dsn = dsn or os.environ.get("PGVECTOR_DSN", "")
        self.table = table or os.environ.get("COUNCIL_PG_TABLE", "council_arctic_raw_v1")
        if not _TABLE_PATTERN.fullmatch(self.table):
            raise ValueError("COUNCIL_PG_TABLE은 영문자·숫자·밑줄만 사용할 수 있습니다")
        self._connect_factory = connect_factory
        self._query_encoder = query_encoder
        self._encoder_lock = threading.Lock()
        self.rrf_k = int(os.environ.get("COUNCIL_RRF_K", "60"))
        self.ef_search = max(1, int(os.environ.get("COUNCIL_PG_EF_SEARCH", "200")))
        self.dense_weight = float(os.environ.get("COUNCIL_DENSE_WEIGHT", "1.0"))
        self.text_weight = float(os.environ.get("COUNCIL_TEXT_WEIGHT", os.environ.get("COUNCIL_BM25_WEIGHT", "1.0")))

    def _connect(self):
        if not self.dsn:
            raise RuntimeError("PGVECTOR_DSN이 설정되지 않았습니다")
        if self._connect_factory is not None:
            return self._connect_factory(self.dsn)
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg.rows import dict_row

        connection = psycopg.connect(self.dsn, row_factory=dict_row)
        register_vector(connection)
        return connection

    def _encoder(self):
        if self._query_encoder is not None:
            return self._query_encoder
        with self._encoder_lock:
            if self._query_encoder is not None:
                return self._query_encoder
            import torch
            from sentence_transformers import SentenceTransformer

            device_setting = os.environ.get("COUNCIL_DEVICE", "auto")
            device = "cuda" if device_setting == "auto" and torch.cuda.is_available() else (
                "cpu" if device_setting == "auto" else device_setting
            )
            model_name = os.environ.get("COUNCIL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
            revision = os.environ.get("COUNCIL_EMBEDDING_REVISION", DEFAULT_EMBEDDING_REVISION)
            kwargs: dict[str, Any] = {
                "device": device,
                "local_files_only": os.environ.get("COUNCIL_ALLOW_DOWNLOAD", "0") != "1",
                "revision": revision,
            }
            cache = os.environ.get("COUNCIL_MODEL_CACHE")
            if cache:
                kwargs["cache_folder"] = str(Path(cache).expanduser().resolve())
            self._query_encoder = SentenceTransformer(model_name, **kwargs)
            self._query_encoder.max_seq_length = int(os.environ.get("COUNCIL_MAX_SEQ_LENGTH", "2048"))
            return self._query_encoder

    def embed_query(self, query: str) -> list[float]:
        model = self._encoder()
        kwargs: dict[str, Any] = {
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        if "query" in getattr(model, "prompts", {}):
            kwargs["prompt_name"] = "query"
        vector = model.encode([query], **kwargs)[0]
        return vector.astype("float32").tolist() if hasattr(vector, "astype") else list(vector)

    @staticmethod
    def _where(filters: Any | None) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if filters is None:
            return "", values
        mapping = (
            ("council", "council", "="),
            ("year", "year", "="),
            ("committee", "committee", "ILIKE"),
            ("date_from", "meeting_date", ">="),
            ("date_to", "meeting_date", "<="),
        )
        for attr, column, operator in mapping:
            value = getattr(filters, attr, None) if not isinstance(filters, dict) else filters.get(attr)
            if value is None:
                continue
            clauses.append(f"{column} {operator} %s")
            values.append(f"%{value}%" if operator == "ILIKE" else value)
        return (" AND " + " AND ".join(clauses)) if clauses else "", values

    def validate(self) -> dict[str, Any]:
        columns = (
            "chunk_id, parent_chunk_id, document_id, council, meeting_date, year, "
            "committee, meeting_name, agenda_title, priority, document, metadata, embedding"
        )
        with self._connect() as connection:
            connection.execute(f"SELECT {columns} FROM {self.table} LIMIT 0")
            row = connection.execute(
                f"SELECT vector_dims(embedding) AS dimensions FROM {self.table} WHERE embedding IS NOT NULL LIMIT 1"
            ).fetchone()
        if row is None:
            raise RuntimeError(f"pgvector 테이블 {self.table}에 임베딩 행이 없습니다")
        dimensions = row.get("dimensions") if isinstance(row, dict) else row[0]
        return {"table": self.table, "vector_dimensions": int(dimensions)}

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        candidate_k: int | None = None,
        filters: Any | None = None,
        include_parent: bool = False,
    ) -> dict[str, Any]:
        del include_parent  # Parent text is not stored in the pgvector table.
        started = time.perf_counter()
        vector = self.embed_query(query)
        embedding_ms = (time.perf_counter() - started) * 1000
        candidate_k = max(top_k, candidate_k or top_k * 5)
        where_sql, where_values = self._where(filters)
        full_text_query = build_full_text_query(query)
        fields = (
            "chunk_id, parent_chunk_id, document_id, council, meeting_date, year, "
            "committee, meeting_name, agenda_title, priority, document, metadata"
        )
        dense_sql = f"""
            SELECT {fields}, 1 - (embedding <=> %s::vector) AS similarity
            FROM {self.table}
            WHERE embedding IS NOT NULL{where_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        text_sql = f"""
            WITH q AS (SELECT websearch_to_tsquery('simple', %s) AS term)
            SELECT {fields}, ts_rank_cd(to_tsvector('simple', coalesce(document, '')), q.term) AS text_rank_score
            FROM {self.table}, q
            WHERE to_tsvector('simple', coalesce(document, '')) @@ q.term{where_sql}
            ORDER BY text_rank_score DESC
            LIMIT %s
        """
        with self._connect() as connection:
            connection.execute(
                "SELECT set_config('hnsw.ef_search', %s, true)",
                (str(max(self.ef_search, candidate_k)),),
            )
            dense_rows = connection.execute(
                dense_sql, [vector, *where_values, vector, candidate_k]
            ).fetchall()
            text_rows = connection.execute(
                text_sql, [full_text_query, *where_values, candidate_k]
            ).fetchall()
        fused = reciprocal_rank_fusion(
            [dict(row) for row in dense_rows],
            [dict(row) for row in text_rows],
            limit=top_k,
            rrf_k=self.rrf_k,
            dense_weight=self.dense_weight,
            text_weight=self.text_weight,
        )
        return {
            "query": query,
            "backend": "postgresql_pgvector_full_text",
            "retrieval_mode": "hybrid_rrf",
            "result_count": len(fused),
            "results": [self._public_row(row) for row in fused],
            "search_ms": round((time.perf_counter() - started) * 1000, 3),
            "query_embedding_ms": round(embedding_ms, 3),
        }

    @staticmethod
    def _public_row(row: dict[str, Any]) -> dict[str, Any]:
        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            import json
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        merged_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        for field in ("council", "meeting_date", "year", "committee", "meeting_name", "agenda_title", "priority", "parent_chunk_id", "document_id"):
            if row.get(field) is not None:
                merged_metadata.setdefault(field, row[field].isoformat() if isinstance(row[field], (date, datetime)) else row[field])
        return {
            "rank": row.get("rank"),
            "chunk_id": str(row.get("chunk_id") or ""),
            "parent_chunk_id": row.get("parent_chunk_id"),
            "document_id": row.get("document_id"),
            "similarity": row.get("similarity"),
            "retrieval_score": row.get("retrieval_score"),
            "retrieval_sources": row.get("retrieval_sources") or [],
            "text": str(row.get("document") or row.get("text") or ""),
            "metadata": merged_metadata,
            "parent_context": None,
        }

    def monthly_counts(self, query: str, filters: Any | None = None) -> list[dict[str, Any]]:
        where_sql, where_values = self._where(filters)
        full_text_query = build_full_text_query(query)
        statement = f"""
            WITH q AS (SELECT websearch_to_tsquery('simple', %s) AS term)
            SELECT date_trunc('month', meeting_date)::date AS month,
                   min(meeting_date)::date AS first_observation_date,
                   count(DISTINCT document_id)::bigint AS document_count,
                   count(DISTINCT council)::bigint AS council_count,
                   array_agg(DISTINCT council) FILTER (WHERE council IS NOT NULL) AS councils
            FROM {self.table}, q
            WHERE meeting_date IS NOT NULL
              AND to_tsvector('simple', coalesce(document, '')) @@ q.term{where_sql}
            GROUP BY 1
            ORDER BY 1
        """
        with self._connect() as connection:
            rows = connection.execute(statement, [full_text_query, *where_values]).fetchall()
        result = []
        for row in rows:
            value = dict(row) if isinstance(row, dict) else {
                "month": row[0], "first_observation_date": row[1],
                "document_count": row[2], "council_count": row[3], "councils": row[4],
            }
            for field in ("month", "first_observation_date"):
                month = value.get(field)
                value[field] = month.isoformat() if isinstance(month, (date, datetime)) else str(month)
            value["document_count"] = int(value.get("document_count") or 0)
            value["council_count"] = int(value.get("council_count") or 0)
            value["councils"] = value.get("councils") or []
            result.append(value)
        return result

    def council_first_observations(
        self, query: str, filters: Any | None = None,
    ) -> list[dict[str, Any]]:
        where_sql, where_values = self._where(filters)
        full_text_query = build_full_text_query(query)
        statement = f"""
            WITH q AS (SELECT websearch_to_tsquery('simple', %s) AS term)
            SELECT council, min(meeting_date)::date AS first_observation_date,
                   count(DISTINCT document_id)::bigint AS document_count
            FROM {self.table}, q
            WHERE meeting_date IS NOT NULL AND council IS NOT NULL
              AND to_tsvector('simple', coalesce(document, '')) @@ q.term{where_sql}
            GROUP BY council
            ORDER BY first_observation_date, council
        """
        with self._connect() as connection:
            rows = connection.execute(statement, [full_text_query, *where_values]).fetchall()
        result = []
        for row in rows:
            value = dict(row) if isinstance(row, dict) else {
                "council": row[0], "first_observation_date": row[1], "document_count": row[2]
            }
            first_date = value.get("first_observation_date")
            value["first_observation_date"] = (
                first_date.isoformat() if isinstance(first_date, (date, datetime)) else str(first_date)
            )
            value["document_count"] = int(value.get("document_count") or 0)
            result.append(value)
        return result

    def get_document(self, document_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT chunk_id, parent_chunk_id, document_id, council, meeting_date, year,
                           committee, meeting_name, agenda_title, priority, document, metadata
                    FROM {self.table} WHERE document_id = %s
                    ORDER BY meeting_date, chunk_id LIMIT %s""",
                (document_id, limit),
            ).fetchall()
        return [self._public_row(dict(row)) for row in rows]

    def get_evidence(self, chunk_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                f"""SELECT chunk_id, parent_chunk_id, document_id, council, meeting_date, year,
                           committee, meeting_name, agenda_title, priority, document, metadata
                    FROM {self.table} WHERE chunk_id = %s LIMIT 1""",
                (chunk_id,),
            ).fetchone()
        return self._public_row(dict(row)) if row is not None else None

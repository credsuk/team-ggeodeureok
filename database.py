"""PostgreSQL/pgvector search with optional SQLite FTS5 BM25 + RRF.

The existing PostgreSQL table and API result fields are retained. If the BM25
index has not been built yet, search falls back to dense vector retrieval.
"""

from __future__ import annotations

import logging
import os
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import asyncpg

from config import settings

logger = logging.getLogger(__name__)


def _table_name() -> str:
    """Accept a simple schema-qualified table name from config only."""
    table = str(settings.db_table)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", table):
        raise ValueError(f"잘못된 DB 테이블명: {table!r}")
    return table


async def connect_db(database_url: str):
    """PostgreSQL 연결 풀을 생성하고 연결을 확인한다."""
    table = _table_name()
    pool = await asyncpg.create_pool(
        dsn=database_url,
        min_size=1,
        max_size=5,
        command_timeout=30,
    )
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
            await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
    except Exception:
        await pool.close()
        raise
    return pool


async def close_db(pool):
    """DB 연결 풀을 종료한다."""
    if pool is not None:
        await pool.close()


def _vector_literal(vector) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"


def _where_filters(council: str | None, year: int | None, start_index: int = 2):
    clauses: list[str] = []
    params: list[Any] = []
    if council:
        params.append(council)
        clauses.append(f"council = ${start_index + len(params) - 1}")
    if year is not None:
        params.append(int(year))
        clauses.append(f"year = ${start_index + len(params) - 1}")
    return clauses, params


def _row_payload(row: Any) -> dict[str, Any]:
    document = row.get("document") or ""
    meeting_date = row.get("meeting_date")
    chunk_id = str(row.get("chunk_id") or "")
    return {
        # Legacy UI/API keys retained for compatibility.
        "row_id": chunk_id,
        "date": str(meeting_date) if meeting_date is not None else "",
        "assembly": row.get("council") or "",
        "speaker": row.get("committee") or "",
        "agenda": row.get("agenda_title") or "",
        "summary": document,
        "preview": to_preview(document),
        "similarity": round(float(row.get("similarity") or 0.0), 4),
        # Canonical fields used by the new retrieval code.
        "chunk_id": chunk_id,
        "meeting_date": str(meeting_date) if meeting_date is not None else "",
        "council": row.get("council") or "",
        "committee": row.get("committee") or "",
        "meeting_name": row.get("meeting_name") or "",
        "agenda_title": row.get("agenda_title") or "",
        "document": document,
        "year": row.get("year"),
    }


async def _fetch_dense(pool, vector_str: str, candidate_k: int, council: str | None, year: int | None):
    table = _table_name()
    clauses, filter_params = _where_filters(council, year)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    # $1 is always the query vector; optional filters begin at $2.
    params = [vector_str, *filter_params, candidate_k]
    limit_param = len(params)
    sql = f"""
        SELECT chunk_id::text AS chunk_id, meeting_date, council, committee,
               meeting_name, agenda_title, document, year,
               (1 - (embedding <=> $1::vector)) AS similarity
        FROM {table}
        {where_sql}
        ORDER BY embedding <=> $1::vector
        LIMIT ${limit_param}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    return [_row_payload(row) for row in rows]


async def _hydrate_bm25(pool, hits, council: str | None, year: int | None):
    if not hits:
        return []
    table = _table_name()
    ids = [str(hit.chunk_id) for hit in hits]
    clauses = ["chunk_id::text = ANY($1::text[])"]
    params: list[Any] = [ids]
    if council:
        params.append(council)
        clauses.append(f"council = ${len(params)}")
    if year is not None:
        params.append(int(year))
        clauses.append(f"year = ${len(params)}")
    sql = f"""
        SELECT chunk_id::text AS chunk_id, meeting_date, council, committee,
               meeting_name, agenda_title, document, year,
               0.0::float8 AS similarity
        FROM {table}
        WHERE {' AND '.join(clauses)}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    by_id = {str(row["chunk_id"]): row for row in rows}
    hydrated = []
    for hit in hits:
        row = by_id.get(str(hit.chunk_id))
        if row is None:
            continue
        item = _row_payload(row)
        item["bm25_score"] = float(hit.score)
        hydrated.append(item)
    return hydrated


async def search_minutes(
    pool,
    vector,
    limit: int = 5,
    council: str | None = None,
    year: int | None = None,
    query: str | None = None,
):
    """Search existing pgvector data; use BM25+RRF when an index is available."""
    if limit < 1:
        raise ValueError("limit은 1 이상이어야 합니다")
    vector_str = _vector_literal(vector)
    candidate_k = min(max(limit * 10, 50), 100)
    dense_rows = await _fetch_dense(pool, vector_str, candidate_k, council, year)

    mode = os.getenv("COUNCIL_RETRIEVAL_MODE", "hybrid").strip().lower()
    index_path = Path(os.getenv(
        "COUNCIL_BM25_INDEX", "data/bm25/arctic_raw_v1/bm25.sqlite3"
    )).expanduser()
    if not index_path.is_absolute():
        index_path = Path(__file__).resolve().parent / index_path

    if mode != "hybrid" or not query or not index_path.is_file():
        if mode == "hybrid" and query and not index_path.is_file():
            logger.info("BM25 인덱스가 없어 dense 검색으로 처리합니다: %s", index_path)
        return dense_rows[:limit]

    try:
        from src.bm25_runtime import BM25Index, reciprocal_rank_fusion

        bm25_hits, _tokens = BM25Index(index_path).search(
            query=query,
            top_k=candidate_k,
            council=council,
            year=year,
        )
        bm25_rows = await _hydrate_bm25(pool, bm25_hits, council, year)
        if not bm25_rows:
            return dense_rows[:limit]
        return reciprocal_rank_fusion(
            dense_rows=dense_rows,
            bm25_rows=bm25_rows,
            limit=limit,
            rrf_k=int(os.getenv("COUNCIL_RRF_K", "60")),
            dense_weight=float(os.getenv("COUNCIL_DENSE_WEIGHT", "1.0")),
            bm25_weight=float(os.getenv("COUNCIL_BM25_WEIGHT", "1.0")),
        )
    except Exception as exc:
        # Missing/incompatible index must not break the existing UI/API.
        logger.warning("BM25 검색 실패; dense 검색으로 대체합니다: %s", exc)
        return dense_rows[:limit]


def to_preview(value, max_chars: int = 1800):
    """HTML 태그를 제거하고 텍스트 미리보기를 만든다."""
    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts = []

        def handle_data(self, data):
            self.parts.append(data)

    if value is None:
        return ""
    parser = TextExtractor()
    try:
        parser.feed(str(value))
        return " ".join(" ".join(parser.parts).split())[:max_chars]
    except Exception:
        return unescape(str(value))[:max_chars]

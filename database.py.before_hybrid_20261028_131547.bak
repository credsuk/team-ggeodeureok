from __future__ import annotations

import re
import asyncpg

from config import settings


def _query_tokens(query: str, max_tokens: int = 8) -> list[str]:
    """Extract simple Korean/English/numeric terms without requiring an external index."""
    tokens = re.findall(r"[0-9A-Za-z가-힣]{2,}", query.casefold())
    result = []
    seen = set()
    for token in tokens:
        if token not in seen:
            seen.add(token)
            result.append(token)
        if len(result) >= max_tokens:
            break
    return result


async def connect_db(database_url: str):
    pool = await asyncpg.create_pool(
        dsn=database_url,
        min_size=1,
        max_size=5,
        command_timeout=60,
    )
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
            await conn.fetchval(
                "SELECT 1 FROM public.council_arctic_raw_v1 LIMIT 1"
            )
    except Exception:
        await pool.close()
        raise
    return pool


async def close_db(pool):
    if pool is not None:
        await pool.close()


def to_preview(value, max_chars=1800):
    if value is None:
        return ""
    return " ".join(str(value).split())[:max_chars]


def _row_to_result(row):
    document = row.get("document") or ""
    return {
        "chunk_id": str(row["chunk_id"]),
        "row_id": str(row["chunk_id"]),
        "date": str(row["meeting_date"]) if row["meeting_date"] else "",
        "assembly": row["council"] or "",
        "committee": row["committee"] or "",
        "meetingName": row["meeting_name"] or "",
        "agenda": row["agenda_title"] or "",
        "speaker": "",
        "summary": "",
        "text": to_preview(document),
        "preview": to_preview(document),
        "parent": to_preview(document, 5000),
        "similarity": round(float(row.get("similarity") or 0.0), 4),
    }


def _rrf_merge(dense_rows, keyword_rows, limit, rrf_k=60):
    """Merge dense and keyword candidates by rank, not by incomparable raw scores."""
    merged = {}
    for source, rows in (("dense", dense_rows), ("keyword", keyword_rows)):
        for rank, row in enumerate(rows, start=1):
            chunk_id = str(row["chunk_id"])
            current = merged.setdefault(chunk_id, dict(row))
            current["rrf_score"] = current.get("rrf_score", 0.0) + 1.0 / (rrf_k + rank)
            current[f"{source}_rank"] = rank
            sources = set(current.get("retrieval_sources") or [])
            sources.add(source)
            current["retrieval_sources"] = sorted(sources)
            if source == "dense":
                current["dense_similarity"] = float(row.get("similarity") or 0.0)
            else:
                current["keyword_score"] = float(row.get("keyword_score") or 0.0)
                # Keep dense metadata/document if already present; otherwise use keyword row.
                for key in ("document", "council", "committee", "meeting_date",
                            "meeting_name", "agenda_title"):
                    if not current.get(key) and row.get(key):
                        current[key] = row[key]

    ranked = sorted(
        merged.values(),
        key=lambda row: (
            -float(row.get("rrf_score") or 0.0),
            int(row.get("dense_rank") or 10**9),
            int(row.get("keyword_rank") or 10**9),
            str(row.get("chunk_id") or ""),
        ),
    )[:limit]

    results = []
    for rank, row in enumerate(ranked, start=1):
        item = _row_to_result(row)
        item["rrf_rank"] = rank
        item["rrf_score"] = round(float(row["rrf_score"]), 9)
        item["dense_rank"] = row.get("dense_rank")
        item["keyword_rank"] = row.get("keyword_rank")
        item["keyword_score"] = round(float(row.get("keyword_score") or 0.0), 3)
        item["retrieval_sources"] = row.get("retrieval_sources", [])
        item["similarity"] = round(
            float(row.get("dense_similarity", row.get("similarity") or 0.0)), 4
        )
        results.append(item)
    return results


async def search_minutes(pool, vector, limit=5, council=None, year=None, query=None):
    """
    Hybrid retrieval without adding a DB index or creating a SQLite FTS file.
    Caution: the keyword branch can scan the 901k-row table and may be slow.
    """
    vector_str = "[" + ",".join(f"{float(v):.8f}" for v in vector) + "]"
    candidate_k = min(max(limit * 10, 50), 100)

    dense_sql = """
        SELECT
            chunk_id,
            meeting_date,
            council,
            committee,
            meeting_name,
            agenda_title,
            document,
            1 - (embedding <=> $1::vector) AS similarity
        FROM public.council_arctic_raw_v1
        WHERE ($2::text IS NULL OR council = $2)
          AND ($3::integer IS NULL OR year = $3)
        ORDER BY embedding <=> $1::vector
        LIMIT $4
    """

    tokens = _query_tokens(query or "")
    async with pool.acquire() as conn:
        dense_records = await conn.fetch(
            dense_sql, vector_str, council, year, candidate_k
        )
        dense_rows = [dict(row) for row in dense_records]

        if not tokens:
            return [_row_to_result(row) for row in dense_rows[:limit]]

        # Use parameterized ILIKE terms. No database or filesystem index is created.
        # LIMIT bounds returned candidates, but PostgreSQL may still scan many rows.
        patterns = [f"%{token}%" for token in tokens]
        token_conditions = []
        score_terms = []
        for i in range(len(patterns)):
            p = f"${i + 1}"
            token_conditions.append(
                f"(document ILIKE {p} OR COALESCE(agenda_title, '') ILIKE {p})"
            )
            score_terms.append(
                f"(CASE WHEN document ILIKE {p} THEN 1 ELSE 0 END "
                f"+ CASE WHEN COALESCE(agenda_title, '') ILIKE {p} THEN 2 ELSE 0 END)"
            )

        council_pos = len(patterns) + 1
        year_pos = len(patterns) + 2
        limit_pos = len(patterns) + 3
        keyword_sql = f"""
            SELECT
                chunk_id,
                meeting_date,
                council,
                committee,
                meeting_name,
                agenda_title,
                document,
                ({' + '.join(score_terms)})::float AS keyword_score
            FROM public.council_arctic_raw_v1
            WHERE ({' OR '.join(token_conditions)})
              AND (${council_pos}::text IS NULL OR council = ${council_pos})
              AND (${year_pos}::integer IS NULL OR year = ${year_pos})
            ORDER BY keyword_score DESC, meeting_date DESC NULLS LAST
            LIMIT ${limit_pos}
        """
        keyword_records = await conn.fetch(
            keyword_sql, *patterns, council, year, candidate_k
        )

    keyword_rows = [dict(row) for row in keyword_records]
    return _rrf_merge(dense_rows, keyword_rows, limit)

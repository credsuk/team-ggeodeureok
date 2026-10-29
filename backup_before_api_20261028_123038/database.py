
import asyncpg

from config import settings


async def connect_db(database_url: str):
    """PostgreSQL 연결 풀을 생성하고 연결을 확인한다."""
    pool = await asyncpg.create_pool(
        dsn=database_url,
        min_size=1,
        max_size=5,
        command_timeout=30,
    )

    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
            await conn.fetchval(
                f"SELECT COUNT(*) FROM {settings.db_table}"
            )
    except Exception:
        await pool.close()
        raise

    return pool


async def close_db(pool):
    """DB 연결 풀을 종료한다."""
    if pool is not None:
        await pool.close()


async def search_minutes(pool, vector, limit: int):
    """검색어 벡터와 유사한 회의록을 조회한다."""
    vector_str = "[" + ",".join(
        f"{float(value):.8f}" for value in vector
    ) + "]"

    sql = f"""
        SELECT
            row_id,
            date,
            assembly,
            speaker,
            agenda,
            summary,
            mints_html,
            (1 - (embedding <=> $1::vector)) AS similarity
        FROM {settings.db_table}
        ORDER BY embedding <=> $1::vector
        LIMIT $2;
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, vector_str, limit)

    results = []

    for row in rows:
        results.append({
            "row_id": row["row_id"],
            "date": (
                str(row["date"])
                if row["date"] is not None
                else ""
            ),
            "assembly": row["assembly"] or "",
            "speaker": row["speaker"] or "",
            "agenda": row["agenda"] or "",
            "summary": row["summary"] or "",
            "preview": to_preview(row["mints_html"]),
            "similarity": round(float(row["similarity"]), 4),
        })

    return results


def to_preview(value, max_chars: int = 1800):
    """HTML 회의록에서 태그를 제거하고 텍스트만 추출한다."""
    from html import unescape
    from html.parser import HTMLParser

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

# DB 기본 코드 -----------------------------------------------------------------------------------------------
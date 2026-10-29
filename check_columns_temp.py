import asyncio
from config import settings
import asyncpg

async def main():
    conn = await asyncpg.connect(settings.database_url)
    try:
        rows = await conn.fetch("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'council_arctic_raw_v1'
            ORDER BY ordinal_position
        """)
        for row in rows:
            print(f"{row['column_name']}: {row['data_type']}")
    finally:
        await conn.close()

asyncio.run(main())

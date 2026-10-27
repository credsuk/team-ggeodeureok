
import sqlite3
from pathlib import Path

# 1. DB 경로
DB_FILE = Path(
    r"C:\mini_team\_db\chroma_snowflake_regions\chroma.sqlite3"
)

# 2. 읽기 전용 연결
db_uri = DB_FILE.as_uri() + "?mode=ro"

with sqlite3.connect(db_uri, uri=True) as conn:

    # 3. 저장된 청크 1,000개 확인
    rows = conn.execute("""
        SELECT LENGTH(string_value)
        FROM embedding_metadata
        WHERE key = 'chroma:document'
          AND string_value IS NOT NULL
        LIMIT 1000
    """).fetchall()

    lengths = [row[0] for row in rows]

    # 4. 결과 출력
    if lengths:
        print("=== 청크 길이 검증 ===")
        print("검증 청크 수:", len(lengths))
        print("최소 길이:", min(lengths))
        print("최대 길이:", max(lengths))
        print("평균 길이:", round(sum(lengths) / len(lengths), 2))
        print("1000자 초과:", sum(x > 1000 for x in lengths))
    else:
        print("저장된 청크 텍스트가 없습니다.")

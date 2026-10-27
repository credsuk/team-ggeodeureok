
import sqlite3
from pathlib import Path

# 1. 실제 DB 경로
DB_FILE = Path(
    r"C:\mini_team\_db\chroma_snowflake_regions\chroma.sqlite3"
)

# 2. DB 존재 확인
if not DB_FILE.is_file():
    raise FileNotFoundError(DB_FILE)

print("DB 파일:", DB_FILE)
print("DB 크기(MB):",
      round(DB_FILE.stat().st_size / 1024**2, 2))

# 3. SQLite 읽기 전용 연결
db_uri = DB_FILE.as_uri() + "?mode=ro"

with sqlite3.connect(db_uri, uri=True) as conn:

    # 컬렉션 목록 확인
    collections = conn.execute(
        "SELECT id, name FROM collections"
    ).fetchall()

    print("\n컬렉션 수:", len(collections))

    for col_id, name in collections:
        print("컬렉션명:", name)
        print("컬렉션 ID:", col_id)

    # 저장된 전체 청크 수 확인
    count = conn.execute(
        "SELECT COUNT(*) FROM embeddings"
    ).fetchone()[0]

    print("\n전체 저장 청크:", f"{count:,}개")

print("\nDB 기본 검증 완료!")


from pathlib import Path

ROOT = Path(r"C:\mini_team\_db")

for db_file in ROOT.rglob("chroma.sqlite3"):
    print("DB 경로:", db_file.parent)
    print("파일 크기(MB):",
          round(db_file.stat().st_size / 1024**2, 2))
    print("-" * 50)



# 4. 저장된 청크 길이 확인
with sqlite3.connect(db_uri, uri=True) as conn:

    # 저장된 문서 중 무작위로 1,000개 추출
    rows = conn.execute("""
        SELECT
            LENGTH(string_value) AS text_length
        FROM embedding_metadata
        WHERE key = 'chroma:document'
          AND string_value IS NOT NULL
        LIMIT 1000
    """).fetchall()

    lengths = [row[0] for row in rows]

    if lengths:
        print("\n=== 청크 길이 검증 ===")
        print("검증 청크 수:", len(lengths))
        print("최소 길이:", min(lengths))
        print("최대 길이:", max(lengths))
        print("평균 길이:", round(sum(lengths) / len(lengths), 2))

        over_1000 = sum(x > 1000 for x in lengths)
        print("1000자 초과:", over_1000)
    else:
        print("저장된 문서 텍스트를 찾지 못했습니다.")

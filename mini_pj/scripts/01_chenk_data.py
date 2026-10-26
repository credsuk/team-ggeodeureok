# 청킹 데이터 구조 및 건수 확인

import pandas as pd
from pathlib import Path

# 청킹 CSV 파일 폴더
DATA_DIR = Path(r"data\processed\chunked_by_region")  # 실제 경로로 수정

# CSV 파일 목록
files = sorted(DATA_DIR.rglob("*.csv"))

print("CSV 파일 수:", len(files))

total = 0

for file in files[:3]:
    df = pd.read_csv(file, encoding="utf-8-sig")

    print("\n파일명:", file.name)
    print("데이터 수:", len(df))
    print("컬럼명:", df.columns.tolist())

    print("\n첫 번째 데이터:")
    print(df.head(1).to_string(index=False))

    total += len(df)

print("\n확인한 파일의 전체 청크 수:", total)
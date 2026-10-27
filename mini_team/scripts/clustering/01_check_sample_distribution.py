
import pandas as pd
from pathlib import Path

FILE = Path(
    r"C:\mini_team\scripts\clustering\results\cluster_sample_metadata.csv"
)

df = pd.read_csv(FILE, encoding="utf-8-sig")

print("\n=== 연도별 표본 ===")
print(df.groupby("year").size())

print("\n=== 지역·연도별 표본 ===")
result = (
    df.groupby(["year", "assembly"])
    .size()
    .reset_index(name="count")
    .sort_values(["year", "assembly"])
)

print(result.to_string(index=False))

print("\n전체 표본:", len(df))
print("지역·연도 조합:", len(result))

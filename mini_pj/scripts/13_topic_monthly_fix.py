
from pathlib import Path
from datetime import datetime
import pandas as pd
import time

ROOT = Path(r"C:\mini_pj\results\full_topic_modeling")
ASSIGN_DIR = ROOT / "assignments"
OUTPUT = ROOT / "웹_월별_토픽집계.csv"

start = time.perf_counter()
print(f"=== 월별 집계 복구 시작: {datetime.now():%Y-%m-%d %H:%M:%S} ===")

files = sorted(ASSIGN_DIR.glob("*_토픽할당.csv"))
print(f"토픽 할당 파일: {len(files)}개")

if not files:
    raise FileNotFoundError("토픽 할당 CSV를 찾지 못했습니다.")

groups = []
total_rows = 0
valid_rows = 0

for i, path in enumerate(files, 1):
    t0 = time.perf_counter()

    df = pd.read_csv(
        path,
        dtype={"date": "string"},
        low_memory=False
    )
    total_rows += len(df)

    # 20240214 → 2024-02-14 → 2024-02
    dates = pd.to_datetime(
        df["date"].str.strip(),
        format="%Y%m%d",
        errors="coerce"
    )

    df["month"] = dates.dt.strftime("%Y-%m")

    valid = df.dropna(subset=["month", "topic_id"])
    valid_rows += len(valid)

    group = (
        valid.groupby(["month", "topic_id"])
        .size()
        .reset_index(name="count")
    )

    groups.append(group)

    print(
        f"[{i}/{len(files)}] {path.name} | "
        f"{len(df):,}건 | 유효 {len(valid):,}건 | "
        f"{time.perf_counter()-t0:.2f}초"
    )

result = pd.concat(groups, ignore_index=True)

result = (
    result.groupby(["month", "topic_id"], as_index=False)["count"]
    .sum()
    .sort_values(["month", "topic_id"])
)

# 원본 집계 파일 보존
if OUTPUT.exists():
    backup = ROOT / "웹_월별_토픽집계_기존백업.csv"
    if not backup.exists():
        OUTPUT.replace(backup)

result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

elapsed = time.perf_counter() - start

print("\n=== 월별 집계 복구 완료 ===")
print(f"전체 처리: {total_rows:,}건")
print(f"유효 날짜: {valid_rows:,}건")
print(f"날짜 누락·오류: {total_rows-valid_rows:,}건")
print(f"집계 합계: {result['count'].sum():,}건")
print(f"집계 행 수: {len(result):,}건")
print(f"저장 위치: {OUTPUT}")
print(f"전체 소요 시간: {elapsed:.2f}초 / {elapsed/60:.2f}분")


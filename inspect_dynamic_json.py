import json
from pathlib import Path

root = Path("/app/outputs/dynamic_issues")
files = sorted(
    root.glob("query_*/05_dynamic_result.json"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

print("FILES:", len(files))
if not files:
    raise SystemExit("NO_RESULT_JSON")

path = files[0]
data = json.loads(path.read_text(encoding="utf-8"))
print("FILE:", path)

for key in (
    "issue", "summary", "propagation", "monthly_series",
    "forecast", "quality", "evidence",
):
    value = data.get(key)
    print("\nFIELD:", key, "TYPE:", type(value).__name__)

    if isinstance(value, dict):
        print("KEYS:", list(value.keys()))
    elif isinstance(value, list):
        print("COUNT:", len(value))
        if value and isinstance(value[0], dict):
            print("FIRST_ITEM_KEYS:", list(value[0].keys()))
            print("FIRST_ITEM_SAMPLE:", json.dumps(
                value[0], ensure_ascii=False
            )[:500])
    else:
        print("VALUE:", str(value)[:200])

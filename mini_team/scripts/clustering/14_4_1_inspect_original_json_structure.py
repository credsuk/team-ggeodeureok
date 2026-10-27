import json
from pathlib import Path

path = Path(r"C:\mini_team\org2\2025_광역 회의록\2025_충청남도의회.json")

with open(path, "r", encoding="utf-8-sig") as f:
    data = json.load(f)

print("최상위 자료형:", type(data).__name__)

if isinstance(data, dict):
    print("최상위 키:", list(data.keys()))
    for key, value in data.items():
        print(f"{key}: {type(value).__name__}")
        if isinstance(value, list) and value:
            print("  첫 번째 원소 자료형:", type(value[0]).__name__)
            if isinstance(value[0], dict):
                print("  첫 번째 원소 키:", list(value[0].keys()))

elif isinstance(data, list):
    print("전체 레코드 수:", len(data))
    if data:
        print("첫 번째 원소 자료형:", type(data[0]).__name__)
        if isinstance(data[0], dict):
            print("첫 번째 원소 키:", list(data[0].keys()))
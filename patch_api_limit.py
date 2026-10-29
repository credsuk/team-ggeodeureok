from pathlib import Path
p = Path("main.py")
s = p.read_text(encoding="utf-8-sig")
old = 'limit: int = Field(default=5, ge=1, le=10)'
new = 'limit: int = Field(default=5, ge=1, le=20)'
if old not in s:
    raise SystemExit("main.py의 limit 설정을 찾지 못했습니다. 파일은 변경하지 않았습니다.")
p.write_text(s.replace(old, new, 1), encoding="utf-8")
print("완료: API 검색 최대 건수를 20건으로 설정했습니다.")

from pathlib import Path # 파일과 폴더의 경로를 다루는 기능
BASE_DIR = Path(__file__).resolve().parent # 프로젝트 현재 기본 경로
FRONTEND_DIR = BASE_DIR / "frontend" # JS파일과 HTML파일 폴더 이동으로 코드 추가

import time
# 자바스크립트 코드에 시간을 넣어서 페이지 로드할때마다 강제 새로고침(버튼 미작동 오류 해결)

# 프론트엔드 기본 코드 -----------------------------------------------------------------------------------------------
import os

with open(FRONTEND_DIR / "index.html", "r", encoding="utf-8") as f:
    html = f.read()

with open(FRONTEND_DIR / "app.js", "r", encoding="utf-8") as f:
    js = f.read() 

target = '<script src="app.js"></script>'
if target in html:
    standalone = html.replace(target, f'<script defer src="app.js?v={int(time.time())}"></script>\n<script>\n{js}\n</script>') 
    with open(FRONTEND_DIR / "dashboard_standalone.html", "w", encoding="utf-8") as f:
        f.write(standalone)
    print("dashboard_standalone.html created successfully! Size:", os.path.getsize(FRONTEND_DIR / "dashboard_standalone.html"))
else:
    print("Target script tag not found!")

# 프론트엔드 기본 코드 -----------------------------------------------------------------------------------------------



# FastAPI 프런트엔드 연결 코드
from fastapi.staticfiles import StaticFiles

if not (FRONTEND_DIR / "index.html").is_file():
    raise RuntimeError(
        f"HTML 파일을 찾을 수 없습니다: {FRONTEND_DIR / 'index.html'}"
    )

# FastAPI에서 HTML, JavaScript, CSS, 이미지 같은 정적 파일을 웹 브라우저에 제공하는 기능
# app.js 파일이 있다면 브라우저가 서버에 요청해서 해당 파일을 받을 수 있게 함
frontend_app = StaticFiles(
    directory=str(FRONTEND_DIR),
    html=True,
)

import os

with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

with open("app.js", "r", encoding="utf-8") as f:
    js = f.read()

target = '<script src="app.js"></script>'
if target in html:
    standalone = html.replace(target, f'<script>\n{js}\n</script>')
    with open("dashboard_standalone.html", "w", encoding="utf-8") as f:
        f.write(standalone)
    print("dashboard_standalone.html created successfully! Size:", os.path.getsize("dashboard_standalone.html"))
else:
    print("Target script tag not found!")

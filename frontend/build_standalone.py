import os

def load_env():
    env_vars = {}
    for env_path in [".env", "../.env"]:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        env_vars[k.strip()] = v.strip().strip('"').strip("'")
    return env_vars

env = load_env()
carto_key = env.get("CARTO_API_KEY", "")

# 1. Sync local env.js from .env (Git-ignored)
if carto_key:
    with open("env.js", "w", encoding="utf-8") as f:
        f.write(f'// Local Environment Configuration (Generated from .env, Git-ignored)\nwindow.CARTO_API_KEY = "{carto_key}";\n')
    print(f"env.js successfully generated from .env (Key: {carto_key[:8]}...{carto_key[-6:]})")

with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

with open("app.js", "r", encoding="utf-8") as f:
    js = f.read()

app_target = '<script src="app.js"></script>'
if app_target in html:
    standalone = html.replace(app_target, f'<script>\n{js}\n</script>')
    with open("dashboard_standalone.html", "w", encoding="utf-8") as f:
        f.write(standalone)
    print("dashboard_standalone.html created successfully! Size:", os.path.getsize("dashboard_standalone.html"))
else:
    print("Target script tag not found!")

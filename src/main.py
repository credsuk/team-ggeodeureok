from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, MutableMapping
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

SRC_ROOT = Path(__file__).resolve().parent
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def load_env_file(path: Path, environ: MutableMapping[str, str] | None = None) -> int:
    """Load simple KEY=VALUE entries, without replacing shell-provided values."""
    target = os.environ if environ is None else environ
    if not path.is_file():
        return 0
    loaded = 0
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit() or key in target:
            continue
        try:
            parts = shlex.split(raw_value, comments=True, posix=True)
            value = " ".join(parts)
        except ValueError:
            value = raw_value.strip().strip("\"'")
        target[key] = value
        loaded += 1
    return loaded


def _local_http_url(url: str) -> tuple[str, str] | None:
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "http" or host not in LOCAL_HOSTS:
            return None
        port = parsed.port or 11434
        bind_host = f"[{host}]" if ":" in host else host
        return host, f"{bind_host}:{port}"
    except ValueError:
        return None


def ollama_is_reachable(url: str, timeout: float = 1.0) -> bool:
    try:
        with urlopen(url.rstrip("/") + "/api/tags", timeout=timeout) as response:
            return 200 <= response.status < 300
    except (OSError, URLError, ValueError):
        return False


def _print_log_tail(path: Path, *, lines: int = 80) -> None:
    print(f"--- 로그 {path} (마지막 {lines}줄) ---")
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        print("\n".join(content[-lines:]) if content else "(로그 내용 없음)")
    except OSError as exc:
        print(f"로그 파일을 읽지 못했습니다: {type(exc).__name__}: {exc}")


def _auto_ollama_enabled(environ: MutableMapping[str, str]) -> bool:
    return environ.get("AUTO_START_OLLAMA", "1").strip().lower() not in {"0", "false", "no", "off"}


def ensure_ollama(
    environ: MutableMapping[str, str] | None = None,
    *,
    wait_seconds: float = 10.0,
) -> bool:
    """Start a local Ollama server if requested and not already running."""
    env = os.environ if environ is None else environ
    url = env.get("OLLAMA_URL", "http://127.0.0.1:11434")
    if not _auto_ollama_enabled(env):
        print("Ollama 자동 시작은 AUTO_START_OLLAMA 설정으로 꺼져 있습니다.")
        return ollama_is_reachable(url)
    local = _local_http_url(url)
    if local is None:
        print(f"오류: 로컬 Ollama 자동 시작에 사용할 수 없는 주소입니다: {url}")
        return False
    if ollama_is_reachable(url):
        print("Ollama 서버가 이미 실행 중입니다.")
        return True

    executable = shutil.which("ollama") or shutil.which("ollama.exe")
    if not executable:
        print("오류: ollama 실행 파일을 찾을 수 없습니다. API 서버를 시작할 수 없습니다.")
        return False

    env_for_ollama = dict(env)
    env_for_ollama["OLLAMA_HOST"] = local[1]
    if executable.lower().endswith(".exe") and Path("/proc/version").is_file():
        if "microsoft" in Path("/proc/version").read_text(errors="ignore").lower():
            shared = env_for_ollama.get("WSLENV", "").split(":")
            if not any(value.split("/", 1)[0] == "OLLAMA_HOST" for value in shared):
                shared.append("OLLAMA_HOST")
            env_for_ollama["WSLENV"] = ":".join(value for value in shared if value)
    log_dir = SRC_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "ollama.log"
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            options: dict[str, Any] = {
                "cwd": str(SRC_ROOT),
                "env": env_for_ollama,
                "stdout": log_file,
                "stderr": subprocess.STDOUT,
            }
            if os.name == "nt":
                options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            else:
                options["start_new_session"] = True
            subprocess.Popen([executable, "serve"], **options)
    except OSError as exc:
        print(f"오류: Ollama 실행에 실패했습니다: {type(exc).__name__}: {exc}")
        _print_log_tail(log_path)
        return False

    print(f"Ollama 서버 시작을 요청했습니다. 로그: {log_path}")
    deadline = time.monotonic() + max(wait_seconds, 0.0)
    while time.monotonic() < deadline:
        if ollama_is_reachable(url, timeout=0.75):
            print("Ollama 서버 연결을 확인했습니다.")
            return True
        time.sleep(0.5)
    print("오류: Ollama가 제한 시간 안에 응답하지 않았습니다.")
    _print_log_tail(log_path)
    return False


def prepare_web_bundle(environ: MutableMapping[str, str] | None = None) -> bool:
    """Build the cached issue catalog when its inputs are already available."""
    env = os.environ if environ is None else environ
    web_dir = Path(env.get("COUNCIL_FINAL_WEB_DIR", SRC_ROOT / "outputs" / "web_final")).expanduser()
    if (web_dir / "issue_catalog.json").is_file():
        print("대표 이슈 웹 번들이 준비되어 있습니다.")
        return True
    source_web = Path(env.get("COUNCIL_WEB_DIR", SRC_ROOT / "outputs" / "web")).expanduser()
    forecast = Path(env.get(
        "COUNCIL_MULTI_FORECAST",
        SRC_ROOT / "outputs" / "forecast" / "multi_issue_v1" / "model" / "multi_issue_forecast.json",
    )).expanduser()
    if not forecast.is_file() or not source_web.is_dir() or not any(source_web.glob("*.json")):
        print("오류: 대표 이슈 결과 파일이 없어 웹 번들을 만들 수 없습니다. 분석 결과와 예측 파일을 먼저 준비해 주세요.")
        return False
    command = [sys.executable, "-u", "src/34_build_final_web_bundle.py",
               "--web-dir", str(source_web.resolve()), "--multi-forecast", str(forecast.resolve()),
               "--output-dir", str(web_dir.resolve())]
    try:
        subprocess.run(command, cwd=SRC_ROOT, check=True, capture_output=True, text=True,
                       timeout=int(env.get("COUNCIL_BUNDLE_TIMEOUT_SECONDS", "120")))
    except subprocess.CalledProcessError as exc:
        print(f"오류: 번들 생성 실패 (exit code {exc.returncode}). 실행 로그:")
        for label, value in (("stdout", exc.stdout), ("stderr", exc.stderr)):
            if value:
                text = value.decode(errors="replace") if isinstance(value, bytes) else value
                print(f"--- {label} ---")
                print(text[-12000:])
        return False
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"오류: 대표 이슈 번들 자동 생성에 실패했습니다: {type(exc).__name__}: {exc}")
        return False
    ready = (web_dir / "issue_catalog.json").is_file()
    if not ready:
        print("오류: 번들 생성 후 카탈로그 파일을 찾지 못했습니다.")
        return False
    print("대표 이슈 웹 번들 생성 완료.")
    return True


def validate_api_prerequisites() -> None:
    """Load the API's shared search runtime and verify the configured PG table."""
    dsn = os.environ.get("PGVECTOR_DSN")
    if not dsn:
        raise RuntimeError("PGVECTOR_DSN이 설정되지 않았습니다.")
    parent_path = Path(os.environ.get(
        "COUNCIL_PARENT_PATH", SRC_ROOT / "data" / "canonical_v3" / "chunks_v1" / "parents.parquet"
    )).expanduser().resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(f"부모 문맥 파일이 없습니다: {parent_path}")

    import psycopg
    from psycopg import sql
    from api.dependencies import get_api_service

    service = get_api_service()
    payloads = service._load_payloads()
    if not payloads:
        raise RuntimeError("대표 이슈 payload를 읽지 못했습니다.")
    search = service.search_service()
    if search is None:
        raise RuntimeError("검색 서비스를 초기화하지 못했습니다.")

    table_name = os.environ.get("COUNCIL_PG_TABLE", "council_arctic_raw_v1")
    statement = sql.SQL(
        "SELECT chunk_id, document, metadata, embedding FROM {} LIMIT 0"
    ).format(sql.Identifier(table_name))
    with psycopg.connect(dsn, connect_timeout=5) as connection:
        connection.execute(statement)
    print("대표 이슈 자료·검색 모델·pgvector 연결 확인 완료.")


def main() -> None:
    try:
        os.chdir(SRC_ROOT)
        if str(SRC_ROOT) not in sys.path:
            sys.path.insert(0, str(SRC_ROOT))

        loaded = load_env_file(SRC_ROOT / ".env")
        print(f"프로젝트 설정 로드 완료 (.env 항목 {loaded}개).")
        ollama_ready = ensure_ollama()
        if _auto_ollama_enabled(os.environ) and not ollama_ready:
            raise RuntimeError("Ollama 준비에 실패했습니다. 위 Ollama 로그를 확인하세요.")
        if not prepare_web_bundle():
            raise RuntimeError("대표 이슈 웹 번들 준비에 실패했습니다. 위 로그를 확인하세요.")
        validate_api_prerequisites()

        import uvicorn

        host = os.environ.get("HOST", "127.0.0.1")
        port = int(os.environ.get("PORT", "8000"))
        # Uvicorn prints the URL only after its socket binds successfully.
        uvicorn.run("api.main:app", host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        print("서버 시작이 사용자에 의해 취소되었습니다.")
    except Exception as exc:
        print(f"오류: 시작 단계가 실패해 API 서버를 시작하지 않았습니다: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stdout)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

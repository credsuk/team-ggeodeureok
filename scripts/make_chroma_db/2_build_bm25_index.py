from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import re
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "data" / "org_jsonl"
DEFAULT_DB_PATH = ROOT / "data" / "bm25_index" / "assembly_bm25.sqlite3"
BATCH_SIZE = 1000


@contextmanager
def sqlite_connection(path: Path) -> Iterator[sqlite3.Connection]:
    """Commit/rollback like a connection context manager, then always close it."""
    conn = sqlite3.connect(path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _windows_lock_holders(paths: list[Path]) -> list[str]:
    """Ask Windows Restart Manager which processes use the registered paths."""
    if os.name != "nt":
        return ["unavailable (not running on Windows)"]
    try:
        import ctypes
        from ctypes import wintypes

        class _FileTime(ctypes.Structure):
            _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

        class _UniqueProcess(ctypes.Structure):
            _fields_ = [("dwProcessId", wintypes.DWORD), ("ProcessStartTime", _FileTime)]

        class _ProcessInfo(ctypes.Structure):
            _fields_ = [
                ("Process", _UniqueProcess),
                ("strAppName", wintypes.WCHAR * 256),
                ("strServiceShortName", wintypes.WCHAR * 64),
                ("ApplicationType", ctypes.c_int),
                ("AppStatus", wintypes.ULONG),
                ("TSSessionId", wintypes.DWORD),
                ("bRestartable", wintypes.BOOL),
            ]

        rm = ctypes.WinDLL("rstrtmgr.dll")
        rm.RmStartSession.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD, ctypes.c_wchar_p]
        rm.RmStartSession.restype = wintypes.DWORD
        rm.RmRegisterResources.argtypes = [
            wintypes.DWORD, wintypes.UINT, ctypes.POINTER(ctypes.c_wchar_p),
            wintypes.UINT, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_wchar_p),
        ]
        rm.RmRegisterResources.restype = wintypes.DWORD
        rm.RmGetList.argtypes = [
            wintypes.DWORD, ctypes.POINTER(wintypes.UINT), ctypes.POINTER(wintypes.UINT),
            ctypes.POINTER(_ProcessInfo), ctypes.POINTER(wintypes.DWORD),
        ]
        rm.RmGetList.restype = wintypes.DWORD
        rm.RmEndSession.argtypes = [wintypes.DWORD]
        rm.RmEndSession.restype = wintypes.DWORD
        session = wintypes.DWORD()
        session_key = ctypes.create_unicode_buffer(33)
        rc = rm.RmStartSession(ctypes.byref(session), 0, session_key)
        if rc != 0:
            return [f"Restart Manager start failed (code={rc})"]
        try:
            names = (ctypes.c_wchar_p * len(paths))(*(str(path.resolve()) for path in paths))
            rc = rm.RmRegisterResources(session, len(paths), names, 0, None, 0, None)
            if rc != 0:
                return [f"Restart Manager resource registration failed (code={rc})"]

            needed = wintypes.UINT()
            count = wintypes.UINT()
            reboot_reasons = wintypes.DWORD()
            rc = rm.RmGetList(session, ctypes.byref(needed), ctypes.byref(count), None,
                              ctypes.byref(reboot_reasons))
            if rc == 0 and needed.value == 0:
                return []
            if rc not in (0, 234):  # ERROR_MORE_DATA = 234
                return [f"Restart Manager process query failed (code={rc})"]

            count = wintypes.UINT(needed.value)
            processes = (_ProcessInfo * count.value)()
            rc = rm.RmGetList(session, ctypes.byref(needed), ctypes.byref(count), processes,
                              ctypes.byref(reboot_reasons))
            if rc != 0:
                return [f"Restart Manager process list failed (code={rc})"]
            return [
                f"pid={processes[i].Process.dwProcessId} app={processes[i].strAppName!r}"
                for i in range(count.value)
            ]
        finally:
            rm.RmEndSession(session)
    except Exception as exc:
        return [f"Restart Manager diagnostic failed ({type(exc).__name__}: {exc})"]


def _file_state(path: Path) -> tuple[bool, int | None]:
    try:
        return True, path.stat().st_size
    except FileNotFoundError:
        return False, None
    except OSError:
        return path.exists(), None


def format_lock_diagnostic(stage: str, temp_path: Path, target_path: Path,
                           error: OSError, process_id: int | None = None) -> str:
    temp_exists, temp_size = _file_state(temp_path)
    target_exists, target_size = _file_state(target_path)
    winerror = getattr(error, "winerror", None)
    holders = _windows_lock_holders([temp_path, target_path]) if winerror == 32 else ["not queried"]
    return (
        "BM25 파일 교체 진단: "
        f"stage={stage}; pid={process_id or os.getpid()}; "
        f"winerror={winerror if winerror is not None else 'N/A'}; "
        f"errno={getattr(error, 'errno', None)}; error={error}; "
        f"temp_path={temp_path.resolve()}; temp_exists={temp_exists}; temp_size={temp_size}; "
        f"target_path={target_path.resolve()}; target_exists={target_exists}; "
        f"target_size={target_size}; lock_holders={holders}"
    )


def make_record_id(record: dict[str, Any]) -> str:
    source_file = record.get("source_file")
    row_id = record.get("row_id")
    if not isinstance(source_file, str) or not source_file.strip():
        raise ValueError("source_file이 비어 있거나 문자열이 아닙니다.")
    if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id < 1:
        raise ValueError(f"row_id는 1 이상의 정수여야 합니다: {row_id!r}")
    return f"{Path(source_file).stem}:{row_id}"


def record_text(record: dict[str, Any]) -> str:
    text = record.get("utterance") or record.get("summary")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("BM25 색인을 만들 utterance 또는 summary 텍스트가 없습니다.")
    return text.strip()


def initialize_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite_connection(path) as conn:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("""
            CREATE VIRTUAL TABLE bm25_docs USING fts5(
                chroma_id UNINDEXED,
                source_file UNINDEXED,
                row_id UNINDEXED,
                text,
                tokenize='unicode61'
            )
        """)
        conn.execute("PRAGMA user_version=1")


def iter_records(input_dir: Path, max_records: int | None = None):
    count = 0
    paths = sorted(input_dir.rglob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"입력 JSONL이 없습니다: {input_dir}")
    for path in paths:
        expected_row_id = 0
        with path.open("r", encoding="utf-8-sig") as source:
            for line_no, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSON 오류: {path}:{line_no}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"JSON 객체가 아닙니다: {path}:{line_no}")
                chroma_id = make_record_id(record)
                text = record_text(record)
                source_file = str(record["source_file"])
                row_id = str(record["row_id"])
                expected_row_id += 1
                if record["row_id"] != expected_row_id:
                    raise ValueError(
                        f"{path.name}: JSONL 레코드 순번 {expected_row_id}와 "
                        f"row_id={row_id}가 일치하지 않습니다."
                    )
                yield chroma_id, source_file, row_id, text
                count += 1
                if max_records is not None and count >= max_records:
                    return


def build_index(input_dir: Path, db_path: Path, max_records: int | None = None,
                batch_size: int = BATCH_SIZE) -> int:
    if max_records is not None and max_records < 1:
        raise ValueError("max_records는 1 이상이어야 합니다.")
    if batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다.")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{db_path.name}.", suffix=".tmp", dir=db_path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    total = 0
    stage = "initialize temporary database"
    try:
        initialize_database(temp_path)
        stage = "populate and close temporary database"
        with sqlite_connection(temp_path) as conn:
            batch = []
            for record in iter_records(input_dir, max_records):
                batch.append(record)
                if len(batch) >= batch_size:
                    conn.executemany(
                        "INSERT INTO bm25_docs(chroma_id, source_file, row_id, text) VALUES (?, ?, ?, ?)",
                        batch,
                    )
                    total += len(batch)
                    batch.clear()
                    if total % 50_000 == 0:
                        print(f"BM25 색인 중: {total:,}건", flush=True)
            if batch:
                conn.executemany(
                    "INSERT INTO bm25_docs(chroma_id, source_file, row_id, text) VALUES (?, ?, ?, ?)",
                    batch,
                )
                total += len(batch)
        stage = "atomic replace"
        try:
            os.replace(temp_path, db_path)
        except OSError as exc:
            print(format_lock_diagnostic(stage, temp_path, db_path, exc), file=sys.stderr)
            raise
    except Exception as exc:
        print(
            f"BM25 빌드 실패 진단: stage={stage}; inserted={total}; "
            f"temp_path={temp_path.resolve()}; target_path={db_path.resolve()}; "
            f"exception={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as cleanup_exc:
            print(
                format_lock_diagnostic("temporary-file cleanup", temp_path, db_path, cleanup_exc),
                file=sys.stderr,
            )
        raise
    print(f"BM25 색인 완료: {total:,}건 -> {db_path}")
    return total


def make_fts_query(query: str) -> str:
    terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
    if not terms:
        return ""
    quoted = ['"' + term.replace('"', '""') + '"' for term in terms]
    return " OR ".join(quoted)


def search_bm25(db_path: Path, query: str, limit: int = 10) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit는 1 이상이어야 합니다.")
    fts_query = make_fts_query(query)
    if not fts_query:
        return []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT chroma_id, source_file, row_id,
                   bm25(bm25_docs) AS score,
                   snippet(bm25_docs, 3, '[', ']', ' … ', 18) AS excerpt
            FROM bm25_docs
            WHERE bm25_docs MATCH ?
            ORDER BY score ASC
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="org_jsonl 원문 발언으로 SQLite FTS5 BM25 색인을 생성하거나 검색합니다."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--max-records", type=int, default=None,
                        help="시험 색인용 상한. 지정하면 부분 색인으로 DB를 새로 만듭니다.")
    parser.add_argument("--query", default=None, help="기존 BM25 DB 검색어")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    try:
        if args.query is not None:
            results = search_bm25(args.db, args.query, args.limit)
            for rank, item in enumerate(results, start=1):
                print(f"{rank}. {item['chroma_id']} | {item['source_file']} row_id={item['row_id']} | score={item['score']:.6f}")
                print(f"   {item['excerpt']}")
            print(f"검색 결과: {len(results)}건")
        else:
            build_index(args.input_dir, args.db, args.max_records)
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

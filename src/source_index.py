from __future__ import annotations

import argparse
import csv
import io
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from typing import Any, Iterator, TextIO

REQUIRED_COLUMNS = {"MTG_DE", "RASMBLY_NM", "MTGNM", "MINTS_NM", "agenda_title", "utterance"}


@contextmanager
def sqlite_connection(path: Path | str, *, timeout: float = 5.0) -> Iterator[sqlite3.Connection]:
    """Commit/rollback within the block, then explicitly release the file handle."""
    conn = sqlite3.connect(path, timeout=timeout)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


class _CountingLines:
    """Track UTF-8 byte positions at CSV record boundaries, including quoted newlines."""

    def __init__(self, source: TextIO):
        self.source = source
        self.byte_count = 0

    def __iter__(self) -> "_CountingLines":
        return self

    def __next__(self) -> str:
        line = self.source.readline()
        if line == "":
            raise StopIteration
        self.byte_count += len(line.encode("utf-8"))
        return line


def _normalize_date(value: str) -> str:
    digits = "".join(ch for ch in (value or "") if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else digits


def build_source_index(input_dir: Path | str, db_path: Path | str, batch_size: int = 5000) -> int:
    """Index row metadata/byte offsets only; do not copy utterance text into SQLite."""
    input_dir, db_path = Path(input_dir), Path(db_path)
    paths = sorted(input_dir.rglob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"원본 TXT 파일이 없습니다: {input_dir}")
    if batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다.")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{db_path.name}.", suffix=".tmp", dir=db_path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    total = 0
    stage = "create temporary database schema"
    try:
        with sqlite_connection(tmp_path) as conn:
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("""CREATE TABLE source_metadata (
                doc_id TEXT PRIMARY KEY,
                source_file TEXT NOT NULL,
                row_id INTEGER NOT NULL,
                byte_offset INTEGER NOT NULL,
                date TEXT NOT NULL,
                assembly TEXT NOT NULL,
                meeting_title TEXT NOT NULL,
                minutes_title TEXT NOT NULL,
                agenda TEXT NOT NULL
            )""")
            conn.execute("CREATE INDEX idx_source_metadata_date ON source_metadata(date)")
            conn.execute("CREATE INDEX idx_source_metadata_assembly_date ON source_metadata(assembly, date)")
            for path in paths:
                stage = f"read source file {path.name}"
                with path.open("r", encoding="utf-8", newline="") as source:
                    counted = _CountingLines(source)
                    reader = csv.DictReader(counted, delimiter="\t")
                    if reader.fieldnames:
                        reader.fieldnames = [name.lstrip("\ufeff") for name in reader.fieldnames]
                    headers = set(reader.fieldnames or [])
                    missing = REQUIRED_COLUMNS - headers
                    if missing:
                        raise ValueError(f"필수 TXT 열이 없습니다: {path.name}: {sorted(missing)}")
                    batch: list[tuple[Any, ...]] = []
                    record_start = counted.byte_count
                    for row_id, row in enumerate(reader, start=1):
                        doc_id = f"{path.stem}:{row_id}"
                        batch.append((
                            doc_id, path.stem, row_id, record_start,
                            _normalize_date(row.get("MTG_DE", "")),
                            (row.get("RASMBLY_NM") or "").strip(),
                            (row.get("MTGNM") or "").strip(),
                            (row.get("MINTS_NM") or "").strip(),
                            (row.get("agenda_title") or "").strip(),
                        ))
                        record_start = counted.byte_count
                        if len(batch) >= batch_size:
                            stage = f"insert batch from {path.name}"
                            conn.executemany("INSERT INTO source_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
                            total += len(batch)
                            batch.clear()
                    if batch:
                        stage = f"insert final batch from {path.name}"
                        conn.executemany("INSERT INTO source_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
                        total += len(batch)
        stage = "atomic replace"
        os.replace(tmp_path, db_path)
    except Exception as exc:
        print(
            f"source index build failed: stage={stage}; indexed={total}; "
            f"temp_path={tmp_path.resolve()}; target_path={db_path.resolve()}; "
            f"pid={os.getpid()}; original_exception={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            print(
                "source index cleanup failed: "
                f"stage=cleanup-after-{stage}; temp_path={tmp_path.resolve()}; "
                f"target_path={db_path.resolve()}; pid={os.getpid()}; "
                f"winerror={getattr(cleanup_exc, 'winerror', None)}; "
                f"errno={getattr(cleanup_exc, 'errno', None)}; "
                f"temp_exists={tmp_path.exists()}; "
                f"original_exception={type(exc).__name__}: {exc}; "
                f"cleanup_exception={type(cleanup_exc).__name__}: {cleanup_exc}",
                file=sys.stderr,
            )
        raise
    return total


def ensure_source_index(input_dir: Path | str, db_path: Path | str) -> bool:
    """Build the source sidecar only when absent; never overwrite an existing index silently."""
    input_dir, db_path = Path(input_dir), Path(db_path)
    if db_path.exists():
        try:
            with sqlite_connection(db_path) as conn:
                columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(source_metadata)").fetchall()
                }
                missing = {
                    "doc_id", "source_file", "row_id", "byte_offset", "date", "assembly",
                    "meeting_title", "minutes_title", "agenda",
                } - columns
                has_rows = (
                    conn.execute("SELECT 1 FROM source_metadata LIMIT 1").fetchone() is not None
                    if "doc_id" in columns else False
                )
        except sqlite3.Error as exc:
            raise RuntimeError(
                f"소스 메타데이터 DB가 있지만 읽을 수 없습니다. 자동 재생성은 하지 않았습니다: {db_path}"
            ) from exc
        if missing or not has_rows:
            raise RuntimeError(
                f"소스 메타데이터 DB가 있지만 유효하지 않습니다. 자동 재생성은 하지 않았습니다: "
                f"{db_path}; missing_columns={sorted(missing)}; has_rows={has_rows}"
            )
        return False
    build_source_index(input_dir, db_path)
    return True


def lookup_source_metadata(db_path: Path | str, doc_ids: list[str]) -> list[dict[str, Any]]:
    if not doc_ids:
        return []
    with sqlite_connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        placeholders = ",".join("?" for _ in doc_ids)
        rows = conn.execute(
            f"SELECT * FROM source_metadata WHERE doc_id IN ({placeholders})", doc_ids
        ).fetchall()
    by_id = {row["doc_id"]: dict(row) for row in rows}
    return [by_id[item] for item in doc_ids if item in by_id]


class SourceTextReader:
    """Read exact utterance text from original TSV/TXT only for selected row IDs."""

    def __init__(self, metadata_db: Path | str, source_dir: Path | str):
        self.metadata_db = Path(metadata_db)
        self.source_dir = Path(source_dir)
        self._source_paths: dict[str, Path] = {}

    def _resolve_source_path(self, source_file: str) -> Path:
        cached = self._source_paths.get(source_file)
        if cached is not None:
            return cached
        direct = self.source_dir / f"{source_file}.txt"
        if direct.is_file():
            self._source_paths[source_file] = direct
            return direct
        matches = list(self.source_dir.rglob(f"{source_file}.txt"))
        if not matches:
            raise FileNotFoundError(f"원본 TXT를 찾을 수 없습니다: {direct}")
        if len(matches) > 1:
            raise ValueError(
                f"같은 이름의 원본 TXT가 여러 개라 구분할 수 없습니다: {source_file}.txt"
            )
        self._source_paths[source_file] = matches[0]
        return matches[0]

    def read(self, doc_ids: list[str]) -> dict[str, str]:
        rows = lookup_source_metadata(self.metadata_db, doc_ids)
        result: dict[str, str] = {}
        headers_by_file: dict[str, list[str]] = {}
        for row in rows:
            path = self._resolve_source_path(row["source_file"])
            key = row["source_file"]
            if key not in headers_by_file:
                with path.open("r", encoding="utf-8-sig", newline="") as source:
                    headers = next(csv.reader(source, delimiter="\t"), [])
                headers_by_file[key] = [name.lstrip("\ufeff") for name in headers]
            try:
                utterance_index = headers_by_file[key].index("utterance")
            except ValueError as exc:
                raise ValueError(f"TXT에 utterance 열이 없습니다: {path.name}") from exc
            with path.open("rb") as raw:
                raw.seek(int(row["byte_offset"]))
                with io.TextIOWrapper(raw, encoding="utf-8", newline="") as source:
                    values = next(csv.reader(source, delimiter="\t"), [])
            result[row["doc_id"]] = values[utterance_index] if utterance_index < len(values) else ""
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="TXT에서 회의 메타데이터/행 오프셋만 SQLite로 인덱싱합니다.")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--input-dir", type=Path, default=root / "data" / "to_text")
    parser.add_argument("--db", type=Path, default=root / "data" / "source_metadata.sqlite3")
    parser.add_argument("--force", action="store_true", help="기존 인덱스가 있어도 다시 생성합니다.")
    args = parser.parse_args()
    if args.force:
        count = build_source_index(args.input_dir, args.db)
        print(f"회의 메타데이터 인덱싱 완료: {count:,}건 -> {args.db}")
    elif ensure_source_index(args.input_dir, args.db):
        print(f"회의 메타데이터 인덱싱 완료 -> {args.db}")
    else:
        print(f"기존 유효한 회의 메타데이터 인덱스를 재사용합니다: {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

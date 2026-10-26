#!/usr/bin/env python3
"""원본 지방의회 JSON을 신뢰 가능한 canonical 문서·부모·검색 청크로 변환한다.

외부 API나 LLM을 호출하지 않는다. RASMBLY_NM, MTG_DE, MTGNM, MINTS_NM은
원본 JSON에서 직접 상속하고 MINTS_HTML만 로컬에서 텍스트로 정제한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import ijson
import pyarrow as pa
import pyarrow.parquet as pq
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATTERN = re.compile(r"^(?P<year>20\d{2})_(?P<council>.+?)(?: \(\d+\))?\.json$")
VERSION = "raw-json-canonical-v1.1"
UI_EXACT_LINES = frozenset({"접기", "펼치기"})

DOCUMENT_SCHEMA = pa.schema([
    ("document_id", pa.string()),
    ("council", pa.string()),
    ("meeting_date", pa.string()),
    ("year", pa.int16()),
    ("committee", pa.string()),
    ("meeting_name", pa.string()),
    ("source_file", pa.string()),
    ("source_record_no", pa.int64()),
    ("text", pa.string()),
    ("text_sha256", pa.string()),
    ("html_sha256", pa.string()),
    ("char_length", pa.int32()),
    ("metadata_source", pa.string()),
    ("metadata_version", pa.string()),
])

PARENT_SCHEMA = pa.schema([
    ("parent_chunk_id", pa.string()),
    ("document_id", pa.string()),
    ("council", pa.string()),
    ("meeting_date", pa.string()),
    ("year", pa.int16()),
    ("committee", pa.string()),
    ("meeting_name", pa.string()),
    ("source_file", pa.string()),
    ("source_record_no", pa.int64()),
    ("parent_index", pa.int32()),
    ("char_start", pa.int32()),
    ("char_end", pa.int32()),
    ("text", pa.string()),
    ("text_sha256", pa.string()),
    ("char_length", pa.int32()),
    ("metadata_source", pa.string()),
    ("metadata_version", pa.string()),
])

CHUNK_SCHEMA = pa.schema([
    ("chunk_id", pa.string()),
    ("document_id", pa.string()),
    ("parent_chunk_id", pa.string()),
    ("council", pa.string()),
    ("meeting_date", pa.string()),
    ("year", pa.int16()),
    ("committee", pa.string()),
    ("meeting_name", pa.string()),
    ("agenda_title", pa.string()),
    ("speaker", pa.string()),
    ("speaker_role", pa.string()),
    ("speaker_type", pa.string()),
    ("chunk_type", pa.string()),
    ("priority", pa.string()),
    ("source_file", pa.string()),
    ("source_record_no", pa.int64()),
    ("source_line_no", pa.int64()),
    ("parent_index", pa.int32()),
    ("child_index", pa.int32()),
    ("char_start", pa.int32()),
    ("char_end", pa.int32()),
    ("text", pa.string()),
    ("embedding_text", pa.string()),
    ("text_sha256", pa.string()),
    ("char_length", pa.int32()),
    ("replacement_char_count", pa.int16()),
    ("metadata_source", pa.string()),
    ("metadata_version", pa.string()),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path.home() / "projects" / "council-minutes" / "data" / "raw",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "canonical" / "raw_v1",
    )
    parser.add_argument(
        "--key",
        action="append",
        help="처리할 연도_의회 키. 여러 번 지정 가능하며 생략하면 전체 처리",
    )
    parser.add_argument("--max-records-per-key", type=int)
    parser.add_argument("--parent-chars", type=int, default=3600)
    parser.add_argument("--child-chars", type=int, default=900)
    parser.add_argument("--child-overlap", type=int, default=150)
    parser.add_argument("--min-tail-chars", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_value(value: object) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def normalize_date(value: object) -> str | None:
    digits = re.sub(r"[^0-9]", "", str(value or ""))
    if len(digits) < 8:
        return None
    try:
        return datetime.strptime(digits[:8], "%Y%m%d").date().isoformat()
    except ValueError:
        return None


def html_to_clean_text(value: object) -> tuple[str, int]:
    if value is None:
        return "", 0
    soup = BeautifulSoup(str(value), "lxml")
    for tag in soup(("script", "style", "noscript")):
        tag.decompose()
    for tag in soup.find_all("br"):
        tag.replace_with("\n")
    raw_text = soup.get_text("\n")
    lines = []
    removed_ui_lines = 0
    for line in raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", line).strip()
        if not line:
            continue
        if line in UI_EXACT_LINES:
            removed_ui_lines += 1
            continue
        lines.append(line)
    return "\n".join(lines), removed_ui_lines


def stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join("" if part is None else str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]}"


def choose_boundary(text: str, start: int, hard_end: int, minimum_position: int) -> int:
    if hard_end >= len(text):
        return len(text)
    candidates = []
    for marker in ("\n", ".", "?", "!", "。", "？", "！"):
        position = text.rfind(marker, minimum_position, hard_end)
        if position >= minimum_position:
            candidates.append(position + 1)
    return max(candidates) if candidates else hard_end


def make_spans(text: str, max_chars: int, overlap: int, min_tail: int) -> list[tuple[int, int]]:
    if not text:
        return []
    spans = []
    start = 0
    while start < len(text):
        remaining = len(text) - start
        if remaining <= max_chars:
            spans.append((start, len(text)))
            break
        hard_end = min(len(text), start + max_chars)
        minimum_position = start + max(1, int(max_chars * 0.55))
        end = choose_boundary(text, start, hard_end, minimum_position)
        if len(text) - end < min_tail:
            end = len(text)
        if end <= start:
            end = hard_end
        spans.append((start, end))
        if end >= len(text):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return spans


def discover_raw_files(raw_dir: Path) -> dict[str, Path]:
    files = {}
    for path in sorted(raw_dir.glob("*.json")):
        match = RAW_PATTERN.match(path.name)
        if not match:
            continue
        key = f"{match.group('year')}_{match.group('council')}"
        if key in files:
            raise ValueError(f"연도·의회 원본 JSON이 중복됩니다: {key}")
        files[key] = path
    return files


def open_utf8_json(path: Path):
    file = path.open("rb")
    prefix = file.read(4)
    if prefix.startswith(b"\xef\xbb\xbf"):
        file.seek(3)
    elif prefix.startswith(b"\xff\xfe") or prefix.startswith(b"\xfe\xff"):
        file.close()
        raise UnicodeError(f"UTF-16 파일은 UTF-8 변환이 필요합니다: {path}")
    else:
        file.seek(0)
    return file


class BufferedParquetWriter:
    def __init__(self, path: Path, schema: pa.Schema, batch_size: int):
        self.path = path
        self.temporary = path.with_name(path.name + ".building")
        if self.temporary.exists():
            self.temporary.unlink()
        self.writer = pq.ParquetWriter(self.temporary, schema=schema, compression="zstd")
        self.schema = schema
        self.batch_size = batch_size
        self.rows: list[dict] = []
        self.count = 0

    def append(self, row: dict) -> None:
        self.rows.append(row)
        self.count += 1
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.rows.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()

    def commit(self) -> None:
        self.temporary.replace(self.path)

    def abort(self) -> None:
        try:
            self.writer.close()
        finally:
            if self.temporary.exists():
                self.temporary.unlink()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise NotADirectoryError(raw_dir)
    if args.parent_chars < 500 or args.child_chars < 100:
        raise ValueError("--parent-chars는 500 이상, --child-chars는 100 이상이어야 합니다.")
    if not 0 <= args.child_overlap < args.child_chars:
        raise ValueError("--child-overlap은 0 이상 child-chars 미만이어야 합니다.")
    if args.min_tail_chars < 1 or args.batch_size < 1:
        raise ValueError("--min-tail-chars와 --batch-size는 1 이상이어야 합니다.")

    discovered = discover_raw_files(raw_dir)
    selected_keys = args.key or sorted(discovered)
    missing = sorted(set(selected_keys) - set(discovered))
    if missing:
        raise FileNotFoundError(f"원본 JSON 키가 없습니다: {missing}")
    selected = [(key, discovered[key]) for key in selected_keys]

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "documents": output_dir / "documents.parquet",
        "parents": output_dir / "parents.parquet",
        "chunks": output_dir / "chunks.parquet",
        "report": output_dir / "build_report.json",
    }
    for path in paths.values():
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"출력이 존재합니다. 덮어쓰려면 --overwrite: {path}")

    writers = {
        "documents": BufferedParquetWriter(paths["documents"], DOCUMENT_SCHEMA, min(args.batch_size, 1000)),
        "parents": BufferedParquetWriter(paths["parents"], PARENT_SCHEMA, args.batch_size),
        "chunks": BufferedParquetWriter(paths["chunks"], CHUNK_SCHEMA, args.batch_size),
    }
    totals = Counter()
    file_reports = []
    started = time.perf_counter()

    try:
        for file_number, (key, path) in enumerate(selected, start=1):
            print(f"[{file_number}/{len(selected)}] {key} | {path.name}", flush=True)
            file_counts = Counter()
            expected_year, expected_council = key.split("_", 1)
            with open_utf8_json(path) as file:
                for record_index, record in enumerate(ijson.items(file, "item")):
                    if args.max_records_per_key is not None and record_index >= args.max_records_per_key:
                        break
                    file_counts["raw_records"] += 1
                    totals["raw_records"] += 1
                    if not isinstance(record, dict):
                        file_counts["invalid_record_type"] += 1
                        continue

                    council = clean_value(record.get("RASMBLY_NM")) or expected_council
                    meeting_date = normalize_date(record.get("MTG_DE"))
                    committee = clean_value(record.get("MTGNM"))
                    meeting_name = clean_value(record.get("MINTS_NM"))
                    html = str(record.get("MINTS_HTML") or "")
                    if council != expected_council:
                        file_counts["council_mismatch"] += 1
                    if not meeting_date:
                        file_counts["invalid_date"] += 1
                        continue
                    text, removed_ui_lines = html_to_clean_text(html)
                    file_counts["removed_ui_lines"] += removed_ui_lines
                    totals["removed_ui_lines"] += removed_ui_lines
                    if not text:
                        file_counts["empty_text"] += 1
                        continue

                    year = int(meeting_date[:4])
                    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    document_id = stable_id(
                        "doc", key, record_index, meeting_date, committee, meeting_name, text_hash,
                    )
                    source_common = {
                        "document_id": document_id,
                        "council": council,
                        "meeting_date": meeting_date,
                        "year": year,
                        "committee": committee,
                        "meeting_name": meeting_name,
                        "source_file": path.name,
                        "source_record_no": record_index,
                    }
                    writers["documents"].append({
                        **source_common,
                        "text": text,
                        "text_sha256": text_hash,
                        "html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
                        "char_length": len(text),
                        "metadata_source": "competition_raw_json",
                        "metadata_version": VERSION,
                    })
                    file_counts["documents"] += 1
                    file_counts["document_chars"] += len(text)
                    totals["documents"] += 1
                    totals["document_chars"] += len(text)

                    parent_spans = make_spans(text, args.parent_chars, 0, args.min_tail_chars)
                    for parent_index, (parent_start, parent_end) in enumerate(parent_spans):
                        parent_text = text[parent_start:parent_end]
                        parent_id = stable_id("parent", document_id, parent_index, parent_start, parent_end)
                        writers["parents"].append({
                            "parent_chunk_id": parent_id,
                            **source_common,
                            "parent_index": parent_index,
                            "char_start": parent_start,
                            "char_end": parent_end,
                            "text": parent_text,
                            "text_sha256": hashlib.sha256(parent_text.encode("utf-8")).hexdigest(),
                            "char_length": len(parent_text),
                            "metadata_source": "competition_raw_json",
                            "metadata_version": VERSION,
                        })
                        file_counts["parents"] += 1
                        totals["parents"] += 1

                        child_spans = make_spans(
                            parent_text,
                            args.child_chars,
                            args.child_overlap,
                            args.min_tail_chars,
                        )
                        for child_index, (local_start, local_end) in enumerate(child_spans):
                            child_start = parent_start + local_start
                            child_end = parent_start + local_end
                            child_text = text[child_start:child_end]
                            child_id = stable_id(
                                "chunk", parent_id, child_index, child_start, child_end,
                                hashlib.sha256(child_text.encode("utf-8")).hexdigest(),
                            )
                            writers["chunks"].append({
                                "chunk_id": child_id,
                                "document_id": document_id,
                                "parent_chunk_id": parent_id,
                                "council": council,
                                "meeting_date": meeting_date,
                                "year": year,
                                "committee": committee,
                                "meeting_name": meeting_name,
                                "agenda_title": None,
                                "speaker": None,
                                "speaker_role": None,
                                "speaker_type": None,
                                "chunk_type": "child",
                                "priority": None,
                                "source_file": path.name,
                                "source_record_no": record_index,
                                "source_line_no": record_index,
                                "parent_index": parent_index,
                                "child_index": child_index,
                                "char_start": child_start,
                                "char_end": child_end,
                                "text": child_text,
                                "embedding_text": child_text,
                                "text_sha256": hashlib.sha256(child_text.encode("utf-8")).hexdigest(),
                                "char_length": len(child_text),
                                "replacement_char_count": child_text.count("\ufffd"),
                                "metadata_source": "competition_raw_json",
                                "metadata_version": VERSION,
                            })
                            file_counts["chunks"] += 1
                            totals["chunks"] += 1

                    if file_counts["raw_records"] % 100 == 0:
                        print(
                            f"    records={file_counts['raw_records']:,} | "
                            f"documents={file_counts['documents']:,} | chunks={file_counts['chunks']:,}",
                            flush=True,
                        )

            file_reports.append({
                "key": key,
                "source_file": path.name,
                **dict(file_counts),
            })
            print(
                f"    완료: documents={file_counts['documents']:,}, "
                f"parents={file_counts['parents']:,}, chunks={file_counts['chunks']:,}",
                flush=True,
            )

        for writer in writers.values():
            writer.close()
        for writer in writers.values():
            writer.commit()
    except Exception:
        for writer in writers.values():
            writer.abort()
        raise

    report = {
        "version": VERSION,
        "security": "원본 회의록과 생성 결과를 외부로 전송하지 않음",
        "raw_dir": str(raw_dir),
        "selected_keys": selected_keys,
        "parameters": {
            "parent_chars": args.parent_chars,
            "child_chars": args.child_chars,
            "child_overlap": args.child_overlap,
            "min_tail_chars": args.min_tail_chars,
            "max_records_per_key": args.max_records_per_key,
        },
        "totals": dict(totals),
        "files": file_reports,
        "outputs": {name: str(path) for name, path in paths.items() if name != "report"},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "raw_text_in_report": False,
    }
    paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 원본 JSON canonical 생성 완료 ===")
    print(f"문서: {totals['documents']:,}")
    print(f"부모 청크: {totals['parents']:,}")
    print(f"검색 청크: {totals['chunks']:,}")
    print(f"출력 폴더: {output_dir}")
    print(f"보고서: {paths['report']}")


if __name__ == "__main__":
    main()

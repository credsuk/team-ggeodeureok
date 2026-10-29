#!/usr/bin/env python3
"""수정 회의록과 신규 metadata를 결합하기 전에 입력 전체를 검사한다.

원문은 보고서에 저장하지 않는다. 파일 구조, 인코딩, 깨진 문자, JSONL
스키마와 결합키 후보만 기록한다.
"""

from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IGNORED_NAMES = {".gitkeep"}
JOIN_KEY_CANDIDATES = (
    "chunk_id",
    "document_id",
    "parent_id",
    "source_file",
    "source_record_no",
    "date",
    "meeting_date",
    "assembly",
    "council",
    "meeting_name",
    "committee",
    "agenda_title",
    "speaker_name",
    "speaker_role",
    "speaker_type",
    "utterance",
    "text",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=PROJECT_ROOT / "data" / "input"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "restart_inventory",
    )
    parser.add_argument("--read-block-mib", type=int, default=4)
    return parser.parse_args()


def is_ignored(path: Path) -> bool:
    return path.name in IGNORED_NAMES or ":Zone.Identifier" in path.name


def classify(path: Path, root: Path) -> str:
    relative_parts = [part.lower() for part in path.relative_to(root).parts[:-1]]
    joined = "/".join(relative_parts)
    if "metadata_jsonl_python" in joined:
        return "metadata"
    if "llm_rules_outputs" in joined:
        return "parsing_rule"
    if "output_json_sample" in joined:
        return "sample_output"
    if "2024_" in joined or "2025_" in joined or "회의록" in joined:
        return "meeting_corpus"
    return "other"


def detect_encoding(path: Path) -> tuple[str, str]:
    with path.open("rb") as file:
        sample = file.read(1024 * 1024)
    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", "utf-8-sig"
    if sample.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return "utf-32", "utf-32"
    if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16", "utf-16"
    for encoding in ("utf-8", "cp949", "euc-kr"):
        try:
            sample.decode(encoding)
            return encoding, encoding
        except UnicodeDecodeError:
            pass
    return "utf-8-with-replacement", "utf-8"


def inspect_text_file(path: Path, block_size: int) -> dict[str, Any]:
    encoding_label, decoder_encoding = detect_encoding(path)
    decoder = codecs.getincrementaldecoder(decoder_encoding)(errors="replace")
    digest = hashlib.sha256()
    byte_count = 0
    char_count = 0
    line_count = 0
    replacement_count = 0
    null_byte_count = 0

    with path.open("rb") as file:
        while raw := file.read(block_size):
            digest.update(raw)
            byte_count += len(raw)
            null_byte_count += raw.count(b"\x00")
            text = decoder.decode(raw)
            char_count += len(text)
            line_count += text.count("\n")
            replacement_count += text.count("\ufffd")
        tail = decoder.decode(b"", final=True)
        char_count += len(tail)
        line_count += tail.count("\n")
        replacement_count += tail.count("\ufffd")

    return {
        "encoding": encoding_label,
        "size_bytes": byte_count,
        "char_count": char_count,
        "row_or_line_count": line_count + (1 if char_count else 0),
        "replacement_char_count": replacement_count,
        "null_byte_count": null_byte_count,
        "sha256": digest.hexdigest(),
    }


def inspect_tsv(path: Path) -> tuple[dict[str, Any], Counter[str]]:
    encoding_label, encoding = detect_encoding(path)
    keys: Counter[str] = Counter()
    rows = 0
    invalid_width = 0
    replacement_count = 0
    with path.open("r", encoding=encoding, errors="replace", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        header = next(reader, [])
        keys.update(header)
        width = len(header)
        for row in reader:
            rows += 1
            invalid_width += int(len(row) != width)
            replacement_count += sum(value.count("\ufffd") for value in row)
    return ({
        "encoding": encoding_label,
        "size_bytes": path.stat().st_size,
        "char_count": None,
        "row_or_line_count": rows,
        "replacement_char_count": replacement_count,
        "null_byte_count": None,
        "sha256": file_sha256(path),
        "column_count": len(header),
        "invalid_width_rows": invalid_width,
    }, keys)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def json_object_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        yield from map(str, value.keys())


def inspect_jsonl(path: Path) -> tuple[dict[str, Any], Counter[str]]:
    encoding_label, encoding = detect_encoding(path)
    keys: Counter[str] = Counter()
    rows = 0
    invalid_rows = 0
    non_object_rows = 0
    replacement_count = 0
    with path.open("r", encoding=encoding, errors="replace") as file:
        for line in file:
            if not line.strip():
                continue
            rows += 1
            replacement_count += line.count("\ufffd")
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows += 1
                continue
            if not isinstance(value, dict):
                non_object_rows += 1
                continue
            keys.update(json_object_keys(value))
    return ({
        "encoding": encoding_label,
        "size_bytes": path.stat().st_size,
        "char_count": None,
        "row_or_line_count": rows,
        "replacement_char_count": replacement_count,
        "null_byte_count": None,
        "sha256": file_sha256(path),
        "invalid_json_rows": invalid_rows,
        "non_object_rows": non_object_rows,
    }, keys)


def inspect_json(path: Path) -> tuple[dict[str, Any], Counter[str]]:
    # 규칙/샘플 JSON은 작다는 전제다. 큰 JSON 배열은 여기서 원문을 펼치지 않는다.
    keys: Counter[str] = Counter()
    base = {
        "encoding": "utf-8-sig",
        "size_bytes": path.stat().st_size,
        "char_count": None,
        "row_or_line_count": None,
        "replacement_char_count": None,
        "null_byte_count": None,
        "sha256": file_sha256(path),
    }
    if path.stat().st_size > 64 * 1024 * 1024:
        base["json_status"] = "skipped_large_json_over_64mib"
        return base, keys
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        base["json_status"] = "valid"
        base["top_level_type"] = type(value).__name__
        if isinstance(value, dict):
            keys.update(map(str, value.keys()))
            base["row_or_line_count"] = 1
        elif isinstance(value, list):
            base["row_or_line_count"] = len(value)
            for item in value[:1000]:
                keys.update(json_object_keys(item))
    except Exception as exc:
        base["json_status"] = f"invalid: {type(exc).__name__}"
    return base, keys


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(path for path in input_dir.rglob("*") if path.is_file() and not is_ignored(path))
    if not paths:
        raise FileNotFoundError(f"입력 파일이 없습니다: {input_dir}")

    file_rows: list[dict[str, Any]] = []
    schema_by_role: dict[str, Counter[str]] = defaultdict(Counter)
    extensions = Counter()
    roles = Counter()

    for index, path in enumerate(paths, start=1):
        role = classify(path, input_dir)
        suffix = path.suffix.lower()
        print(f"[{index}/{len(paths)}] {role:15s} {path.relative_to(input_dir)}")
        keys: Counter[str] = Counter()
        if suffix in {".jsonl", ".ndjson"}:
            detail, keys = inspect_jsonl(path)
        elif suffix == ".json":
            detail, keys = inspect_json(path)
        elif suffix in {".tsv", ".tab"}:
            detail, keys = inspect_tsv(path)
        else:
            detail = inspect_text_file(path, args.read_block_mib * 1024 * 1024)

        extensions[suffix or "<none>"] += 1
        roles[role] += 1
        schema_by_role[role].update(keys)
        file_rows.append({
            "role": role,
            "relative_path": str(path.relative_to(input_dir)),
            "suffix": suffix,
            **detail,
        })

    metadata_keys = schema_by_role["metadata"]
    join_key_presence = {
        key: {
            "files_present": metadata_keys.get(key, 0),
            "present": metadata_keys.get(key, 0) > 0,
        }
        for key in JOIN_KEY_CANDIDATES
    }
    stable_key_available = any(
        join_key_presence[key]["present"]
        for key in ("chunk_id", "document_id", "source_record_no")
    )

    duplicate_hashes = {
        digest: count
        for digest, count in Counter(row["sha256"] for row in file_rows).items()
        if count > 1
    }
    summary = {
        "input_dir": str(input_dir),
        "file_count": len(file_rows),
        "role_counts": dict(roles),
        "extension_counts": dict(extensions),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in file_rows),
        "total_replacement_chars": sum(
            int(row.get("replacement_char_count") or 0) for row in file_rows
        ),
        "files_with_replacement_chars": [
            {
                "relative_path": row["relative_path"],
                "count": row["replacement_char_count"],
            }
            for row in file_rows
            if int(row.get("replacement_char_count") or 0) > 0
        ],
        "duplicate_file_sha256": duplicate_hashes,
        "metadata_schema": {
            "field_file_frequency": dict(metadata_keys.most_common()),
            "join_key_presence": join_key_presence,
            "stable_join_key_available": stable_key_available,
            "warning": None if stable_key_available else (
                "고정 결합키가 없습니다. 행 순서로 결합하지 말고 텍스트/문서 위치 기반 매핑을 설계해야 합니다."
            ),
        },
        "decision": {
            "safe_to_build_canonical": (
                roles["meeting_corpus"] > 0
                and roles["metadata"] > 0
                and not any(row.get("invalid_json_rows", 0) for row in file_rows)
            ),
            "full_embedding_allowed": False,
            "reason": "canonical 생성과 100~1000행 수동 샘플 검수가 먼저 필요합니다.",
        },
        "raw_text_in_report": False,
    }

    files_path = output_dir / "files_inventory.csv"
    fieldnames = sorted({key for row in file_rows for key in row})
    with files_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(file_rows)

    schema_path = output_dir / "metadata_schema.csv"
    with schema_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["field", "files_present"])
        writer.writeheader()
        for field, count in metadata_keys.most_common():
            writer.writerow({"field": field, "files_present": count})

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 재시작 입력 검사 완료 ===")
    print(f"파일: {len(file_rows):,} | 역할: {dict(roles)}")
    print(f"깨진 문자(U+FFFD): {summary['total_replacement_chars']:,}")
    print(f"metadata 필드: {list(metadata_keys)}")
    print(f"고정 결합키 존재: {stable_key_available}")
    print(f"canonical 생성 사전조건: {summary['decision']['safe_to_build_canonical']}")
    print(f"요약: {summary_path}")
    print(f"파일별 검사: {files_path}")
    print(f"metadata 스키마: {schema_path}")


if __name__ == "__main__":
    main()

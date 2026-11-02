from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_DIR = ROOT / "data" / "metadata_jsonl_python"
DEFAULT_TEXT_DIR = ROOT / "data" / "to_text"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "org_jsonl"

# The source TSV contains the full utterance. Summary is omitted because it is
# redundant for the retrieval corpus; all other metadata fields are retained.
OMIT_METADATA_FIELDS = {"summary"}


def merge_file(metadata_path: Path, text_path: Path, output_path: Path) -> int:
    """Merge one metadata JSONL with its matching TSV, validating row_id order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    count = 0
    try:
        with metadata_path.open("r", encoding="utf-8-sig") as metadata_file, \
                text_path.open("r", encoding="utf-8-sig", newline="") as text_file, \
                os.fdopen(fd, "w", encoding="utf-8", newline="") as output_file:
            reader = csv.DictReader(text_file, delimiter="\t")
            if not reader.fieldnames or "utterance" not in reader.fieldnames:
                raise ValueError(f"TXT에 utterance 열이 없습니다: {text_path}")

            for expected_row_id, metadata_line in enumerate(metadata_file, start=1):
                if not metadata_line.strip():
                    continue
                metadata: dict[str, Any] = json.loads(metadata_line)
                row_id = metadata.get("row_id")
                if row_id != expected_row_id:
                    raise ValueError(
                        f"row_id 불일치: {metadata_path.name}의 {expected_row_id}번째 JSONL 레코드가 "
                        f"row_id={row_id!r}입니다."
                    )
                text_row = next(reader, None)
                if text_row is None:
                    raise ValueError(
                        f"TXT 행 부족: {text_path.name}, JSONL row_id={row_id}"
                    )
                record = {
                    key: value for key, value in metadata.items()
                    if key not in OMIT_METADATA_FIELDS
                }
                record["utterance"] = text_row.get("utterance", "")
                record["source_file"] = text_path.name
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

            extra_text_row = next(reader, None)
            if extra_text_row is not None:
                raise ValueError(
                    f"TXT 행이 JSONL보다 많습니다: {text_path.name}, "
                    f"JSONL={count}행 이상, TXT에 추가 행 존재"
                )

        os.replace(temp_name, output_path)
        return count
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def convert_all(metadata_dir: Path, text_dir: Path, output_dir: Path) -> tuple[int, int]:
    metadata_files = sorted(metadata_dir.rglob("*.jsonl"))
    if not metadata_files:
        raise FileNotFoundError(f"JSONL 입력 파일이 없습니다: {metadata_dir}")

    text_by_stem: dict[str, list[Path]] = {}
    for path in text_dir.rglob("*.txt"):
        text_by_stem.setdefault(path.stem, []).append(path)

    total_records = 0
    for metadata_path in metadata_files:
        matches = text_by_stem.get(metadata_path.stem, [])
        if len(matches) != 1:
            raise FileNotFoundError(
                f"{metadata_path.name}에 대응하는 TXT 파일이 정확히 1개가 아닙니다 "
                f"(발견: {len(matches)}개)."
            )
        output_path = output_dir / f"{metadata_path.stem}.jsonl"
        converted = merge_file(metadata_path, matches[0], output_path)
        total_records += converted
        print(f"완료: {metadata_path.name} -> {output_path.name} ({converted:,}건)")

    unmatched_texts = [
        path for path in text_dir.rglob("*.txt")
        if path.stem not in {item.stem for item in metadata_files}
    ]
    if unmatched_texts:
        raise FileNotFoundError(
            "메타데이터 JSONL과 매칭되지 않는 TXT가 있습니다: "
            + ", ".join(path.name for path in unmatched_texts[:10])
        )
    return len(metadata_files), total_records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="JSONL 메타데이터와 같은 row_id의 TXT 발언을 합쳐 정리 JSONL을 생성합니다."
    )
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--text-dir", type=Path, default=DEFAULT_TEXT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    try:
        file_count, record_count = convert_all(
            args.metadata_dir, args.text_dir, args.output_dir
        )
    except (OSError, ValueError, json.JSONDecodeError, csv.Error) as exc:
        parser.error(str(exc))
    print(
        f"전체 완료: 파일 {file_count}개, 레코드 {record_count:,}건\n"
        f"출력 폴더: {args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

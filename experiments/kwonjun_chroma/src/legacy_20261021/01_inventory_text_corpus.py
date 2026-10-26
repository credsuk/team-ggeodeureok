#!/usr/bin/env python3
"""정제 TXT 회의록 묶음과 파싱 규칙 JSON을 원문 노출 없이 검사한다."""

from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COUNCILS = {
    "서울특별시의회", "부산광역시의회", "대구광역시의회", "인천광역시의회",
    "광주광역시의회", "대전광역시의회", "울산광역시의회", "세종특별자치시의회",
    "경기도의회", "강원특별자치도의회", "충청북도의회", "충청남도의회",
    "전북특별자치도의회", "전라남도의회", "경상북도의회", "경상남도의회",
    "제주특별자치도의회",
}
FILE_PATTERN = re.compile(r"^cleaned_(?P<year>20\d{2})_(?P<council>.+)\.txt$")
STRUCTURE_MARKERS = {
    "html_div": "<div",
    "html_p": "<p",
    "agenda_tag": "[안건]",
    "content_tag": "[내용]",
    "speaker_tag": "[발언자]",
    "meeting_tag": "[회의",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "input")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "inventory_text")
    parser.add_argument("--read-block-mib", type=int, default=4)
    return parser.parse_args()


def plausible_text_score(text: str) -> float:
    if not text:
        return 0.0
    sample = text[:200000]
    printable = sum(character.isprintable() or character in "\r\n\t" for character in sample)
    controls = sum(ord(character) < 32 and character not in "\r\n\t" for character in sample)
    hangul = sum("가" <= character <= "힣" for character in sample)
    return printable / len(sample) + min(hangul / max(len(sample), 1), 0.2) - controls / len(sample) * 5


def detect_encoding(path: Path) -> tuple[str, str]:
    with path.open("rb") as file:
        sample = file.read(1024 * 1024)
    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", "utf-8-sig"
    if sample.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return "utf-32", "utf-32"
    if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16", "utf-16"
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            sample.decode(encoding)
            return encoding, encoding
        except UnicodeDecodeError:
            continue

    # BOM이 없는 UTF-16/32 파일을 위한 검사다. 단순 decode 성공이 아니라
    # 출력 가능한 글자와 한글 비율을 함께 평가해 잘못된 endian 선택을 피한다.
    null_ratio = sample.count(b"\x00") / max(len(sample), 1)
    candidates: list[tuple[float, str]] = []
    if null_ratio >= 0.05:
        for encoding in ("utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be"):
            try:
                decoded = sample.decode(encoding)
                candidates.append((plausible_text_score(decoded), encoding))
            except UnicodeDecodeError:
                continue
    if candidates:
        score, encoding = max(candidates)
        if score >= 0.85:
            return encoding, encoding

    # 일부 바이트만 손상된 UTF-8은 전체 파일을 버리지 않고 교체문자로 표시한다.
    # replacement_char_count가 0보다 크면 후속 단계에서 반드시 검토한다.
    return "utf-8-with-replacement", "utf-8"


def inspect_text(path: Path, root: Path, block_size: int) -> dict:
    match = FILE_PATTERN.match(path.name)
    encoding_label, decoder_encoding = detect_encoding(path)
    digest = hashlib.sha256()
    byte_count = 0
    char_count = 0
    newline_count = 0
    replacement_count = 0
    null_byte_count = 0
    markers = Counter({name: 0 for name in STRUCTURE_MARKERS})
    decoder = codecs.getincrementaldecoder(decoder_encoding)(errors="replace")
    with path.open("rb") as file:
        while raw := file.read(block_size):
            digest.update(raw)
            byte_count += len(raw)
            null_byte_count += raw.count(b"\x00")
            text = decoder.decode(raw)
            char_count += len(text)
            newline_count += text.count("\n")
            replacement_count += text.count("\ufffd")
            lowered = text.lower()
            for name, marker in STRUCTURE_MARKERS.items():
                markers[name] += lowered.count(marker.lower())
        tail = decoder.decode(b"", final=True)
        char_count += len(tail)
        newline_count += tail.count("\n")
        replacement_count += tail.count("\ufffd")

    return {
        "relative_path": str(path.relative_to(root)),
        "file_name": path.name,
        "year": int(match.group("year")) if match else None,
        "council": match.group("council") if match else None,
        "size_bytes": byte_count,
        "encoding": encoding_label,
        "decoder_encoding": decoder_encoding,
        "char_count": char_count,
        "line_count": newline_count + (1 if char_count else 0),
        "replacement_char_count": replacement_count,
        "null_byte_count": null_byte_count,
        "sha256": digest.hexdigest(),
        **markers,
    }


def inspect_rule_json(path: Path, root: Path) -> dict:
    result = {
        "relative_path": str(path.relative_to(root)),
        "file_name": path.name,
        "size_bytes": path.stat().st_size,
        "valid_json": False,
        "top_level_type": None,
        "top_level_keys": [],
        "item_count": None,
        "error": None,
    }
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        result["valid_json"] = True
        result["top_level_type"] = type(value).__name__
        if isinstance(value, dict):
            result["top_level_keys"] = sorted(map(str, value.keys()))
            result["item_count"] = len(value)
        elif isinstance(value, list):
            result["item_count"] = len(value)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_paths = sorted(
        path for path in input_dir.rglob("*.txt")
        if path.is_file() and ":Zone.Identifier" not in path.name
    )
    rule_paths = sorted(
        path for path in input_dir.rglob("*.json")
        if path.is_file() and ":Zone.Identifier" not in path.name
    )
    if not text_paths:
        raise FileNotFoundError(f"정제 TXT 파일을 찾지 못했습니다: {input_dir}")

    rows = []
    for index, path in enumerate(text_paths, start=1):
        print(f"[{index}/{len(text_paths)}] {path.name}")
        rows.append(inspect_text(path, input_dir, args.read_block_mib * 1024 * 1024))
    rules = [inspect_rule_json(path, input_dir) for path in rule_paths]

    councils_by_year: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        if row["year"] and row["council"]:
            councils_by_year[row["year"]].add(row["council"])
    hashes = Counter(row["sha256"] for row in rows if row["sha256"])
    duplicate_hashes = {digest: count for digest, count in hashes.items() if count > 1}

    text_keys = {
        f"{row['year']}_{row['council']}"
        for row in rows if row["year"] and row["council"]
    }
    rule_keys = {
        key
        for rule in rules if rule["valid_json"]
        for key in rule["top_level_keys"]
    }

    summary = {
        "input_dir": str(input_dir),
        "text_file_count": len(rows),
        "rule_json_count": len(rules),
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
        "total_char_count": sum(row["char_count"] or 0 for row in rows),
        "encoding_counts": dict(Counter(row["encoding"] for row in rows)),
        "replacement_char_total": sum(row["replacement_char_count"] or 0 for row in rows),
        "duplicate_file_sha256": duplicate_hashes,
        "files_with_replacement_chars": [
            {"file_name": row["file_name"], "replacement_char_count": row["replacement_char_count"]}
            for row in rows if (row["replacement_char_count"] or 0) > 0
        ],
        "years": {
            str(year): {
                "file_count": len(councils),
                "councils": sorted(councils),
                "missing_from_standard_17": sorted(EXPECTED_COUNCILS - councils),
                "unexpected_names": sorted(councils - EXPECTED_COUNCILS),
            }
            for year, councils in sorted(councils_by_year.items())
        },
        "rule_files": rules,
        "parsing_rule_alignment": {
            "text_key_count": len(text_keys),
            "rule_key_count": len(rule_keys),
            "missing_rule_for_text": sorted(text_keys - rule_keys),
            "rule_without_matching_text": sorted(rule_keys - text_keys),
        },
        "raw_text_in_report": False,
    }

    csv_path = output_dir / "text_files_inventory.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 정제 TXT 데이터 검사 완료 ===")
    print(f"TXT: {len(rows):,}개 | 규칙 JSON: {len(rules):,}개")
    print(f"전체 크기: {summary['total_size_bytes'] / 1024**3:.2f} GiB")
    print(f"전체 문자: {summary['total_char_count']:,}자")
    for year, detail in summary["years"].items():
        print(f"{year}: {detail['file_count']}개 | 누락={detail['missing_from_standard_17']}")
    print(f"summary: {summary_path}")
    print(f"inventory: {csv_path}")


if __name__ == "__main__":
    main()


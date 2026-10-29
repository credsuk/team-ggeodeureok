#!/usr/bin/env python3
"""canonical 발화 데이터를 근거 보존형 부모·자식 청크로 변환한다."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENTENCE_BREAK = re.compile(r"(?<=[.!?。！？])\s+|\n+")

PARENT_SCHEMA = pa.schema([
    pa.field("parent_chunk_id", pa.string(), nullable=False),
    pa.field("document_id", pa.string(), nullable=False),
    pa.field("source_key", pa.string(), nullable=False),
    pa.field("parent_index", pa.int32(), nullable=False),
    pa.field("council", pa.string(), nullable=False),
    pa.field("meeting_date", pa.string(), nullable=False),
    pa.field("year", pa.int16(), nullable=False),
    pa.field("committee", pa.string()),
    pa.field("meeting_name", pa.string()),
    pa.field("agenda_title", pa.string()),
    pa.field("source_row_start", pa.int64(), nullable=False),
    pa.field("source_row_end", pa.int64(), nullable=False),
    pa.field("utterance_count", pa.int32(), nullable=False),
    pa.field("utterance_ids_json", pa.string(), nullable=False),
    pa.field("speakers_json", pa.string(), nullable=False),
    pa.field("policy_categories_json", pa.string(), nullable=False),
    pa.field("core_issues_json", pa.string(), nullable=False),
    pa.field("keywords_json", pa.string(), nullable=False),
    pa.field("substantive_true_count", pa.int32(), nullable=False),
    pa.field("substantive_ratio", pa.float32(), nullable=False),
    pa.field("embedding_candidate", pa.bool_(), nullable=False),
    pa.field("quality_flags_json", pa.string(), nullable=False),
    pa.field("text", pa.string(), nullable=False),
    pa.field("embedding_text_raw", pa.string(), nullable=False),
    pa.field("embedding_text_enriched", pa.string(), nullable=False),
    pa.field("text_sha256", pa.string(), nullable=False),
    pa.field("char_length", pa.int32(), nullable=False),
    pa.field("canonical_version", pa.string(), nullable=False),
])

CHUNK_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string(), nullable=False),
    pa.field("parent_chunk_id", pa.string(), nullable=False),
    pa.field("document_id", pa.string(), nullable=False),
    pa.field("source_key", pa.string(), nullable=False),
    pa.field("parent_index", pa.int32(), nullable=False),
    pa.field("child_index", pa.int32(), nullable=False),
    pa.field("council", pa.string(), nullable=False),
    pa.field("meeting_date", pa.string(), nullable=False),
    pa.field("year", pa.int16(), nullable=False),
    pa.field("committee", pa.string()),
    pa.field("meeting_name", pa.string()),
    pa.field("agenda_title", pa.string()),
    pa.field("source_row_start", pa.int64(), nullable=False),
    pa.field("source_row_end", pa.int64(), nullable=False),
    pa.field("utterance_count", pa.int32(), nullable=False),
    pa.field("utterance_ids_json", pa.string(), nullable=False),
    pa.field("speakers_json", pa.string(), nullable=False),
    pa.field("policy_categories_json", pa.string(), nullable=False),
    pa.field("core_issues_json", pa.string(), nullable=False),
    pa.field("keywords_json", pa.string(), nullable=False),
    pa.field("substantive_true_count", pa.int32(), nullable=False),
    pa.field("substantive_ratio", pa.float32(), nullable=False),
    pa.field("embedding_candidate", pa.bool_(), nullable=False),
    pa.field("quality_flags_json", pa.string(), nullable=False),
    pa.field("chunk_type", pa.string(), nullable=False),
    pa.field("priority", pa.string(), nullable=False),
    pa.field("text", pa.string(), nullable=False),
    pa.field("embedding_text_raw", pa.string(), nullable=False),
    pa.field("embedding_text_enriched", pa.string(), nullable=False),
    pa.field("text_sha256", pa.string(), nullable=False),
    pa.field("char_length", pa.int32(), nullable=False),
    pa.field("canonical_version", pa.string(), nullable=False),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=PROJECT_ROOT / "data" / "canonical_v3" / "utterances.parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "canonical_v3" / "chunks_v1")
    parser.add_argument(
        "--report-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "restart_v3" / "chunking",
    )
    parser.add_argument("--key", help="예: 2024_강원특별자치도의회")
    parser.add_argument("--parent-target-chars", type=int, default=2400)
    parser.add_argument("--parent-max-chars", type=int, default=3200)
    parser.add_argument("--child-target-chars", type=int, default=750)
    parser.add_argument("--child-max-chars", type=int, default=950)
    parser.add_argument("--child-min-chars", type=int, default=180)
    parser.add_argument("--overlap-segments", type=int, default=1)
    parser.add_argument("--batch-rows", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def stable_id(prefix: str, *values: Any) -> str:
    return f"{prefix}_{digest(chr(31).join(clean(value) for value in values))[:32]}"


def unique(values: Iterable[Any]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = clean(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def decode_keywords(value: Any) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        parsed = [part.strip() for part in str(value).split(",")]
    if isinstance(parsed, list):
        return unique(parsed)
    return [clean(parsed)] if clean(parsed) else []


def speaker_label(row: dict[str, Any]) -> str:
    role = clean(row.get("speaker_role"))
    name = clean(row.get("speaker_name"))
    if role and name and role not in name:
        return f"{role} {name}"
    return name or role or "발화자 미상"


def split_long_utterance(text: str, limit: int) -> list[tuple[str, int, int]]:
    if len(text) <= limit:
        return [(text, 0, len(text))]
    sentences = [part.strip() for part in SENTENCE_BREAK.split(text) if part.strip()]
    if len(sentences) <= 1:
        return [(text[start:start + limit], start, min(start + limit, len(text))) for start in range(0, len(text), limit)]
    pieces = []
    current = ""
    current_start = 0
    cursor = 0
    for sentence in sentences:
        position = text.find(sentence, cursor)
        position = cursor if position < 0 else position
        if current and len(current) + 1 + len(sentence) > limit:
            pieces.append((current, current_start, cursor))
            current = ""
        if len(sentence) > limit:
            if current:
                pieces.append((current, current_start, position))
                current = ""
            for start in range(0, len(sentence), limit):
                part = sentence[start:start + limit]
                pieces.append((part, position + start, position + start + len(part)))
            cursor = position + len(sentence)
            continue
        if not current:
            current_start = position
            current = sentence
        else:
            current += " " + sentence
        cursor = position + len(sentence)
    if current:
        pieces.append((current, current_start, len(text)))
    return pieces


def utterance_to_segments(row: dict[str, Any], child_max_chars: int) -> list[dict[str, Any]]:
    utterance = clean(row.get("utterance"))
    label = speaker_label(row)
    content_limit = max(100, child_max_chars - len(label) - 12)
    pieces = split_long_utterance(utterance, content_limit)
    result = []
    for part_index, (piece, start, end) in enumerate(pieces):
        result.append({
            "utterance_id": row["utterance_id"],
            "source_row_id": int(row["source_row_id"]),
            "speaker": label,
            "text": f"[{label}] {piece}",
            "part_index": part_index,
            "char_start": start,
            "char_end": end,
            "is_substantive": row.get("is_substantive") is True,
            "policy_category": clean(row.get("policy_category")),
            "core_issue": clean(row.get("core_issue")),
            "keywords": decode_keywords(row.get("keywords_json")),
        })
    return result


def iter_rows(path: Path, key: str | None) -> Iterator[dict[str, Any]]:
    if key:
        table = pq.read_table(path, filters=[("source_key", "=", key)])
        yield from table.to_pylist()
        return
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=16_384):
        yield from batch.to_pylist()


def group_key(row: dict[str, Any]) -> tuple[str, str]:
    return clean(row.get("document_id")), clean(row.get("agenda_title"))


def json_list(values: Iterable[Any]) -> str:
    return json.dumps(unique(values), ensure_ascii=False, separators=(",", ":"))


def aggregate_segments(
    segments: list[dict[str, Any]],
    base: dict[str, Any],
    parent_id: str,
    parent_index: int,
    child_index: int | None,
) -> dict[str, Any]:
    text = "\n".join(segment["text"] for segment in segments)
    utterance_ids = unique(segment["utterance_id"] for segment in segments)
    speakers = unique(segment["speaker"] for segment in segments)
    categories = unique(segment["policy_category"] for segment in segments)
    issues = unique(segment["core_issue"] for segment in segments)
    keywords = unique(keyword for segment in segments for keyword in segment["keywords"])
    source_rows = [segment["source_row_id"] for segment in segments]
    # 긴 발화가 여러 조각으로 나뉘더라도 한 발화를 여러 번 가중하지 않는다.
    substantive_by_utterance: dict[str, bool] = {}
    for segment in segments:
        substantive_by_utterance[segment["utterance_id"]] = (
            substantive_by_utterance.get(segment["utterance_id"], False)
            or bool(segment["is_substantive"])
        )
    substantive_true = sum(substantive_by_utterance.values())
    substantive_ratio = substantive_true / len(utterance_ids) if utterance_ids else 0.0
    candidate = substantive_true > 0
    agenda = clean(base.get("agenda_title"))
    context = []
    for label, value in (
        ("의회", base.get("council")),
        ("위원회", base.get("committee")),
        ("회의록", base.get("meeting_name")),
        ("안건", agenda),
    ):
        rendered = clean(value)
        if rendered:
            context.append(f"[{label}] {rendered}")
    raw_embedding = "\n".join(context + ["[발언]", text])
    enriched_parts = []
    if categories:
        enriched_parts.append(f"[정책분야] {', '.join(categories)}")
    if issues:
        enriched_parts.append(f"[핵심이슈] {', '.join(issues)}")
    if keywords:
        enriched_parts.append(f"[키워드] {', '.join(keywords)}")
    enriched_embedding = "\n".join(enriched_parts + [raw_embedding])
    flags = []
    if len(text) < 80:
        flags.append("short_under_80")
    elif len(text) < 180:
        flags.append("short_under_180")
    if len(utterance_ids) == 1:
        flags.append("single_utterance")
    if 0.0 < substantive_ratio < 1.0:
        flags.append("mixed_substantive_context")
    if not candidate:
        flags.append("non_substantive")
    common = {
        "parent_chunk_id": parent_id,
        "document_id": base["document_id"],
        "source_key": base["source_key"],
        "parent_index": parent_index,
        "council": base["council"],
        "meeting_date": base["meeting_date"],
        "year": int(base["year"]),
        "committee": clean(base.get("committee")) or None,
        "meeting_name": clean(base.get("meeting_name")) or None,
        "agenda_title": agenda or None,
        "source_row_start": min(source_rows),
        "source_row_end": max(source_rows),
        "utterance_count": len(utterance_ids),
        "utterance_ids_json": json.dumps(utterance_ids, ensure_ascii=False, separators=(",", ":")),
        "speakers_json": json.dumps(speakers, ensure_ascii=False, separators=(",", ":")),
        "policy_categories_json": json.dumps(categories, ensure_ascii=False, separators=(",", ":")),
        "core_issues_json": json.dumps(issues, ensure_ascii=False, separators=(",", ":")),
        "keywords_json": json.dumps(keywords, ensure_ascii=False, separators=(",", ":")),
        "substantive_true_count": substantive_true,
        "substantive_ratio": float(substantive_ratio),
        "embedding_candidate": candidate,
        "quality_flags_json": json.dumps(flags, ensure_ascii=False, separators=(",", ":")),
        "text": text,
        "embedding_text_raw": raw_embedding,
        "embedding_text_enriched": enriched_embedding,
        "text_sha256": digest(text),
        "char_length": len(text),
        "canonical_version": "parent_child_v1",
    }
    if child_index is None:
        return common
    if not candidate:
        priority = "low"
    elif substantive_ratio >= 0.5 and len(text) >= 80:
        priority = "high"
    else:
        priority = "medium"
    chunk_id = stable_id("chunk", parent_id, child_index, common["text_sha256"])
    return {
        "chunk_id": chunk_id,
        **common,
        "child_index": child_index,
        "chunk_type": "policy" if candidate else "context",
        "priority": priority,
    }


def make_children(segments: list[dict[str, Any]], target: int, maximum: int, minimum: int, overlap: int) -> list[list[dict[str, Any]]]:
    chunks = []
    start = 0
    while start < len(segments):
        current = []
        current_length = 0
        index = start
        while index < len(segments):
            segment_length = len(segments[index]["text"]) + (1 if current else 0)
            if current and current_length + segment_length > maximum:
                break
            current.append(segments[index])
            current_length += segment_length
            index += 1
            if current_length >= target:
                break
        chunks.append(current)
        if index >= len(segments):
            break
        next_start = max(start + 1, index - overlap)
        start = next_start

    if len(chunks) >= 2:
        last_length = len("\n".join(item["text"] for item in chunks[-1]))
        previous_keys = {
            (item["utterance_id"], item["part_index"], item["char_start"], item["char_end"])
            for item in chunks[-2]
        }
        unique_tail = [
            item for item in chunks[-1]
            if (item["utterance_id"], item["part_index"], item["char_start"], item["char_end"])
            not in previous_keys
        ]
        merged = chunks[-2] + unique_tail
        merged_length = len("\n".join(item["text"] for item in merged))
        if last_length < minimum and merged_length <= maximum:
            chunks[-2] = merged
            chunks.pop()
    return chunks


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if args.parent_target_chars > args.parent_max_chars:
        raise ValueError("parent-target-chars는 parent-max-chars 이하여야 합니다")
    if args.child_target_chars > args.child_max_chars:
        raise ValueError("child-target-chars는 child-max-chars 이하여야 합니다")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    parents_path = output_dir / "parents.parquet"
    chunks_path = output_dir / "chunks.parquet"
    if (parents_path.exists() or chunks_path.exists()) and not args.overwrite:
        raise FileExistsError("출력 파일이 이미 있습니다. 교체하려면 --overwrite를 사용하세요")
    parent_partial = output_dir / "parents.partial.parquet"
    chunk_partial = output_dir / "chunks.partial.parquet"
    for path in (parent_partial, chunk_partial):
        if path.exists():
            path.unlink()

    parent_writer = pq.ParquetWriter(parent_partial, PARENT_SCHEMA, compression="zstd")
    chunk_writer = pq.ParquetWriter(chunk_partial, CHUNK_SCHEMA, compression="zstd")
    parent_buffer = []
    chunk_buffer = []
    parent_counts: Counter[str] = Counter()
    chunk_counts: Counter[str] = Counter()
    length_stats = {"parent": [], "child": []}
    sample_rows = []
    parent_index_by_document: defaultdict[str, int] = defaultdict(int)
    current_key = None
    current_base = None
    current_segments: list[dict[str, Any]] = []
    utterance_rows = 0

    def write_buffers(force: bool = False) -> None:
        nonlocal parent_buffer, chunk_buffer
        if parent_buffer and (force or len(parent_buffer) >= args.batch_rows):
            parent_writer.write_table(pa.Table.from_pylist(parent_buffer, schema=PARENT_SCHEMA))
            parent_buffer = []
        if chunk_buffer and (force or len(chunk_buffer) >= args.batch_rows):
            chunk_writer.write_table(pa.Table.from_pylist(chunk_buffer, schema=CHUNK_SCHEMA))
            chunk_buffer = []

    def flush_parent() -> None:
        nonlocal current_segments, current_base
        if not current_segments or current_base is None:
            return
        document_id = current_base["document_id"]
        parent_index = parent_index_by_document[document_id]
        parent_index_by_document[document_id] += 1
        provisional_text = "\n".join(segment["text"] for segment in current_segments)
        parent_id = stable_id(
            "parent", document_id, parent_index,
            current_segments[0]["source_row_id"], current_segments[-1]["source_row_id"],
            digest(provisional_text),
        )
        parent_row = aggregate_segments(current_segments, current_base, parent_id, parent_index, None)
        parent_buffer.append(parent_row)
        parent_counts["total"] += 1
        parent_counts["candidate"] += int(parent_row["embedding_candidate"])
        length_stats["parent"].append(parent_row["char_length"])
        children = make_children(
            current_segments, args.child_target_chars, args.child_max_chars,
            args.child_min_chars, args.overlap_segments,
        )
        for child_index, child_segments in enumerate(children):
            child_row = aggregate_segments(
                child_segments, current_base, parent_id, parent_index, child_index
            )
            chunk_buffer.append(child_row)
            chunk_counts["total"] += 1
            chunk_counts["candidate"] += int(child_row["embedding_candidate"])
            chunk_counts[f"priority_{child_row['priority']}"] += 1
            length_stats["child"].append(child_row["char_length"])
            if len(sample_rows) < 100:
                sample_rows.append({
                    "chunk_id": child_row["chunk_id"],
                    "source_key": child_row["source_key"],
                    "meeting_date": child_row["meeting_date"],
                    "committee": child_row["committee"],
                    "meeting_name": child_row["meeting_name"],
                    "agenda_title": child_row["agenda_title"],
                    "source_row_start": child_row["source_row_start"],
                    "source_row_end": child_row["source_row_end"],
                    "char_length": child_row["char_length"],
                    "priority": child_row["priority"],
                    "embedding_candidate": child_row["embedding_candidate"],
                    "text_preview": child_row["text"][:500],
                })
        write_buffers()
        current_segments = []

    try:
        for row in iter_rows(input_path, args.key):
            utterance_rows += 1
            if args.progress_every > 0 and utterance_rows % args.progress_every == 0:
                print(
                    f"진행: 발화 {utterance_rows:,} | "
                    f"부모 {parent_counts['total']:,} | 자식 {chunk_counts['total']:,}",
                    flush=True,
                )
            row_key = group_key(row)
            segments = utterance_to_segments(row, args.child_max_chars)
            if current_key is not None and row_key != current_key:
                flush_parent()
            current_key = row_key
            current_base = row
            for segment in segments:
                added_length = len(segment["text"]) + (1 if current_segments else 0)
                current_length = len("\n".join(item["text"] for item in current_segments))
                reaches_target = (
                    current_length >= args.parent_target_chars
                    and current_length + added_length > args.parent_target_chars
                )
                exceeds_maximum = current_length + added_length > args.parent_max_chars
                if current_segments and (reaches_target or exceeds_maximum):
                    flush_parent()
                    current_base = row
                current_segments.append(segment)
        flush_parent()
        write_buffers(force=True)
    finally:
        parent_writer.close()
        chunk_writer.close()

    if parents_path.exists() and args.overwrite:
        parents_path.unlink()
    if chunks_path.exists() and args.overwrite:
        chunks_path.unlink()
    parent_partial.replace(parents_path)
    chunk_partial.replace(chunks_path)

    def describe(values: list[int]) -> dict[str, Any]:
        if not values:
            return {"min": None, "mean": None, "max": None}
        return {
            "min": min(values),
            "mean": round(sum(values) / len(values), 2),
            "max": max(values),
        }

    summary = {
        "input": str(input_path),
        "source_key_filter": args.key,
        "utterance_rows": utterance_rows,
        "parent_counts": dict(parent_counts),
        "chunk_counts": dict(chunk_counts),
        "parent_length": describe(length_stats["parent"]),
        "child_length": describe(length_stats["child"]),
        "settings": {
            "parent_target_chars": args.parent_target_chars,
            "parent_max_chars": args.parent_max_chars,
            "child_target_chars": args.child_target_chars,
            "child_max_chars": args.child_max_chars,
            "child_min_chars": args.child_min_chars,
            "overlap_segments": args.overlap_segments,
            "progress_every": args.progress_every,
        },
        "parents": str(parents_path),
        "chunks": str(chunks_path),
    }
    suffix = args.key or "all"
    summary_path = report_dir / f"chunking_{suffix}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    sample_path = report_dir / f"chunk_samples_{suffix}.csv"
    if sample_rows:
        with sample_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(sample_rows[0]))
            writer.writeheader()
            writer.writerows(sample_rows)

    print("\n=== 부모·자식 청킹 완료 ===")
    print(f"발화 행: {utterance_rows:,}")
    print(f"부모 청크: {parent_counts['total']:,} | 후보: {parent_counts['candidate']:,}")
    print(f"자식 청크: {chunk_counts['total']:,} | 후보: {chunk_counts['candidate']:,}")
    print(f"자식 우선순위: high={chunk_counts['priority_high']:,} medium={chunk_counts['priority_medium']:,} low={chunk_counts['priority_low']:,}")
    print(f"부모 길이: {summary['parent_length']}")
    print(f"자식 길이: {summary['child_length']}")
    print(f"parents: {parents_path}")
    print(f"chunks: {chunks_path}")
    print(f"샘플: {sample_path}")
    print(f"요약: {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""정책 청크의 안건 메타데이터를 보정하고 기존 Chroma 메타데이터를 갱신한다.

벡터와 semantic_text는 변경하지 않는다. 잘못 연결된 절차 안건을 원문 근거의
제목으로 표시하지 않도록 raw/effective/confidence를 분리한다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as error:
    raise SystemExit("pyarrow가 필요합니다.") from error


SCRIPT_VERSION = "2026-10-20-agenda-metadata-v1"

GENERIC_RE = re.compile(
    r"^(?:심사된\s*안건|부의(?:된)?\s*안건|상정된\s*안건|처리안건|보고사항|"
    r"의사일정|휴회의\s*건|회의록\s*서명의원(?:\s*선출)?(?:의\s*건)?|"
    r"회기\s*(?:결정|운영)(?:의\s*건)?|개의|개회|정회|산회|질의[·ㆍ\s]*답변)$"
)
PROCEDURAL_RE = re.compile(
    r"휴회의\s*건|의사일정|회기\s*(?:결정|운영)|서명의원|의석\s*배정|"
    r"위원장\s*선임|부위원장\s*선임|개의|개회|정회|산회"
)
BROAD_REPORT_RE = re.compile(
    r"^(?:[가-하]\.?\s*)?.{0,40}(?:소관|실국|본부|위원회)?\s*"
    r"(?:주요)?업무(?:추진)?(?:계획)?\s*보고(?:의\s*건)?$|"
    r"^(?:[가-하]\.?\s*)?[가-힣A-Za-z·]{2,30}\s*소관$"
)
POLICY_TITLE_RE = re.compile(
    r"조례안|예산안|동의안|건의안|결의안|계획안|승인안|추경|정책|사업|"
    r"지원|대책|개선|확대|강화|구축|조성|활성화|육성|보호|예방|감축|"
    r"인구|저출생|출산|청년|주거|교육|복지|교통|안전|환경|산업|농업|의료"
)

CHROMA_FIELDS = (
    "embedding_row", "chunk_id", "parent_chunk_id", "doc_id", "council",
    "committee", "meeting_name", "meeting_date", "agenda_title", "speaker_label",
    "evidence_ref", "char_start", "char_end", "chunk_type", "priority",
    "agenda_title_raw", "agenda_title_effective", "agenda_confidence",
    "agenda_reason", "evidence_title",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="안건 메타데이터 보정")
    parser.add_argument(
        "--input",
        default="data/index/arctic_ko_semantic_test_v2/chunk_metadata.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/index/arctic_ko_semantic_test_v2/chunk_metadata_enriched.parquet",
    )
    parser.add_argument("--report-dir", default="outputs/agenda_metadata")
    parser.add_argument("--update-chroma", action="store_true")
    parser.add_argument("--chroma-dir", default="data/chroma/arctic_policy_v1")
    parser.add_argument("--collection", default="council_policy_arctic_v1")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-count", type=int, default=200)
    return parser.parse_args()


def text(value) -> str:
    return "" if value is None else str(value).strip()


def normalize(value) -> str:
    result = unicodedata.normalize("NFKC", text(value))
    result = re.sub(r"\s+", " ", result).strip(" ;,·ㆍ-|\n\t")
    return result


def strip_numbering(value: str) -> str:
    return re.sub(
        r"^\s*(?:(?:제\s*)?\d{1,3}\s*(?:항|번)?\s*[.)]?|[가-하]\s*[.)])\s*",
        "",
        value,
    ).strip()


def classify_agenda(row: dict) -> tuple[str, str, str, str]:
    raw = normalize(row.get("agenda_title"))
    topic_hint = normalize(row.get("semantic_topic_hint"))
    if topic_hint:
        return raw, topic_hint[:300], "high", "semantic_topic_hint"
    if not raw:
        return raw, "", "none", "missing"

    parts = [normalize(part) for part in re.split(r"[;\n]+", raw) if normalize(part)]
    cleaned = []
    removed = 0
    for part in parts:
        candidate = strip_numbering(part)
        if not candidate or GENERIC_RE.fullmatch(candidate):
            removed += 1
            continue
        if PROCEDURAL_RE.search(candidate) and not POLICY_TITLE_RE.search(candidate):
            removed += 1
            continue
        if candidate not in cleaned:
            cleaned.append(candidate)

    if not cleaned:
        return raw, "", "none", "procedural_or_generic"
    if len(cleaned) > 1:
        return raw, "", "low", "multiple_agendas"

    effective = cleaned[0][:300]
    if BROAD_REPORT_RE.fullmatch(effective):
        return raw, effective, "medium", "broad_report"
    if POLICY_TITLE_RE.search(effective):
        return raw, effective, "high", "policy_title"
    if removed:
        return raw, effective, "medium", "mixed_with_procedural"
    return raw, effective, "medium", "unverified_title"


def set_or_append(table: pa.Table, name: str, values: list[str]) -> pa.Table:
    array = pa.array(values, type=pa.string())
    index = table.schema.get_field_index(name)
    return table.set_column(index, name, array) if index >= 0 else table.append_column(name, array)


def chroma_metadata(row: dict) -> dict:
    output = {"embedding_row": int(row.get("embedding_row") or 0)}
    for name in CHROMA_FIELDS:
        if name in {"embedding_row", "char_start", "char_end"}:
            continue
        value = row.get(name)
        if value is not None and text(value):
            output[name] = text(value)
    for name in ("char_start", "char_end"):
        value = row.get(name)
        if value is not None and text(value):
            try:
                output[name] = int(value)
            except (TypeError, ValueError):
                output[name] = text(value)
    date = text(row.get("meeting_date")).replace("-", "").replace(".", "")
    if len(date) >= 4 and date[:4].isdigit():
        output["meeting_year"] = int(date[:4])
    if len(date) == 8 and date.isdigit():
        output["meeting_date_int"] = int(date)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def update_chroma(args: argparse.Namespace, rows: list[dict]) -> None:
    try:
        import chromadb
    except ImportError as error:
        raise SystemExit("--update-chroma에는 chromadb가 필요합니다.") from error
    chroma_dir = Path(args.chroma_dir).expanduser().resolve()
    if not chroma_dir.exists():
        raise SystemExit(f"Chroma 디렉터리가 없습니다: {chroma_dir}")
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(args.collection, embedding_function=None)
    if collection.count() != len(rows):
        raise SystemExit(
            f"Chroma/Parquet 행 수 불일치: {collection.count():,} != {len(rows):,}"
        )
    for start in range(0, len(rows), args.batch_size):
        end = min(start + args.batch_size, len(rows))
        collection.update(
            ids=[text(row.get("chunk_id")) for row in rows[start:end]],
            metadatas=[chroma_metadata(row) for row in rows[start:end]],
        )
        if end % (args.batch_size * 10) == 0 or end == len(rows):
            print(f"Chroma 메타데이터 갱신 {end:,}/{len(rows):,}", flush=True)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.sample_count < 0:
        raise SystemExit("batch-size는 1 이상, sample-count는 0 이상이어야 합니다.")
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"입력 파일이 없습니다: {input_path}")
    if input_path == output_path:
        raise SystemExit("입력과 출력 경로는 달라야 합니다.")
    started = time.time()
    table = pq.read_table(input_path)
    required = {"embedding_row", "chunk_id", "meeting_name", "agenda_title"}
    missing = required - set(table.schema.names)
    if missing:
        raise SystemExit("필수 컬럼 누락: " + ", ".join(sorted(missing)))
    rows = table.to_pylist()
    raw_values, effective_values, confidence_values, reason_values, evidence_titles = [], [], [], [], []
    counters = Counter()
    samples = []
    for row in rows:
        raw, effective, confidence, reason = classify_agenda(row)
        evidence_title = effective or text(row.get("meeting_name"))
        raw_values.append(raw)
        effective_values.append(effective)
        confidence_values.append(confidence)
        reason_values.append(reason)
        evidence_titles.append(evidence_title)
        counters[f"confidence_{confidence}"] += 1
        counters[f"reason_{reason}"] += 1
        if raw != effective:
            rank = hashlib.sha1(text(row.get("chunk_id")).encode()).hexdigest()
            samples.append((rank, {
                "chunk_id": text(row.get("chunk_id")),
                "council": text(row.get("council")),
                "meeting_date": text(row.get("meeting_date")),
                "meeting_name": text(row.get("meeting_name")),
                "agenda_title_raw": raw,
                "agenda_title_effective": effective,
                "agenda_confidence": confidence,
                "agenda_reason": reason,
                "semantic_preview": " ".join(text(row.get("semantic_text")).split())[:500],
            }))

    for name, values in (
        ("agenda_title_raw", raw_values),
        ("agenda_title_effective", effective_values),
        ("agenda_confidence", confidence_values),
        ("agenda_reason", reason_values),
        ("evidence_title", evidence_titles),
    ):
        table = set_or_append(table, name, values)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="zstd", use_dictionary=True)
    enriched_rows = table.to_pylist()
    if args.update_chroma:
        update_chroma(args, enriched_rows)

    report_dir.mkdir(parents=True, exist_ok=True)
    samples.sort(key=lambda item: item[0])
    write_csv(report_dir / "agenda_metadata_samples.csv", [row for _, row in samples[:args.sample_count]])
    summary = {
        "script_version": SCRIPT_VERSION,
        "security": "로컬 규칙 기반 메타데이터 보정; 외부 데이터 전송 없음",
        "input": str(input_path),
        "output": str(output_path),
        "row_count": len(rows),
        "changed_or_suppressed_count": len(samples),
        "counts": dict(sorted(counters.items())),
        "chroma_updated": args.update_chroma,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    atomic_json(report_dir / "agenda_metadata_summary.json", summary)
    print("\n안건 메타데이터 보정 완료")
    print(f"전체: {len(rows):,}")
    print(f"원본과 표시값이 달라진 행: {len(samples):,}")
    print(f"confidence: high={counters['confidence_high']:,}, medium={counters['confidence_medium']:,}, low={counters['confidence_low']:,}, none={counters['confidence_none']:,}")
    print(f"보정 Parquet: {output_path}")
    print(f"요약: {report_dir / 'agenda_metadata_summary.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""원본 JSON의 위원회·회의록명을 canonical 발언에 연결한다.

발언을 외부 서비스로 전송하지 않는다. 동일 날짜의 원본 회의록 HTML에
정규화된 발언 앵커가 포함되는지 확인하며, 결과 보고서에는 원문을 기록하지 않는다.
먼저 --key로 한 연도×의회만 시험한 뒤 전체 실행하는 것을 권장한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import ijson
import pyarrow as pa
import pyarrow.parquet as pq
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATTERN = re.compile(r"^(?P<year>20\d{2})_(?P<council>.+)\.json$")
MAPPING_SCHEMA = pa.schema([
    ("chunk_id", pa.string()),
    ("document_id", pa.string()),
    ("committee", pa.string()),
    ("meeting_name", pa.string()),
    ("match_method", pa.string()),
    ("match_score", pa.int16()),
])


@dataclass
class RawDocument:
    document_id: str
    date: str
    committee: str | None
    meeting_name: str | None
    normalized_text: str
    normalized_agenda_hints: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=PROJECT_ROOT / "data" / "canonical" / "corpus.parquet",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path.home() / "projects" / "council-minutes" / "data" / "raw",
    )
    parser.add_argument(
        "--key",
        help="시험할 단일 키. 예: 2024_강원특별자치도의회. 생략하면 전체 실행",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=PROJECT_ROOT / "data" / "canonical" / "meeting_metadata_map.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "canonical" / "corpus_enriched.parquet",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "meeting_enrichment" / "enrichment_report.json",
    )
    parser.add_argument("--min-match-ratio", type=float, default=0.995)
    parser.add_argument("--memory-limit", default="6GB")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_value(value: object) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def normalize_date(value: object) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"[^0-9]", "", str(value))
    if len(digits) < 8:
        return None
    try:
        return datetime.strptime(digits[:8], "%Y%m%d").date().isoformat()
    except ValueError:
        return None


def normalize_for_match(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"[^0-9A-Za-z가-힣]", "", str(value)).lower()


def html_to_text(value: object) -> str:
    if value is None:
        return ""
    soup = BeautifulSoup(str(value), "lxml")
    for tag in soup(("script", "style", "noscript")):
        tag.decompose()
    return soup.get_text(" ", strip=True)


def stable_document_id(key: str, index: int, date: str, committee: str | None, meeting_name: str | None) -> str:
    payload = "\x1f".join((key, str(index), date, committee or "", meeting_name or ""))
    return "rawdoc_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def open_utf8_json(path: Path):
    file = path.open("rb")
    prefix = file.read(4)
    if prefix.startswith(b"\xef\xbb\xbf"):
        file.seek(3)
    elif prefix.startswith(b"\xff\xfe") or prefix.startswith(b"\xfe\xff"):
        file.close()
        raise UnicodeError(f"UTF-16 원본은 먼저 UTF-8로 변환해야 합니다: {path}")
    else:
        file.seek(0)
    return file


def load_raw_documents(path: Path, key: str, expected_council: str) -> tuple[list[RawDocument], dict]:
    documents = []
    counters = Counter()
    with open_utf8_json(path) as file:
        for index, record in enumerate(ijson.items(file, "item")):
            counters["raw_records"] += 1
            if not isinstance(record, dict):
                counters["invalid_record_type"] += 1
                continue
            council = clean_value(record.get("RASMBLY_NM"))
            date = normalize_date(record.get("MTG_DE"))
            committee = clean_value(record.get("MTGNM"))
            meeting_name = clean_value(record.get("MINTS_NM"))
            html = record.get("MINTS_HTML")
            if council and council != expected_council:
                counters["raw_council_mismatch"] += 1
            if not date:
                counters["raw_invalid_date"] += 1
                continue
            if not meeting_name:
                counters["raw_missing_meeting_name"] += 1
            if not committee:
                counters["raw_missing_committee"] += 1
            text = normalize_for_match(html_to_text(html))
            if not text:
                counters["raw_empty_html_text"] += 1
            document_id = stable_document_id(key, index, date, committee, meeting_name)
            documents.append(RawDocument(
                document_id=document_id,
                date=date,
                committee=committee,
                meeting_name=meeting_name,
                normalized_text=text,
                normalized_agenda_hints=normalize_for_match(f"{committee or ''} {meeting_name or ''}"),
            ))
    counters["usable_documents"] = len(documents)
    return documents, dict(counters)


def make_anchors(normalized_text: str) -> list[str]:
    if not normalized_text:
        return []
    if len(normalized_text) <= 240:
        return [normalized_text]
    width = 80
    middle = len(normalized_text) // 2
    return [
        normalized_text[:width],
        normalized_text[max(0, middle - width // 2): middle + width // 2],
        normalized_text[-width:],
    ]


def make_fine_anchors(normalized_text: str) -> list[str]:
    """일부 문자가 달라진 긴 발언도 찾되 짧은 상투어 오탐은 피한다."""
    if len(normalized_text) < 64:
        return []
    width = 32
    last = len(normalized_text) - width
    starts = {0, last, last // 4, last // 2, (last * 3) // 4}
    return [normalized_text[start:start + width] for start in sorted(starts)]


def anchor_hits(document: RawDocument, anchors: list[str]) -> int:
    return sum(bool(anchor) and anchor in document.normalized_text for anchor in anchors)


def match_document(
    date: str,
    agenda: str | None,
    speaker: str | None,
    utterance: str,
    candidates: list[RawDocument],
    previous_document_id: str | None,
    membership_cache: dict[tuple[str, str, str], frozenset[str]],
) -> tuple[RawDocument | None, str, int]:
    if not candidates:
        return None, "no_date_candidate", 0
    normalized_utterance = normalize_for_match(utterance)
    normalized_agenda = normalize_for_match(agenda)
    normalized_speaker = normalize_for_match(speaker)
    anchors = make_anchors(normalized_utterance)
    fine_anchors = make_fine_anchors(normalized_utterance)

    if len(candidates) == 1:
        candidate = candidates[0]
        hits = anchor_hits(candidate, anchors)
        return candidate, "date_unique_text" if hits else "date_unique_only", min(32767, 100 * hits)

    if previous_document_id:
        previous = next((doc for doc in candidates if doc.document_id == previous_document_id), None)
        if previous:
            hits = anchor_hits(previous, anchors)
            if hits:
                return previous, "previous_document_text", min(32767, 100 * hits + 10)

    def cached_document_ids(kind: str, value: str) -> frozenset[str]:
        if not value:
            return frozenset()
        cache_key = (date, kind, value)
        if cache_key not in membership_cache:
            membership_cache[cache_key] = frozenset(
                document.document_id for document in candidates if value in document.normalized_text
            )
        return membership_cache[cache_key]

    agenda_document_ids = cached_document_ids("agenda", normalized_agenda)
    speaker_document_ids = cached_document_ids("speaker", normalized_speaker)

    # 안건명이 같은 날짜의 회의록 하나에만 존재하면 텍스트 전체 점수 계산보다
    # 먼저 확정할 수 있다. 반복 안건 검색 결과는 날짜별로 캐시된다.
    agenda_speaker_document_ids = agenda_document_ids & speaker_document_ids
    if len(agenda_speaker_document_ids) == 1:
        document_id = next(iter(agenda_speaker_document_ids))
        return next(doc for doc in candidates if doc.document_id == document_id), "agenda_speaker_unique", 25
    if len(agenda_document_ids) == 1:
        document_id = next(iter(agenda_document_ids))
        return next(doc for doc in candidates if doc.document_id == document_id), "agenda_unique", 20

    scored = []
    for document in candidates:
        hits = anchor_hits(document, anchors)
        fine_hits = anchor_hits(document, fine_anchors)
        exact = int(
            bool(normalized_utterance)
            and len(normalized_utterance) <= 600
            and normalized_utterance in document.normalized_text
        )
        agenda_hit = int(document.document_id in agenda_document_ids)
        speaker_hit = int(document.document_id in speaker_document_ids)
        score = exact * 1000 + hits * 100 + fine_hits * 30 + agenda_hit * 20 + speaker_hit * 5
        scored.append((score, hits, fine_hits, exact, agenda_hit, speaker_hit, document))
    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5]), reverse=True)
    top = scored[0]
    second_score = scored[1][0]
    if top[0] > second_score and (top[1] > 0 or top[2] >= 2 or top[3] > 0):
        method = "unique_fine_text_score" if top[1] == 0 and top[2] >= 2 else "unique_text_score"
        return top[6], method, min(32767, top[0])

    speaker_candidates = [
        document for document in candidates
        if len(normalized_speaker) >= 2 and document.document_id in speaker_document_ids
    ]
    if len(speaker_candidates) == 1:
        return speaker_candidates[0], "speaker_unique", 5
    return None, "ambiguous_or_no_text_match", 0


def find_raw_files(raw_dir: Path) -> dict[str, Path]:
    result = {}
    for path in sorted(raw_dir.glob("*.json")):
        match = RAW_PATTERN.match(path.name)
        if not match:
            continue
        key = f"{match.group('year')}_{match.group('council')}"
        if key in result:
            raise ValueError(f"원본 JSON 키가 중복되었습니다: {key}")
        result[key] = path
    return result


def corpus_groups(connection: duckdb.DuckDBPyConnection, corpus: Path) -> list[tuple[str, int, str, int]]:
    rows = connection.execute(
        f"""
        SELECT CAST(year AS INTEGER), council, COUNT(*)
        FROM read_parquet({sql_string(str(corpus))})
        GROUP BY year, council
        ORDER BY year, council
        """
    ).fetchall()
    return [(f"{int(year)}_{council}", int(year), str(council), int(count)) for year, council, count in rows]


def process_group(
    connection: duckdb.DuckDBPyConnection,
    corpus: Path,
    key: str,
    year: int,
    council: str,
    expected_rows: int,
    raw_path: Path,
    writer: pq.ParquetWriter,
) -> dict:
    started = time.perf_counter()
    documents, raw_stats = load_raw_documents(raw_path, key, council)
    documents_by_date: dict[str, list[RawDocument]] = defaultdict(list)
    for document in documents:
        documents_by_date[document.date].append(document)

    table = connection.execute(
        f"""
        SELECT chunk_id, meeting_date, agenda_title, speaker, text
        FROM read_parquet({sql_string(str(corpus))})
        WHERE year = ? AND council = ?
        ORDER BY source_file, source_line_no
        """,
        [year, council],
    ).to_arrow_table()
    if table.num_rows != expected_rows:
        raise RuntimeError(f"그룹 행 수 불일치: {key}, expected={expected_rows}, actual={table.num_rows}")
    data = table.to_pydict()
    previous_by_date: dict[str, str] = {}
    membership_cache: dict[tuple[str, str, str], frozenset[str]] = {}
    results = []
    documents_by_id = {document.document_id: document for document in documents}

    for index in range(table.num_rows):
        date = str(data["meeting_date"][index])
        candidates = documents_by_date.get(date, [])
        matched, method, score = match_document(
            date=date,
            agenda=data["agenda_title"][index],
            speaker=data["speaker"][index],
            utterance=str(data["text"][index]),
            candidates=candidates,
            previous_document_id=previous_by_date.get(date),
            membership_cache=membership_cache,
        )
        if matched:
            previous_by_date[date] = matched.document_id
        results.append({
            "chunk_id": str(data["chunk_id"][index]),
            "date": date,
            "agenda_key": normalize_for_match(data["agenda_title"][index]),
            "speaker_key": normalize_for_match(data["speaker"][index]),
            "text_length": len(normalize_for_match(data["text"][index])),
            "document": matched,
            "method": method,
            "score": score,
        })
        if (index + 1) % 10_000 == 0 or index + 1 == table.num_rows:
            elapsed = time.perf_counter() - started
            speed = (index + 1) / elapsed if elapsed else 0.0
            print(
                f"    progress={index + 1:,}/{table.num_rows:,} "
                f"({(index + 1) / table.num_rows:.1%}) | {speed:,.1f} rows/s",
                flush=True,
            )

    # 같은 날짜·같은 안건 또는 발화자의 직접 매칭 결과가 한 회의록으로만
    # 모이면, 반복되는 짧은 발언도 그 회의록으로 보완한다.
    def build_consensus(field: str, minimum_key_length: int, minimum_evidence: int) -> dict[tuple[str, str], str]:
        evidence: dict[tuple[str, str], Counter] = defaultdict(Counter)
        for row in results:
            key_value = row[field]
            if row["document"] is not None and len(key_value) >= minimum_key_length:
                evidence[(row["date"], key_value)][row["document"].document_id] += 1
        consensus = {}
        for key_value, counts in evidence.items():
            if len(counts) == 1 and sum(counts.values()) >= minimum_evidence:
                consensus[key_value] = next(iter(counts))
        return consensus

    agenda_consensus = build_consensus("agenda_key", minimum_key_length=4, minimum_evidence=2)
    speaker_consensus = build_consensus("speaker_key", minimum_key_length=2, minimum_evidence=3)
    for row in results:
        if row["document"] is not None:
            continue
        document_id = agenda_consensus.get((row["date"], row["agenda_key"]))
        method = "agenda_context_consensus"
        score = 15
        if document_id is None:
            document_id = speaker_consensus.get((row["date"], row["speaker_key"]))
            method = "speaker_context_consensus"
            score = 4
        if document_id is not None:
            row["document"] = documents_by_id[document_id]
            row["method"] = method
            row["score"] = score

    # 연속된 미매칭 구간의 앞뒤가 동일 회의록으로 확정됐을 때만 사이를 채운다.
    index = 0
    while index < len(results):
        if results[index]["document"] is not None:
            index += 1
            continue
        start_index = index
        date = results[index]["date"]
        while index < len(results) and results[index]["document"] is None and results[index]["date"] == date:
            index += 1
        end_index = index
        left = results[start_index - 1] if start_index > 0 and results[start_index - 1]["date"] == date else None
        right = results[end_index] if end_index < len(results) and results[end_index]["date"] == date else None
        if (
            left is not None
            and right is not None
            and left["document"] is not None
            and right["document"] is not None
            and left["document"].document_id == right["document"].document_id
        ):
            for fill_index in range(start_index, end_index):
                results[fill_index]["document"] = left["document"]
                results[fill_index]["method"] = "neighbor_sandwich"
                results[fill_index]["score"] = 3

    counters = Counter()
    method_counts = Counter()
    unmatched_chunk_ids = []
    unmatched_length_buckets = Counter()
    mapping_rows = []
    for row in results:
        counters["rows"] += 1
        method_counts[row["method"]] += 1
        matched = row["document"]
        if matched is None:
            counters["unmatched_rows"] += 1
            length = row["text_length"]
            bucket = "0-31" if length < 32 else "32-79" if length < 80 else "80-239" if length < 240 else "240+"
            unmatched_length_buckets[bucket] += 1
            if len(unmatched_chunk_ids) < 20:
                unmatched_chunk_ids.append(row["chunk_id"])
            continue
        counters["matched_rows"] += 1
        mapping_rows.append({
            "chunk_id": row["chunk_id"],
            "document_id": matched.document_id,
            "committee": matched.committee,
            "meeting_name": matched.meeting_name,
            "match_method": row["method"],
            "match_score": row["score"],
        })
        if len(mapping_rows) >= 20_000:
            writer.write_table(pa.Table.from_pylist(mapping_rows, schema=MAPPING_SCHEMA))
            mapping_rows.clear()
    if mapping_rows:
        writer.write_table(pa.Table.from_pylist(mapping_rows, schema=MAPPING_SCHEMA))

    return {
        "key": key,
        "raw_file": raw_path.name,
        "corpus_rows": table.num_rows,
        "raw_document_stats": raw_stats,
        "raw_date_count": len(documents_by_date),
        "matched_rows": counters["matched_rows"],
        "unmatched_rows": counters["unmatched_rows"],
        "match_ratio": round(counters["matched_rows"] / counters["rows"], 8) if counters["rows"] else 0.0,
        "method_counts": dict(method_counts),
        "unmatched_normalized_text_length_buckets": dict(unmatched_length_buckets),
        "unmatched_chunk_id_sample": unmatched_chunk_ids,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "raw_text_in_report": False,
    }


def build_enriched_corpus(
    connection: duckdb.DuckDBPyConnection,
    corpus: Path,
    mapping: Path,
    output: Path,
) -> dict:
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        temporary.unlink()
    corpus_sql = sql_string(str(corpus))
    mapping_sql = sql_string(str(mapping))
    temporary_sql = sql_string(str(temporary))
    connection.execute(
        f"""
        COPY (
            SELECT
                c.chunk_id,
                COALESCE(m.document_id, c.document_id) AS document_id,
                c.parent_chunk_id,
                c.council,
                c.meeting_date,
                c.year,
                m.committee AS committee,
                m.meeting_name AS meeting_name,
                c.agenda_title,
                c.speaker,
                c.speaker_role,
                c.speaker_type,
                c.chunk_type,
                c.priority,
                c.source_file,
                c.source_line_no,
                c.text,
                c.embedding_text,
                c.text_sha256,
                c.char_length,
                c.replacement_char_count,
                c.metadata_source,
                'team-cleaned-tsv-v1+raw-json-meeting-v1' AS metadata_version
            FROM read_parquet({corpus_sql}) AS c
            LEFT JOIN read_parquet({mapping_sql}) AS m USING (chunk_id)
            ORDER BY c.source_file, c.source_line_no
        ) TO {temporary_sql} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 10000)
        """
    )
    original_rows, enriched_rows, missing_meeting, changed_embedding = connection.execute(
        f"""
        SELECT
            (SELECT COUNT(*) FROM read_parquet({corpus_sql})),
            (SELECT COUNT(*) FROM read_parquet({temporary_sql})),
            (SELECT COUNT(*) FROM read_parquet({temporary_sql}) WHERE meeting_name IS NULL OR committee IS NULL),
            (
                SELECT COUNT(*)
                FROM read_parquet({corpus_sql}) o
                JOIN read_parquet({temporary_sql}) e USING (chunk_id)
                WHERE o.embedding_text IS DISTINCT FROM e.embedding_text
            )
        """
    ).fetchone()
    if int(original_rows) != int(enriched_rows):
        temporary.unlink()
        raise RuntimeError(f"보강 전후 행 수가 다릅니다: {original_rows} != {enriched_rows}")
    if int(changed_embedding):
        temporary.unlink()
        raise RuntimeError(f"embedding_text가 변경된 행이 있습니다: {changed_embedding}")
    temporary.replace(output)
    return {
        "original_rows": int(original_rows),
        "enriched_rows": int(enriched_rows),
        "rows_missing_meeting_or_committee": int(missing_meeting),
        "embedding_text_changed_rows": int(changed_embedding),
        "output_file": str(output),
        "output_size_bytes": output.stat().st_size,
    }


def main() -> None:
    args = parse_args()
    corpus = args.corpus.expanduser().resolve()
    raw_dir = args.raw_dir.expanduser().resolve()
    if not corpus.exists():
        raise FileNotFoundError(corpus)
    if not raw_dir.is_dir():
        raise NotADirectoryError(raw_dir)
    if not 0 <= args.min_match_ratio <= 1:
        raise ValueError("--min-match-ratio는 0~1 사이여야 합니다.")

    connection = duckdb.connect()
    connection.execute(f"SET memory_limit={sql_string(args.memory_limit)}")
    connection.execute(f"SET threads={args.threads}")
    all_groups = corpus_groups(connection, corpus)
    raw_files = find_raw_files(raw_dir)
    corpus_keys = {key for key, _, _, _ in all_groups}
    extra_raw = sorted(set(raw_files) - corpus_keys)
    if args.key:
        groups = [group for group in all_groups if group[0] == args.key]
        if not groups:
            raise ValueError(f"corpus에 없는 --key입니다: {args.key}")
        # 단일 의회 테스트에서는 선택한 원본만 검사한다. 무관한 의회의 원본
        # 누락 때문에 특정 의회 테스트까지 중단되어서는 안 된다.
        missing_raw = sorted({args.key} - set(raw_files))
    else:
        groups = all_groups
        missing_raw = sorted(corpus_keys - set(raw_files))

    if missing_raw:
        raise FileNotFoundError(f"corpus에 대응하는 원본 JSON이 없습니다: {missing_raw}")

    is_test = bool(args.key)
    if is_test:
        safe_key = re.sub(r"[^0-9A-Za-z가-힣_-]", "_", args.key)
        mapping_path = PROJECT_ROOT / "outputs" / "meeting_enrichment_test" / f"mapping_{safe_key}.parquet"
        report_path = PROJECT_ROOT / "outputs" / "meeting_enrichment_test" / f"report_{safe_key}.json"
        output_path = None
    else:
        mapping_path = args.mapping.expanduser().resolve()
        report_path = args.report.expanduser().resolve()
        output_path = args.output.expanduser().resolve()
        for path in (mapping_path, output_path):
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"출력이 이미 존재합니다. 덮어쓰려면 --overwrite: {path}")
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if mapping_path.exists():
        mapping_path.unlink()

    temporary_mapping = mapping_path.with_name(mapping_path.name + ".building")
    if temporary_mapping.exists():
        temporary_mapping.unlink()
    writer = pq.ParquetWriter(temporary_mapping, schema=MAPPING_SCHEMA, compression="zstd")
    reports = []
    started = time.perf_counter()
    try:
        for index, (key, year, council, expected_rows) in enumerate(groups, start=1):
            print(f"[{index}/{len(groups)}] {key} | corpus={expected_rows:,}")
            group_report = process_group(
                connection=connection,
                corpus=corpus,
                key=key,
                year=year,
                council=council,
                expected_rows=expected_rows,
                raw_path=raw_files[key],
                writer=writer,
            )
            reports.append(group_report)
            print(
                f"  matched={group_report['matched_rows']:,} / "
                f"{group_report['corpus_rows']:,} ({group_report['match_ratio']:.2%})"
            )
    except Exception:
        writer.close()
        if temporary_mapping.exists():
            temporary_mapping.unlink()
        raise
    else:
        writer.close()
    temporary_mapping.replace(mapping_path)

    total_rows = sum(report["corpus_rows"] for report in reports)
    matched_rows = sum(report["matched_rows"] for report in reports)
    match_ratio = matched_rows / total_rows if total_rows else 0.0
    report = {
        "mode": "single_key_test" if is_test else "full",
        "tested_keys": [group[0] for group in groups],
        "corpus_group_count": len(corpus_keys),
        "processed_group_count": len(groups),
        "raw_json_count": len(raw_files),
        "extra_raw_json_keys_ignored": extra_raw,
        "missing_raw_json_keys": missing_raw,
        "total_rows": total_rows,
        "matched_rows": matched_rows,
        "unmatched_rows": total_rows - matched_rows,
        "match_ratio": round(match_ratio, 8),
        "minimum_required_match_ratio": args.min_match_ratio,
        "passed_match_threshold": match_ratio >= args.min_match_ratio,
        "mapping_file": str(mapping_path),
        "groups": reports,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "raw_text_in_report": False,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if is_test:
        connection.close()
        print("\n=== 단일 그룹 회의록 메타데이터 매칭 시험 완료 ===")
        print(f"매칭: {matched_rows:,}/{total_rows:,} ({match_ratio:.2%})")
        print(f"mapping: {mapping_path}")
        print(f"report: {report_path}")
        return

    if match_ratio < args.min_match_ratio:
        connection.close()
        raise RuntimeError(
            f"전체 매칭률이 기준보다 낮아 enriched corpus를 만들지 않았습니다: "
            f"{match_ratio:.2%} < {args.min_match_ratio:.2%}. report={report_path}"
        )
    enrichment = build_enriched_corpus(connection, corpus, mapping_path, output_path)
    connection.close()
    report["enrichment"] = enrichment
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 전체 회의록 메타데이터 보강 완료 ===")
    print(f"매칭: {matched_rows:,}/{total_rows:,} ({match_ratio:.2%})")
    print(f"미보강: {enrichment['rows_missing_meeting_or_committee']:,}")
    print(f"embedding_text 변경: {enrichment['embedding_text_changed_rows']:,}")
    print(f"enriched corpus: {output_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()

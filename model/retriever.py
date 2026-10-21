"""Metadata-filtered, date/council/agenda-level retrieval with Chroma evidence."""
import csv
import json
import re

import chromadb
from chromadb.utils import embedding_functions

from _constant import CHROMA_PATH, COLLECTION_NAME, DATASETS, EMBEDDING_MODEL


def _norm(value):
    return re.sub(r"[\W_]+", "", str(value or "").lower(), flags=re.UNICODE)


def _date_key(value):
    text = str(value or "")
    match = re.search(r"(20\d{2})\D{0,4}(\d{1,2})\D{0,4}(\d{1,2})", text)
    if match:
        return tuple(int(part) for part in match.groups())
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 8:
        return (int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
    return (9999, 12, 31)


def _terms(item):
    value = item.get("keywords", [])
    if isinstance(value, str):
        try: value = json.loads(value)
        except json.JSONDecodeError: value = [value]
    return value if isinstance(value, list) else []


class HybridRetriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        embedding = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        try: self.collection = self.client.get_collection(COLLECTION_NAME, embedding_function=embedding)
        except Exception as exc:
            raise RuntimeError(f"Chroma 컬렉션 '{COLLECTION_NAME}'이 없습니다. 먼저 0_make_chroma_db.py를 실행하세요.") from exc
        self.records = []
        self.source_counts = {}
        for ds in DATASETS:
            metadata_by_id = {}
            with ds["metadata_path"].open("r", encoding="utf-8-sig") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line); metadata_by_id[item["row_id"]] = item
            seen = set()
            with ds["transcript_path"].open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                required = {"date", "assembly", "agenda_title", "utterance"}
                if not required.issubset(set(reader.fieldnames or [])):
                    raise ValueError(f"{ds['transcript_path']}: 7열 TXT 형식이 필요합니다")
                row_id = 0
                for row in reader:
                    if not row or not any(str(v or "").strip() for v in row.values()): continue
                    row_id += 1
                    item = metadata_by_id.get(row_id)
                    if item is None: continue
                    record = {**item, "dataset_key": ds["key"], "date": row["date"], "assembly": row["assembly"],
                              "agenda_title": row.get("agenda_title", ""), "row_id": row_id}
                    self.records.append(record); seen.add(row_id)
            missing = set(metadata_by_id) - seen
            if missing: raise ValueError(f"{ds['key']}: TXT에서 메타데이터 row_id {len(missing)}건을 찾지 못했습니다")
            self.source_counts[ds["key"]] = len(metadata_by_id)

    def matching_events(self, keyword, start_year=None, end_year=None):
        needle = _norm(keyword)
        events = {}
        for row in self.records:
            date = str(row.get("date", "")); year = date[:4]
            if start_year and year < str(start_year): continue
            if end_year and year > str(end_year): continue
            haystack = " ".join([str(row.get(k, "")) for k in ("core_issue", "summary", "agenda", "agenda_title")] + [str(t) for t in _terms(row)])
            if not needle or needle not in _norm(haystack): continue
            agenda_title = str(row.get("agenda_title", ""))
            event_key = (row["dataset_key"], row["assembly"], date, agenda_title)
            event = events.setdefault(event_key, {"dataset_key": row["dataset_key"], "assembly": row["assembly"],
                "date": date, "agenda_title": agenda_title,
                "row_ids": [], "matched_records": 0})
            event["row_ids"].append(row["row_id"]); event["matched_records"] += 1
        return sorted(events.values(), key=lambda e: (_date_key(e["date"]), str(e["assembly"]), str(e["agenda_title"])))

    def evidence_for_events(self, keyword, events):
        for event in events: event["evidence"] = ""
        # Chroma's $in supports bounded row-id batches; choose one source chunk per date/agenda event.
        groups = {}
        for event in events: groups.setdefault(event["dataset_key"], []).append(event)
        for dataset_key, group in groups.items():
            for event in group:
                ids = event["row_ids"]
                for start in range(0, len(ids), 300):
                    result = self.collection.get(where={"$and": [{"dataset_key": dataset_key}, {"row_id": {"$in": ids[start:start+300]}}]}, include=["documents", "metadatas"])
                    for doc, meta in zip(result.get("documents") or [], result.get("metadatas") or []):
                        if meta and meta.get("assembly") == event["assembly"] and meta.get("date") == event["date"] and meta.get("agenda_title", "") == event["agenda_title"]:
                            if not event["evidence"] or len(doc) > len(event["evidence"]): event["evidence"] = doc
        return events


"""Read-only hybrid candidate retrieval from the completed Chroma collection."""
import math
import re
from collections import defaultdict

import chromadb
from chromadb.utils import embedding_functions

from _constant import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL

DENSE_TOP_K = 40
LEXICAL_TOP_K = 40
LEXICAL_PAGE_SIZE = 500
LEXICAL_SCAN_CAP = 5000
RRF_K = 60


def _norm(value):
    return re.sub(r"[\W_]+", "", str(value or "").lower(), flags=re.UNICODE)


def _meta_value(meta, *names):
    for name in names:
        value = meta.get(name)
        if value not in (None, ""):
            return value
    return ""


def normalize_metadata(meta):
    """Expose stable names while preserving original Chroma metadata."""
    result = dict(meta or {})
    result["date"] = str(_meta_value(result, "date", "MTG_DE"))
    result["assembly"] = str(_meta_value(result, "assembly", "RASMBLY_NM"))
    result["agenda_title"] = str(_meta_value(result, "agenda_title", "agenda", "MTGNM"))
    result["meeting_title"] = str(_meta_value(result, "MINTS_NM", "MTGNM", "meeting_title"))
    result["speaker_role"] = str(_meta_value(result, "speaker_role"))
    result["speaker_name"] = str(_meta_value(result, "speaker_name"))
    result["speaker_type"] = str(_meta_value(result, "speaker_type"))
    return result


def _year(value):
    match = re.search(r"(20\d{2})", str(value or ""))
    return match.group(1) if match else ""


class HybridRetriever:
    """Combines Chroma dense retrieval with exact lexical matches from its documents."""
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        embedding = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        try:
            self.collection = self.client.get_collection(COLLECTION_NAME, embedding_function=embedding)
        except Exception as exc:
            raise RuntimeError(f"Chroma 컬렉션 '{COLLECTION_NAME}'을 열 수 없습니다. 색인 완료 여부를 확인하세요.") from exc

    def _matches_filters(self, meta, params):
        date_year = _year(_meta_value(meta, "date", "MTG_DE"))
        start, end = params.get("start_year"), params.get("end_year")
        if start and (not date_year or date_year < start):
            return False
        if end and (not date_year or date_year > end):
            return False
        councils = params.get("councils") or []
        if councils:
            assembly = _norm(_meta_value(meta, "assembly", "RASMBLY_NM"))
            if assembly not in {_norm(item) for item in councils}:
                return False
        return True

    def _dense_candidates(self, queries, limit, params):
        found = {}
        count = self.collection.count()
        if count < 1:
            raise RuntimeError("Chroma 컬렉션이 비어 있습니다. 색인 완료 후 실행하세요.")
        n_results = min(limit, count)
        for query in queries:
            if not query.strip():
                continue
            result = self.collection.query(query_texts=[query], n_results=n_results,
                                           include=["documents", "metadatas", "distances"])
            ids = (result.get("ids") or [[]])[0]
            docs = (result.get("documents") or [[]])[0]
            metas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            for rank, (doc_id, doc, meta) in enumerate(zip(ids, docs, metas), 1):
                meta = normalize_metadata(meta)
                if not self._matches_filters(meta, params):
                    continue
                item = found.setdefault(doc_id, {"id": doc_id, "text": doc or "", "metadata": meta,
                                                  "rrf": 0.0, "retrieval_sources": []})
                item["rrf"] += 1.0 / (RRF_K + rank)
                item["retrieval_sources"].append("dense")
                if rank <= n_results and distances:
                    item["distance"] = float(distances[rank - 1])
        return found

    def _lexical_candidates(self, terms, params):
        # Chroma supplies exact-term candidates; BM25 ranks that lexical candidate set.
        by_id = {}
        clean_terms = list(dict.fromkeys(str(term).strip() for term in terms if len(_norm(term)) >= 2))
        for term in clean_terms:
            offset = 0
            scanned = 0
            while scanned < LEXICAL_SCAN_CAP:
                page_size = min(LEXICAL_PAGE_SIZE, LEXICAL_SCAN_CAP - scanned)
                page = self.collection.get(
                    where_document={"$contains": term}, limit=page_size, offset=offset,
                    include=["documents", "metadatas"],
                )
                ids = page.get("ids") or []
                docs = page.get("documents") or []
                metas = page.get("metadatas") or []
                if not ids:
                    break
                for doc_id, doc, meta in zip(ids, docs, metas):
                    meta = normalize_metadata(meta)
                    if not self._matches_filters(meta, params):
                        continue
                    item = by_id.setdefault(doc_id, {"id": doc_id, "text": doc or "", "metadata": meta,
                                                      "terms": set()})
                    item["terms"].add(term)
                scanned += len(ids)
                offset += len(ids)
                if len(ids) < page_size:
                    break
        if not by_id:
            return {}
        count = len(by_id)
        avg_length = sum(max(1, len(item["text"].split())) for item in by_id.values()) / count
        document_frequency = {
            term: sum(1 for item in by_id.values() if term in item["terms"])
            for term in clean_terms
        }
        ranked = []
        k1, b = 1.2, 0.75
        for item in by_id.values():
            length = max(1, len(item["text"].split()))
            score = 0.0
            for term in item["terms"]:
                tf = item["text"].count(term)
                df = document_frequency.get(term, 0)
                if tf and df:
                    idf = math.log(1.0 + (count - df + 0.5) / (df + 0.5))
                    score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * length / avg_length))
            item["lexical_score"] = score
            ranked.append(item)
        ranked.sort(key=lambda item: item["lexical_score"], reverse=True)
        result = {}
        for rank, item in enumerate(ranked[:LEXICAL_TOP_K], 1):
            item.pop("terms", None)
            item["rrf"] = 1.0 / (RRF_K + rank)
            item["retrieval_sources"] = ["bm25"]
            result[item["id"]] = item
        return result
    def search(self, params, top_k=30):
        queries = list(dict.fromkeys([params.get("query", ""), params.get("topic", ""),
                                      *params.get("include_terms", [])]))
        terms = list(dict.fromkeys([params.get("topic", ""), *params.get("include_terms", [])]))
        dense = self._dense_candidates(queries, max(DENSE_TOP_K, top_k), params)
        lexical = self._lexical_candidates(terms, params)
        candidates = dense
        for doc_id, item in lexical.items():
            if doc_id in candidates:
                candidates[doc_id]["rrf"] += item["rrf"]
                candidates[doc_id]["retrieval_sources"].extend(item["retrieval_sources"])
            else:
                candidates[doc_id] = item
        ordered = sorted(candidates.values(), key=lambda item: item["rrf"], reverse=True)[:max(top_k, 1)]
        return ordered

    def scan_all_lexical(self, params):
        """Exhaustively collect Chroma chunks containing the query's explicit topic terms."""
        terms = list(dict.fromkeys([params.get("topic", ""), *params.get("include_terms", [])]))
        found = {}
        for term in terms:
            term = str(term).strip()
            if len(_norm(term)) < 2:
                continue
            offset = 0
            while True:
                page = self.collection.get(
                    where_document={"$contains": term}, limit=LEXICAL_PAGE_SIZE, offset=offset,
                    include=["documents", "metadatas"],
                )
                ids = page.get("ids") or []
                docs = page.get("documents") or []
                metas = page.get("metadatas") or []
                if not ids:
                    break
                for doc_id, doc, meta in zip(ids, docs, metas):
                    meta = normalize_metadata(meta)
                    if self._matches_filters(meta, params):
                        found.setdefault(doc_id, {"id": doc_id, "text": doc or "", "metadata": meta,
                                                  "rrf": 0.0, "retrieval_sources": ["exact_topic_scan"]})
                offset += len(ids)
                if len(ids) < LEXICAL_PAGE_SIZE:
                    break
        return list(found.values())

    def expand_rows(self, candidates):
        """Rebuild each source utterance from all its Chroma chunks, in bounded DB reads."""
        source_rows = {}
        for candidate in candidates:
            meta = candidate["metadata"]
            dataset_key, row_id = meta.get("dataset_key"), meta.get("row_id")
            if dataset_key in (None, "") or row_id in (None, ""):
                continue
            key = (str(dataset_key), int(row_id))
            current = source_rows.setdefault(key, {
                "id": f"{dataset_key}:{row_id}", "metadata": meta, "chunks": [],
                "rrf": candidate.get("rrf", 0.0), "rerank_score": candidate.get("rerank_score"),
                "retrieval_sources": set(),
            })
            current["rrf"] = max(current["rrf"], candidate.get("rrf", 0.0))
            if candidate.get("rerank_score") is not None:
                previous = current.get("rerank_score")
                current["rerank_score"] = candidate["rerank_score"] if previous is None else max(previous, candidate["rerank_score"])
            current["retrieval_sources"].update(candidate.get("retrieval_sources", []))

        by_dataset = defaultdict(list)
        for dataset_key, row_id in source_rows:
            by_dataset[dataset_key].append(row_id)
        for dataset_key, row_ids in by_dataset.items():
            for offset in range(0, len(row_ids), 300):
                batch = row_ids[offset:offset + 300]
                result = self.collection.get(
                    where={"$and": [{"dataset_key": dataset_key}, {"row_id": {"$in": batch}}]},
                    include=["documents", "metadatas"],
                )
                for doc_id, doc, raw_meta in zip(result.get("ids") or [], result.get("documents") or [], result.get("metadatas") or []):
                    meta = normalize_metadata(raw_meta)
                    key = (str(dataset_key), int(meta.get("row_id")))
                    if key in source_rows:
                        source_rows[key]["chunks"].append((int(meta.get("chunk_index", 0)), doc_id, doc or ""))

        expanded = []
        for row in source_rows.values():
            row["chunks"].sort(key=lambda item: item[0])
            if not row["chunks"]:
                continue
            row["text"] = "".join(item[2] for item in row["chunks"])
            row["chunk_ids"] = [item[1] for item in row["chunks"]]
            row["retrieval_sources"] = sorted(row["retrieval_sources"])
            del row["chunks"]
            expanded.append(row)
        return expanded

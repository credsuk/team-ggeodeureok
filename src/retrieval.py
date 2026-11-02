from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from src.config import Settings
from src.models import Evidence, QueryIntent
from src.ranking import reciprocal_rank_fusion
from src.source_index import SourceTextReader, lookup_source_metadata, sqlite_connection


def build_chroma_filter(intent: QueryIntent) -> dict[str, Any] | None:
    # The existing Chroma index stores date as a string; Chroma range operators are numeric-only.
    # Filter date bounds against returned metadata and the joined SQLite metadata instead.
    if intent.assemblies:
        return {"assembly": {"$in": list(intent.assemblies)}}
    return None


def metadata_matches_scope(metadata: dict[str, Any], intent: QueryIntent) -> bool:
    if intent.assemblies and metadata.get("assembly") not in intent.assemblies:
        return False
    raw_date = str(metadata.get("date", ""))
    date = "".join(character for character in raw_date if character.isdigit())
    if intent.date_start and (not date or date < intent.date_start):
        return False
    if intent.date_end and (not date or date > intent.date_end):
        return False
    return True


def make_fts_query(query: str) -> str:
    terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
    if not terms:
        return ""
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


class BM25Repository:
    """Queries the existing FTS5 index and joins row metadata for date/region filtering."""

    def __init__(self, bm25_db: Path | str, source_metadata_db: Path | str):
        self.bm25_db = Path(bm25_db)
        self.source_metadata_db = Path(source_metadata_db)
        for path in (self.bm25_db, self.source_metadata_db):
            if not path.is_file():
                raise FileNotFoundError(f"필수 검색 인덱스가 없습니다: {path}")

    def search(self, query: str, intent: QueryIntent, limit: int, chronological: bool = False) -> list[dict[str, Any]]:
        if limit < 1:
            return []
        match = make_fts_query(query)
        if not match:
            return []
        filters: list[str] = []
        parameters: list[Any] = [match]
        if intent.assemblies:
            filters.append(f"sm.assembly IN ({','.join('?' for _ in intent.assemblies)})")
            parameters.extend(intent.assemblies)
        if intent.date_start:
            filters.append("sm.date >= ?")
            parameters.append(intent.date_start)
        if intent.date_end:
            filters.append("sm.date <= ?")
            parameters.append(intent.date_end)
        order = "sm.date ASC, bm25(bm25_docs) ASC" if chronological else "bm25(bm25_docs) ASC"
        extra = " AND " + " AND ".join(filters) if filters else ""
        sql = f"""SELECT bm25_docs.chroma_id AS id, bm25_docs.source_file, CAST(bm25_docs.row_id AS INTEGER) AS row_id,
                         bm25(bm25_docs) AS score, sm.date, sm.assembly, sm.meeting_title,
                         sm.minutes_title, sm.agenda
                  FROM bm25_docs
                  JOIN meta.source_metadata AS sm ON sm.doc_id=bm25_docs.chroma_id
                  WHERE bm25_docs MATCH ?{extra}
                  ORDER BY {order} LIMIT ?"""
        parameters.append(limit)
        with sqlite_connection(self.bm25_db, timeout=30) as conn:
            conn.execute("ATTACH DATABASE ? AS meta", (str(self.source_metadata_db),))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, parameters).fetchall()
        return [dict(row) for row in rows]


class HybridRetriever:
    def __init__(self, collection: Any, encoder: Any, bm25: BM25Repository,
                 source_reader: SourceTextReader, settings: Settings):
        self.collection = collection
        self.encoder = encoder
        self.bm25 = bm25
        self.source_reader = source_reader
        self.settings = settings

    def embed_query(self, text: str) -> list[float]:
        vectors = self.encoder.encode(
            [text], prompt_name="query", normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return vectors[0].tolist() if hasattr(vectors[0], "tolist") else list(vectors[0])

    def retrieve(self, intent: QueryIntent, query_embedding: list[float] | None = None) -> list[Evidence]:
        query_text = " ".join([intent.issue, *intent.keywords]).strip()
        vector = query_embedding or self.embed_query(query_text)
        count = int(self.collection.count())
        if count <= 0:
            return []
        where = build_chroma_filter(intent)
        requested_dense = self.settings.dense_limit
        if intent.date_start or intent.date_end:
            requested_dense = max(requested_dense, self.settings.candidate_limit)
        query_args: dict[str, Any] = {
            "query_embeddings": [vector],
            "n_results": min(requested_dense, count),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_args["where"] = where
        dense_result = self.collection.query(**query_args)
        dense_ids = dense_result.get("ids", [[]])[0]
        dense_distances = dense_result.get("distances", [[]])[0]
        dense_metadata = dense_result.get("metadatas", [[]])[0]
        dense = [
            {"id": source_id, "score": 1.0 - float(distance)}
            for source_id, distance, metadata in zip(dense_ids, dense_distances, dense_metadata)
            if metadata_matches_scope(metadata or {}, intent)
        ]
        lexical = self.bm25.search(query_text, intent, self.settings.lexical_limit)
        temporal = self.bm25.search(query_text, intent, self.settings.candidate_limit * 4, chronological=True)
        fused = reciprocal_rank_fusion(dense, lexical)
        fused_by_id = {item.id: item for item in fused}
        chronological_by_assembly: dict[str, dict[str, Any]] = {}
        for item in temporal:
            chronological_by_assembly.setdefault(item["assembly"], item)
        reserved = list(chronological_by_assembly.values())
        reserve_ids = {item["id"] for item in reserved}
        ranked = [item for item in fused if item.id not in reserve_ids]
        selected = ranked[:max(0, self.settings.candidate_limit - len(reserve_ids))]
        selected_ids = list(dict.fromkeys([item.id for item in selected] + [item["id"] for item in reserved]))
        if not selected_ids:
            return []
        metadata_result = self.collection.get(ids=selected_ids, include=["metadatas"])
        metadata_by_id = {
            source_id: (metadata or {})
            for source_id, metadata in zip(metadata_result.get("ids", []), metadata_result.get("metadatas", []))
        }
        source_rows = {row["doc_id"]: row for row in lookup_source_metadata(self.settings.source_metadata_db, selected_ids)}
        texts = self.source_reader.read(selected_ids)
        evidence: list[Evidence] = []
        for source_id in selected_ids:
            text = texts.get(source_id, "")
            if not text.strip():
                continue
            metadata = metadata_by_id.get(source_id, {})
            source = source_rows.get(source_id, {})
            fusion = fused_by_id.get(source_id)
            evidence.append(Evidence(
                id=source_id,
                text=text,
                date=str(metadata.get("date") or source.get("date") or ""),
                assembly=str(metadata.get("assembly") or source.get("assembly") or ""),
                meeting_title=str(source.get("meeting_title") or ""),
                minutes_title=str(source.get("minutes_title") or ""),
                agenda=str(metadata.get("agenda") or source.get("agenda") or ""),
                source_file=str(source.get("source_file") or ""),
                row_id=int(source["row_id"]) if source.get("row_id") is not None else None,
                score=fusion.rrf_score if fusion else 0.0,
                metadata=metadata,
            ))
        return evidence


def load_retriever(settings: Settings, client: Any | None = None) -> HybridRetriever:
    if not settings.chroma_dir.is_dir():
        raise FileNotFoundError(f"Chroma DB 폴더가 없습니다: {settings.chroma_dir}")
    for path in (settings.bm25_db, settings.source_metadata_db):
        if not path.is_file():
            raise FileNotFoundError(f"필수 검색 인덱스가 없습니다: {path}")
    source_dir = settings.data_dir / "to_text"
    if not source_dir.is_dir() or not any(source_dir.rglob("*.txt")):
        raise FileNotFoundError(f"원본 TXT 폴더/파일이 없습니다: {source_dir}")
    if not settings.allow_model_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import chromadb
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("앱 의존성이 없습니다. src/requirements-app.txt를 설치하세요.") from exc
    if settings.embedding_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("EMBEDDING_DEVICE=cuda 이지만 CUDA GPU를 사용할 수 없습니다.")
    client = client or chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        collection = client.get_collection(settings.collection_name)
    except Exception as exc:
        raise RuntimeError(f"Chroma 컬렉션을 찾을 수 없습니다: {settings.collection_name}") from exc
    existing_model = (collection.metadata or {}).get("embedding_model")
    if existing_model != settings.embedding_model:
        raise RuntimeError(f"임베딩 모델 불일치: Chroma={existing_model!r}, 설정={settings.embedding_model!r}")
    encoder = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
    bm25 = BM25Repository(settings.bm25_db, settings.source_metadata_db)
    source_reader = SourceTextReader(settings.source_metadata_db, source_dir)
    return HybridRetriever(collection, encoder, bm25, source_reader, settings)


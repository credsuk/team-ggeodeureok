from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import json
from difflib import SequenceMatcher
from pathlib import Path
import re
import sqlite3
import unicodedata
from typing import Any

from src.core.models import AnalysisResult, QueryIntent


def _question_hash(question: str) -> str:
    normalized = unicodedata.normalize("NFKC", question).casefold()
    normalized = " ".join(re.sub(r"[^\w]+", " ", normalized).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


class AnalysisCache:
    """SQLite is the authoritative store for reusable structured answers."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS analysis_cache (
                    cache_key TEXT PRIMARY KEY,
                    intent_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    data_version TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS question_cache (
                    query_hash TEXT NOT NULL,
                    as_of_date TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    data_version TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    PRIMARY KEY(query_hash, as_of_date, data_version, prompt_version, model_id)
                )"""
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a transaction and always close its SQLite handle on exit."""
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def get(
        self,
        intent: QueryIntent,
        *,
        data_version: str,
        prompt_version: str,
        model_id: str,
    ) -> AnalysisResult | None:
        return self.get_by_key(
            intent.cache_key, data_version=data_version,
            prompt_version=prompt_version, model_id=model_id,
        )

    def get_by_question(
        self,
        question: str,
        *,
        as_of_date: str,
        data_version: str,
        prompt_version: str,
        model_id: str,
    ) -> AnalysisResult | None:
        query_hash = _question_hash(question)
        if not query_hash:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """SELECT cache_key FROM question_cache
                   WHERE query_hash=? AND as_of_date=? AND data_version=?
                     AND prompt_version=? AND model_id=?""",
                (query_hash, as_of_date, data_version, prompt_version, model_id),
            ).fetchone()
        if row is None:
            return None
        return self.get_by_key(
            row[0], data_version=data_version,
            prompt_version=prompt_version, model_id=model_id,
        )

    def remember_question(
        self,
        question: str,
        intent: QueryIntent,
        *,
        as_of_date: str,
        data_version: str,
        prompt_version: str,
        model_id: str,
    ) -> None:
        query_hash = _question_hash(question)
        if not query_hash:
            return
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO question_cache
                   (query_hash, as_of_date, cache_key, data_version, prompt_version, model_id)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(query_hash, as_of_date, data_version, prompt_version, model_id)
                   DO UPDATE SET cache_key=excluded.cache_key""",
                (query_hash, as_of_date, intent.cache_key, data_version, prompt_version, model_id),
            )

    def get_by_key(
        self,
        cache_key: str,
        *,
        data_version: str,
        prompt_version: str,
        model_id: str,
    ) -> AnalysisResult | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT result_json FROM analysis_cache
                   WHERE cache_key=? AND data_version=? AND prompt_version=? AND model_id=?""",
                (cache_key, data_version, prompt_version, model_id),
            ).fetchone()
        if row is None:
            return None
        result = AnalysisResult.from_dict(json.loads(row[0]))
        result.cache_hit = True
        return result

    def put(
        self,
        intent: QueryIntent,
        result: AnalysisResult,
        *,
        data_version: str,
        prompt_version: str,
        model_id: str,
        question: str | None = None,
    ) -> None:
        payload: dict[str, Any] = result.to_dict()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO analysis_cache
                   (cache_key, intent_json, result_json, data_version, prompt_version, model_id)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(cache_key) DO UPDATE SET
                     intent_json=excluded.intent_json,
                     result_json=excluded.result_json,
                     data_version=excluded.data_version,
                     prompt_version=excluded.prompt_version,
                     model_id=excluded.model_id,
                     created_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')""",
                (
                    intent.cache_key,
                    json.dumps(intent.to_dict(), ensure_ascii=False, sort_keys=True),
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    data_version,
                    prompt_version,
                    model_id,
                ),
            )
            query_hash = _question_hash(question or "")
            if query_hash:
                conn.execute(
                    """INSERT INTO question_cache
                       (query_hash, as_of_date, cache_key, data_version, prompt_version, model_id)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(query_hash, as_of_date, data_version, prompt_version, model_id)
                       DO UPDATE SET cache_key=excluded.cache_key""",
                    (
                        query_hash, result.forecast_as_of or intent.as_of_date or "",
                        intent.cache_key, data_version, prompt_version, model_id,
                    ),
                )

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM analysis_cache").fetchone()[0])

    def clear(self) -> int:
        """Delete cached answers and return the number of removed rows."""
        with self._connect() as conn:
            count = int(conn.execute("SELECT COUNT(*) FROM analysis_cache").fetchone()[0])
            conn.execute("DELETE FROM analysis_cache")
            conn.execute("DELETE FROM question_cache")
        return count


class SemanticQuestionCache:
    """Chroma locates paraphrased questions; SQLite remains the answer source of truth."""

    def __init__(self, collection: Any, similarity_threshold: float, data_version: str,
                 prompt_version: str, model_id: str):
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold는 0~1 사이여야 합니다.")
        self.collection = collection
        self.threshold = similarity_threshold
        self.data_version = data_version
        self.prompt_version = prompt_version
        self.model_id = model_id

    @staticmethod
    def _scope_key(intent: QueryIntent) -> str:
        payload = {
            "assemblies": sorted(intent.assemblies),
            "date_start": intent.date_start,
            "date_end": intent.date_end,
            "request_type": intent.request_type,
            "as_of_date": intent.as_of_date,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _where(self, intent: QueryIntent) -> dict[str, Any]:
        values = {
            "scope_key": self._scope_key(intent),
            "data_version": self.data_version,
            "prompt_version": self.prompt_version,
            "model_id": self.model_id,
        }
        return {"$and": [{key: value} for key, value in values.items()]}

    def find(self, embedding: list[float], intent: QueryIntent) -> str | None:
        result = self.collection.query(
            query_embeddings=[embedding], n_results=5, where=self._where(intent),
            include=["metadatas", "distances"],
        )
        ids = result.get("ids", [[]])
        distances = result.get("distances", [[]])
        metadatas = result.get("metadatas", [[]])
        if not ids or not ids[0]:
            return None
        for distance, metadata in zip(distances[0], metadatas[0]):
            if 1.0 - float(distance) >= self.threshold:
                candidate_issue = str((metadata or {}).get("issue", "")).casefold().strip()
                current_issue = " ".join(intent.issue.casefold().split())
                candidate_issue = " ".join(candidate_issue.split())
                if candidate_issue and (
                    candidate_issue == current_issue
                    or SequenceMatcher(None, candidate_issue, current_issue).ratio() >= 0.62
                ):
                    cache_key = (metadata or {}).get("cache_key")
                    if isinstance(cache_key, str) and cache_key:
                        return cache_key
        return None

    def remember(self, embedding: list[float], query_text: str, cache_key: str, intent: QueryIntent) -> None:
        import uuid
        self.collection.upsert(
            ids=[uuid.uuid4().hex],
            embeddings=[embedding],
            documents=[query_text],
            metadatas=[{
                "cache_key": cache_key,
                "issue": " ".join(intent.issue.casefold().split()),
                "scope_key": self._scope_key(intent),
                "data_version": self.data_version,
                "prompt_version": self.prompt_version,
                "model_id": self.model_id,
            }],
        )

    def clear(self, batch_size: int = 1000) -> int:
        """Delete all cached-question vectors without dropping collection metadata."""
        if batch_size < 1:
            raise ValueError("batch_size는 1 이상이어야 합니다.")
        deleted = 0
        while True:
            result = self.collection.get(limit=batch_size, include=["metadatas"])
            ids = result.get("ids", [])
            if not ids:
                return deleted
            self.collection.delete(ids=ids)
            deleted += len(ids)


#!/usr/bin/env python3
"""최종 임베딩 아티팩트와 검색 백엔드가 공유하는 런타임 모듈."""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
DEFAULT_REVISION = "55ec6e9358a56d56af759bc8372e970caf8c305f"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def json_safe(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return json_safe(value.item())
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def date_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)[:10]


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    vectors: Path
    metadata: Path
    summary: Path

    @classmethod
    def resolve(cls, embedding_dir: Path) -> "ArtifactPaths":
        root = embedding_dir.expanduser().resolve()
        paths = cls(
            root=root,
            vectors=root / "embeddings.f16.npy",
            metadata=root / "chunk_metadata.parquet",
            summary=root / "embedding_summary.json",
        )
        missing = [str(path) for path in (paths.vectors, paths.metadata) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"최종 임베딩 파일이 없습니다: {missing}")
        return paths


class EmbeddingArtifact:
    def __init__(self, embedding_dir: Path):
        self.paths = ArtifactPaths.resolve(embedding_dir)
        self.vectors = np.load(self.paths.vectors, mmap_mode="r")
        self.parquet = pq.ParquetFile(self.paths.metadata)
        self.columns = self.parquet.schema_arrow.names
        self.summary = (
            json.loads(self.paths.summary.read_text(encoding="utf-8"))
            if self.paths.summary.is_file() else {}
        )
        if self.vectors.ndim != 2:
            raise ValueError(f"벡터 배열은 2차원이어야 합니다: {self.vectors.shape}")
        if len(self.vectors) != self.parquet.metadata.num_rows:
            raise ValueError(
                f"벡터/metadata 행 불일치: {len(self.vectors):,} != "
                f"{self.parquet.metadata.num_rows:,}"
            )

    @property
    def row_count(self) -> int:
        return len(self.vectors)

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def model_name(self) -> str:
        return str(self.summary.get("model_name") or DEFAULT_MODEL)

    @property
    def revision(self) -> str | None:
        return self.summary.get("revision") or DEFAULT_REVISION

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "dimension": self.dimension,
            "model_name": self.model_name,
            "revision": self.revision,
            "text_view": self.summary.get("text_view"),
            "selected_digest_sha256": self.summary.get("selected_digest_sha256"),
        }

    def chunk_ids(self, limit: int | None = None) -> list[str]:
        table = pq.read_table(self.paths.metadata, columns=["chunk_id"])
        values = table["chunk_id"].to_pylist()
        if limit is not None:
            values = values[:limit]
        return [str(value) for value in values]


@dataclass
class SearchFilters:
    council: str | None = None
    year: int | None = None
    committee: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    priorities: list[str] | None = None
    policy_category: str | None = None

    def matches(self, metadata: dict[str, Any]) -> bool:
        if self.council and str(metadata.get("council") or "") != self.council:
            return False
        if self.year is not None and int(metadata.get("year") or -1) != self.year:
            return False
        if self.committee and self.committee not in str(metadata.get("committee") or ""):
            return False
        current_date = date_string(metadata.get("meeting_date"))
        if self.date_from and (not current_date or current_date < self.date_from):
            return False
        if self.date_to and (not current_date or current_date > self.date_to):
            return False
        if self.priorities and str(metadata.get("priority") or "") not in self.priorities:
            return False
        if self.policy_category:
            categories = str(
                metadata.get("policy_categories_json")
                or metadata.get("policy_category") or ""
            )
            if self.policy_category not in categories:
                return False
        return True

    def chroma_where(self) -> dict[str, Any] | None:
        conditions: list[dict[str, Any]] = []
        if self.council:
            conditions.append({"council": self.council})
        if self.year is not None:
            conditions.append({"year": self.year})
        if self.date_from:
            conditions.append({"meeting_date_int": {"$gte": int(self.date_from.replace("-", "")[:8])}})
        if self.date_to:
            conditions.append({"meeting_date_int": {"$lte": int(self.date_to.replace("-", "")[:8])}})
        if self.priorities and len(self.priorities) == 1:
            conditions.append({"priority": self.priorities[0]})
        elif self.priorities:
            conditions.append({"priority": {"$in": self.priorities}})
        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}


@dataclass
class SearchResult:
    rank: int
    chunk_id: str
    similarity: float
    text: str
    metadata: dict[str, Any]
    parent_context: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QueryEncoder:
    def __init__(
        self,
        artifact: EmbeddingArtifact,
        device: str = "auto",
        model_cache: Path | None = None,
        local_files_only: bool = True,
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        resolved_device = (
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto" else device
        )
        kwargs: dict[str, Any] = {
            "device": resolved_device,
            "local_files_only": local_files_only,
        }
        if model_cache:
            kwargs["cache_folder"] = str(model_cache.expanduser().resolve())
        if artifact.revision:
            kwargs["revision"] = artifact.revision
        self.model = SentenceTransformer(artifact.model_name, **kwargs)
        self.model.max_seq_length = int(artifact.summary.get("max_seq_length") or 2048)
        self.device = resolved_device
        self.model_name = artifact.model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        kwargs: dict[str, Any] = {
            "batch_size": min(16, max(1, len(texts))),
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        if "query" in getattr(self.model, "prompts", {}):
            kwargs["prompt_name"] = "query"
        return self.model.encode(texts, **kwargs).astype(np.float32)


def rows_by_chunk_ids(metadata_path: Path, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not chunk_ids:
        return {}
    connection = duckdb.connect()
    table = connection.execute(
        """
        SELECT * FROM read_parquet(?)
        WHERE chunk_id IN (SELECT * FROM UNNEST(?))
        """,
        [str(metadata_path), chunk_ids],
    ).fetch_arrow_table()
    connection.close()
    return {str(row["chunk_id"]): row for row in table.to_pylist()}


def parent_contexts(parent_path: Path | None, parent_ids: Iterable[str]) -> dict[str, str]:
    ids = sorted({str(value) for value in parent_ids if value})
    if not parent_path or not parent_path.is_file() or not ids:
        return {}
    connection = duckdb.connect()
    table = connection.execute(
        """
        SELECT parent_chunk_id, text FROM read_parquet(?)
        WHERE parent_chunk_id IN (SELECT * FROM UNNEST(?))
        """,
        [str(parent_path.expanduser().resolve()), ids],
    ).fetch_arrow_table()
    connection.close()
    return {
        str(row["parent_chunk_id"]): str(row.get("text") or "")
        for row in table.to_pylist()
    }


def exact_topk(
    vectors: np.ndarray,
    queries: np.ndarray,
    top_k: int,
    limit: int | None = None,
    block_rows: int = 8192,
) -> tuple[np.ndarray, np.ndarray, float]:
    """float16 memmap을 전체 float32로 복사하지 않고 exact inner-product 검색."""
    row_count = min(len(vectors), limit) if limit else len(vectors)
    query_count = len(queries)
    best_scores = np.full((query_count, top_k), -np.inf, dtype=np.float32)
    best_indices = np.full((query_count, top_k), -1, dtype=np.int64)
    started = time.perf_counter()
    for start in range(0, row_count, block_rows):
        end = min(start + block_rows, row_count)
        block = np.asarray(vectors[start:end], dtype=np.float32)
        scores = queries @ block.T
        block_indices = np.arange(start, end, dtype=np.int64)
        for query_index in range(query_count):
            merged_scores = np.concatenate((best_scores[query_index], scores[query_index]))
            merged_indices = np.concatenate((best_indices[query_index], block_indices))
            count = min(top_k, len(merged_scores))
            positions = np.argpartition(merged_scores, -count)[-count:]
            order = positions[np.argsort(merged_scores[positions])[::-1]]
            best_scores[query_index, :count] = merged_scores[order]
            best_indices[query_index, :count] = merged_indices[order]
    return best_indices, best_scores, (time.perf_counter() - started) * 1000


class CouncilSearchService:
    def __init__(
        self,
        backend: str,
        embedding_dir: Path,
        parent_path: Path | None = None,
        chroma_dir: Path | None = None,
        collection: str = "council_chunks_final_v1",
        pg_dsn: str | None = None,
        pg_table: str = "council_chunks_final_v1",
        pg_ef_search: int = 100,
        device: str = "auto",
        model_cache: Path | None = None,
        local_files_only: bool = True,
    ):
        if backend not in {"exact", "chroma", "pgvector"}:
            raise ValueError(f"지원하지 않는 backend: {backend}")
        self.backend = backend
        self.artifact = EmbeddingArtifact(embedding_dir)
        self.parent_path = parent_path
        self.encoder = QueryEncoder(
            self.artifact, device=device, model_cache=model_cache,
            local_files_only=local_files_only,
        )
        self.collection_name = collection
        self._chunk_ids: list[str] | None = None
        self.pg_dsn = pg_dsn or os.environ.get("PGVECTOR_DSN")
        self.pg_table = pg_table
        self.pg_ef_search = pg_ef_search
        self.chroma = None
        if backend == "chroma":
            import chromadb

            if not chroma_dir:
                raise ValueError("Chroma 검색에는 chroma_dir가 필요합니다")
            client = chromadb.PersistentClient(path=str(chroma_dir.expanduser().resolve()))
            self.chroma = client.get_collection(collection, embedding_function=None)
        if backend == "pgvector" and not self.pg_dsn:
            raise ValueError("pgvector 검색에는 PGVECTOR_DSN 또는 pg_dsn이 필요합니다")

    def _search_exact(self, query_vector: np.ndarray, candidate_k: int) -> list[dict[str, Any]]:
        indices, scores, _ = exact_topk(
            self.artifact.vectors, query_vector[None, :], candidate_k
        )
        if self._chunk_ids is None:
            self._chunk_ids = self.artifact.chunk_ids()
        valid = [(int(index), float(score)) for index, score in zip(indices[0], scores[0]) if index >= 0]
        ids = [self._chunk_ids[index] for index, _ in valid]
        rows = rows_by_chunk_ids(self.artifact.paths.metadata, ids)
        return [
            {
                "chunk_id": chunk_id,
                "similarity": score,
                "text": str(rows.get(chunk_id, {}).get("text") or ""),
                "metadata": {
                    key: value for key, value in rows.get(chunk_id, {}).items()
                    if key not in {"text", "embedding_text"}
                },
            }
            for chunk_id, (_, score) in zip(ids, valid)
        ]

    def _search_chroma(
        self, query_vector: np.ndarray, candidate_k: int, filters: SearchFilters
    ) -> list[dict[str, Any]]:
        assert self.chroma is not None
        query_kwargs: dict[str, Any] = dict(
            query_embeddings=[query_vector.tolist()],
            n_results=min(candidate_k, self.chroma.count()),
            include=["documents", "metadatas", "distances"],
        )
        where = filters.chroma_where()
        if where:
            query_kwargs["where"] = where
        result = self.chroma.query(**query_kwargs)
        rows = []
        for chunk_id, document, metadata, distance in zip(
            result["ids"][0], result["documents"][0],
            result["metadatas"][0], result["distances"][0],
        ):
            rows.append({
                "chunk_id": str(chunk_id),
                "similarity": 1.0 - float(distance),
                "text": str(document or ""),
                "metadata": dict(metadata or {}),
            })
        return rows

    def _search_pgvector(
        self, query_vector: np.ndarray, candidate_k: int, filters: SearchFilters
    ) -> list[dict[str, Any]]:
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg import sql

        conditions = []
        parameters: list[Any] = []
        if filters.council:
            conditions.append(sql.SQL("council = %s")); parameters.append(filters.council)
        if filters.year is not None:
            conditions.append(sql.SQL("year = %s")); parameters.append(filters.year)
        if filters.committee:
            conditions.append(sql.SQL("committee ILIKE %s")); parameters.append(f"%{filters.committee}%")
        if filters.date_from:
            conditions.append(sql.SQL("meeting_date >= %s")); parameters.append(filters.date_from)
        if filters.date_to:
            conditions.append(sql.SQL("meeting_date <= %s")); parameters.append(filters.date_to)
        if filters.priorities:
            conditions.append(sql.SQL("priority = ANY(%s)")); parameters.append(filters.priorities)
        where = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions) if conditions else sql.SQL("")
        statement = sql.SQL(
            "SELECT chunk_id, document, metadata, 1 - (embedding <=> %s) AS similarity "
            "FROM {}{} ORDER BY embedding <=> %s LIMIT %s"
        ).format(sql.Identifier(self.pg_table), where)
        values = [query_vector, *parameters, query_vector, candidate_k]
        with psycopg.connect(self.pg_dsn) as connection:
            register_vector(connection)
            # PostgreSQL SET 문은 psycopg 바인딩 파라미터($1)를 허용하지 않는다.
            # set_config()를 사용하면 값 연결 없이 세션 설정을 안전하게 적용할 수 있다.
            connection.execute(
                "SELECT set_config('hnsw.ef_search', %s, false)",
                (str(self.pg_ef_search),),
            )
            connection.execute(
                "SELECT set_config('hnsw.iterative_scan', 'strict_order', false)"
            )
            rows = connection.execute(statement, values).fetchall()
        return [
            {
                "chunk_id": str(row[0]),
                "text": str(row[1] or ""),
                "metadata": dict(row[2] or {}),
                "similarity": float(row[3]),
            }
            for row in rows
        ]

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int | None = None,
        filters: SearchFilters | None = None,
        include_parent: bool = True,
    ) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("검색어가 비었습니다")
        filters = filters or SearchFilters()
        candidate_k = max(top_k, candidate_k or top_k * 10)
        started = time.perf_counter()
        query_vector = self.encoder.encode([query])[0]
        encoded_ms = (time.perf_counter() - started) * 1000
        search_started = time.perf_counter()
        if self.backend == "exact":
            candidates = self._search_exact(query_vector, candidate_k)
        elif self.backend == "chroma":
            candidates = self._search_chroma(query_vector, candidate_k, filters)
        else:
            candidates = self._search_pgvector(query_vector, candidate_k, filters)
        search_ms = (time.perf_counter() - search_started) * 1000
        selected = [row for row in candidates if filters.matches(row["metadata"])][:top_k]
        contexts = parent_contexts(
            self.parent_path,
            [str(row["metadata"].get("parent_chunk_id") or "") for row in selected],
        ) if include_parent else {}
        results = []
        for rank, row in enumerate(selected, start=1):
            metadata = {key: json_safe(value) for key, value in row["metadata"].items()}
            parent_id = str(metadata.get("parent_chunk_id") or "")
            result = SearchResult(
                rank=rank,
                chunk_id=row["chunk_id"],
                similarity=round(float(row["similarity"]), 6),
                text=row["text"],
                metadata=metadata,
                parent_context=contexts.get(parent_id),
            )
            results.append(result.to_dict())
        return {
            "query": query,
            "backend": self.backend,
            "model": self.encoder.model_name,
            "filters": asdict(filters),
            "top_k": top_k,
            "candidate_k": candidate_k,
            "query_embedding_ms": round(encoded_ms, 3),
            "search_ms": round(search_ms, 3),
            "result_count": len(results),
            "results": results,
        }


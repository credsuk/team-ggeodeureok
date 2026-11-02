from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env_file(path: Path | str) -> None:
    """Load simple KEY=VALUE settings, preserving values already set by the shell."""
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key or not key.replace("_", "").isalnum() or not key[0].isalpha():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _path_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    candidate = Path(raw).expanduser() if raw else default
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()


@dataclass(frozen=True)
class Settings:
    root: Path
    data_dir: Path
    chroma_dir: Path
    collection_name: str
    embedding_model: str
    bm25_db: Path
    source_metadata_db: Path
    cache_db: Path
    ollama_url: str
    ollama_model: str
    ollama_keep_alive: str
    reranker_model: str
    embedding_device: str
    reranker_device: str
    allow_model_download: bool
    data_version: str
    prompt_version: str
    cache_similarity: float
    dense_limit: int
    lexical_limit: int
    candidate_limit: int
    rerank_limit: int
    answer_evidence_limit: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_env_file(ROOT / "src" / ".env")
        root = ROOT
        data_dir = _path_env("DATA_DIR", root / "data")
        return cls(
            root=root,
            data_dir=data_dir,
            chroma_dir=_path_env("CHROMA_DIR", data_dir / "chroma_db"),
            collection_name=os.environ.get("CHROMA_COLLECTION", "assembly_metadata"),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "dragonkue/snowflake-arctic-embed-l-v2.0-ko"),
            bm25_db=_path_env("BM25_DB", data_dir / "bm25_index" / "assembly_bm25.sqlite3"),
            source_metadata_db=_path_env("SOURCE_METADATA_DB", data_dir / "source_metadata.sqlite3"),
            cache_db=_path_env("ANSWER_CACHE_DB", data_dir / "answer_cache.sqlite3"),
            ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
            ollama_model=os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b"),
            ollama_keep_alive=os.environ.get("OLLAMA_KEEP_ALIVE", "5m"),
            reranker_model=os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            embedding_device=os.environ.get("EMBEDDING_DEVICE", "cpu"),
            reranker_device=os.environ.get("RERANKER_DEVICE", "cpu"),
            allow_model_download=os.environ.get("ALLOW_MODEL_DOWNLOAD", "0").strip().lower() in {"1", "true", "yes"},
            data_version=os.environ.get("DATA_VERSION", "2024-2025-v1"),
            prompt_version=os.environ.get("PROMPT_VERSION", "local-rag-v8"),
            cache_similarity=float(os.environ.get("CACHE_SIMILARITY", "0.94")),
            dense_limit=int(os.environ.get("DENSE_LIMIT", "80")),
            lexical_limit=int(os.environ.get("LEXICAL_LIMIT", "120")),
            candidate_limit=int(os.environ.get("CANDIDATE_LIMIT", "300")),
            rerank_limit=int(os.environ.get("RERANK_LIMIT", "120")),
            answer_evidence_limit=int(os.environ.get("ANSWER_EVIDENCE_LIMIT", "24")),
        )

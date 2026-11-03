from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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


def _weight_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name}는 0 이상의 숫자여야 합니다.") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name}는 유한한 0 이상의 숫자여야 합니다.")
    return value


def _weight_list_env(name: str) -> tuple[float, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return ()
    values = tuple(_weight_env_value(name, item.strip()) for item in raw.split(","))
    if not values:
        return ()
    return values


def _weight_env_value(name: str, raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}는 쉼표로 구분한 0 이상의 숫자 목록이어야 합니다.") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name}의 각 값은 유한한 0 이상의 숫자여야 합니다.")
    return value


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
    reranker_models: tuple[str, ...]
    reranker_weights: tuple[float, ...]
    embedding_device: str
    reranker_device: str
    allow_model_download: bool
    data_version: str
    prompt_version: str
    cache_similarity: float
    dense_limit: int
    lexical_limit: int
    dense_rrf_weight: float
    lexical_rrf_weight: float
    candidate_limit: int
    rerank_limit: int
    answer_evidence_limit: int

    def __post_init__(self) -> None:
        if self.dense_rrf_weight == 0 and self.lexical_rrf_weight == 0:
            raise ValueError("DENSE_RRF_WEIGHT와 BM25_RRF_WEIGHT 중 하나는 0보다 커야 합니다.")
        if not self.reranker_models or len(self.reranker_models) != len(self.reranker_weights):
            raise ValueError("RERANKER_MODELS와 RERANKER_WEIGHTS의 항목 수가 맞지 않습니다.")
        if any(not math.isfinite(weight) or weight < 0 for weight in self.reranker_weights):
            raise ValueError("RERANKER_WEIGHTS는 유한한 0 이상의 숫자여야 합니다.")
        if not any(self.reranker_weights):
            raise ValueError("RERANKER_WEIGHTS 중 하나는 0보다 커야 합니다.")

    @property
    def cache_identity(self) -> str:
        identity = {
            "llm": self.ollama_model,
            "embedding": self.embedding_model,
            "rerankers": list(zip(self.reranker_models, self.reranker_weights)),
            "devices": {
                "embedding": self.embedding_device,
                "reranker": self.reranker_device,
            },
            "search": {
                "dense_weight": self.dense_rrf_weight,
                "bm25_weight": self.lexical_rrf_weight,
                "dense_limit": self.dense_limit,
                "lexical_limit": self.lexical_limit,
                "candidate_limit": self.candidate_limit,
                "rerank_limit": self.rerank_limit,
                "answer_evidence_limit": self.answer_evidence_limit,
            },
        }
        return json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_env(cls) -> "Settings":
        env_file = ROOT / "src" / ".env"
        if not env_file.is_file():
            env_file = ROOT / "src" / ".env.example"
        load_env_file(env_file)
        root = ROOT
        data_dir = _path_env("DATA_DIR", root / "data")
        reranker_model = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        configured_rerankers = tuple(
            name.strip() for name in os.environ.get("RERANKER_MODELS", "").split(",") if name.strip()
        )
        reranker_models = configured_rerankers or (reranker_model,)
        reranker_weights = _weight_list_env("RERANKER_WEIGHTS") or (1.0,) * len(reranker_models)
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
            reranker_model=reranker_model,
            reranker_models=reranker_models,
            reranker_weights=reranker_weights,
            embedding_device=os.environ.get("EMBEDDING_DEVICE", "cpu"),
            reranker_device=os.environ.get("RERANKER_DEVICE", "cpu"),
            allow_model_download=os.environ.get("ALLOW_MODEL_DOWNLOAD", "0").strip().lower() in {"1", "true", "yes"},
            data_version=os.environ.get("DATA_VERSION", "2024-2025-v1"),
            prompt_version=os.environ.get("PROMPT_VERSION", "local-rag-v8"),
            cache_similarity=float(os.environ.get("CACHE_SIMILARITY", "0.94")),
            dense_limit=int(os.environ.get("DENSE_LIMIT", "80")),
            lexical_limit=int(os.environ.get("LEXICAL_LIMIT", "120")),
            dense_rrf_weight=_weight_env("DENSE_RRF_WEIGHT", 1.0),
            lexical_rrf_weight=_weight_env("BM25_RRF_WEIGHT", 1.0),
            candidate_limit=int(os.environ.get("CANDIDATE_LIMIT", "300")),
            rerank_limit=int(os.environ.get("RERANK_LIMIT", "120")),
            answer_evidence_limit=int(os.environ.get("ANSWER_EVIDENCE_LIMIT", "24")),
        )

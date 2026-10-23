"""Paths and model settings for the council-record RAG pipeline."""
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# All matching JSONL/TXT pairs below these roots are indexed automatically.
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata_jsonl_python"
TRANSCRIPT_ROOT = PROJECT_ROOT / "data" / "to_text"


def _dataset_key_prefix(relative_path):
    """Create a stable year/category prefix from the source folder path."""
    scope = next(
        (part for part in relative_path.parts[:-1] if re.match(r"^\d{4}_", part)),
        None,
    )
    if scope is None:
        return "dataset", 0

    year, label = scope.split("_", 1)
    label = re.sub(r"\s*회의록$", "", label).strip()
    category = {"광역": "gwangyeok", "기초": "gicho"}.get(label, label.replace(" ", "_"))
    return f"{year}_{category}", relative_path.parts.index(scope) + 1


def _discover_datasets(metadata_root, transcript_root):
    """Recursively pair unique same-name JSONL/TXT files under the two roots."""
    if not metadata_root.is_dir() or not transcript_root.is_dir():
        raise FileNotFoundError(f"설정된 데이터 경로가 없습니다: {metadata_root}, {transcript_root}")
    metadata_files = {}
    transcript_files = {}
    for path in metadata_root.rglob("*.jsonl"):
        if path.stem in metadata_files:
            raise ValueError(f"JSONL 파일 이름이 중복되어 자동 짝짓기를 할 수 없습니다: {path.stem}")
        metadata_files[path.stem] = path
    for path in transcript_root.rglob("*.txt"):
        if path.stem in transcript_files:
            raise ValueError(f"TXT 파일 이름이 중복되어 자동 짝짓기를 할 수 없습니다: {path.stem}")
        transcript_files[path.stem] = path

    missing_transcripts = sorted(metadata_files.keys() - transcript_files.keys())
    missing_metadata = sorted(transcript_files.keys() - metadata_files.keys())
    if missing_transcripts or missing_metadata:
        raise ValueError(
            f"동일한 이름의 JSONL/TXT 파일 쌍이 아닙니다. "
            f"TXT 누락={missing_transcripts[:10]}, JSONL 누락={missing_metadata[:10]}"
        )
    if not metadata_files:
        raise ValueError(f"처리할 JSONL/TXT 파일 쌍이 없습니다: {metadata_root}, {transcript_root}")

    datasets = []
    for stem in sorted(metadata_files):
        metadata_path = metadata_files[stem]
        transcript_path = transcript_files[stem]
        relative_path = metadata_path.relative_to(metadata_root).with_suffix("")
        prefix, suffix_start = _dataset_key_prefix(relative_path)
        suffix_parts = relative_path.parts[suffix_start:]
        generated_key = "__".join([prefix, *suffix_parts]) if suffix_parts else prefix
        datasets.append({
            "key": generated_key,
            "metadata_path": metadata_path,
            "transcript_path": transcript_path,
        })
    return datasets


DATASETS = _discover_datasets(METADATA_ROOT, TRANSCRIPT_ROOT)
CHROMA_PATH = PROJECT_ROOT / "data" / "vector_db"
COLLECTION_NAME = "council_minutes"

EMBEDDING_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
# Multilingual reranker; compare candidates on the labeled Korean evaluation set.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_API_KEY = "not-needed"
LLM_MODEL_NAME = "gemma4:12b"


# RecursiveCharacterTextSplitter measures characters, not model tokens.
# Conservative character-based starting point for Korean embedding input limits.
# Tune with retrieval examples; the splitter does not count model tokens.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
INSERT_BATCH_SIZE = 256
INDEX_ONLY_SUBSTANTIVE = False
TOP_K_RETRIEVAL = 30
TOP_N_RESULTS = 100

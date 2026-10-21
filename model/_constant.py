"""Paths and model settings for the council-record RAG pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add one entry for each council/year pair. dataset_key must be unique and stable.
DATASETS = [
    {
        "key": "2024_gangwon",
        "metadata_path": PROJECT_ROOT / "data" / "metadata_jsonl_python" / "2024_광역 회의록" / "cleaned_2024_강원특별자치도의회.jsonl",
        "transcript_path": PROJECT_ROOT / "data" / "to_text" / "2024_광역 회의록" / "cleaned_2024_강원특별자치도의회.txt",
    },
]
CHROMA_PATH = PROJECT_ROOT / "data" / "vector_db"
COLLECTION_NAME = "council_minutes"

EMBEDDING_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
RERANKER_MODEL = "BAAI/bge-reranker-large"
LLM_BASE_URL = "http://localhost:8000/v1"
LLM_API_KEY = "EMPTY"
LLM_MODEL_NAME = "local-model"

# RecursiveCharacterTextSplitter measures characters, not model tokens.
# Conservative character-based starting point for Korean embedding input limits.
# Tune with retrieval examples; the splitter does not count model tokens.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
INSERT_BATCH_SIZE = 256
INDEX_ONLY_SUBSTANTIVE = False
TOP_K_RETRIEVAL = 30
TOP_N_RESULTS = 100

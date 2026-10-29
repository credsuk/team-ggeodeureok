import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def get_env(name, default=""):
    return os.getenv(name, default).strip()

class Settings:
    def __init__(self):
        self.database_url = get_env("DATABASE_URL")
        self.db_table = get_env("DB_TABLE", "public.council_arctic_raw_v1")
        self.embedding_model_path = get_env("EMBEDDING_MODEL_PATH")
        self.ollama_base_url = get_env("OLLAMA_BASE_URL", "http://host.docker.internal:11434").rstrip("/")
        self.ollama_model = get_env("OLLAMA_MODEL", "qwen3:8b")

        if self.db_table != "public.council_arctic_raw_v1":
            raise RuntimeError("DB_TABLE must be public.council_arctic_raw_v1")
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is missing")
        if not self.embedding_model_path:
            raise RuntimeError("EMBEDDING_MODEL_PATH is missing")
        if not self.ollama_model:
            raise RuntimeError("OLLAMA_MODEL is missing")

settings = Settings()

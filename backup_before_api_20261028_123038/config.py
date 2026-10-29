
import os
from pathlib import Path # 파일과 폴더의 경로를 다루는 기능
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent # 프로젝트 현재 기본 경로
load_dotenv(BASE_DIR / ".env") # 환경변수 로드


def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


class Settings:
    def __init__(self):
        self.database_url = get_env("DATABASE_URL")
        self.embedding_model_path = get_env("EMBEDDING_MODEL_PATH")
        self.llm_model_path = get_env("LLM_MODEL_PATH")

        self.db_table = get_env(
            "DB_TABLE",
            "mini_project_meeting_notes",
        )

        self.allowed_tables = {
            "mini_project_meeting_notes",
        }

        # 외부 모델 다운로드 방지
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.validate()

    def validate(self):
        if not self.database_url:
            raise RuntimeError("DATABASE_URL이 설정되지 않았습니다.")

        if self.db_table not in self.allowed_tables:
            raise RuntimeError("허용되지 않은 DB_TABLE 설정입니다.")

        if not self.embedding_model_path:
            raise RuntimeError(
                "EMBEDDING_MODEL_PATH가 설정되지 않았습니다."
            )

        if not self.llm_model_path:
            raise RuntimeError(
                "LLM_MODEL_PATH가 설정되지 않았습니다."
            )

        if not Path(self.embedding_model_path).is_dir():
            raise RuntimeError(
                "임베딩 모델 폴더를 찾을 수 없습니다."
            )

        if not Path(self.llm_model_path).is_dir():
            raise RuntimeError(
                "LLM 모델 폴더를 찾을 수 없습니다."
            )


settings = Settings()

# config 기본 코드 -----------------------------------------------------------------------------------------------
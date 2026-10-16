"""
프로젝트 설정 및 기본 파라미터 정의
"""
import os
from pathlib import Path
from typing import Optional

# 기본 디렉토리 경로
BASE_DIR = Path(__file__).resolve().parent.parent

# 데이터 디렉토리 경로 (data/text 또는 text 자동 감지)
if (BASE_DIR / "data" / "text").exists():
    DATA_DIR = BASE_DIR / "data" / "text"
elif (BASE_DIR / "text").exists():
    DATA_DIR = BASE_DIR / "text"
else:
    DATA_DIR = BASE_DIR / "data"

CHROMA_DIR = BASE_DIR / "chroma_db"

# ChromaDB 기본 컬렉션 설정
COLLECTION_NAME = "local_council_minutes"
DISTANCE_METRIC = "cosine"

# 청킹(Chunking) 하이퍼파라미터
CHUNK_SIZE = 800          # 장문 발언 서브청킹 기준 크기 (문자 수)
CHUNK_OVERLAP = 100       # 서브청킹 간 중첩 크기
MIN_CHAR_LEN = 40         # 단순 진행용 초단문 필터링 기준 (문자 수)

# 배치 적재 파라미터
BATCH_SIZE = 500          # ChromaDB add 1회당 전송 문서 수

# -------------------------------------------------------------
# 추천 임베딩 모델 프리셋 (별칭 지원)
# -------------------------------------------------------------
EMBEDDING_PRESETS = {
    # 1. 한국어 특화 SBERT (기본값, 768차원, 빠르고 안정적인 성능)
    "ko-sbert": "jhgan/ko-sbert-nli",
    # 2. 한국어 RoBERTa 기반 SimCSE (768차원, 문장 의미 유사도 우수)
    "kosimcse": "BM-K/KoSimCSE-roberta-multitask",
    # 3. 고성능 다국어/한국어 SOTA 모델 (1024차원, 대용량 VRAM 권장)
    "bge-m3": "BAAI/bge-m3",
    # 4. BGE-M3 한국어 미세조정 버전 (1024차원)
    "bge-m3-ko": "dragonkyle/bge-m3-ko",
    # 5. 다국어 E5 Base 모델 (768차원)
    "e5-base": "intfloat/multilingual-e5-base",
    # 6. 서울대 한국어 SBERT (768차원)
    "kr-sbert": "snunlp/KR-SBERT-V40K-klue-subword",
}

# 기본 임베딩 모델 (환경변수 EMBEDDING_MODEL_NAME이 지정되어 있으면 최우선 적용)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", EMBEDDING_PRESETS["ko-sbert"])


def resolve_embedding_model(name_or_alias: Optional[str] = None) -> str:
    """
    임베딩 모델 식별자 또는 프리셋 별칭을 전체 HuggingFace 모델 경로로 변환합니다.
    예:
      'ko-sbert' -> 'jhgan/ko-sbert-nli'
      'kosimcse' -> 'BM-K/KoSimCSE-roberta-multitask'
      'BAAI/bge-m3' -> 'BAAI/bge-m3' (그대로 유지)
      None -> EMBEDDING_MODEL_NAME
    """
    if not name_or_alias:
        name_or_alias = EMBEDDING_MODEL_NAME

    # 별칭 매핑 시도 (소문자 치환)
    key = name_or_alias.strip().lower()
    return EMBEDDING_PRESETS.get(key, name_or_alias.strip())


def get_model_slug(model_name: str) -> str:
    """임베딩 모델명에서 컬렉션명으로 쓰기 적합한 안전한 슬러그 생성"""
    slug = model_name.split("/")[-1].lower()
    return slug.replace("-", "_").replace(".", "_")

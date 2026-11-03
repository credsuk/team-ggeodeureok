"""지방의회 회의록 LLM 메타데이터 추출(metadata_llm) 공통 환경 설정 및 상수 정의 모듈.

- 분석 대상 의회(NAME) 및 연도 접두어(YEAR_NAME)
- TSV 분할(split) 기본 입력 파일 및 출력 디렉토리 경로
- LLM 메타데이터 추출(extract) 기본 입력, 출력 JSONL 경로 및 배치 크기
- 로컬 LLM(Ollama) 모델 설정 및 정책 카테고리/스키마 정의
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. 대상 의회 및 연도 설정
# ---------------------------------------------------------------------------
YEAR = "2024"
NAME = "강원특별자치도의회"
YEAR_NAME = f"{YEAR}_{NAME}"

# ---------------------------------------------------------------------------
# 2. 회의록 TSV 분할(split) 설정
# ---------------------------------------------------------------------------
# 입력 후보 경로 (data/to_text/, text/, data/to_text2/ 등 자동 탐색 지원)
_candidate_inputs = [
    Path("data") / "to_text" / f"{YEAR}_광역 회의록" / f"cleaned_{YEAR_NAME}.txt",
    Path("data") / "to_text2" / f"{YEAR}_광역 회의록" / f"cleaned_{YEAR_NAME}.txt",
    Path("text") / f"cleaned_{YEAR_NAME}.txt",
    Path("data") / f"{YEAR}_광역 회의록" / f"cleaned_{YEAR_NAME}.jsonl",
]
SPLIT_DEFAULT_INPUT = next((p for p in _candidate_inputs if p.exists()), _candidate_inputs[0])

# 분할 파일 저장 폴더
SPLIT_DEFAULT_OUTPUT_DIR = Path("data") / "to_text" / f"{YEAR}_광역 회의록" / "split" / NAME

# 분할 시 파일당 최대 데이터 행 수 및 문자 수
SPLIT_DEFAULT_ROWS_PER_FILE = 20
SPLIT_DEFAULT_MAX_CHARS = 10000

# ---------------------------------------------------------------------------
# 3. LLM 메타데이터 추출(extract) 설정
# ---------------------------------------------------------------------------
EXTRACT_DEFAULT_INPUT = SPLIT_DEFAULT_OUTPUT_DIR
EXTRACT_DEFAULT_OUTPUT = Path("metadata") / f"cleaned_{YEAR_NAME}_metadata.jsonl"
EXTRACT_DEFAULT_ROWS_PER_BATCH = 20
EXTRACT_DEFAULT_MAX_CHARS = 10000
EXTRACT_DEFAULT_MAX_BATCHES = 0  # 0이면 전체 배치 처리

# ---------------------------------------------------------------------------
# 4. 로컬 LLM (Ollama) 모델 및 정책 분류 설정
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
# 추천 모델: gemma4:e2b, gemma4:e4b, qwen2.5-coder:7b, huihui_ai/exaone3.5-abliterated
METADATA_MODEL_NAME = "gemma4:e2b"

# 정책 분야 표준 카테고리 8종
POLICY_CATEGORIES = [
    "경제/산업",
    "농림/수산",
    "보건/복지",
    "교육/문화",
    "환경/안전",
    "도시/교통",
    "자치/행정",
    "기타",
]

from pathlib import Path


NAME = "전북특별자치도의회"
YEAR_NAME = "2025_" + NAME

SPLIT_DEFAULT_INPUT = Path(r"data\to_text\2025_광역 회의록\cleaned_" + YEAR_NAME + ".txt")
SPLIT_DEFAULT_OUTPUT_DIR = Path(r"data\to_text\2025_광역 회의록\split\"+ NAME + ")
SPLIT_DEFAULT_ROWS_PER_FILE = 20
SPLIT_DEFAULT_MAX_CHARS = 10000



EXTRACT_DEFAULT_INPUT = Path(r"data\to_text\2025_광역 회의록\split\"+ NAME +")
EXTRACT_DEFAULT_OUTPUT = Path(r"metadata\cleaned_" + YEAR_NAME + "_metadata.jsonl")
EXTRACT_DEFAULT_ROWS_PER_BATCH = 20
EXTRACT_DEFAULT_MAX_CHARS = 10000
EXTRACT_DEFAULT_MAX_BATCHES = 0  # 0 means process every batch.
#!/usr/bin/env python3
"""실행에 필요한 Hugging Face 모델을 model/huggingface/에 내려받고, Ollama 모델 설치 명령을 안내한다.

    python scripts/download_models.py               # 임베딩 + 재순위 모델
    python scripts/download_models.py --comparison  # 실험 비교용 모델(KURE, bge-m3)까지
    python scripts/download_models.py --ollama      # Ollama 모델도 ollama pull로 설치

회의록 데이터는 외부로 전송하지 않습니다. 이 스크립트는 모델 파일만 내려받습니다.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "model" / "huggingface"

# (모델 ID, 고정 revision 또는 None, 용도)
REQUIRED = [
    ("dragonkue/snowflake-arctic-embed-l-v2.0-ko", "55ec6e9358a56d56af759bc8372e970caf8c305f", "청크·질의 임베딩"),
    ("BAAI/bge-reranker-v2-m3", None, "검색 후보 재순위"),
]
COMPARISON = [
    ("nlpai-lab/KURE-v1", None, "임베딩 모델 비교 실험"),
    ("BAAI/bge-m3", None, "임베딩 모델 비교 실험"),
]
OLLAMA_MODELS = [
    ("hf.co/mykor/A.X-4.0-Light-gguf:Q4_K_M", "설명 Agent 기본 모델"),
    ("exaone3.5:7.8b", "설명 Agent 대안 / LLM 비교"),
    ("gemma4:e2b", "정책 메타데이터 추출 (노트북 02)"),
    ("qwen2.5-coder:14b", "HTML 파싱 규칙 생성 (노트북 01)"),
]
OLLAMA_COMPARISON = [
    ("hf.co/mradermacher/kanana-1.5-8b-instruct-2505-GGUF:Q4_K_M", "LLM 비교 (노트북 06)"),
    ("hf.co/DevQuasar/trillionlabs.Tri-7B-GGUF:Q4_K_M", "LLM 비교 (노트북 06)"),
    ("hf.co/rippertnt/HyperCLOVAX-SEED-Text-Instruct-1.5B-Q4_K_M-GGUF:Q4_K_M", "LLM 비교 (노트북 06)"),
    ("hf.co/LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF:Q4_K_M", "LLM 비교 (노트북 06)"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comparison", action="store_true", help="비교 실험용 모델도 내려받기")
    parser.add_argument("--ollama", action="store_true", help="ollama pull 실행")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for model_id, revision, purpose in REQUIRED + (COMPARISON if args.comparison else []):
        print(f"[HF] {model_id} ({purpose}) → {CACHE_DIR}")
        snapshot_download(model_id, revision=revision, cache_dir=CACHE_DIR)

    ollama_models = OLLAMA_MODELS + (OLLAMA_COMPARISON if args.comparison else [])
    if args.ollama:
        if not shutil.which("ollama"):
            raise SystemExit("ollama 실행 파일이 없습니다. https://ollama.com 에서 설치하세요.")
        for name, purpose in ollama_models:
            print(f"[Ollama] {name} ({purpose})")
            subprocess.run(["ollama", "pull", name], check=True)
    else:
        print("\nOllama 모델은 다음 명령으로 설치합니다 (또는 --ollama):")
        for name, purpose in ollama_models:
            print(f"  ollama pull {name}    # {purpose}")
    print(f"\n완료. .env의 COUNCIL_MODEL_CACHE를 {CACHE_DIR} 로 지정하세요.")


if __name__ == "__main__":
    main()

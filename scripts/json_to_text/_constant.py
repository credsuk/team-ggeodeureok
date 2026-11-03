"""지방의회 회의록 전처리(json_to_text) 공통 환경 설정 및 상수 정의 모듈.

- 대상 연도(target) 및 원본 데이터 폴더(org_dir)
- 샘플 HTML 추출 폴더(sample_dir)
- CSS Selector 파싱 규칙 출력 폴더(rules_output_dir) 및 통합 파일(all_rule_file)
- 최종 정제된 텍스트/TSV 저장 폴더(text_dir)
- 로컬 LLM(Ollama OpenAI 호환 API) 클라이언트 및 모델 지정
"""
import os
from pathlib import Path
from openai import OpenAI

# ---------------------------------------------------------------------------
# 1. 대상 연도 및 경로 설정 (Directory Paths)
# ---------------------------------------------------------------------------
target = "2024"

# 원본 JSON 회의록 폴더: origin/ 우선 탐색 후 data/org/ 대체 지원
_candidate_org_dirs = [
    Path(f"./origin/{target}_광역 회의록"),
    Path(f"./data/org/{target}_광역 회의록"),
    Path(f"../origin/{target}_광역 회의록"),
    Path(f"../../origin/{target}_광역 회의록"),
]
org_dir = str(next((p for p in _candidate_org_dirs if p.exists()), Path(f"./origin/{target}_광역 회의록")))

# LLM 파싱 규칙 생성용 샘플 TXT 파일들이 저장될 폴더 경로
sample_dir = f"./data/sample/{target}_광역 회의록"

# 각 의회별 회의록 파일당 추출할 HTML 본문 샘플 개수
records_num = 3

# LLM이 생성한 의회별 파싱 규칙(개별 JSON 및 디버그 로그) 저장 폴더 경로
rules_output_dir = "./data/llm_rules_outputs/광역 회의록"

# 개별 의회 파싱 규칙을 하나로 집약한 최종 통합 규칙 JSON 파일 경로
all_rule_file = f"./data/to_text/llm_rules_outputs/consolidated_parsing_rules_{target}.json"

# 최종 HTML 태그 정제 및 화자 연속 발언 병합 결과 TSV 파일들이 저장될 폴더 경로
text_dir = f"./data/to_text2/{target}_광역 회의록"

# ---------------------------------------------------------------------------
# 2. 로컬 LLM (Ollama) 모델 및 클라이언트 설정
# ---------------------------------------------------------------------------
# 규칙 생성에 활용할 모델명 (Ollama에서 사전에 pull 필요)
# 추천 모델: "qwen2.5-coder:7b", "qwen2.5-coder:14b", "huihui_ai/exaone3.5-abliterated"
make_model_name = "qwen2.5-coder:14b"

# Ollama 로컬 서버의 OpenAI 호환 엔드포인트 클라이언트
client = OpenAI(
    base_url="http://localhost:11434/v1", 
    api_key="not-needed" 
)

#!/usr/bin/env python3
"""2단계: 로컬 LLM(Ollama)을 활용한 정책 의제 및 메타데이터 추출기 (extract_metadata_llm.py).

[주요 기능]
1. split_council_tsv.py로 분할된 배치 TSV 파일들(또는 원본 TSV)을 읽어 발언문 단위로 순회합니다.
2. 로컬 LLM(Ollama)을 호출하여 정책 분야, 핵심 쟁점, 2~4어절 구체적 정책 복합키워드, 요약을 추출합니다.
3. [체크포인트 이어하기]: 이미 처리된 row_id는 자동으로 건너뛰며, 중단 후 재실행 시 기존 JSONL 파일에 실시간 추가(append)됩니다.
4. 엄격한 금지 단어(Negative Constraints) 필터링: '사업', '예산', '조례' 등 포괄적 1단어 명사를 배제합니다.
5. 단순 의전 발언(개식, 폐식, 출석확인 등)은 is_substantive=false로 자동 태깅합니다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests
import _constant

# ---------------------------------------------------------------------------
# LLM 프롬프트 및 정책 분야 정의
# ---------------------------------------------------------------------------
EXTRACTION_SYSTEM_PROMPT = """당신은 대한민국 지방의회 정책 분석 및 메타데이터 구축 전문가입니다.
주어진 지방의회 회의록 발언을 정밀하게 분석하여, 향후 검색 및 필터링 메타데이터로 활용할 수 있도록 JSON 포맷으로만 응답하세요.

[⚠️ 절대 금지 키워드 (Negative Constraints)]
아래와 같이 모든 회의록에 흔히 등장하는 1단어짜리 포괄적 명사, 기관명, 직함은 'keywords'에 절대로 포함하지 마세요:
- 금지 단어: 사업, 예산, 지원, 조례, 조례안, 의견, 업무, 행정, 의회, 의원, 위원, 발언, 질의, 답변, 감사, 생각, 회의, 내용, 부분

[✅ 권장 키워드 형식]
2~4개 단어가 결합된 '구체적이고 실질적인 정책 의제 복합명사'로 추출하세요:
- 좋은 예시: "농지제도 혁신", "지역 소멸 대응", "미래 인구정책", "일자리 산업 육성", "학령인구 감소", "교육복지 강화", "공공의료 확충"

[출력 포맷 (반드시 아래 순수 JSON 형식으로만 응답할 것, 코드블록 없이 순수 JSON만 출력)]
{
  "is_substantive": true,
  "policy_category": "경제/산업",
  "core_issue": "농지 규제 완화를 통한 농가 소득 증대 및 농지제도 혁신 방안",
  "keywords": ["농지제도 혁신", "농업 규제 완화", "경자유전 원칙 개편"],
  "summary": "해방 이후 지속된 농지 소유 규제를 완화하여 청년농 유입과 농업 경쟁력을 강화해야 한다고 주장함."
}

※ policy_category 후보: "경제/산업", "농림/수산", "보건/복지", "교육/문화", "환경/안전", "도시/교통", "자치/행정", "기타" 중 정확히 1개 선택.
※ 만약 단순 의전 멘트(개식 선포, 방청객 환영, 단순 이석 보고 등)로 실질적 정책 논의가 없다면 is_substantive를 false로 하고 keywords는 []로 비워두세요.
"""


def clean_and_parse_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """LLM 출력 문자열에서 코드블록 마크다운 등을 정리하고 안전하게 JSON 객체로 파싱합니다."""
    if not raw_text:
        return None

    text = raw_text.strip()
    if "```" in text:
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = text.replace('```', '').strip()

    match = re.search(r'(\{[\s\S]*\})', text)
    if match:
        json_candidate = match.group(1).strip()
    else:
        json_candidate = text

    try:
        return json.loads(json_candidate)
    except Exception:
        pass

    try:
        # 후행 쉼표 제거 시도
        fixed = re.sub(r',\s*\}', '}', json_candidate)
        fixed = re.sub(r',\s*\]', ']', fixed)
        return json.loads(fixed)
    except Exception:
        return None


def call_ollama(
    prompt: str,
    system_prompt: str,
    model_name: str = _constant.METADATA_MODEL_NAME,
    base_url: str = _constant.OLLAMA_BASE_URL,
    timeout: int = 60,
) -> Optional[str]:
    """Ollama 로컬 API를 호출하여 생성 결과를 문자열로 반환합니다."""
    url = f"{base_url}/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code == 200:
            return resp.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"  [LLM 호출 오류] {e}")
    return None


def load_processed_row_ids(output_path: Path) -> Set[int]:
    """기존 출력 JSONL 파일이 있는 경우 이미 처리된 row_id들을 로드하여 체크포인트를 구성합니다."""
    processed = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if "row_id" in record:
                        processed.add(int(record["row_id"]))
                except Exception:
                    continue
    return processed


def extract_metadata_from_tsv(
    input_path: Path = _constant.EXTRACT_DEFAULT_INPUT,
    output_path: Path = _constant.EXTRACT_DEFAULT_OUTPUT,
    model_name: str = _constant.METADATA_MODEL_NAME,
    max_records: int = _constant.EXTRACT_DEFAULT_MAX_BATCHES,
) -> int:
    """TSV 파일(또는 폴더 내 분할 TSV들)을 읽어 Ollama LLM으로 메타데이터를 추출합니다."""
    # 입력 대상 파일 목록 수집
    if input_path.is_file():
        tsv_files = [input_path]
    elif input_path.is_dir():
        tsv_files = sorted(input_path.glob("part_*.tsv"))
        if not tsv_files:
            tsv_files = sorted(input_path.glob("*.tsv")) + sorted(input_path.glob("*.txt"))
    else:
        raise FileNotFoundError(f"입력 경로를 찾을 수 없습니다: {input_path}")

    if not tsv_files:
        print(f"[경고] 처리할 TSV 파일이 없습니다: {input_path}")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_ids = load_processed_row_ids(output_path)
    if processed_ids:
        print(f"[체크포인트 로드] 이미 처리된 {len(processed_ids):,}건을 감지하여 이어서 진행합니다.")

    extracted_count = 0
    total_scanned = 0

    with output_path.open("a", encoding="utf-8") as out_f:
        for tsv_file in tsv_files:
            with tsv_file.open("r", encoding="utf-8-sig", newline="") as in_f:
                reader = csv.DictReader(in_f, delimiter="\t")
                for row in reader:
                    total_scanned += 1
                    row_id_val = int(row.get("row_id", total_scanned))

                    # 이미 처리된 행은 건너뛰기
                    if row_id_val in processed_ids:
                        continue

                    utterance = row.get("utterance", "").strip()
                    if not utterance or len(utterance) < 15:
                        continue

                    speaker_name = row.get("speaker_name", "")
                    speaker_role = row.get("speaker_role", "")
                    agenda_title = row.get("agenda_title", "")
                    speaker_info = f"{speaker_role} {speaker_name} (안건: {agenda_title})"

                    user_prompt = f"""[발언자 정보]: {speaker_info}
[발언 내용]:
{utterance[:2000]}
"""
                    llm_output = call_ollama(
                        prompt=user_prompt,
                        system_prompt=EXTRACTION_SYSTEM_PROMPT,
                        model_name=model_name,
                    )
                    parsed = clean_and_parse_json(llm_output) if llm_output else None

                    # 파싱 실패 시 기본 안전 레코드 구성
                    if not parsed or not isinstance(parsed, dict):
                        parsed = {
                            "is_substantive": len(utterance) > 50,
                            "policy_category": "기타",
                            "core_issue": agenda_title[:80] if agenda_title else "일반 안건 논의",
                            "keywords": [],
                            "summary": utterance[:100] + "...",
                        }

                    # 최종 메타데이터 레코드 조합
                    result_record = {
                        "row_id": row_id_val,
                        "date": row.get("date") or row.get("MTG_DE", ""),
                        "assembly": row.get("assembly") or row.get("RASMBLY_NM", ""),
                        "speaker_role": speaker_role,
                        "speaker_name": speaker_name,
                        "speaker_type": row.get("speaker_type", ""),
                        "agenda_title": agenda_title,
                        "utterance": utterance,
                        "is_substantive": parsed.get("is_substantive", True),
                        "policy_category": parsed.get("policy_category", "기타"),
                        "core_issue": parsed.get("core_issue", ""),
                        "keywords": parsed.get("keywords", []),
                        "summary": parsed.get("summary", ""),
                    }

                    # JSONL 실시간 파일 쓰기 (flush)
                    out_f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    processed_ids.add(row_id_val)
                    extracted_count += 1

                    if extracted_count % 5 == 0 or extracted_count == 1:
                        print(
                            f"  [추출 진행] row_id={row_id_val} | 카테고리={result_record['policy_category']} | "
                            f"핵심이슈='{result_record['core_issue'][:25]}...' | 키워드={result_record['keywords']}"
                        )

                    if max_records and extracted_count >= max_records:
                        print(f"[제한 도달] 지정된 최대 추출 건수({max_records}건)에 도달하여 마감합니다.")
                        break

            if max_records and extracted_count >= max_records:
                break

    print(
        f"\n[추출 완료] 이번 실행에서 {extracted_count:,}건을 성공적으로 추출 및 저장했습니다.\n"
        f"누적 결과 저장 경로: {output_path.resolve()}"
    )
    return extracted_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="로컬 LLM을 사용하여 회의록 발언에서 정책 메타데이터(JSONL)를 추출합니다."
    )
    parser.add_argument(
        "--input", type=Path, default=_constant.EXTRACT_DEFAULT_INPUT,
        help=f"입력 TSV 파일 또는 분할 폴더 경로 (기본값: {_constant.EXTRACT_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", type=Path, default=_constant.EXTRACT_DEFAULT_OUTPUT,
        help=f"출력 JSONL 파일 경로 (기본값: {_constant.EXTRACT_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model", type=str, default=_constant.METADATA_MODEL_NAME,
        help=f"Ollama 로컬 LLM 모델명 (기본값: {_constant.METADATA_MODEL_NAME})",
    )
    parser.add_argument(
        "--sample", type=int, default=_constant.EXTRACT_DEFAULT_MAX_BATCHES,
        help="추출할 최대 발언 레코드 수 (0이면 전체 처리)",
    )
    args = parser.parse_args()

    try:
        extract_metadata_from_tsv(
            input_path=args.input,
            output_path=args.output,
            model_name=args.model,
            max_records=args.sample,
        )
        return 0
    except Exception as exc:
        print(f"[오류 발생] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

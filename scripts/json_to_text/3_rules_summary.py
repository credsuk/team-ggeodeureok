"""3단계: 의회별 CSS Selector 파싱 규칙 통합 및 검증 (3_rules_summary.py).

2단계(2_rules_make.py)에서 생성된 개별 의회 파싱 규칙(rule_*.json)을 순회하여
엄격한 CSS Selector 유효성(agenda, speaker, utterance)을 검증하고,
필요 시 LLM 보정 또는 기본값 대체(fallback)를 적용하여 단일 통합 규칙 JSON 파일로 병합합니다.
"""
from __future__ import annotations

import glob
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Any

from openai import OpenAI
import _constant as constant
from pipeline_timing import start_step, record_file, finish_step


# 한글 포함 여부 검사용 정규식 (선택자에 한글 설명이 들어가는 오염 방지)
KOREAN_PATTERN = re.compile(r'[가-힣]')

# CSS 선택자 기본값 (선택자 탐색 실패 시 적용)
FALLBACK_SELECTORS = {
    "agenda": "div",
    "speaker": "span.spkrbld",
    "utterance": "p",
}


def validate_rule(rule_dict: dict) -> bool:
    """추출된 CSS Selector 규칙이 유효한지(한글 없음, 슬래시 없음, 필수 필드 존재) 검사합니다."""
    for key in ("agenda", "speaker", "utterance"):
        val = rule_dict.get(key, "")
        if not isinstance(val, str) or not val.strip():
            return False
        # 한글이나 자연어 슬래시('/') 설명이 포함된 경우 잘못된 선택자로 판단
        if KOREAN_PATTERN.search(val) or "/" in val:
            return False
    return True


def normalize_council_name(filename: str) -> str:
    """규칙 파일명에서 접두어(rule_sample_, rule_, sample_)를 제거하여 의회 고유명을 추출합니다."""
    stem = Path(filename).stem
    for prefix in ("rule_sample_", "rule_", "sample_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    return stem


def generate_strict_parsing_rules(
    input_dir: str = constant.rules_output_dir,
    output_file: str = constant.all_rule_file,
    model_name: str = constant.make_model_name,
) -> Dict[str, Any]:
    """개별 의회 규칙 파일들을 모아 유효성을 검증하고 단일 통합 규칙 JSON으로 저장합니다.
    
    1. rules_output_dir 내 rule_*.json 파일들을 탐색
    2. 파일에 이미 유효한 CSS 선택자가 있으면 즉시 반영
    3. 선택자가 불완전하거나 한글이 섞여있으면 LLM을 통해 정제 시도
    4. 여전히 유효하지 않으면 표준 fallback 선택자 적용
    5. 결과를 all_rule_file에 저장
    """
    run_id, step_started_at = start_step("3")
    client = constant.client

    # rule_*.json 또는 rule_sample_*.json 규칙 파일 검색
    rule_files = sorted(glob.glob(os.path.join(input_dir, "rule_*.json")))
    if not rule_files:
        rule_files = sorted(glob.glob(os.path.join(input_dir, "**", "rule_*.json"), recursive=True))

    if not rule_files:
        print(f"[경고] '{input_dir}' 폴더에서 검증할 규칙 JSON(rule_*.json) 파일을 찾을 수 없습니다.")
        finish_step(run_id, "3", step_started_at)
        return {}

    print(f"[3단계 시작] 총 {len(rule_files)}개 규칙 파일에서 통합 파싱 규칙을 추출 및 검증합니다...")

    # 기존 통합 규칙 파일이 있다면 이전 결과를 보존하여 덮어쓰기 유실 방지
    consolidated_rules: Dict[str, Any] = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                prev = json.load(f)
                if isinstance(prev, dict):
                    consolidated_rules.update(prev)
                    print(f"  [기존 규칙 로드] {len(consolidated_rules)}개 의회 규칙을 유지합니다.")
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [기존 규칙 로드 경고] {e}")

    # LLM 정제 프롬프트 정의
    system_prompt = (
        "You are a strict CSS selector parser and cleaner. "
        "Extract pure CSS selectors (e.g. 'div.agnd_0', 'span.spkrbld', 'p') from given data. "
        "NEVER output Korean words, slashes, or explanations. "
        "Output ONLY a JSON object with keys: agenda, speaker, utterance."
    )

    for file_path in rule_files:
        file_started = time.perf_counter()
        file_started_at = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        status = "error"
        item_count = 1
        council_name = normalize_council_name(os.path.basename(file_path))

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_json = json.load(f)

            # JSON 구조 평탄화 (의회명이 키로 감싸져 있는 경우 처리)
            candidate = raw_json
            if isinstance(raw_json, dict) and council_name not in raw_json and len(raw_json) == 1:
                candidate = next(iter(raw_json.values()))

            if isinstance(candidate, dict):
                direct_rule = {
                    key: candidate.get(key, "")
                    for key in ("agenda", "speaker", "utterance")
                }
                # 특수 모드(__TEXT_NODE_...__)이거나 정상 CSS 선택자인 경우 바로 반영
                is_text_node_mode = direct_rule.get("speaker") == "__TEXT_NODE_SPEAKER__"
                if is_text_node_mode or validate_rule(direct_rule):
                    consolidated_rules[council_name] = direct_rule
                    print(f"  [규칙 승인] '{council_name}' -> {direct_rule}")
                    status = "success"
                    record_file(run_id, "3", file_path, time.perf_counter() - file_started,
                                status, item_count, file_started_at)
                    continue

            # 유효하지 않은 경우 LLM 정제 시도
            user_prompt = f"""다음 의회 규칙 데이터에서 순수 CSS 선택자를 추출하여 JSON으로 반환해:
{json.dumps(candidate, ensure_ascii=False)[:1500]}
응답 형식: {{"agenda": "...", "speaker": "...", "utterance": "..."}}"""

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=200,
                    response_format={"type": "json_object"},
                )
                llm_text = (response.choices[0].message.content or "").strip()
                cleaned_json = json.loads(llm_text)
                if validate_rule(cleaned_json):
                    consolidated_rules[council_name] = cleaned_json
                    print(f"  [LLM 정제 성공] '{council_name}' -> {cleaned_json}")
                    status = "success"
                else:
                    raise ValueError("LLM 반환 규칙 유효성 검사 실패")
            except Exception as llm_err:
                print(f"  [경고] '{council_name}' LLM 정제 실패 ({llm_err}). 기본 fallback 적용.")
                consolidated_rules[council_name] = dict(FALLBACK_SELECTORS)
                status = "fallback"

        except Exception as e:
            print(f"  [오류] '{council_name}': {e}. 기본 fallback 적용.")
            consolidated_rules[council_name] = dict(FALLBACK_SELECTORS)
            status = "fallback"

        record_file(run_id, "3", file_path, time.perf_counter() - file_started,
                    status, item_count, file_started_at)

    # 최종 통합 규칙 파일 저장
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(consolidated_rules, f, ensure_ascii=False, indent=2)

    print(f"\n[3단계 완료] 총 {len(consolidated_rules)}개 의회 규칙이 저장되었습니다 -> {output_file}")
    finish_step(run_id, "3", step_started_at)
    return consolidated_rules


if __name__ == "__main__":
    generate_strict_parsing_rules()

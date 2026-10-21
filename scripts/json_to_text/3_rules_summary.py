import os
import glob
import json
import re
import time
from openai import OpenAI
import _constant as constant
from pipeline_timing import start_step, record_file, finish_step

def generate_strict_parsing_rules(
    input_dir=constant.rules_output_dir,
    output_file=constant.all_rule_file,
    model_name=constant.make_model_name,
):
    run_id, step_started_at = start_step("3")
    client = constant.client

    # 디버그 JSON은 규칙 형식이 아니므로 검증을 통과한 규칙 파일만 읽는다.
    rule_files = glob.glob(os.path.join(input_dir, "rule_sample_*.json"))
    
    if not rule_files:
        print(f"[{input_dir}] 폴더에서 읽을 JSON 파일을 찾을 수 없습니다.")
        finish_step(run_id, "3", step_started_at)
        return

    print(f"총 {len(rule_files)}개의 JSON 파일에서 엄격한 파싱 규칙 추출을 시작합니다...\n")

    korean_pattern = re.compile(r'[가-힣]')

    # JSON 데이터 분석에 맞게 시스템 프롬프트 수정
    system_prompt = (
        "You are a strict data parsing system. "
        "Analyze the provided JSON rule data and extract HTML CSS selectors (like 'div.class', 'span#id', 'p'). "
        "NEVER output Korean words, slashes(/), or explanations in the JSON values. "
        "If you cannot find the selector, output 'div' for agenda, 'span' for speaker, and 'p' for utterance."
    )

    # 새로 규칙을 만든 일부 기관만으로 통합 파일을 덮어쓰면 기존 기관
    # 규칙이 사라지므로 이전 결과를 바탕으로 갱신한다.
    consolidated_rules = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                previous_rules = json.load(f)
            if isinstance(previous_rules, dict):
                consolidated_rules.update(previous_rules)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[경고] 기존 통합 규칙을 읽지 못했습니다: {e}")

    for file_path in rule_files:
        file_started = time.perf_counter()
        file_started_at = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        status = "error"
        item_count = 0
        filename = os.path.basename(file_path)
        # 샘플 파일명은 sample_YYYY_기관명 형태지만 원본 JSON은 YYYY_기관명이다.
        # 접두어가 남으면 4_to_csv_2.py의 원본 파일 키와 매칭되지 않는다.
        council_name = os.path.splitext(filename)[0]
        for prefix in ("rule_sample_", "rule_"):
            if council_name.startswith(prefix):
                council_name = council_name[len(prefix):]
                break
        if council_name.startswith("sample_"):
            council_name = council_name[len("sample_"):]

        try:
            # JSON 파일로 읽어 문자열로 변환
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_json = json.load(f)
                item_count = 1
                content = json.dumps(raw_json, ensure_ascii=False)

            # 개별 규칙 파일에 이미 선택자가 있으면 LLM 재요청 없이 직접 반영한다.
            # 디버그/부분 규칙처럼 비어 있는 값은 기존 통합 규칙을 유지한다.
            candidate = raw_json
            if isinstance(raw_json, dict) and council_name not in raw_json and len(raw_json) == 1:
                candidate = next(iter(raw_json.values()))
            if isinstance(candidate, dict):
                direct_rule = {key: candidate.get(key, "") for key in ("agenda", "speaker", "utterance")}
                if all(isinstance(value, str) and value.strip() and not korean_pattern.search(value) and '/' not in value for value in direct_rule.values()):
                    consolidated_rules[council_name] = direct_rule
                    print(f"[규칙 반영] '{council_name}' -> {direct_rule}")
                    record_file(run_id, "3", file_path, time.perf_counter() - file_started,
                                "success", item_count, file_started_at)
                    continue

            # 프롬프트에 JSON 데이터를 분석하라는 명시적 지시 추가
            user_prompt = f"""
다음 JSON 데이터를 분석하여 3가지 요소의 CSS 선택자를 JSON으로 추출해.
[제약사항] value 값에는 절대 한글이나 '/'가 들어가면 안 됨. 오직 영문과 CSS 기호(. # >)만 사용.

Good Example:
{{ "agenda": "div.agnd_0", "speaker": "span.spkrbld", "utterance": "p" }}

Bad Example (DO NOT DO THIS):
{{ "agenda": "안건 제목 태그", "speaker": "발언자/클래스", "utterance": "발언 내용" }}

[분석할 JSON 데이터]
{content[:2000]}
"""
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={ "type": "json_object" }
            )

            llm_result = response.choices[0].message.content.strip()
            
            if "```json" in llm_result:
                llm_result = llm_result.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_result:
                llm_result = llm_result.split("```")[1].strip()

            rule_json = json.loads(llm_result)
            
            is_valid = True
            for key in ["agenda", "speaker", "utterance"]:
                val = rule_json.get(key, "")
                if not val or korean_pattern.search(val) or '/' in val:
                    is_valid = False
                    break
            
            if is_valid:
                consolidated_rules[council_name] = rule_json
                print(f"[성공] '{council_name}' -> {rule_json}")
                status = "success"
            else:
                print(f"[경고] '{council_name}' 추출 값에 한글/오류 포함됨. 기본값으로 덮어씁니다.")
                consolidated_rules[council_name] = {
                    "agenda": "div",  
                    "speaker": "span.spkrbld", 
                    "utterance": "p"
                }
                status = "fallback"

        except json.JSONDecodeError:
            print(f"[실패] '{council_name}' - JSON 파싱 실패. 기본값 주입.")
            consolidated_rules[council_name] = {"agenda": "div", "speaker": "span.spkrbld", "utterance": "p"}
            status = "fallback"
        except Exception as e:
            print(f"[오류] '{council_name}': {e}")
        record_file(run_id, "3", file_path, time.perf_counter() - file_started,
                    status, item_count, file_started_at)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(consolidated_rules, f, ensure_ascii=False, indent=4)
        
    print(f"\n최종 규칙 정리가 완료되었습니다. '{output_file}'을 확인하세요.")
    finish_step(run_id, "3", step_started_at)

if __name__ == "__main__":
    generate_strict_parsing_rules()

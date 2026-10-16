"""
지방의회 회의록에서 로컬 LLM(Ollama)을 활용하여 구체적 정책 의제 및 복합 키워드(메타데이터용)를 추출하는 스크립트
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
from tqdm import tqdm


# 시스템 프롬프트: 포괄적 단어 엄격 배제 및 2~4어절 구체적 정책 복합키워드 추출 강제
EXTRACTION_SYSTEM_PROMPT = """당신은 대한민국 지방의회 정책 분석 및 메타데이터 구축 전문가입니다.
주어진 지방의회 회의록 발언을 정밀하게 분석하여, 향후 검색 및 필터링 메타데이터로 활용할 수 있도록 JSON 포맷으로만 응답하세요.

[⚠️ 절대 금지 키워드 (Negative Constraints)]
아래와 같이 모든 회의록에 흔히 등장하는 1단어짜리 포괄적 명사, 기관명, 직함은 'keywords'에 절대로 포함하지 마세요:
- 금지 단어: 사업, 예산, 지원, 조례, 조례안, 강원특별자치도, 의견, 업무, 행정, 강원도, 의회, 의원, 위원, 발언, 질의, 답변, 감사, 생각, 회의, 내용, 부분

[✅ 권장 키워드 형식]
2~4개 단어가 결합된 '구체적이고 실질적인 정책 의제 복합명사'로 추출하세요:
- 좋은 예시: "농지제도 혁신", "지역 소멸 대응", "미래 인구정책", "일자리 산업 육성", "학령인구 감소", "교육복지 강화", "반도체 클러스터", "공공의료 확충", "해안침식 방지", "정주여건 개선"

[출력 포맷 (반드시 아래 순수 JSON 형식으로만 응답할 것, 마크다운 코드블록 없이 순수 JSON만 출력)]
{
  "is_substantive": true,
  "policy_category": "경제/산업",
  "core_issue": "농지 규제 완화를 통한 농가 소득 증대 및 농지제도 혁신 방안",
  "keywords": ["농지제도 혁신", "농업 규제 완화", "경자유전 원칙 개편"],
  "summary": "해방 이후 지속된 농지 소유 규제를 완화하여 청년농 유입과 농업 경쟁력을 강화해야 한다고 주장함."
}

※ policy_category 후보: "경제/산업", "농림/수산", "보건/복지", "교육/문화", "환경/안전", "도시/교통", "자치/행정", "기타" 중 정확히 1개 선택.
※ 만약 단순 의전 멘트(개의 선포, 방청객 환영, 단순 이석 보고 등)로 실질적 정책 논의가 없다면 is_substantive를 false로 하고 keywords는 []로 비워두세요.
"""


def clean_and_parse_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """LLM 출력 문자열에서 유연하고 안전하게 JSON을 파싱"""
    if not raw_text:
        return None

    text = raw_text.strip()
    # 마크다운 백틱 제거 (```json ... ```)
    if "```" in text:
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = text.replace('```', '').strip()

    # JSON 본체 { ... } 추출
    match = re.search(r'(\{[\s\S]*\})', text)
    if match:
        json_candidate = match.group(1).strip()
    else:
        json_candidate = text

    # 1차 파싱 시도
    try:
        return json.loads(json_candidate)
    except Exception:
        pass

    # 2차: 제어문자 및 trailing comma 정규식 보정
    try:
        fixed = re.sub(r',\s*\}', '}', json_candidate)
        fixed = re.sub(r',\s*\]', ']', fixed)
        return json.loads(fixed)
    except Exception:
        return None


def get_available_ollama_models(base_url: str) -> List[str]:
    """Ollama 서버의 가용한 모델 목록 반환"""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        if resp.status_code == 200:
            return [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []


def check_ollama_model(base_url: str, model_name: str) -> bool:
    """Ollama 서버에서 해당 모델이 사용 가능한지 확인"""
    models = get_available_ollama_models(base_url)
    return any(model_name.lower() in m.lower() for m in models)


def extract_metadata_with_llm(
    text: str,
    speaker_info: str,
    model_name: str,
    base_url: str = "http://localhost:11434",
    timeout: int = 60,
) -> Optional[Dict[str, Any]]:
    """단일 발언 텍스트에서 로컬 LLM을 호출하여 JSON 메타데이터 추출"""
    
    user_prompt = f"""[발언자 정보]: {speaker_info}
[발언 내용]:
{text}

위 발언을 분석하여 시스템 지침에 따라 JSON으로만 응답하세요."""

    payload = {
        "model": model_name,
        "prompt": f"{EXTRACTION_SYSTEM_PROMPT}\n\n{user_prompt}",
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
        }
    }

    try:
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        raw_output = resp.json().get("response", "").strip()
        return clean_and_parse_json(raw_output)
    except Exception:
        return None


def process_council_file(
    input_file: str,
    output_file: str,
    model_name: str = "gemma2:2b",
    base_url: str = "http://localhost:11434",
    sample_limit: int = 10,
    min_char_len: int = 80,
):
    print("=" * 65)
    print(" [로컬 LLM 기반 의회 회의록 정책 키워드 & 메타데이터 추출기] ")
    print("=" * 65)
    print(f" * 입력 데이터 : {input_file}")
    print(f" * 출력 파일   : {output_file}")
    print(f" * 요청 모델   : {model_name}")
    print(f" * Ollama URL  : {base_url}")
    print(f" * 처리 대상   : {'전체 처리' if sample_limit <= 0 else f'최대 {sample_limit}건 샘플'}")
    print(f" * 최소 글자수 : {min_char_len}자 미만 제외")
    print("=" * 65)

    # 1. 모델 가용성 확인 및 자동 폴백 안내
    available_models = get_available_ollama_models(base_url)
    if not available_models:
        print(f"[오류] Ollama 서버({base_url})에 연결할 수 없거나 실행 중이지 않습니다.")
        print("▶ 해결: 'ollama serve' 또는 Ollama 데스크톱 앱을 실행해 주세요.")
        return

    selected_model = model_name
    if not check_ollama_model(base_url, model_name):
        print(f"[알림] 요청하신 모델 '{model_name}'이 로컬 Ollama에 설치되어 있지 않습니다.")
        print(f"       현재 사용 가능한 로컬 모델: {available_models}")

        # 로컬 모델 중 추천 폴백
        fallback = None
        for m in available_models:
            if "gemma" in m.lower():
                fallback = m
                break
        if not fallback:
            # 7b 계열 우선 (품질 우수)
            for m in available_models:
                if "7b" in m.lower():
                    fallback = m
                    break
        if not fallback:
            fallback = available_models[0]

        print(f"▶ 대체 모델로 '{fallback}'을(를) 자동 선택하여 진행합니다.")
        print(f"  (원하시는 젬마 모델을 쓰시려면 터미널에서 'ollama pull {model_name}'을 실행하세요)\n")
        selected_model = fallback

    # 2. 파일 읽기 및 유효 발언 필터링
    records = []
    with open(input_file, "r", encoding="utf-8-sig") as f:
        header = f.readline()
        for row_id, line in enumerate(f, start=1):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 7:
                utterance = parts[6].strip()
                if len(utterance) >= min_char_len:
                    records.append({
                        "row_id": row_id,
                        "date": parts[0],
                        "assembly": parts[1],
                        "agenda_title": parts[2],
                        "speaker_role": parts[3],
                        "speaker_name": parts[4],
                        "speaker_type": parts[5],
                        "utterance": utterance,
                    })
                    if 0 < sample_limit <= len(records):
                        break

    print(f">> 총 {len(records)}건의 유효 발언을 추출 대상으로 선정했습니다.\n")

    # 3. LLM 메타데이터 추출 실행
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    results = []
    success_count = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        with tqdm(total=len(records), desc=f"[{selected_model}] 메타데이터 추출", unit="건") as pbar:
            for r in records:
                speaker_info = f"{r['assembly']} {r['speaker_role']} {r['speaker_name']} ({r['speaker_type']}) - 안건: {r['agenda_title']}"
                meta = extract_metadata_with_llm(
                    text=r["utterance"][:1200],  # 발언 핵심 전반부 1,200자 전달
                    speaker_info=speaker_info,
                    model_name=selected_model,
                    base_url=base_url,
                )

                if meta and isinstance(meta, dict):
                    # 카테고리 정제
                    cat = meta.get("policy_category", "기타")
                    if "|" in cat:
                        cat = cat.split("|")[0].strip()

                    combined = {
                        "row_id": r["row_id"],
                        "date": r["date"],
                        "assembly": r["assembly"],
                        "speaker": f"{r['speaker_role']} {r['speaker_name']}",
                        "agenda": r["agenda_title"],
                        "is_substantive": meta.get("is_substantive", True),
                        "policy_category": cat,
                        "core_issue": meta.get("core_issue", ""),
                        "keywords": meta.get("keywords", []),
                        "summary": meta.get("summary", ""),
                        "utterance_preview": r["utterance"][:100] + "...",
                    }
                    success_count += 1
                else:
                    combined = {
                        "row_id": r["row_id"],
                        "date": r["date"],
                        "assembly": r["assembly"],
                        "speaker": f"{r['speaker_role']} {r['speaker_name']}",
                        "agenda": r["agenda_title"],
                        "is_substantive": False,
                        "policy_category": "미상",
                        "core_issue": "추출 실패",
                        "keywords": [],
                        "summary": "",
                        "utterance_preview": r["utterance"][:100] + "...",
                    }

                out_f.write(json.dumps(combined, ensure_ascii=False) + "\n")
                out_f.flush()  # 실시간 디스크 저장
                results.append(combined)
                pbar.update(1)

    print("\n" + "=" * 65)
    print(" [추출 완료 결과 보고서] ")
    print("=" * 65)
    print(f" * 처리 성공 건수 : {success_count} / {len(records)} 건")
    print(f" * 결과 저장 파일 : {output_file}")
    print("=" * 65)

    # 4. 상위 추출 결과 샘플 출력
    substantive_items = [item for item in results if item.get("is_substantive") and item.get("keywords")]
    print(f"\n>> [추출된 주요 정책 의제 및 복합 키워드 샘플 (상위 {min(5, len(substantive_items))}건)]:\n")
    for idx, item in enumerate(substantive_items[:5], 1):
        print(f"[{idx}] {item['speaker']} ({item['date']}) | 정책분야: [{item['policy_category']}]")
        print(f"    - 핵심 쟁점 : {item['core_issue']}")
        print(f"    - 복합 키워드 : {', '.join(item['keywords'])}")
        print(f"    - 발언 요약 : {item['summary']}")
        print("-" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama 로컬 LLM 회의록 메타데이터 추출기")
    parser.add_argument(
        "--input",
        type=str,
        default="text/cleaned_2024_강원특별자치도의회.txt",
        help="입력 TSV 회의록 파일",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gangwon_policy_metadata.jsonl",
        help="출력 JSONL 파일 경로",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="gemma2:2b",
        help="사용할 Ollama 모델명 (기본: gemma2:2b, 설치된 qwen2.5-coder:7b 등 가능)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="테스트할 발언 개수 (0이면 전체 처리, 기본: 5)",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=80,
        help="최소 발언 길이 필터링 (기본: 80자)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API 서버 주소 (기본: http://localhost:11434)",
    )

    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    process_council_file(
        input_file=args.input,
        output_file=args.output,
        model_name=args.model,
        base_url=args.url,
        sample_limit=args.sample,
        min_char_len=args.min_len,
    )

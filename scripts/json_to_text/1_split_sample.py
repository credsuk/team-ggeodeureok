"""1단계: 지방의회 회의록 원본 JSON에서 LLM 규칙 학습용 본문 HTML 샘플 추출 (1_split_sample.py).

[주요 기능]
1. 원본 회의록 JSON의 `MINTS_HTML` 필드에서 실제 발언/안건 본문 영역을 탐색합니다.
2. 상단 메뉴, 네비게이션, 푸터, 스크립트 등 불필요한 노이즈 태그를 정밀 제거합니다.
3. 발언 마커(○, ◯, 의원/위원장 직함 등) 및 CSS 힌트(speaker, transcript 등)를 기반으로 본문 컨테이너를 스코어링하여 선택합니다.
4. LLM 입력 토큰 한도를 초과하지 않도록 문맥 구조는 보존하면서 긴 발언 본문을 안전하게 축약(shorten)합니다.
5. 각 의회별 대표 본문 HTML 샘플을 `sample_{파일명}.txt` 형태로 저장합니다.
"""
from __future__ import annotations

import glob
import html
import json
import os
import random
import re
import time
import traceback
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, Comment, Tag
import _constant as constant
from pipeline_timing import start_step, record_file, finish_step


# ---------------------------------------------------------------------------
# 정규식 힌트 패턴 정의 (Structural & Semantic Hints)
# ---------------------------------------------------------------------------
# 본문 컨테이너 식별에 도움을 주는 CSS 클래스/ID 패턴
TRANSCRIPT_HINT = re.compile(
    r"(?:sMan|speaker|speech|utterance|statement|minutes|record[-_]?content|record[-_]?body|transcript|view[-_]?content|con[-_]?view|an\d+)",
    re.IGNORECASE,
)

# 발언자 시작 기호 및 주요 직함 패턴
SPEECH_HINT = re.compile(
    r"(?:○\s*[^\n]{2,30}|◯\s*[^\n]{2,30}|(?:위원장|의장|의원|국장|과장|교육감|부위원장)\s*[가-힣]{2,5}\s*(?:의원|위원|:|\s))"
)

# 제거 대상 노이즈 클래스/ID 패턴
NOISE = re.compile(
    r"(?:sidebar|navbar|navigation|breadcrumb|footer|header|toolbar|menu|pagination|related|sns)",
    re.IGNORECASE,
)


def prepare_html(raw_html: str) -> BeautifulSoup:
    """HTML 엔티티 변환 및 노이즈 요소(스크립트, 스타일, 네비게이션 등)를 사전 제거합니다."""
    # HTML 이스케이프 해제
    if "&lt;" in raw_html or "&gt;" in raw_html:
        raw_html = html.unescape(raw_html)
        
    soup = BeautifulSoup(raw_html, "html.parser")
    
    # HTML 주석 노드 제거
    for node in soup.find_all(string=lambda s: isinstance(s, Comment)):
        node.extract()
        
    # 명백한 불필요 태그 일괄 제거
    for node in soup(["script", "style", "noscript", "meta", "link", "header", "nav", "footer"]):
        node.decompose()

    # aside 및 form 태그 중 본문과 무관한 노이즈 태그 선별 제거
    for tag in list(soup.find_all(["aside", "form"])):
        if tag.name is None or not isinstance(tag.attrs, dict):
            continue
        attrs = " ".join(tag.get("class") or []) + " " + (tag.get("id") or "")
        if NOISE.search(attrs) or (tag.name == "aside" and not TRANSCRIPT_HINT.search(attrs)):
            tag.decompose()

    # 클래스/ID에 노이즈 키워드가 포함된 일반 태그 제거
    for tag in list(soup.find_all(True)):
        if tag.name is None or not isinstance(tag.attrs, dict):
            continue
        classes = tag.get("class") or []
        element_id = tag.get("id") or ""
        attrs = " ".join(classes) + " " + element_id
        if NOISE.search(attrs) and not TRANSCRIPT_HINT.search(attrs):
            tag.decompose()

    return soup


def find_transcript_region(soup: BeautifulSoup) -> Optional[Tag]:
    """DOM 구조와 텍스트 의미적 증거를 평가하여 실제 회의록 발언이 포함된 최적의 본문 컨테이너를 탐색합니다."""
    candidates = []
    
    for tag in soup.find_all(["main", "article", "section", "div", "table"]):
        txt = tag.get_text(" ", strip=True)
        # 내용이 지나치게 짧은 컨테이너는 제외
        if len(txt) < 120:
            continue
            
        attrs = " ".join(tag.get("class") or []) + " " + (tag.get("id") or "")
        marker_count = len(SPEECH_HINT.findall(txt))
        structure = bool(TRANSCRIPT_HINT.search(attrs))
        descendant_hint = len(tag.find_all(attrs={"class": TRANSCRIPT_HINT})) + len(
            tag.find_all(attrs={"id": TRANSCRIPT_HINT})
        )
        
        # 발언자 기호나 구조적 힌트가 부족한 대형 div는 제외
        if not (marker_count >= 2 or descendant_hint >= 2 or (structure and marker_count >= 1)):
            continue

        # 발언자 마커 빈도, 클래스 힌트, 텍스트 밀도를 종합한 스코어 산출
        score = min(marker_count, 30) * 3 + min(descendant_hint, 30) * 2 + (8 if structure else 0)
        score -= min(len(txt) // 12000, 10)
        candidates.append((score, -len(str(tag)), tag))

    if not candidates:
        return None
        
    # 가장 높은 스코어를 가진 가장 집약적인 컨테이너 반환
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def shorten_long_text(region: Tag, limit: int = 300) -> str:
    """HTML DOM 태그 구조는 완전히 보존하면서 긴 텍스트 노드의 내용만 지정 길이로 축약합니다."""
    copy_soup = BeautifulSoup(str(region), "html.parser")
    for node in list(copy_soup.find_all(string=True)):
        value = str(node)
        if len(value) > limit:
            node.replace_with(value[:limit] + "…[발언 내용 일부 생략]")
    return str(copy_soup)


def extract_samples_from_folder(
    input_dir: str = constant.org_dir,
    output_dir: str = constant.sample_dir,
    num_records: int = constant.records_num,
) -> None:
    """지정된 폴더의 모든 회의록 JSON에서 LLM 분석용 HTML 본문 샘플을 추출하여 저장합니다."""
    run_id, step_started_at = start_step("1")
    os.makedirs(output_dir, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if not files:
        # 서브디렉토리 검색 시도
        files = sorted(glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True))

    if not files:
        print(f"[오류] JSON 파일 없음: {input_dir}")
        finish_step(run_id, "1", step_started_at)
        return

    print(f"[1단계 시작] 전체 {len(files)}개 파일 | 입력: {input_dir} | 출력: {output_dir}")

    for path in files:
        file_started = time.perf_counter()
        file_started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        base = os.path.splitext(os.path.basename(path))[0]
        status = "error"
        item_count = 0
        
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                
            records = data if isinstance(data, list) else [data]
            item_count = len(records)
            
            # 앞뒤 의전/행정 기록을 피하고 중간 실질 심사 회의록 위주로 후보 선정
            middle = records[5:-5] if len(records) > 10 else records
            indices = list(range(len(middle)))
            random.shuffle(indices)
            
            samples = []
            for idx in indices:
                record = middle[idx]
                if not isinstance(record, dict) or not isinstance(record.get("MINTS_HTML"), str):
                    continue
                try:
                    soup = prepare_html(record["MINTS_HTML"])
                    region = find_transcript_region(soup)
                    if region is None:
                        continue
                    reduced = shorten_long_text(region)
                except Exception as record_exc:
                    print(f"  [레코드 오류] {base} (idx={idx}): {record_exc}")
                    continue

                samples.append(reduced)
                if len(samples) >= num_records:
                    break

            if samples:
                out = os.path.join(output_dir, f"sample_{base}.txt")
                with open(out, "w", encoding="utf-8") as f:
                    for i, sample in enumerate(samples, start=1):
                        f.write(f"--- [Random Sample Record {i} from {os.path.basename(path)}] ---\n")
                        f.write(sample + "\n\n")
                print(f"  [성공] {base}: {len(samples)}건 본문 후보 저장 -> {out}")
                status = "success"
            else:
                print(f"  [확인 필요] {base}: 발언 본문 후보를 찾지 못함 (MINTS_HTML 구조 점검 필요)")
                status = "no_sample"
                
        except Exception as exc:
            print(f"  [오류] {base}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            record_file(
                run_id, "1", path, time.perf_counter() - file_started,
                status, item_count, file_started_at,
            )

    finish_step(run_id, "1", step_started_at)


if __name__ == "__main__":
    extract_samples_from_folder(
        input_dir=constant.org_dir,
        output_dir=constant.sample_dir,
        num_records=constant.records_num,
    )

"""4단계: 통합 파싱 규칙 기반 전체 회의록 텍스트(TSV) 추출 및 정제 (4_to_txt.py).

[주요 기능]
1. 3단계에서 생성된 통합 파싱 규칙(consolidated_parsing_rules_*.json)을 로드합니다.
2. 각 지방의회 원본 JSON 회의록을 순회하며 HTML을 파싱합니다.
3. CSS Selector 기반 및 특수 순차 텍스트 노드(__TEXT_NODE_*) 구조를 모두 지원하여
   안건명(agenda), 발언자(speaker: 직함/성명/소속분류), 발언 본문(utterance)을 정밀 추출합니다.
4. 의전 멘트('(개의)', '(박수)' 등)를 제거하고, 공백이 분절된 텍스트('의 원 여 러 분')를 자간 복원합니다.
5. 동일 발언자의 연속된 발언은 하나의 문단으로 자연스럽게 병합합니다.
6. UTF-8 TSV (탭 구분 텍스트) 형식으로 최종 저장합니다.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import time
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Tuple

from bs4 import BeautifulSoup, Tag
import pandas as pd
import _constant as constant
from pipeline_timing import start_step, record_file, finish_step


class DynamicCouncilPreprocessor:
    """지방의회 HTML 회의록에서 안건, 발언자, 발언문을 추출하고 정제하는 전처리기 클래스."""

    def __init__(self, min_char_len: int = 30):
        # 유효한 실질 발언으로 간주할 최소 문자 수
        self.min_char_len = min_char_len
        
        # 회의록에 흔히 등장하는 의전 및 장내 상황 괄호 멘트 정규식
        self.formality_patterns = [
            r"\([^\)]*(?:개식|폐식|개의|산회|인사|기립|묵념|박수|마이크)[^\)]*\)",
        ]
        self.combined_regex = re.compile("|".join(self.formality_patterns), re.MULTILINE)

    def fix_fragmented_spacing(self, text: str) -> str:
        """한 글자씩 공백으로 쪼개진 텍스트('의 원 여 러 분')를 단어로 자연스럽게 병합합니다."""
        # 구두점 앞 공백 제거
        text = re.sub(r'\s+([,.\?!])', r'\1', text)
        tokens = text.split()
        fixed_tokens = []
        temp_word = ""

        for token in tokens:
            pure_text = re.sub(r'[^가-힣A-Za-z0-9]', '', token)
            # 1글자 단편인 경우 임시 버퍼에 누적
            if len(pure_text) <= 1:
                temp_word += token
            else:
                if temp_word:
                    fixed_tokens.append(temp_word)
                    temp_word = ""
                fixed_tokens.append(token)

        if temp_word:
            fixed_tokens.append(temp_word)

        return " ".join(fixed_tokens)

    def clean_text(self, text: str) -> str:
        """의전 괄호 제거, 자간 분절 복원, 중복 공백 정규화를 수행합니다."""
        cleaned = self.combined_regex.sub(" ", text)
        cleaned = self.fix_fragmented_spacing(cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    def parse_speaker(self, raw_speaker: str) -> Tuple[str, str, str]:
        """발언자 원문 문자열(예: '○ 위원장 이상정')에서 직함(role), 성명(name), 소속구분(speaker_type)을 분리합니다."""
        # 발언 기호(○, □, ■, - 등) 제거
        text = re.sub(r"^[○□■\-\s]+", "", raw_speaker).strip()
        tokens = text.split()

        if len(tokens) >= 2:
            role, name = tokens[0], "".join(tokens[1:])
        elif len(tokens) == 1:
            role, name = "미상", tokens[0]
        else:
            return "미상", "미상", "기타"

        # 의회 소속 여부 판단 (의원, 위원장 등은 '의회', 도지사/교육감 등은 '집행부')
        is_council = any(r in role for r in ["의원", "의장", "부의장", "위원장", "부위원장", "위원"])
        speaker_type = "의회" if is_council else "집행부"
        return role, name, speaker_type

    def extract_following_text(self, spk_tag: Tag, speaker_ids: set) -> str:
        """CSS 블록이 분리된 경우, 현재 발언자 태그 뒤에 이어지는 인접 형제 텍스트 노드들을 다음 발언자 전까지 수집합니다."""
        parts = []
        for sibling in spk_tag.next_siblings:
            if isinstance(sibling, Tag):
                # 다음 발언자 요소를 만나면 수집 중단
                if id(sibling) in speaker_ids or any(id(child) in speaker_ids for child in sibling.find_all(True)):
                    break
                parts.append(sibling.get_text(" ", strip=True))
            else:
                parts.append(str(sibling))
        return self.clean_text(" ".join(part for part in parts if part))

    def process_record(
        self,
        record: dict,
        rule: dict,
        filename_info: str,
        source_record_no: int,
    ) -> List[Dict[str, Any]]:
        """단일 회의록 레코드(JSON)의 MINTS_HTML에서 발언 목록을 추출합니다."""
        results = []
        html_content = record.get("MINTS_HTML", "")
        if not html_content:
            return results

        soup = BeautifulSoup(html_content, "html.parser")

        # 스크립트, 사이드바 등 파싱 방해 요소 제거
        for tag in soup(["script", "style", "form", "table", "ul.skipUl", "#topFunction", "#mySidenav2"]):
            tag.decompose()

        agenda_selector = rule.get("agenda", "div")
        speaker_selector = rule.get("speaker", "span.spkrbld")
        utterance_selector = rule.get("utterance", "p")

        text_node_mode = (
            agenda_selector == "__TEXT_NODE_AGENDA__"
            and speaker_selector == "__TEXT_NODE_SPEAKER__"
            and utterance_selector == "__TEXT_NODE_UTTERANCE__"
        )

        if text_node_mode:
            # 텍스트 노드 기반 발언 구조 (태그 없이 <strong> 발언자 + 일반 텍스트로 이어지는 경우)
            speaker_tags = [
                tag for tag in soup.find_all("strong")
                if re.match(r"^\s*[○□■↘]\s*\S", tag.get_text(" ", strip=True))
            ]
            agenda_tags = [
                tag for tag in soup.find_all("strong")
                if re.match(r"^\s*\d+[.)．]\s*\S", tag.get_text(" ", strip=True))
            ]
            utterance_tags = []
            utterance_selector = ""
        else:
            try:
                speaker_tags = soup.select(speaker_selector)
                agenda_tags = soup.select(agenda_selector)
                utterance_tags = soup.select(utterance_selector) if utterance_selector else []
            except Exception:
                return results

        # 발언자 선택자 매칭 실패 시 기본값 fallback 재시도
        if len(speaker_tags) == 0:
            speaker_selector = "span.spkrbld"
            try:
                speaker_tags = soup.select(speaker_selector)
            except Exception:
                pass

            if len(speaker_tags) == 0:
                speaker_tags = [
                    tag for tag in soup.find_all("strong")
                    if tag.get_text(" ", strip=True).startswith(("○", "□", "■", "↘"))
                ]
                if not speaker_tags:
                    return results
                utterance_selector = ""
                agenda_tags = [
                    tag for tag in soup.find_all("strong")
                    if re.match(r"^\s*\d+[.)．]\s*\S", tag.get_text(" ", strip=True))
                ]

        # 단일 순회(DOM traversal)를 통해 각 발언자가 속한 직전 안건명을 매핑
        agenda_ids = {id(tag) for tag in agenda_tags}
        speaker_ids = {id(tag) for tag in speaker_tags}
        utterance_ids = {id(tag) for tag in utterance_tags}
        agenda_by_speaker = {}
        document_order = {}
        speaker_container_counts = {}
        active_agenda = "일반안건"
        order_index = 0

        for node in soup.descendants:
            if not getattr(node, "name", None):
                continue
            node_id = id(node)
            document_order[node_id] = order_index
            order_index += 1
            if node_id in agenda_ids:
                active_agenda = node.get_text(" ", strip=True)[:50] or "일반안건"
            if node_id in speaker_ids:
                agenda_by_speaker[node_id] = active_agenda
                for parent in node.parents:
                    parent_id = id(parent)
                    speaker_container_counts[parent_id] = speaker_container_counts.get(parent_id, 0) + 1
                    if parent_id in utterance_ids:
                        break

        utterance_positions = sorted(
            document_order[id(tag)] for tag in utterance_tags if id(tag) in document_order
        )
        utterance_by_position = {
            document_order[id(tag)]: tag for tag in utterance_tags if id(tag) in document_order
        }

        # 각 발언자별 발언 본문 추출
        for spk_tag in speaker_tags:
            raw_speaker = spk_tag.get_text(strip=True)
            role, name, speaker_type = self.parse_speaker(raw_speaker)
            agenda_title = agenda_by_speaker.get(id(spk_tag), "일반안건")
            utterance_cleaned = ""

            if utterance_selector and utterance_selector != "p":
                next_content = next(
                    (parent for parent in spk_tag.parents
                     if id(parent) in utterance_ids and speaker_container_counts.get(id(parent), 0) <= 1),
                    None,
                )
                if next_content is None:
                    speaker_position = document_order.get(id(spk_tag), -1)
                    next_position_index = bisect_right(utterance_positions, speaker_position)
                    if next_position_index < len(utterance_positions):
                        next_content = utterance_by_position[utterance_positions[next_position_index]]
                if next_content:
                    utterance_cleaned = self.clean_text(next_content.get_text(strip=True))

            # 발언 내용이 비어있거나 불완전하면 인접 텍스트 노드 추출 방식으로 보완
            if not utterance_cleaned or len(utterance_cleaned) < 10:
                utterance_cleaned = self.extract_following_text(spk_tag, speaker_ids)

            if len(utterance_cleaned) >= self.min_char_len:
                results.append({
                    "_source_record_no": source_record_no,
                    "MTG_DE": record.get("MTG_DE", ""),
                    "RASMBLY_NM": record.get("RASMBLY_NM", ""),
                    "MTGNM": record.get("MTGNM", ""),
                    "MINTS_NM": record.get("MINTS_NM", ""),
                    "agenda_title": agenda_title,
                    "speaker_role": role,
                    "speaker_name": name,
                    "speaker_type": speaker_type,
                    "utterance": utterance_cleaned,
                })

        return results


def run_extraction(
    input_dir: str = constant.org_dir,
    output_dir: str = constant.text_dir,
    rules_file: str = constant.all_rule_file,
    selected_councils: Optional[List[str]] = None,
) -> None:
    """통합 파싱 규칙을 적용하여 모든 회의록 JSON 파일을 정제된 TSV 파일로 변환합니다."""
    run_id, step_started_at = start_step("4")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(rules_file):
        print(f"[경고] 통합 규칙 파일 '{rules_file}'이 없습니다. 기본 규칙으로 진행합니다.")
        parsing_rules = {}
    else:
        with open(rules_file, "r", encoding="utf-8") as f:
            parsing_rules = json.load(f)

    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if not json_files:
        json_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True))

    if selected_councils:
        filter_set = set(selected_councils)
        json_files = [
            p for p in json_files
            if os.path.basename(p) in filter_set or os.path.splitext(os.path.basename(p))[0] in filter_set
        ]

    preprocessor = DynamicCouncilPreprocessor(min_char_len=30)
    print(f"[4단계 시작] 전체 {len(json_files)}개 파일 처리 | 입력: {input_dir} | 출력: {output_dir}")

    for file_index, input_file in enumerate(json_files, start=1):
        file_started = time.perf_counter()
        file_started_at = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        file_status = "error"
        item_count = 0
        base_name = os.path.basename(input_file)
        file_key = os.path.splitext(base_name)[0]

        matched_rule = parsing_rules.get(file_key)
        if matched_rule is None:
            # 접두어나 공백 정리 후 매칭 시도
            matched_rule = next(
                (v for k, v in parsing_rules.items() if k in file_key or file_key in k),
                {"agenda": "div", "speaker": "span.spkrbld", "utterance": "p"},
            )

        print(f"[{file_index}/{len(json_files)}] [{base_name}] 전처리 시작...", flush=True)

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            if not isinstance(raw_data, list):
                raw_data = [raw_data]

            item_count = len(raw_data)
            all_utterances = []

            for record_index, record in enumerate(raw_data, start=1):
                extracted = preprocessor.process_record(
                    record, matched_rule, base_name, record_index
                )
                all_utterances.extend(extracted)

            df_clean = pd.DataFrame(all_utterances)

            if not df_clean.empty:
                # 동일 화자의 연속 발언 병합 처리
                condition = (
                    (df_clean['_source_record_no'] != df_clean['_source_record_no'].shift()) |
                    (df_clean['speaker_name'] != df_clean['speaker_name'].shift()) |
                    (df_clean['speaker_type'] != df_clean['speaker_type'].shift()) |
                    (df_clean['agenda_title'] != df_clean['agenda_title'].shift())
                )
                df_clean['group_id'] = condition.cumsum()

                df_merged = df_clean.groupby('group_id').agg({
                    'MTG_DE': 'first',
                    'RASMBLY_NM': 'first',
                    'MTGNM': 'first',
                    'MINTS_NM': 'first',
                    'agenda_title': 'first',
                    'speaker_role': 'first',
                    'speaker_name': 'first',
                    'speaker_type': 'first',
                    'utterance': lambda x: ' '.join(x)
                }).reset_index(drop=True)

                output_path = os.path.join(output_dir, f"cleaned_{file_key}.txt")
                df_merged.to_csv(output_path, sep="\t", index=False, encoding="utf-8-sig")
                print(f"  -> 완료: {len(df_merged)}건 추출 (연속 발언 병합됨) -> {output_path}")
                file_status = "success"
            else:
                print(f"  -> 추출된 유효 데이터 없음: {base_name}")
                file_status = "empty"

        except Exception as e:
            print(f"  -> 파일 처리 오류: {e}")

        record_file(run_id, "4", input_file, time.perf_counter() - file_started,
                    file_status, item_count, file_started_at)

    finish_step(run_id, "4", step_started_at)


if __name__ == "__main__":
    councils = sys.argv[1:] if len(sys.argv) > 1 else None
    run_extraction(selected_councils=councils)

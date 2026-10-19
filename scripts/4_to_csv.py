import os
import glob
import json
import re
import time
import sys
from bisect import bisect_right
from bs4 import BeautifulSoup, Tag
import pandas as pd
import _constant as constant
from pipeline_timing import start_step, record_file, finish_step

class DynamicCouncilPreprocessor:
    def __init__(self, min_char_len: int = 30):
        self.min_char_len = min_char_len
        self.formality_patterns = [
            r"\([^\)]*(?:개식|폐식|개의|산회|인사|기립|묵념|박수|마이크)[^\)]*\)",
        ]
        self.combined_regex = re.compile("|".join(self.formality_patterns), re.MULTILINE)

    def fix_fragmented_spacing(self, text: str) -> str:
        """1글자씩 쪼개진 텍스트('의 원 여 러 분')를 데이터 손실 없이 병합합니다."""
        text = re.sub(r'\s+([,.\?!])', r'\1', text) # 구두점 앞 공백 제거
        tokens = text.split()
        fixed_tokens = []
        temp_word = ""
        
        for token in tokens:
            pure_text = re.sub(r'[^가-힣A-Za-z0-9]', '', token)
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
        cleaned = self.combined_regex.sub(" ", text)
        cleaned = self.fix_fragmented_spacing(cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    def parse_speaker(self, raw_speaker: str) -> tuple:
        text = re.sub(r"^[○□■\-\s]+", "", raw_speaker).strip()
        tokens = text.split()
        
        if len(tokens) >= 2:
            role, name = tokens[0], "".join(tokens[1:])
        elif len(tokens) == 1:
            role, name = "미상", tokens[0]
        else:
            return "미상", "미상", "기타"

        speaker_type = "의회" if any(r in role for r in ["의원", "의장", "부의장", "위원장", "부위원장", "위원"]) else "집행부"
        return role, name, speaker_type

    def extract_following_text(self, spk_tag, speaker_ids: set) -> str:
        """발언자 뒤의 인접 텍스트만 추출하고 다음 발언자에서 멈춘다."""
        parts = []
        for sibling in spk_tag.next_siblings:
            if isinstance(sibling, Tag):
                if id(sibling) in speaker_ids or any(id(child) in speaker_ids for child in sibling.find_all(True)):
                    break
                parts.append(sibling.get_text(" ", strip=True))
            else:
                parts.append(str(sibling))
        return self.clean_text(" ".join(part for part in parts if part))

    def process_record(self, record: dict, rule: dict, filename_info: str) -> list:
        results = []
        html_content = record.get("MINTS_HTML", "")
        if not html_content:
            return results

        soup = BeautifulSoup(html_content, "html.parser")

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

        if len(speaker_tags) == 0:
            print(f"  [경고] '{filename_info}'에서 발언자({speaker_selector})를 찾지 못해 기본값(span.spkrbld)으로 재시도합니다.")
            speaker_selector = "span.spkrbld"
            try:
                speaker_tags = soup.select(speaker_selector)
            except Exception:
                pass
            
            if len(speaker_tags) == 0:
                # 일부 회의록은 발언자와 발언 내용이 strong + 일반 텍스트 노드로만
                # 구성되어 CSS 발언자 규칙이 없다. 발언자 기호로만 최소 복구한다.
                speaker_tags = [
                    tag for tag in soup.find_all("strong")
                    if tag.get_text(" ", strip=True).startswith(("○", "□", "■", "↘"))
                ]
                if not speaker_tags:
                    print(f"  [실패] 기본 규칙과 텍스트 발언자 탐색 모두 실패: '{filename_info}'")
                    return results
                utterance_selector = ""
                agenda_tags = [
                    tag for tag in soup.find_all("strong")
                    if re.match(r"^\s*\d+[.)．]\s*\S", tag.get_text(" ", strip=True))
                ]
                print(f"  [보완] 텍스트 노드 발언 구조 {len(speaker_tags)}건을 순차 추출합니다.")

        # 발언자마다 문서 앞부분 전체를 역탐색하지 않도록 DOM 순회 1회로
        # 현재 안건을 각 발언자에 연결한다. 대형 회의록에서 반복 탐색 비용이 컸다.
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
        utterance_positions = sorted(document_order[id(tag)] for tag in utterance_tags if id(tag) in document_order)
        utterance_by_position = {
            document_order[id(tag)]: tag for tag in utterance_tags if id(tag) in document_order
        }

        for spk_tag in speaker_tags:
            raw_speaker = spk_tag.get_text(strip=True)
            # 오류가 발생했던 부분 수정 완료
            role, name, speaker_type = self.parse_speaker(raw_speaker)

            agenda_title = agenda_by_speaker.get(id(spk_tag), "일반안건")

            utterance_cleaned = ""
            
            if utterance_selector and utterance_selector != "p":
                # 발언자 본인이 발언 블록 안에 있으면 그 블록을, 아니면
                # 문서 순서상 바로 다음 발언 후보를 인덱스로 찾는다.
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

            if not utterance_cleaned or len(utterance_cleaned) < 10:
                utterance_cleaned = self.extract_following_text(spk_tag, speaker_ids)

            if len(utterance_cleaned) >= self.min_char_len:
                results.append({
                    "date": record.get("MTG_DE", ""),
                    "assembly": record.get("RASMBLY_NM", ""),
                    "agenda_title": agenda_title,
                    "speaker_role": role,
                    "speaker_name": name,
                    "speaker_type": speaker_type,
                    "utterance": utterance_cleaned
                })
        return results

if __name__ == "__main__":
    input_dir = constant.org_dir
    output_dir = constant.text_dir
    rules_file = constant.all_rule_file
    run_id, step_started_at = start_step("4")

    os.makedirs(output_dir, exist_ok=True)

    with open(rules_file, "r", encoding="utf-8") as f:
        parsing_rules = json.load(f)

    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    # org2가 일부 기관만 가진 경우 org 원본에서 빠진 기관 파일을 보충한다.
    fallback_input_dir = input_dir.replace("org2", "org", 1)
    if fallback_input_dir != input_dir and os.path.isdir(fallback_input_dir):
        known_stems = {os.path.splitext(os.path.basename(path))[0] for path in json_files}
        missing_files = [
            path for path in glob.glob(os.path.join(fallback_input_dir, "*.json"))
            if os.path.splitext(os.path.basename(path))[0] not in known_stems
        ]
        json_files.extend(missing_files)
        if missing_files:
            print(f"[보완] org2 누락 파일 {len(missing_files)}개를 org에서 추가합니다.")
    selected_keys = set(sys.argv[1:])
    if selected_keys:
        json_files = [
            path for path in json_files
            if os.path.basename(path) in selected_keys
            or os.path.splitext(os.path.basename(path))[0] in selected_keys
        ]
        unmatched = selected_keys - {
            os.path.basename(path) for path in json_files
        } - {
            os.path.splitext(os.path.basename(path))[0] for path in json_files
        }
        if unmatched:
            print(f"[경고] 입력 파일을 찾지 못했습니다: {', '.join(sorted(unmatched))}")
    preprocessor = DynamicCouncilPreprocessor(min_char_len=30)

    if not json_files:
        print(f"[오류] 원본 JSON 파일을 찾을 수 없습니다: {input_dir}")

    print(f"[시작] 전체 {len(json_files)}개 파일 | 입력: {input_dir} | 출력: {output_dir}")

    for file_index, input_file in enumerate(json_files, start=1):
        file_started = time.perf_counter()
        file_started_at = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        file_status = "error"
        item_count = 0
        base_name = os.path.basename(input_file)
        file_key = os.path.splitext(base_name)[0]

        matched_rule = parsing_rules.get(file_key)
        if matched_rule is None:
            matched_rule = {"agenda": "div", "speaker": "span.spkrbld", "utterance": "p"}
            print(f"  [경고] 통합 규칙에 '{file_key}'가 없어 기본 규칙을 사용합니다.")

        print(f"[{file_index}/{len(json_files)}] [{base_name}] 전처리 시작... 규칙={matched_rule}", flush=True)
        
        try:
            load_started = time.perf_counter()
            with open(input_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            item_count = len(raw_data) if isinstance(raw_data, list) else 0
            print(f"  [JSON 로드 완료] {len(raw_data) if isinstance(raw_data, list) else '형식 오류'}건, "
                  f"{time.perf_counter() - load_started:.1f}초", flush=True)
                
            all_utterances = []
            record_count = len(raw_data) if isinstance(raw_data, list) else 0
            if not isinstance(raw_data, list):
                print(f"  [오류] 최상위 JSON 형식이 배열이 아닙니다: {type(raw_data).__name__}")
                record_file(run_id, "4", input_file, time.perf_counter() - file_started,
                            "error_invalid_json", item_count, file_started_at)
                continue

            for record_index, record in enumerate(raw_data, start=1):
                record_started = time.perf_counter()
                print(f"  [회의록 {record_index}/{record_count}] 처리 중...", flush=True)
                extracted = preprocessor.process_record(record, matched_rule, base_name)
                all_utterances.extend(extracted)
                elapsed = time.perf_counter() - file_started
                estimated_remaining = (elapsed / record_index) * (record_count - record_index)
                print(
                    f"  [회의록 {record_index}/{record_count}] {len(extracted)}건 추출, "
                    f"경과 {time.perf_counter() - record_started:.1f}초 "
                    f"(누적 {len(all_utterances)}건, 파일 경과 {elapsed:.1f}초, "
                    f"예상 남음 {estimated_remaining:.1f}초)",
                    flush=True,
                )

            df_clean = pd.DataFrame(all_utterances)
            
            if not df_clean.empty:
                # 동일 화자(speaker_name, speaker_type 동일) 연속 발언 병합
                condition = (df_clean['speaker_name'] != df_clean['speaker_name'].shift()) | \
                            (df_clean['speaker_type'] != df_clean['speaker_type'].shift())
                df_clean['group_id'] = condition.cumsum()

                df_merged = df_clean.groupby('group_id').agg({
                    'date': 'first',
                    'assembly': 'first',
                    'agenda_title': 'first',
                    'speaker_role': 'first',
                    'speaker_name': 'first',
                    'speaker_type': 'first',
                    'utterance': lambda x: ' '.join(x)
                }).reset_index(drop=True)

                output_path = os.path.join(output_dir, f"cleaned_{file_key}.txt")
                df_merged.to_csv(output_path, sep="\t", index=False, encoding="utf-8-sig")
                print(f"  -> 완료: {len(df_merged)}건 추출 (연속 발언 병합됨) ({output_path})")
                file_status = "success"
            else:
                print(f"  -> 추출된 유효 데이터 없음")
                file_status = "empty"
            print(f"  -> 파일 소요 시간: {time.perf_counter() - file_started:.1f}초", flush=True)
        except Exception as e:
             print(f"  -> 파일 처리 중 치명적 오류 발생: {e}")
             print(f"  -> 실패까지 소요 시간: {time.perf_counter() - file_started:.1f}초", flush=True)
        record_file(run_id, "4", input_file, time.perf_counter() - file_started,
                    file_status, item_count, file_started_at)

    finish_step(run_id, "4", step_started_at)

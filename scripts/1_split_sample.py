"""지방의회 회의록 JSON에서 LLM 규칙 학습용 본문 HTML 샘플 생성.

- MINTS_HTML 값에 실제 발언/안건 본문이 들어 있는지 확인하고 추출한다.
- 원본의 첫 10,000자를 무작정 절단하지 않는다.
- 기존 sample_*.txt 출력명 및 _constant 설정을 유지한다.
"""
import glob
import html
import json
import os
import random
import re
import time
import traceback
from bs4 import BeautifulSoup, Comment
import _constant as constant
from pipeline_timing import start_step, record_file, finish_step

# 타 사이트에서도 공통으로 사용되는 구조적 힌트. 정확한 선택자는 2단계 LLM이 판단한다.
TRANSCRIPT_HINT = re.compile(r"(?:sMan|speaker|speech|utterance|statement|minutes|record[-_]?content|record[-_]?body|transcript|view[-_]?content|con[-_]?view|an\d+)", re.I)
SPEECH_HINT = re.compile(r"(?:○\s*[^\n]{2,30}|◯\s*[^\n]{2,30}|(?:위원장|의장|의원|국장|과장|교육감|부위원장)\s*[가-힣]{2,5}\s*(?:의원|위원|:|\s))")
NOISE = re.compile(r"(?:sidebar|navbar|navigation|breadcrumb|footer|header|toolbar|menu|pagination|related|sns)", re.I)


def prepare_html(raw_html: str):
    raw_html = html.unescape(raw_html) if '&lt;' in raw_html or '&gt;' in raw_html else raw_html
    soup = BeautifulSoup(raw_html, 'html.parser')
    for node in soup.find_all(string=lambda s: isinstance(s, Comment)):
        node.extract()
    for node in soup(['script', 'style', 'noscript', 'meta', 'link', 'header', 'nav', 'footer']):
        node.decompose()
    # Aside 또는 form도 의회 사이트에 따라 본문을 담을 수 있어 조건부로 제거한다.
    for tag in list(soup.find_all(['aside', 'form'])):
        # 상위 aside/form 제거 시 내부 태그도 분해되므로 살아 있는 노드만 접근한다.
        if tag.name is None or not isinstance(tag.attrs, dict):
            continue
        attrs = ' '.join(tag.get('class') or []) + ' ' + (tag.get('id') or '')
        if NOISE.search(attrs) or (tag.name == 'aside' and not TRANSCRIPT_HINT.search(attrs)):
            tag.decompose()
    for tag in list(soup.find_all(True)):
        # decompose()는 자식 태그까지 분해한다. 미리 수집한 자식은 attrs=None이 된다.
        if tag.name is None or not isinstance(tag.attrs, dict):
            continue
        classes = tag.get('class') or []
        element_id = tag.get('id') or ''
        attrs = ' '.join(classes) + ' ' + element_id
        if NOISE.search(attrs) and not TRANSCRIPT_HINT.search(attrs):
            tag.decompose()
    return soup


def find_transcript_region(soup):
    """증거가 있을 때만 본문 영역을 반환. 불확실하면 None."""
    candidates = []
    for tag in soup.find_all(['main', 'article', 'section', 'div', 'table']):
        txt = tag.get_text(' ', strip=True)
        if len(txt) < 120:
            continue
        attrs = ' '.join(tag.get('class') or []) + ' ' + (tag.get('id') or '')
        marker_count = len(SPEECH_HINT.findall(txt))
        structure = bool(TRANSCRIPT_HINT.search(attrs))
        descendant_hint = len(tag.find_all(attrs={'class': TRANSCRIPT_HINT})) + len(tag.find_all(attrs={'id': TRANSCRIPT_HINT}))
        # 구조적/언어적 증거가 없는 단순한 대형 div는 본문으로 선택하지 않는다.
        if not (marker_count >= 2 or descendant_hint >= 2 or (structure and marker_count >= 1)):
            continue
        # 무조건 큰 부모 요소를 선택하지 않도록 내용과 밀도를 함께 평가
        score = min(marker_count, 30) * 3 + min(descendant_hint, 30) * 2 + (8 if structure else 0)
        score -= min(len(txt) // 12000, 10)
        candidates.append((score, -len(str(tag)), tag))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def shorten_long_text(region, limit=300):
    """문장 선두는 보존하고 길이만 제한한다. 원본 노드는 변경하지 않도록 copy 사용."""
    copy = BeautifulSoup(str(region), 'html.parser')
    for node in list(copy.find_all(string=True)):
        value = str(node)
        if len(value) > limit:
            node.replace_with(value[:limit] + '…[발언 내용 일부 생략]')
    return str(copy)


def extract_samples_from_folder(input_dir='./input_data', output_dir='./sample_outputs', num_records=2):
    run_id, step_started_at = start_step("1")
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, '*.json')))
    if not files:
        print(f"[오류] JSON 파일 없음: {input_dir}")
        finish_step(run_id, "1", step_started_at)
        return
    for path in files:
        file_started = time.perf_counter()
        file_started_at = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        base = os.path.splitext(os.path.basename(path))[0]
        status = "error"
        item_count = 0
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            records = data if isinstance(data, list) else [data]
            item_count = len(records)
            middle = records[5:-5] if len(records) > 10 else records
            # 랜덤 '추출'이 아니라, 랜덤 순서로 검토하여 본문이 확인되는 n개를 찾음
            indices = list(range(len(middle)))
            random.shuffle(indices)
            samples = []
            for idx in indices:
                record = middle[idx]
                if not isinstance(record, dict) or not isinstance(record.get('MINTS_HTML'), str):
                    continue
                try:
                    soup = prepare_html(record['MINTS_HTML'])
                    region = find_transcript_region(soup)
                    if region is None:
                        continue
                    reduced = shorten_long_text(region)
                except Exception as record_exc:
                    print(f'[레코드 오류] {base} / index={idx}: {type(record_exc).__name__}: {record_exc}')
                    traceback.print_exc()
                    continue
                # HTML 태그 중간에서 자르지 않는다. 긴 경우 먼저 노드 텍스트를 축약.
                samples.append(reduced)
                if len(samples) >= num_records:
                    break
            if samples:
                out = os.path.join(output_dir, f'sample_{base}.txt')
                with open(out, 'w', encoding='utf-8') as f:
                    for i, sample in enumerate(samples, start=1):
                        f.write(f'--- [Random Sample Record {i} from {os.path.basename(path)}] ---\n')
                        f.write(sample + '\n\n')
                print(f'[성공] {base}: {len(samples)}건 본문 후보 저장 -> {out}')
                status = "success"
            else:
                print(f'[확인 필요] {base}: 발언 본문 후보를 찾지 못함. 원본 MINTS_HTML 구조 확인 필요')
                status = "no_sample"
        except Exception as exc:
            print(f'[오류] {base}: {type(exc).__name__}: {exc}')
            traceback.print_exc()
        finally:
            record_file(run_id, "1", path, time.perf_counter() - file_started,
                        status, item_count, file_started_at)
    finish_step(run_id, "1", step_started_at)


if __name__ == '__main__':
    extract_samples_from_folder(
        input_dir=constant.org_dir,
        output_dir=constant.sample_dir,
        num_records=constant.records_num,
    )

# -*- coding: utf-8 -*-
"""2차·3차 프롬프트 전달 진단. 기존 실험 CSV/XLSX는 읽기 전용으로 사용합니다.

실행: C:\\Users\\Admin\\miniconda3\\envs\\mini\\python.exe C:\\mini_pj\\scripts\\evaluation_02_prompt_check.py
"""
import ast
import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import requests
import chromadb

ROOT = Path(r'C:\mini_pj')
SCRIPTS = ROOT / 'scripts'
BASE = ROOT / 'results' / 'rag_evaluation'
SECOND = BASE / 'bertopic_2차_optimized_llm'
THIRD = BASE / 'bertopic_3차_prompt'
OUT = BASE / 'prompt_diagnostic'
DB_DIR = ROOT / '_db' / 'chroma_personal'
COLLECTION = 'personal_policy_chunks'
QUESTION = '청년 일자리'
MODEL = 'exaone3.5:2.4b'
OLLAMA_URL = 'http://127.0.0.1:11434/api/generate'
RUN_LLM = True  # 프롬프트 비교만 하려면 False


def load_prompt_function(path):
    """원본 스크립트의 prompt_for만 추출: main()이나 임베딩 모델을 로드하지 않음."""
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prompt_for'), None)
    if node is None:
        raise RuntimeError(f'prompt_for 함수를 찾지 못함: {path}')
    ns = {}
    code = compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec')
    exec(code, ns)
    return ns['prompt_for']


def read_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def select_reference_rows():
    result_path = SECOND / '모델결과_누적.csv'
    evidence_path = SECOND / '근거회의록_누적.csv'
    if not result_path.exists() or not evidence_path.exists():
        raise FileNotFoundError('2차 실험 모델결과/근거회의록 CSV를 확인하세요: ' + str(SECOND))
    matches = [r for r in read_csv(result_path) if r.get('키워드') == QUESTION and r.get('모델') == MODEL]
    if not matches:
        raise ValueError(f'2차 결과에 {QUESTION} / {MODEL} 조합이 없습니다.')
    reference = matches[0]
    experiment_id = reference['실험ID']
    evid = [r for r in read_csv(evidence_path) if r.get('실험ID') == experiment_id]
    evid.sort(key=lambda x: int(x.get('근거 번호') or 0))
    if not evid:
        raise ValueError('2차 실험에 연결된 근거가 없습니다.')
    return reference, evid


def lookup_evidence(rows):
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION)
    ids = [r.get('chunk_id', '') for r in rows]
    if any(not x for x in ids):
        raise ValueError('근거 CSV에 chunk_id가 비어 있습니다.')
    found = collection.get(ids=ids, include=['documents', 'metadatas'])
    mapping = {uid: (doc, meta or {}) for uid, doc, meta in zip(found['ids'], found['documents'], found['metadatas'])}
    if len(mapping) != len(set(ids)):
        raise ValueError('Chroma DB에서 일부 근거를 찾지 못했습니다.')
    evid = []
    for r in rows:
        doc, meta = mapping[r['chunk_id']]
        date = str(r.get('일자') or '')
        if not date:
            val = ''.join(c for c in str(meta.get('date', '')) if c.isdigit())
            date = f'{val[:4]}-{val[4:6]}-{val[6:8]}' if len(val) >= 8 else ''
        evid.append({
            'date': date,
            'assembly': r.get('의회') or str(meta.get('assembly') or ''),
            'meeting': r.get('원본 회의록명') or str(meta.get('meeting_title') or ''),
            'agenda': r.get('안건') or str(meta.get('agenda_title') or ''),
            'text': doc or '',
        })
    return evid


def parse_topics(row):
    """CSV에 남은 토픽 문구를 복원. 유사도 수치는 CSV에 없어 0으로 표기."""
    raw = row.get('BERTopic 관련 토픽') or ''
    result = []
    for item in raw.split(' | '):
        topic, sep, words = item.partition(':')
        if not sep:
            continue
        try:
            result.append({'topic_id': int(topic.strip()), 'keywords': words.strip(), 'score': 0.0})
        except ValueError:
            continue
    return result


def ollama_call(prompt):
    started = time.perf_counter()
    payload = {'model': MODEL, 'prompt': prompt, 'stream': False, 'think': False,
               'options': {'temperature': 0, 'num_predict': 320, 'num_ctx': 8192,}}
   
    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return {
        'response': (data.get('response') or '').strip(),
        'elapsed_seconds': round(time.perf_counter() - started, 3),
        'load_seconds': round(data.get('load_duration', 0) / 1e9, 3),
        'generated_tokens': data.get('eval_count', 0),
        'generation_seconds': round(data.get('eval_duration', 0) / 1e9, 3),
    }


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    second_py = SCRIPTS / '09_optimized_llm_evaluate.py'
    third_py = SCRIPTS / '10_prompt_evaluate.py'
    for p in (second_py, third_py):
        if not p.exists():
            raise FileNotFoundError(f'필요한 실험 코드가 없습니다: {p}')
    f2, f3 = load_prompt_function(second_py), load_prompt_function(third_py)
    source_result, evidence_rows = select_reference_rows()
    evidence = lookup_evidence(evidence_rows)
    # 검색 근거 15개 중 상위 5개만 사용
    evidence = evidence[:5]
    
    chosen = parse_topics(source_result)
    if not chosen:
        raise ValueError('2차 결과에 관련 토픽 정보가 없습니다.')
    p2 = f2(QUESTION, chosen, evidence)
    p3 = f3(QUESTION, chosen, evidence)
    digest = lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest()
    print(f'질문: {QUESTION} / 모델: {MODEL} / 실제 근거: {len(evidence)}개')
    print('주의: 토픽 유사도 점수는 CSV에 없어 두 프롬프트 모두 0.0으로 복원합니다.')
    print('2차 프롬프트:', len(p2), '자 / SHA256:', digest(p2)[:16])
    print('3차 프롬프트:', len(p3), '자 / SHA256:', digest(p3)[:16])
    print('두 프롬프트 동일 여부:', p2 == p3)
    print('3차 강화 지시문 포함:', '출력 직전 200자 이내인지 점검' in p3)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    (OUT / f'2차_프롬프트_{stamp}.txt').write_text(p2, encoding='utf-8-sig')
    (OUT / f'3차_프롬프트_{stamp}.txt').write_text(p3, encoding='utf-8-sig')
    report = {'timestamp': stamp, 'question': QUESTION, 'model': MODEL,
              'evidence_count': len(evidence), 'topic_count': len(chosen),
              'topic_scores_reconstructed_as_zero': True,
              'prompt2_length': len(p2), 'prompt3_length': len(p3),
              'prompt2_sha256': digest(p2), 'prompt3_sha256': digest(p3),
              'prompts_identical': p2 == p3, 'llm_executed': RUN_LLM}
    if RUN_LLM:
        print('\n[2차 프롬프트로 실제 Ollama 호출]')
        a = ollama_call(p2)
        print('응답 길이:', len(a['response']), '/ LLM 호출:', a['elapsed_seconds'], '초')
        print('\n[3차 프롬프트로 실제 Ollama 호출]')
        b = ollama_call(p3)
        print('응답 길이:', len(b['response']), '/ LLM 호출:', b['elapsed_seconds'], '초')
        print('\n응답 본문 동일 여부:', a['response'] == b['response'])
        print('200자 통과:', len(a['response']) <= 200, '/', len(b['response']) <= 200)
        report.update({'second_response': a, 'third_response': b,
                       'responses_identical': a['response'] == b['response']})
    report['total_seconds'] = round(time.perf_counter() - start, 3)
    dest = OUT / f'프롬프트_진단_{stamp}.json'
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8-sig')
    print('\n진단 보고서:', dest)
    print('총 소요 시간:', report['total_seconds'], '초')
    print('기존 실험 결과 CSV/Excel 수정 없음')


if __name__ == '__main__':
    main()

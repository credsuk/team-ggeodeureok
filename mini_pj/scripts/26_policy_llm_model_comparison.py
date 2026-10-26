# -*- coding: utf-8 -*-
"""26번 | 로컬 LLM 3종 정책 전망 공정 비교 (25번 결과 재사용).

실행:
  python 26_policy_llm_model_comparison.py --input-dir "C:\\mini_pj\\results\\policy_issue_search\\20261025_000801_청년_일자리"
  python 26_policy_llm_model_comparison.py --input-dir "..." --timeout 240 --num-predict 650

주의: 자동 평가점수는 형식 및 입력근거 연결 검사에 한정된다.
      전망 정확도, 사실성, 인과적 전파, 정책 채택 여부를 증명하지 않는다.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

MODELS = ['exaone3.5:2.4b', 'qwen3:8b', 'gemma3:4b']
DEFAULT_INPUT = r'C:\mini_pj\results\policy_issue_search\20261025_000801_청년_일자리'


def log(msg):
    print(f'[{datetime.now():%H:%M:%S}] {msg}', flush=True)


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f'필수 결과 파일 없음: {path}')
    return pd.read_csv(path, encoding='utf-8-sig', low_memory=False)


def string(value):
    if pd.isna(value):
        return ''
    return str(value).strip()


def trimmed(text, limit=180):
    return re.sub(r'\s+', ' ', string(text))[:limit]


def rows_by_month(df):
    # 24개월 전체를 넣어야 단일 급등월만으로 추세를 왜곡하지 않음.
    results = []
    for _, r in df.sort_values('month').iterrows():
        results.append({
            '월': string(r['month']),
            '직접후보_청크': int(r['직접관련_자동후보_청크']),
            '전체_청크': int(r['all_chunks']),
            '직접후보비율_pct': round(float(r['직접관련_자동후보_비율_pct']), 4),
        })
    return results


def select_evidence(df, n=12):
    """근거 다양성 확보: 날짜(초기/최근)와 정책 수단 다양성, 출처 다양성을 고려."""
    d = df[df['review_candidate_label'].eq('직접관련_후보')].copy()
    d = d.dropna(subset=['date', 'assembly', 'text_preview'])
    d = d.sort_values(['date', 'assembly', 'chunk_id']).copy()
    if d.empty:
        return []
    # 시간 구간 및 정책 수단별로 후보 분산; 같은 회의록 source_id는 중복 억제
    d['year'] = d['date'].astype(str).str[:4]
    d['mechanism'] = d['mechanisms_detected'].fillna('미분류').astype(str).str.split('|').str[0]
    pools = [d.head(8), d[d['year'].eq('2024')].tail(8), d[d['year'].eq('2025')].head(8), d.tail(12)]
    for _, g in d.groupby('mechanism', sort=True):
        pools.append(g.head(2))
    sampled = pd.concat(pools).drop_duplicates('chunk_id')
    sampled = sampled.drop_duplicates('source_id')
    if len(sampled) < n:
        sampled = pd.concat([sampled, d]).drop_duplicates('chunk_id').drop_duplicates('source_id')
    sampled = sampled.head(n).sort_values('date')
    evidence = []
    for i, (_, r) in enumerate(sampled.iterrows(), start=1):
        evidence.append({
            'id': f'E{i:02d}', '날짜': string(r['date']),
            '의회': string(r['assembly']), '회의록': string(r['meeting_title']),
            'chunk_id': string(r['chunk_id']), '정책수단_자동탐지': string(r['mechanisms_detected']),
            '원문': trimmed(r['text_preview'], 420),
            '핵심구간_100자': trimmed(r['core_evidence_text_100'], 100),
            '검증상태': '자동후보_수동미검증',
        })
    return evidence


def build_input(root, evidence_n):
    timeline = root / 'timeline_analysis'
    relevance = root / 'relevance_validation'
    monthly = read_csv(timeline / '월별_정책관련성_추이.csv')
    regional = read_csv(timeline / '지역월별_정책관련성_추이.csv')
    first = read_csv(timeline / '지역별_최초관측_구분.csv')
    cross = read_csv(timeline / '지역간_후속논의_검토후보.csv')
    source = read_csv(relevance / '전체_정책관련성_검토결과.csv')
    summary_path = timeline / '시계열_분석요약.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8-sig')) if summary_path.exists() else {}
    evidence = select_evidence(source, evidence_n)
    if not evidence:
        raise ValueError('직접 관련 자동 후보 근거가 없어 전망을 생성할 수 없습니다.')
    d0 = first[first['관측범위'].eq('직접관련_자동후보')].sort_values('date')
    if d0.empty:
        d0 = first.sort_values('date')
    earliest = d0.iloc[0]
    regional_top = regional.groupby('assembly', as_index=False).agg(
        직접후보_청크=('직접관련_자동후보_청크', 'sum'), 전체_청크=('all_chunks', 'sum'))
    regional_top['정규화비율_pct'] = (regional_top['직접후보_청크'] / regional_top['전체_청크'].replace(0, float('nan')) * 100).round(4)
    regional_top = regional_top.sort_values('직접후보_청크', ascending=False).head(6).fillna(0)
    pairs = []
    for _, row in cross.head(6).iterrows():
        pairs.append({'id': string(row['candidate_id']),
                      '선행관측': f"{row['earlier_date']} {row['earlier_assembly']}",
                      '후속관측': f"{row['later_date']} {row['later_assembly']}",
                      '유사정책수단': string(row['specific_common_mechanisms']),
                      '상태': '자동 유사 후보·전파 미확인'})
    data = {
        '이슈': '청년 일자리', '관측기간': '2024~2025년 보유 지방의회 회의록',
        '집계단위': '회의록 청크 (정책 시행 또는 회의 건수 아님)',
        '첫_자동직접후보': {'날짜': string(earliest['date']), '의회': string(earliest['assembly'])},
        '전체어휘일치_청크': int(summary.get('full_lexical_matches', len(source))),
        '직접관련_자동후보_청크': int((source['review_candidate_label'] == '직접관련_후보').sum()),
        '수동확정_청크': int(summary.get('manual_verified', 0)),
        '월별_직접후보': rows_by_month(monthly),
        '지역별_자동후보상위': regional_top.to_dict('records'),
        '타지역_유사정책_후속관측_후보': pairs,
        '근거': evidence,
        '해석주의': ['자료 수집량 및 회기 영향 고려', '청크별 자동 분류 미검증',
                '지역 간 유사 논의는 정책 전파를 입증하지 않음',
                '2026년 관측 데이터가 없으므로 미래는 조건부 추정',
                '근거 내용에 없는 수치·정책·공청회 실적을 생성하지 말 것'],
    }
    return data


def prompt_parts(data):
    system = ("당신은 한국 지방의회 회의록 기반 정책 동향 분석가다. "
              "오직 제공된 JSON 자료를 활용하고 미확인 사실은 확정하지 않는다. "
              "미래 예측은 조건부로 작성하되, 자동 분류와 정책 전파의 미확정성을 반드시 존중한다. "
              "응답은 마크다운·설명 없이 JSON 객체 하나만 반환한다.")
    instructions = """동일 조건에서 모델별 성능을 비교한다. 다음 JSON 객체의 열쇠를 정확히 사용하라.
{
 "forecast": "150자 이내 한국어 서술형 미래 전망 한 문단",
 "trend_interpretation": "청크량 및 정규화 비율을 고려한 180자 이내 과거 관측 해석",
 "forecast_basis": "제공된 통계 및 근거에 기반한 전망 논리 180자 이내",
 "evidence_ids": ["E01", "E02"],
 "uncertainty": "한계나 대안 가능성 120자 이내"
}
규칙:
- forecast는 반드시 150자 이하(공백 포함)이며 단정 대신 '가능성', '경우' 등 조건부 표현 사용.
- 근거 ID는 제공된 근거 목록에서만 선택(최대 3개). 원문이 전망에 직접 관련될 때만 선택.
- 2024~2025년은 관측 기간, 2026년은 미관측 예측 대상이다.
- '최초 도입', '정책 전파 확정', '공청회 실시 확정' 등의 미입증 표현 사용 금지.
- 회의록 청크 빈도 증감이 곧 정책 채택량 증가라는 해석 금지.
- 자동 후보를 수동 확인된 근거로 표현하지 말 것.
자료(JSON):
""" + json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    return system, instructions


def request_json(base_url, api_path, payload, timeout):
    url = base_url.rstrip('/') + api_path
    req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode('utf-8'))


def available_models(base_url, timeout=12):
    with urllib.request.urlopen(base_url.rstrip('/') + '/api/tags', timeout=timeout) as res:
        data = json.loads(res.read().decode('utf-8'))
    return [m.get('name', '') for m in data.get('models', [])]


def parse_model_response(raw):
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.S).strip()
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned, flags=re.I).strip()
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        start = cleaned.find('{')
        if start < 0:
            return None
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


def evaluate(parsed, evidence_ids):
    # 기계적 적합성 검사만 계산. 의미적 정확성 점수는 배제.
    result = {'json_valid': isinstance(parsed, dict), 'forecast_chars': None,
              'forecast_under_150': False, 'evidence_ids_valid': False,
              'evidence_ids_used': 0, 'required_fields_present': False,
              'caution_word_present': False, 'format_score_100': 0,
              'human_review_required': True}
    if not isinstance(parsed, dict):
        return result
    mandatory = ['forecast', 'trend_interpretation', 'forecast_basis', 'evidence_ids', 'uncertainty']
    result['required_fields_present'] = all(k in parsed for k in mandatory)
    forecast = parsed.get('forecast', '')
    if not isinstance(forecast, str):
        forecast = ''
    result['forecast_chars'] = len(forecast)
    result['forecast_under_150'] = bool(forecast) and len(forecast) <= 150
    ids = parsed.get('evidence_ids', [])
    valid_ids = isinstance(ids, list) and 1 <= len(ids) <= 3 and all(isinstance(i, str) and i in evidence_ids for i in ids)
    result['evidence_ids_valid'] = valid_ids
    result['evidence_ids_used'] = len(ids) if isinstance(ids, list) else 0
    result['caution_word_present'] = any(x in forecast for x in ['가능성', '경우', '조건', '전망', '예상', '수 있'])
    result['format_score_100'] = (25 + 20 * result['required_fields_present']
                                  + 25 * result['forecast_under_150']
                                  + 20 * result['evidence_ids_valid']
                                  + 10 * result['caution_word_present'])
    return result


def safe_save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def run():
    ap = argparse.ArgumentParser(description='26 정책 전망: Ollama 로컬 3모델 동일 조건 비교')
    ap.add_argument('--input-dir', default=DEFAULT_INPUT)
    ap.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    ap.add_argument('--timeout', type=int, default=240)
    ap.add_argument('--num-predict', type=int, default=700)
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--evidence-n', type=int, default=12)
    ap.add_argument('--models', nargs='+', default=MODELS)
    ap.add_argument('--dry-run', action='store_true', help='모델 호출 없이 파일 연결·근거·프롬프트 검사')
    args = ap.parse_args()
    started = time.perf_counter()
    root = Path(args.input_dir)
    dest = root / 'llm_model_comparison_26'
    dest.mkdir(parents=True, exist_ok=True)
    log('=== 26 정책 LLM 모델 비교 시작 ===')
    t = time.perf_counter()
    data = build_input(root, max(3, min(args.evidence_n, 24)))
    system, user = prompt_parts(data)
    safe_save_json(dest / '동일입력_근거_시계열.json', data)
    (dest / '동일프롬프트.txt').write_text('SYSTEM\n' + system + '\n\nUSER\n' + user, encoding='utf-8')
    pd.DataFrame(data['근거']).to_csv(dest / 'LLM_입력_근거목록.csv', index=False, encoding='utf-8-sig')
    log(f'[1/3] 자료 준비: 24개월, 근거 {len(data["근거"])}개 / {time.perf_counter()-t:.2f}초')
    if args.dry_run:
        log(f'[DRY RUN] Ollama 호출 생략, 프롬프트 글자수 {len(system)+len(user):,} / 전체 {time.perf_counter()-started:.2f}초')
        log(f'자료 경로: {dest}')
        return 0

    try:
        installed = available_models(args.ollama_url)
    except Exception as exc:
        log(f'[오류] Ollama 서버 연결 실패: {exc}')
        log('해결: Ollama 실행 상태 확인, 또는 ollama serve 명령 확인')
        return 2
    missing = [m for m in args.models if m not in installed]
    if missing:
        log(f'[중단] 설치되지 않은 모델: {missing}')
        log('설치 모델: ' + ', '.join(installed))
        log('해결: ollama pull <모델명> 으로 준비하거나 --models로 지정')
        return 2
    log('[2/3] 모델 확인 완료: ' + ', '.join(args.models))

    all_records = []
    allowed_ids = {x['id'] for x in data['근거']}
    for idx, model in enumerate(args.models, start=1):
        t = time.perf_counter()
        log(f'  모델 [{idx}/{len(args.models)}] {model} 시작')
        error, raw, parsed, metrics = '', '', None, {}
        try:
            response = request_json(args.ollama_url, '/api/chat', {
                'model': model, 'stream': False,
                'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                'format': 'json',
                'options': {'temperature': args.temperature, 'num_predict': args.num_predict, 'seed': 42},
                'keep_alive': '5m',
                'think': False,
            }, timeout=args.timeout)
            raw = response.get('message', {}).get('content', '')
            parsed = parse_model_response(raw)
            metrics = {k: response.get(k) for k in ('total_duration', 'load_duration',
                        'prompt_eval_count', 'prompt_eval_duration', 'eval_count', 'eval_duration')}
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        elapsed = round(time.perf_counter() - t, 3)
        checks = evaluate(parsed, allowed_ids)
        record = {
            'model': model, 'started_at': datetime.now().isoformat(timespec='seconds'),
            'elapsed_seconds': elapsed, 'error': error,
            'forecast': parsed.get('forecast', '') if isinstance(parsed, dict) else '',
            'trend_interpretation': parsed.get('trend_interpretation', '') if isinstance(parsed, dict) else '',
            'forecast_basis': parsed.get('forecast_basis', '') if isinstance(parsed, dict) else '',
            'evidence_ids': json.dumps(parsed.get('evidence_ids', []), ensure_ascii=False) if isinstance(parsed, dict) else '[]',
            'uncertainty': parsed.get('uncertainty', '') if isinstance(parsed, dict) else '',
            **checks,
            'prompt_tokens': metrics.get('prompt_eval_count'),
            'generation_tokens': metrics.get('eval_count'),
            'generation_tokens_per_sec': (round(metrics.get('eval_count', 0) / (metrics.get('eval_duration') / 1e9), 2)
                                          if metrics.get('eval_duration') and metrics.get('eval_count') else None),
        }
        all_records.append(record)
        model_code = re.sub(r'[^a-zA-Z0-9]+', '_', model)
        safe_save_json(dest / f'{model_code}_결과.json', {'record': record, 'parsed': parsed,
                       'raw_response': raw, 'ollama_metrics': metrics})
        pd.DataFrame(all_records).to_csv(dest / '3모델_비교결과.csv', index=False, encoding='utf-8-sig')
        log(f'  → {model} {elapsed:.2f}초 / 전망 {checks["forecast_chars"]}자 / '
            f'형식점수 {checks["format_score_100"]}/100 / '
            f'근거ID 검사 {checks["evidence_ids_valid"]}' + (f' / 오류 {error}' if error else ''))

    result_summary = {
        'issue': data['이슈'], 'models': args.models,
        'observed_period': '2024~2025',
        'manual_verified': False,
        'forecast_target': '2026 이후 조건부 전망',
        'prompt_identical': True, 'model_options': {'temperature': args.temperature,
             'num_predict': args.num_predict, 'seed': 42, 'format': 'json', 'think': False},
        'count': len(all_records), 'results': all_records,
        'limitations': [
            '형식점수는 자동 형식 준수 점수이며 정책 전망 정확도·사실성 점수가 아님',
            '근거 ID 유효성은 해당 문장이 전망을 실제 지지하는지 검증하지 않음',
            '수동검토 0건: 정책 수단·최초 시점·지역 간 전파 미확정',
            '같은 질문 1개, 실행 1회로 우수 모델을 확정할 수 없음',
            '향후 다른 정책 이슈와 반복 실행으로 모델별 편차 평가 필요',
        ],
        'elapsed_seconds': round(time.perf_counter() - started, 3),
    }
    safe_save_json(dest / '비교_요약.json', result_summary)
    log(f'[3/3] 종료: 전체 {result_summary["elapsed_seconds"]:.2f}초 / 저장: {dest}')
    log('주의: 점수는 형식 준수만 평가. 내용의 사실성·전망 타당성은 별도 검토 필요.')
    return 0 if all(not x['error'] for x in all_records) else 1


if __name__ == '__main__':
    try:
        sys.exit(run())
    except Exception as exc:
        log(f'[중단] {type(exc).__name__}: {exc}')
        sys.exit(2)

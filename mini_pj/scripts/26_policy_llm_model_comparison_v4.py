# -*- coding: utf-8 -*-
"""26번 V4 | 로컬 LLM 3종 정책 전망 공정 비교 (25번 결과 재사용).

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


def compact_for_llm(data):
    """원시 회의록 재출력을 줄이되, 24개월 통계와 근거 ID를 보존한다.

    원래 입력 전체는 별도 JSON/CSV로 저장하며, 모델은 이 짧은 사본만 받는다.
    """
    evidence = []
    for e in data['근거']:
        evidence.append({
            'id':e['id'], '날짜':e['날짜'], '의회':e['의회'],
            '정책수단':trimmed(e['정책수단_자동탐지'], 35),
            '발췌':trimmed(e['핵심구간_100자'] or e['원문'], 100),
            '검증':'자동후보_수동미검증',
        })
    return {
        '이슈': data['이슈'], '관측기간': data['관측기간'],
        '집계단위':data['집계단위'],
        '최초_자동후보':data['첫_자동직접후보'],
        '전체어휘일치_청크':data['전체어휘일치_청크'],
        '직접관련_자동후보_청크':data['직접관련_자동후보_청크'],
        '수동확정_청크':data['수동확정_청크'],
        '월별_직접후보':data['월별_직접후보'],
        '지역상위':[
            {'assembly':x['assembly'], '직접후보_청크':int(x['직접후보_청크']),
             '정규화비율_pct':float(x['정규화비율_pct'])}
            for x in data['지역별_자동후보상위']
        ],
        '유사_후속관측_후보':[
            {'선행':p['선행관측'],'후속':p['후속관측'],
             '수단':trimmed(p['유사정책수단'],45),'전파확정':False}
            for p in data['타지역_유사정책_후속관측_후보'][:5]
        ],
        '근거_발췌':evidence,
        '주의':'청크 자동후보는 수동 미검증. 빈도 변화는 회기/수집량 영향. 지역 간 인과 전파 미입증.',
    }


OUTPUT_SCHEMA = {
    'type':'object',
    'properties':{
        'forecast':{'type':'string', 'description':'200자 이내 미래에 대한 조건부 전망'},
        'trend_interpretation':{'type':'string', 'description':'180자 이내 과거 추세 해석'},
        'forecast_basis':{'type':'string', 'description':'180자 이내 미래 전망의 이유'},
        'evidence_ids':{'type':'array', 'items':{'type':'string'}, 'minItems':1, 'maxItems':3},
        'uncertainty':{'type':'string', 'description':'120자 이내 불확실성'},
    },
    'required':['forecast','trend_interpretation','forecast_basis','evidence_ids','uncertainty'],
    'additionalProperties':False,
}


def prompt_parts(data):
    compact = compact_for_llm(data)
    system = (
        '당신은 지방의회 자료를 이용해 2026년 이후 정책 방향을 **조건부로 예측**하는 분석가다. '
        '과거 회의록 요약, 자료 목록 재출력, 입력 JSON 복사, 장문의 설명은 금지한다. '
        '2024~2025년 기록은 자동 분류된 청크이며 수동 검증이 없고 전파 인과관계는 미확인이다. '
        '다섯 개 키만 가진 JSON 객체 하나를 반환한다.'
    )
    user = (
        '작업: 다음 과거 관측자료를 근거로 청년 일자리 정책의 **향후 방향**을 예측한다. '
        '반드시 미래 행동 가능성이나 정책 변화 전망을 forecast에 작성하라. '
        '현재 내용만 설명하거나 입력 기록을 되풀이하면 실패다.\n'
        '출력할 JSON 키는 forecast, trend_interpretation, forecast_basis, evidence_ids, uncertainty 다섯 개뿐이다.\n'
        'forecast: 200자 이내 한국어 한 문단. "~할 가능성이 있다"처럼 조건부 미래 전망.\n'
        'trend_interpretation: 180자 이내 과거 관측 추세. 월별 건수와 분모 변화 함께 고려.\n'
        'forecast_basis: 180자 이내 제공된 근거에 연결한 예측 이유.\n'
        'evidence_ids: 제공한 E번호 가운데 관련된 1~3개만. 없는 번호 금지.\n'
        'uncertainty: 120자 이내 수동 미검증·수집량 편향·미래 불확실성.\n'
        '조례 채택, 공청회 실시, 지역 간 정책 전파, 정책 확대를 사실로 확정하지 말 것.\n'
        '원문 전체나 근거 목록을 결과에 재출력하지 말 것.\n'
        '입력 데이터:\n'+json.dumps(compact,ensure_ascii=False,separators=(',',':'))
    )
    return system, user


def retry_messages(system, user, raw, error):
    """재시도는 같은 조건의 오류 수정 프롬프트. 지난 긴 응답은 보내지 않는다."""
    repair = (
        '직전 응답은 평가 실패다. 실패 이유: '+error+'. '
        '기존 응답을 요약하거나 복사하지 마라. '
        'forecast에는 과거 요약이 아니라 미래 정책 방향의 조건부 전망 1문단을 쓴다. '
        '필수 키 forecast, trend_interpretation, forecast_basis, evidence_ids, uncertainty를 전부 채워라. '
        'forecast 200자, trend_interpretation 180자, forecast_basis 180자, uncertainty 120자 이하. '
        '근거 목록 원문 재출력 금지. 허용된 E번호만 1~3개. '
        'JSON 객체 하나만 출력하라.'
    )
    return [
        {'role':'system','content':system},
        {'role':'user','content':user},
        {'role':'user','content':repair},
    ]


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


def normalize_payload(data):
    """서로 다른 모델의 필드명/중첩 구조 차이를 흡수하되 빈 전망은 보충 생성하지 않는다."""
    if not isinstance(data, dict):
        return None
    if isinstance(data.get('result'), dict):
        data = data['result']
    if isinstance(data.get('answer'), dict):
        data = data['answer']
    aliases = {
        'forecast': ('forecast', 'prediction', 'future_forecast', '향후 예측', '향후예측', '예측', '전망'),
        'trend_interpretation': ('trend_interpretation', 'trend', 'trend_analysis', '추세 해석', '시계열해석'),
        'forecast_basis': ('forecast_basis', 'basis', 'reason', '예측 근거', '전망 근거'),
        'evidence_ids': ('evidence_ids', 'references', 'evidence', '근거ID', '근거_ids'),
        'uncertainty': ('uncertainty', 'limitations', 'caveats', '한계', '불확실성'),
    }
    out = {}
    for field, keys in aliases.items():
        out[field] = next((data[k] for k in keys if k in data), [] if field == 'evidence_ids' else '')
    if isinstance(out['evidence_ids'], str):
        out['evidence_ids'] = re.findall(r'E\d{2}', out['evidence_ids'])
    for key in ('forecast','trend_interpretation','forecast_basis','uncertainty'):
        if not isinstance(out[key], str):
            out[key] = ''
        out[key] = out[key].strip()
    return out


def parse_model_response(raw):
    if not isinstance(raw, str):
        return None
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.S).strip()
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned, flags=re.I).strip()
    try:
        result = json.loads(cleaned)
        return normalize_payload(result)
    except json.JSONDecodeError:
        start = cleaned.find('{')
        if start < 0:
            return None
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
            return normalize_payload(obj)
        except json.JSONDecodeError:
            return None


def evaluate(parsed, evidence_ids):
    # 기계적 적합성 검사만 계산. 의미적 정확성 점수는 배제.
    result = {'json_valid': isinstance(parsed, dict), 'forecast_chars': None,
              'forecast_under_200': False, 'evidence_ids_valid': False,
              'evidence_ids_used': 0, 'required_fields_present': False,
              'caution_word_present': False, 'format_score_100': 0,
              'human_review_required': True}
    if not isinstance(parsed, dict):
        return result
    mandatory = ['forecast', 'trend_interpretation', 'forecast_basis', 'evidence_ids', 'uncertainty']
    result['required_fields_present'] = all((parsed.get(k) not in (None, '', [])) for k in mandatory)
    forecast = parsed.get('forecast', '')
    if not isinstance(forecast, str):
        forecast = ''
    result['forecast_chars'] = len(forecast)
    result['forecast_under_200'] = bool(forecast) and len(forecast) <= 200
    ids = parsed.get('evidence_ids', [])
    valid_ids = isinstance(ids, list) and 1 <= len(ids) <= 3 and all(isinstance(i, str) and i in evidence_ids for i in ids)
    result['evidence_ids_valid'] = valid_ids
    result['evidence_ids_used'] = len(ids) if isinstance(ids, list) else 0
    result['caution_word_present'] = any(x in forecast for x in ['가능성', '경우', '조건', '전망', '예상', '수 있'])
    result['format_score_100'] = (25 * bool(result['forecast_chars']) + 20 * result['required_fields_present']
                                  + 25 * result['forecast_under_200']
                                  + 20 * result['evidence_ids_valid']
                                  + 10 * result['caution_word_present'])
    return result


def safe_save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


# 이전 개인 기본 RAG 결과와 동일한 컬럼 순서를 유지한다.
BASE_FIELDS = ["실험", "키워드", "모델", "상태(200자)", "등장시점(검색표본)",
    "타 지방의회 논의순서(검색표본)", "향후 예측(앞 200자 표시)", "원래 예측 전체",
    "실제 글자 수", "200자 준수", "근거 회의록(텍스트 요약)", "근거 수",
    "질문 임베딩(초)", "DB 검색(초)", "LLM(초)", "전체(초)",
    "모델 로드(초)", "출력 토큰", "토큰/초", "오류", "실행시각", "검색근거검증", "실험ID"]
EVIDENCE_FIELDS = ["실험ID", "이슈", "모델", "근거 번호", "일자", "의회", "원본 회의록명",
    "안건", "발언자", "모델 상태", "근거 원문(앞 220자)", "저장된 source_file",
    "row_id", "chunk_id", "검색 거리", "검증 상태"]
ISSUES = ['지역소멸', '청년 일자리', '저출생', '고령화', '기후위기']


def pick_flow(data):
    pairs = data.get('타지역_유사정책_후속관측_후보', [])
    if not pairs:
        return '유사 논의 확인 후보 없음'
    seen = []
    for pair in pairs:
        for key in ('선행관측', '후속관측'):
            value = str(pair.get(key, '')).strip()
            if value and value not in seen:
                seen.append(value)
    seen.sort(key=lambda v: v[:10])
    return ' → '.join(seen[:10]) + ' (자동 유사 후보, 전파 미확인)'


def build_evidence_rows(evidence, model, experiment_id):
    rows = []
    for ev in evidence:
        rows.append({'실험ID': experiment_id, '이슈': '청년 일자리', '모델': model,
            '근거 번호': ev['id'], '일자': ev['날짜'], '의회': ev['의회'],
            '원본 회의록명': ev['회의록'], '안건': '', '발언자': '',
            '모델 상태': '자동후보_수동미검증', '근거 원문(앞 220자)': ev['원문'][:220],
            '저장된 source_file': '', 'row_id': '', 'chunk_id': ev['chunk_id'],
            '검색 거리': '', '검증 상태': '자동후보_원본JSON미대조'})
    return rows


def export_workbook(dest, records, evidences):
    # 기존 RAG와 동일한 4개 시트. 수동 품질점수를 만들어 넣지 않는다.
    try:
        import xlsxwriter  # noqa: F401
    except ImportError:
        log('Excel 저장 생략: pip install xlsxwriter')
        return
    frame = pd.DataFrame(records, columns=BASE_FIELDS)
    ev_frame = pd.DataFrame(evidences, columns=EVIDENCE_FIELDS)
    rank = frame[['키워드','모델','상태(200자)','LLM(초)','200자 준수']].copy()
    rank.columns = ['이슈키워드','모델','상태','LLM 초','200자 준수']
    rank['예측 타당성(40) 입력'] = ''
    rank['근거 일치성(30) 입력'] = ''
    rank['형식(15) 자동'] = (rank['200자 준수'] == '통과').astype(int)*15
    rank['속도(15) 자동'] = 0.0
    for issue, group in rank.groupby('이슈키워드'):
        valid = group[(group['200자 준수'] == '통과') & (pd.to_numeric(group['LLM 초'], errors='coerce') > 0)]
        if len(valid):
            minimum = pd.to_numeric(valid['LLM 초']).min()
            for idx in valid.index:
                rank.loc[idx, '속도(15) 자동'] = round(minimum/float(rank.loc[idx,'LLM 초'])*15, 2)
    rank['총점(100)'] = '미평가'
    rank['키워드 내 순위'] = '미평가'
    rank['평가 비고'] = '수동 정확성 평가 전; 자동 형식/속도 점수만 산정'
    rank = rank[['이슈키워드','모델','상태','예측 타당성(40) 입력',
        '근거 일치성(30) 입력','형식(15) 자동','속도(15) 자동','총점(100)',
        '키워드 내 순위','LLM 초','200자 준수','평가 비고']]
    rules = pd.DataFrame([
        ['분석방식','26번: 24번 관련성 자동분류 + 25번 시계열 + Ollama 3모델; Chroma 재검색 없음'],
        ['비교 범위','청년 일자리 1개 이슈만 분석 자료 확보; 다른 4개 이슈는 미실험'],
        ['예측 타당성','0~40점 수동 검토; 아직 미평가'],
        ['근거 일치성','0~30점 수동 검토; 원문 JSON 대조 전 확정 금지'],
        ['형식 준수','원래 모델 응답의 전망 문장 200자 이하이면 15점'],
        ['속도','같은 이슈에서 유효 출력 모델 최단 LLM 시간/모델 LLM 시간*15'],
        ['총점','수동 점수(40+30) 확보 후 형식(15)+속도(15)와 합산'],
        ['등장시점','2024~25 보유 자료 내 자동 직접 관련 후보 최초 관측; 기본 RAG는 검색표본 최초'],
        ['지역 흐름','지역 간 유사 논의 자동 후보; 실제 정책 전파 아님'],
        ['처리시간','26번은 기계학습/임베딩/DB 검색 선행완료: 질문임베딩/DB검색=0(재사용), 사전처리 비용 별도'],
        ['입력 차이','기본 RAG는 Top-K 검색표본; 26번은 관련성+시계열+원문 근거. 완전히 같은 입력 아님'],
        ['검증상태','자동 후보이며 수동 확인/원본 JSON 대조 0건'],
    ], columns=['항목','설명 / 채점 원칙'])
    path = dest/'개인_LLM_시계열강화RAG_평가.xlsx'
    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        frame.to_excel(writer, sheet_name='모델결과', index=False)
        rules.to_excel(writer, sheet_name='평가기준', index=False)
        rank.to_excel(writer, sheet_name='키워드별순위', index=False)
        ev_frame.to_excel(writer, sheet_name='근거회의록', index=False)
        for sheet in writer.sheets.values():
            sheet.freeze_panes(1, 0)
            sheet.set_column(0, 2, 22)
            sheet.set_column(3, 22, 25)
    log(f'Excel 저장: {path.name}')


def run():
    ap = argparse.ArgumentParser(description='26번 개정: 개인 기본 RAG와 동일 양식의 시계열 강화 LLM 3모델 비교')
    ap.add_argument('--input-dir', default=DEFAULT_INPUT)
    ap.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    ap.add_argument('--timeout', type=int, default=360)
    ap.add_argument('--num-predict', type=int, default=700)
    ap.add_argument('--retries', type=int, default=1, help='잘못된 JSON/빈 전망 시 오류별 수정 지시로 추가 시도 횟수')
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--evidence-n', type=int, default=12)
    ap.add_argument('--models', nargs='+', default=MODELS)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--output-dir', default='', help='지정하지 않으면 llm_model_comparison_26_v4 별도 디렉터리')
    ap.add_argument('--baseline-xlsx', default='', help='선택: 기존 개인_LLM_RAG_평가누적.xlsx 경로')
    args = ap.parse_args()
    begun_at = datetime.now()
    global_start = time.perf_counter()
    root = Path(args.input_dir)
    dest = Path(args.output_dir) if args.output_dir else root / 'llm_model_comparison_26_v4'
    dest.mkdir(parents=True, exist_ok=True)
    log('=== 26 V4: 3모델 시계열 강화 RAG 비교 시작 ===')
    t = time.perf_counter()
    data = build_input(root, max(3, min(args.evidence_n, 24)))
    system, user = prompt_parts(data)
    first = data['첫_자동직접후보']
    first_text = f"{first['날짜']} / {first['의회']} (자동후보·수동미검증)"
    flow_text = pick_flow(data)
    safe_save_json(dest/'동일입력_근거_시계열.json', data)
    safe_save_json(dest/'모델입력_압축본.json', compact_for_llm(data))
    safe_save_json(dest/'응답_JSON_Schema.json', OUTPUT_SCHEMA)
    (dest/'동일프롬프트.txt').write_text('SYSTEM\n'+system+'\n\nUSER\n'+user, encoding='utf-8')
    pd.DataFrame(data['근거']).to_csv(dest/'LLM_입력_근거목록.csv', index=False, encoding='utf-8-sig')
    prep_sec = round(time.perf_counter()-t, 3)
    log(f'[1/3] 입력 준비: {len(data["월별_직접후보"])}개월, 근거 {len(data["근거"])}개 / {prep_sec:.3f}초')
    if args.dry_run:
        log(f'DRY-RUN 정상: 모델 호출 없음. 결과 경로: {dest}')
        return 0
    try:
        installed = available_models(args.ollama_url)
    except Exception as exc:
        log(f'Ollama 연결 실패: {exc}')
        return 2
    absent = [name for name in args.models if name not in installed]
    if absent:
        log(f'설치 누락: {absent}. ollama list 확인')
        return 2
    log('[2/3] 모델 준비: '+', '.join(args.models))
    experiment_id = begun_at.strftime('%Y%m%d_%H%M%S')
    rows, evrows, details = [], [], []
    allowed = {x['id'] for x in data['근거']}
    for i, model in enumerate(args.models, start=1):
        t = time.perf_counter()
        log(f'  [{i}/{len(args.models)}] {model} 시작')
        raw, parsed, error, metrics = '', None, '', {}
        attempts = []
        # 세 모델 모두 같은 프롬프트·조건·재시도 정책을 적용한다.
        for attempt in range(max(0, args.retries) + 1):
            try:
                response = request_json(args.ollama_url, '/api/chat', {
                    'model': model, 'stream': False,
                    'messages': ([{'role':'system','content':system}, {'role':'user','content':user}]
                                 if attempt == 0 else retry_messages(system,user,raw,error)),
                    'format':OUTPUT_SCHEMA, 'think':False, 'keep_alive':'5m',
                    'options':{'temperature':args.temperature,'num_predict':args.num_predict,'seed':42}
                }, args.timeout)
                message = response.get('message') or {}
                raw = str(message.get('content') or '')
                parsed = parse_model_response(raw)
                metrics = {key:response.get(key) for key in (
                    'total_duration','load_duration','prompt_eval_count','prompt_eval_duration','eval_count','eval_duration')}
                error = ''
                if not isinstance(parsed,dict):
                    error = 'JSON 파싱 실패'
                elif not parsed.get('forecast'):
                    error = 'forecast 빈 문자열 또는 누락'
                elif len(parsed['forecast']) > 200:
                    error = 'forecast 200자 초과'
                elif not evaluate(parsed, allowed)['required_fields_present']:
                    error = '필수 필드 누락 또는 빈 값'
                elif not evaluate(parsed, allowed)['evidence_ids_valid']:
                    error = '근거 ID 유효성 실패'
                elif not evaluate(parsed, allowed)['caution_word_present']:
                    error = '조건부 전망 표현 누락'
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            attempts.append({'attempt':attempt+1, 'raw':raw, 'parsed':parsed,
                             'error':error, 'metrics':metrics})
            if not error:
                break
            if attempt < args.retries:
                log(f'     파싱/검증 실패: {error} → 오류별 수정 지시로 재시도 {attempt+2}/{args.retries+1}')
        # 원본 응답은 잘리거나 덮어쓰이지 않도록 각 시도 모두 저장한다.
        llm_sec = round(time.perf_counter()-t, 3)
        checks = evaluate(parsed, allowed)
        forecast = parsed.get('forecast','') if isinstance(parsed,dict) else ''
        if not isinstance(forecast,str): forecast = ''
        valid = bool(forecast) and len(forecast)<=200 and not error
        state = ('OVER_200' if forecast and len(forecast)>200 else
                 'ERROR' if error or not isinstance(parsed,dict) or not forecast else 'OK')
        used_ids = parsed.get('evidence_ids',[]) if isinstance(parsed,dict) else []
        evidence_text = ' | '.join(f"{e['날짜']} {e['의회']}: {e['회의록']} ({e['chunk_id']})" for e in data['근거'])
        output_tokens = int(metrics.get('eval_count') or 0)
        eval_duration = (metrics.get('eval_duration') or 0)/1e9
        speed = round(output_tokens/eval_duration,2) if eval_duration>0 else None
        load_sec = round((metrics.get('load_duration') or 0)/1e9,3)
        row = {'실험':'26_시계열강화RAG_개인', '키워드':data['이슈'], '모델':model,
            '상태(200자)':state, '등장시점(검색표본)': first_text,
            '타 지방의회 논의순서(검색표본)': flow_text,
            '향후 예측(앞 200자 표시)':forecast[:200], '원래 예측 전체':forecast,
            '실제 글자 수':len(forecast), '200자 준수':'통과' if valid and state=='OK' else '미준수/오류',
            '근거 회의록(텍스트 요약)':evidence_text, '근거 수':len(data['근거']),
            '질문 임베딩(초)':0.0, 'DB 검색(초)':0.0, 'LLM(초)':llm_sec,
            '전체(초)':round(llm_sec+prep_sec/len(args.models),3),
            '모델 로드(초)':load_sec, '출력 토큰':output_tokens, '토큰/초':speed,
            '오류':error or ('JSON 파싱 실패' if not isinstance(parsed,dict) else ''),
            '실행시각':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '검색근거검증':'24/25번 자동후보·수동미검증/원본JSON미대조',
            '실험ID':f'{experiment_id}_{i:02d}'}
        rows.append(row)
        evrows.extend(build_evidence_rows(data['근거'],model,row['실험ID']))
        details.append({'model':model,'raw':raw,'parsed':parsed,'metrics':metrics,
                        'checks':checks,'used_evidence_ids':used_ids, 'error':error,'attempts':attempts})
        model_code = re.sub(r'[^a-zA-Z0-9]+','_',model)
        safe_save_json(dest/f'{model_code}_결과.json',details[-1])
        (dest/f'{model_code}_원시응답.txt').write_text('\n\n'.join(f"=== 시도 {a['attempt']} / {a['error']} ===\n{a['raw']}" for a in attempts),encoding='utf-8')
        pd.DataFrame(rows,columns=BASE_FIELDS).to_csv(dest/'모델결과_26.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(evrows,columns=EVIDENCE_FIELDS).to_csv(dest/'근거회의록_26.csv',index=False,encoding='utf-8-sig')
        log(f'     완료 {llm_sec:.2f}초, {len(forecast)}자, 상태 {state}, {speed or 0} tok/s')
    export_workbook(dest, rows, evrows)
    if args.baseline_xlsx:
        try:
            baseline = pd.read_excel(args.baseline_xlsx, sheet_name='모델결과')
            compared = pd.concat([baseline.reindex(columns=BASE_FIELDS),
                pd.DataFrame(rows,columns=BASE_FIELDS)],ignore_index=True)
            compared.to_csv(dest/'기본RAG_vs_시계열강화_동일양식.csv',index=False,encoding='utf-8-sig')
            log(f'기존 RAG 비교표 병합: {len(baseline)}+{len(rows)}행')
        except Exception as exc:
            log(f'기존 RAG 병합 생략: {exc}')
    summary = {'experiment':'26_시계열강화RAG_개인_V4','query':data['이슈'],
        'observed_months':len(data['월별_직접후보']), 'models':args.models,
        'prompt_identical':True, 'temperature':args.temperature,
        'max_prediction_chars':200,'manually_verified_evidence_count':0,'max_retries':args.retries,'schema_constrained':True,'input_compacted':True,
        'preparation_seconds':prep_sec, 'total_elapsed_seconds':round(time.perf_counter()-global_start,3),
        'results':[{k:r.get(k) for k in ['모델','상태(200자)','실제 글자 수','LLM(초)','모델 로드(초)','출력 토큰','토큰/초','오류']} for r in rows],
        'caveats':['기존 RAG 검색표본 최초 관측과 이번 전체 자동후보 최초 관측은 직접 동등한 측정치가 아님',
            '질문 임베딩·DB 검색 시간 0은 해당 단계 재실행하지 않고 사전 결과를 활용했음을 의미',
            '청년 일자리 이외 4개 이슈는 전처리 결과 미제공으로 평가하지 않음',
            '자동 관련성 후보와 지역 간 유사 후보는 수동 검증되지 않음',
            '형식 준수와 처리 속도만 자동 채점. 사실성·예측 타당성은 미평가']}
    safe_save_json(dest/'비교_요약.json',summary)
    log(f'[3/3] 종료: 전체 {summary["total_elapsed_seconds"]:.2f}초 / {dest}')
    return 0 if all(r['상태(200자)']=='OK' for r in rows) else 1


if __name__ == '__main__':
    try:
        sys.exit(run())
    except Exception as exc:
        log(f'[중단] {type(exc).__name__}: {exc}')
        sys.exit(2)

# -*- coding: utf-8 -*-
"""28 | 27번 5개 이슈 실험 결과를 Gradio용 JSON으로 통합.

Windows / 개인 학원 PC C:\\mini_pj 전용. Ollama/Chroma/임베딩 호출 없음.

실행:
  python C:\\mini_pj\\scripts\\28_policy_result_integration.py
  python C:\\mini_pj\\scripts\\28_policy_result_integration.py --run-dir "C:\\mini_pj\\results\\rag_vs_timeline_5issues\\20261026_012345"

입력 (27번 완료 폴더):
  5질문_3모델_실험결과.csv
  <이슈>/시계열강화_근거.json
  <이슈>/기본RAG_검색근거.json
옵션 입력 (15, 24, 25번):
  results/policy_issue_search/<실행폴더>/전체_어휘일치_근거.csv
  .../relevance_validation/전체_정책관련성_검토결과.csv
  .../timeline_analysis/의정활동_원문언급_검토후보.csv
  .../timeline_analysis/지역간_후속논의_검토후보.csv

출력: results/policy_result_integration/<시각>/
  이슈별 JSON (각 1개), all_issues.json, manifest.json,
  통합_검토표.csv, 비교_모델요약.csv

중요: 최초 관측은 보유한 2024-25 회의록에서 발견한 자동 후보일 뿐,
정책 실제 최초 도입·재논의·공청회 개최·지역 간 인과적 확산을 뜻하지 않음.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DEFAULT_ROOT = Path(r'C:\mini_pj')
ISSUES = ['지역소멸', '청년 일자리', '저출생', '고령화', '기후위기']
MODELS = ['exaone3.5:2.4b', 'qwen3:8b', 'gemma3:4b']
SOURCE_CSV = '5질문_3모델_실험결과.csv'
MAX_FLOW = 17
MAX_EVIDENCE = 12


def log(s):
    print(f'[{datetime.now():%H:%M:%S}] {s}', flush=True)


def safe(s):
    return re.sub(r'[^0-9a-zA-Z가-힣]+', '_', str(s)).strip('_')


def text(v):
    if v is None: return ''
    s = str(v).strip()
    return '' if s.lower() in ('nan','none','nat','null') else s


def date(s):
    s = text(s)
    digits = re.sub(r'\D', '', s)
    if len(digits) >= 8:
        stamp = f'{digits[:4]}-{digits[4:6]}-{digits[6:8]}'
        try:
            datetime.strptime(stamp, '%Y-%m-%d')
            return stamp
        except ValueError:
            pass
    return ''


def load_json(path, default=None):
    if not path.is_file(): return default
    with path.open('r', encoding='utf-8-sig') as f:
        return json.load(f)


def load_csv(path):
    if not path.is_file(): return []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')


def save_csv(path, records, fieldnames):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def latest_27_run(root):
    base = root / 'results' / 'rag_vs_timeline_5issues'
    runs = sorted((p for p in base.glob('*') if p.is_dir() and (p/SOURCE_CSV).exists()),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f'27번 결과 CSV를 찾지 못함: {base}\\<실행시각>\\{SOURCE_CSV}')
    # 최근 실행이 부분 실행이라면 이를 숨기지 않고 명시적으로 선택한다.
    return runs[0]


def search_result_dir(root, issue):
    base = root / 'results' / 'policy_issue_search'
    if not base.is_dir(): return None
    folders = [p for p in base.iterdir() if p.is_dir() and p.name.endswith('_'+safe(issue))
               and (p/'전체_어휘일치_근거.csv').exists()]
    return max(folders, key=lambda p:p.stat().st_mtime) if folders else None


def normalize_evidence(row, scope, fallback_id=''):
    excerpt = text(row.get('core_evidence_text_100') or row.get('핵심구간_100자') or
                   row.get('text_preview') or row.get('text') or row.get('원문'))
    # 100자 미만의 발췌가 원문에서 실제 연속 등장하는지 확인 (있을 때만).
    full = text(row.get('text_preview') or row.get('원문') or row.get('text'))
    excerpt = excerpt[:100]
    literal = (excerpt in full) if full and excerpt else None
    return {'id': text(row.get('chunk_id') or row.get('id') or fallback_id),
            'source_id': text(row.get('source_id')),
            'date': date(row.get('date') or row.get('날짜')),
            'assembly': text(row.get('assembly') or row.get('의회')),
            'meeting_title': text(row.get('meeting_title') or row.get('회의록')),
            'excerpt_100': excerpt,
            'source_scope': scope,
            'excerpt_contiguous_in_available_text': literal,
            'verification_status': '자동추출_원문수동미검증'}


def select_evidence(source_rows, enhanced_evidence, limit=MAX_EVIDENCE):
    """초기 관측 자료 + 지역 다양성 중심. 동일 회의 중복 청크는 축소."""
    if source_rows:
        source = [normalize_evidence(r,'원본_검색결과') for r in source_rows]
    else:
        source = [normalize_evidence(r,'27번_압축근거', f'27_E{i:02d}')
                  for i,r in enumerate(enhanced_evidence,1)]
    source = [r for r in source if r['date'] and r['assembly'] and r['excerpt_100']]
    source.sort(key=lambda r:(r['date'],r['assembly'],r['id']))
    selected, selected_keys, selected_regions = [], set(), set()
    for r in source:
        k=(r['assembly'],r['date'],r['meeting_title'])
        if k in selected_keys: continue
        if len(selected) < min(3,limit) or r['assembly'] not in selected_regions:
            selected.append(r);selected_keys.add(k);selected_regions.add(r['assembly'])
        if len(selected)>=limit:break
    if len(selected)<limit:
        for r in source:
            k=(r['assembly'],r['date'],r['meeting_title'])
            if k not in selected_keys:
                selected.append(r);selected_keys.add(k)
            if len(selected)>=limit:break
    for idx,r in enumerate(selected,1):r['display_id']=f'E{idx:02d}'
    return selected


def make_flow(source_rows, enhanced_data, limit=MAX_FLOW):
    """각 지역 최초 관측만 보여준다. 지역 간 확산 경로로 해석하지 않는다."""
    base=[]
    for r in source_rows:
        dt=date(r.get('date')); assembly=text(r.get('assembly'))
        if dt and assembly:
            base.append((dt, assembly, r))
    base.sort(key=lambda x:(x[0],x[1]))
    result=[];seen=set()
    for dt,assembly,r in base:
        if assembly in seen: continue
        seen.add(assembly)
        result.append({'date':dt,'assembly':assembly,'activity_label':'논의 관측',
                       'activity_status':'어휘·자동분류 기반 관측; 의정활동 유형 미확정',
                       'meeting_title':text(r.get('meeting_title')),
                       'chunk_id':text(r.get('chunk_id')),
                       'evidence_100':text(r.get('core_evidence_text_100') or r.get('text_preview'))[:100]})
        if len(result)>=limit:break
    if result: return result
    # 원본 검색 CSV 없는 경우 27번 강화 결과(first/flow)를 사용하되 근거 수준을 낮춘다.
    flow=text(enhanced_data.get('flow'))
    for part in flow.split('→'):
        hit=re.match(r'\s*(\d{4}-\d{2}-\d{2})\s*/?\s*(.+?)\s*$',part)
        if hit:
            result.append({'date':hit.group(1),'assembly':hit.group(2),
                           'activity_label':'논의 관측','activity_status':'27번_자동_관측',
                           'meeting_title':'','chunk_id':'','evidence_100':''})
    return sorted(result,key=lambda x:(x['date'],x['assembly']))[:limit]


def first_observation(flow, enhanced_data):
    if flow:
        head=flow[0]
        return {'date':head['date'],'assembly':head['assembly'],
                'meeting_title':head.get('meeting_title',''),
                'chunk_id':head.get('chunk_id',''),
                'verification_status':'보유 자료 내 자동 관측; 수동 미검증'}
    m=re.match(r'\s*(\d{4}-\d{2}-\d{2})\s*/\s*(.+)', text(enhanced_data.get('first')))
    if m:
        return {'date':m.group(1),'assembly':m.group(2),
                'meeting_title':'','chunk_id':'',
                'verification_status':'27번_자동관측; 원문 연결 재확인 필요'}
    return None


def policy_events(timeline_dir, issue, max_rows=30):
    if issue!='청년 일자리' or not timeline_dir: return []
    src=load_csv(timeline_dir/'의정활동_원문언급_검토후보.csv')
    out=[];seen=set()
    for r in sorted(src,key=lambda a:date(a.get('date'))):
        key=(date(r.get('date')),text(r.get('assembly')),text(r.get('meeting_title')),
             text(r.get('의정활동_언급후보')))
        if not key[0] or key in seen:continue
        seen.add(key)
        out.append({'date':key[0],'assembly':key[1],
                    'activity_cue':key[3], 'activity_status':'표현 검출 후보 / 실제 실시 미확인',
                    'meeting_title':key[2], 'chunk_id':text(r.get('chunk_id')),
                    'evidence_100':text(r.get('근거_100자'))[:100]})
        if len(out)>=max_rows:break
    return out


def cross_region_candidates(timeline_dir, issue):
    if issue!='청년 일자리' or not timeline_dir:return []
    rows=load_csv(timeline_dir/'지역간_후속논의_검토후보.csv')
    out=[]
    for r in rows:
        out.append({'candidate_id':text(r.get('candidate_id')),
                    'earlier_date':date(r.get('earlier_date')),
                    'earlier_assembly':text(r.get('earlier_assembly')),
                    'later_date':date(r.get('later_date')),
                    'later_assembly':text(r.get('later_assembly')),
                    'common_policy_mechanisms':text(r.get('specific_common_mechanisms')),
                    'status':'자동 유사 논의 후보 / 인과적 전파 미확인',
                    'earlier_chunk_id':text(r.get('earlier_chunk_id')),
                    'later_chunk_id':text(r.get('later_chunk_id'))})
    return sorted(out,key=lambda x:(x['earlier_date'],x['later_date']))


def all_model_results(rows, issue):
    result=[]
    for r in rows:
        if text(r.get('질문'))!=issue:continue
        try: sec=float(text(r.get('LLM(초')))
        except ValueError: sec=None
        forecast=text(r.get('향후 예측'))
        valid= text(r.get('상태'))=='OK' and 0<len(forecast)<=200
        result.append({'model':text(r.get('모델')),'method':text(r.get('방식')),
                       'status':text(r.get('상태')),'forecast_200':forecast,
                       'forecast_length':len(forecast),
                       'under_200':valid,'llm_seconds':sec,
                       'error':text(r.get('오류')),
                       'verification_status':text(r.get('근거 검증 상태'))})
    return result


def choose_forecast(model_results, preferred):
    candidates=[r for r in model_results if r['method']=='시계열 강화 RAG' and r['under_200']]
    if not candidates:
        return {'status':'사용 가능 전망 없음','model':None,'forecast_150':'',
                'forecast_200_original':'', 'needs_150_rewrite':False,
                'verification_status':'자동 생성 전망; 정책 예측 성능 미검증'}
    # 모델 정확도 순위가 아닌 운영상 '선호 모델' 우선. 임의로 전망 품질 점수를 만들지 않음.
    candidates.sort(key=lambda r:(r['model']!=preferred, MODELS.index(r['model']) if r['model'] in MODELS else 99))
    chosen=candidates[0];full=chosen['forecast_200']
    # 150자를 넘을 경우 잘라내지 않는다. 핵심 의미가 잘릴 수 있기 때문.
    return {'status':'표시 가능' if len(full)<=150 else '150자 재작성 필요',
            'model':chosen['model'],'forecast_150':full if len(full)<=150 else '',
            'forecast_200_original':full,'forecast_length':len(full),
            'needs_150_rewrite':len(full)>150,
            'verification_status':'LLM 조건부 전망 / 근거·예측 정확도 수동 미검증'}


def make_issue(issue, run_dir, root, run_rows, preferred):
    issue_dir=run_dir/safe(issue)
    enhanced=load_json(issue_dir/'시계열강화_근거.json',{}) or {}
    if not enhanced:log(f'  주의: {issue} 27번 강화 근거 JSON 누락')
    old_search=search_result_dir(root,issue)
    csv_path=old_search/'전체_어휘일치_근거.csv' if old_search else None
    original=load_csv(csv_path) if csv_path else []
    scope='어휘일치_수동미검증'
    if issue=='청년 일자리' and old_search:
        rel_path=old_search/'relevance_validation'/'전체_정책관련성_검토결과.csv'
        rel=load_csv(rel_path)
        direct=[r for r in rel if text(r.get('review_candidate_label'))=='직접관련_후보']
        if direct:original=direct;scope='직접관련_자동후보_수동미검증'
    flow=make_flow(original,enhanced)
    first=first_observation(flow,enhanced)
    ev=select_evidence(original,enhanced.get('evidence',[]))
    timeline_dir=old_search/'timeline_analysis' if old_search else None
    forecasts=all_model_results(run_rows,issue)
    predicted=choose_forecast(forecasts,preferred)
    warnings=[]
    if not original:warnings.append('원본 어휘일치 CSV를 찾지 못해 27번 압축 근거를 이용함')
    if not enhanced:warnings.append('27번 시계열 강화 입력 파일이 없어 월별 통계가 비어 있을 수 있음')
    if not ev:warnings.append('출처가 연결된 근거 회의록이 없음')
    if not predicted['forecast_150']:warnings.append('150자 이내 검증되지 않은 전망을 임의로 자르거나 생성하지 않음')
    res={'schema_version':'1.0','issue':issue,
         'first_observation':first,
         'regional_discussion_flow':flow,
         'policy_activity_cues':policy_events(timeline_dir,issue),
         'cross_region_similarity_candidates':cross_region_candidates(timeline_dir,issue),
         'forecast':predicted,'evidence_records':ev,
         'monthly_timeline':enhanced.get('timeline',[]),
         'all_model_results':forecasts,
         'data_quality':{'observation_scope':scope,
                         'source_period':'2024~2025년 보유 자료',
                         'manual_verified':False,
                         'causal_policy_transfer_verified':False,
                         'actual_policy_first_adoption_verified':False,
                         'limitations':[
                             '최초 관측은 보유 청크 중 자동 검색·분류의 첫 관측으로 실제 정책 도입 최초가 아님',
                             '지역별 관측 순서는 시간적 나열이지 지역 간 정책 전파 관계가 아님',
                             '활동 유형 후보는 표현 탐지이며 공청회 실시·재논의·조례 통과 확정이 아님',
                             'LLM 생성 전망은 미래 실제 결과에 대한 예측 정확도가 검증되지 않음',
                             '청년 일자리 외 4개 질문은 관련성 분류기가 적용되지 않은 어휘일치 결과'
                         ]},
         'provenance':{'27_run':str(run_dir),'issue_input':str(issue_dir),
                       'optional_search_source':str(old_search) if old_search else None},
         'warnings':warnings}
    return res


def main():
    p=argparse.ArgumentParser(description='27번 모델·시계열 결과 → 28번 Gradio 정책 결과 통합 JSON')
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--run-dir',type=Path,default=None,help='27번 실행 결과 폴더(미지정: 최신 폴더)')
    p.add_argument('--output-dir',type=Path,default=None)
    p.add_argument('--preferred-model',choices=MODELS,default='qwen3:8b',
                   help='내용 정확도 1위 의미가 아닌 우선 표시 모델')
    args=p.parse_args();start=time.perf_counter()
    run_dir=args.run_dir or latest_27_run(args.root)
    source=run_dir/SOURCE_CSV
    if not source.is_file():raise FileNotFoundError(f'27번 결과 CSV 없음: {source}')
    rows=load_csv(source)
    out=args.output_dir or args.root/'results'/'policy_result_integration'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=True)
    log(f'=== 28 정책 결과 통합 시작 === 27번 원본 {len(rows)}행')
    if len(rows)!=30:log(f'주의: 기대 30행이나 실제 {len(rows)}행. 누락 결과는 생성하지 않음')
    issues={};review=[];models=[]
    for i,issue in enumerate(ISSUES,1):
        t=time.perf_counter()
        data=make_issue(issue,run_dir,args.root,rows,args.preferred_model)
        issues[issue]=data
        save_json(out/f'{i:02d}_{safe(issue)}.json',data)
        first=data['first_observation'] or {}
        fc=data['forecast'];review.append({
            '질문':issue,'최초관측일':first.get('date',''),'최초관측의회':first.get('assembly',''),
            '관측_검증수준':data['data_quality']['observation_scope'],
            '지역별관측수':len(data['regional_discussion_flow']),
            '활동표현후보':len(data['policy_activity_cues']),
            '지역간유사후보':len(data['cross_region_similarity_candidates']),
            '전망모델':fc['model'] or '','전망150자':fc['forecast_150'],
            '전망상태':fc['status'],'근거회의록수':len(data['evidence_records']),
            '주의':'; '.join(data['warnings'])})
        for m in data['all_model_results']:
            models.append({'질문':issue,'모델':m['model'],'방식':m['method'],
                           '상태':m['status'],'예측글자수':m['forecast_length'],
                           '200자준수':m['under_200'],'LLM초':m['llm_seconds'],
                           '전망원문':m['forecast_200'],'오류':m['error']})
        log(f'[{i}/5] {issue}: 지역 {len(data["regional_discussion_flow"])}개, 근거 {len(data["evidence_records"])}개, '+
            f'전망 {fc["status"]}, {time.perf_counter()-t:.2f}초')
    save_json(out/'all_issues.json',issues)
    save_csv(out/'통합_검토표.csv',review,
             ['질문','최초관측일','최초관측의회','관측_검증수준','지역별관측수','활동표현후보',
              '지역간유사후보','전망모델','전망150자','전망상태','근거회의록수','주의'])
    save_csv(out/'비교_모델요약.csv',models,
             ['질문','모델','방식','상태','예측글자수','200자준수','LLM초','전망원문','오류'])
    manifest={'created_at':datetime.now().isoformat(timespec='seconds'),
              'source_27_csv':str(source),'output_dir':str(out),
              'issues':ISSUES,'expected_model_results':30,'actual_model_results':len(models),
              'integrated_issue_count':len(issues),
              'forecast_display_ready':sum(bool(x['forecast']['forecast_150']) for x in issues.values()),
              'elapsed_seconds':round(time.perf_counter()-start,3),
              'notice':'자동 검색·관측 결과. 인과적 확산·공청회 개최·정책 채택 확정 금지'}
    save_json(out/'manifest.json',manifest)
    log(f'=== 28 완료: {manifest["elapsed_seconds"]:.2f}초 === 결과: {out}')
    log(f'Gradio용: {out/"all_issues.json"}')

if __name__=='__main__':
    main()

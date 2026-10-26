# -*- coding: utf-8 -*-
"""28 V2: 5개 정책 이슈 통합 결과의 근거 품질 보완 + 25번 흐름 후보 연결.

재임베딩, Chroma, Ollama 호출 없음. 기존 28번 출력 보존.

PowerShell:
 & "C:\\Users\\Admin\\miniconda3\\envs\\mini\\python.exe" "C:\\mini_pj\\scripts\\28_policy_result_integration_v2.py"

선택 옵션:
 --prior-json "C:\\mini_pj\\results\\policy_result_integration\\<28번시각>\\all_issues.json"
 --timeline-dir "C:\\mini_pj\\results\\policy_issue_search\\20261025_000801_청년_일자리\\timeline_analysis"
 --relevance-csv "C:\\mini_pj\\results\\policy_issue_search\\20261025_000801_청년_일자리\\relevance_validation\\전체_정책관련성_검토결과.csv"
 --output-dir "C:\\mini_pj\\results\\policy_result_integration_v2\\test"

추가로 27번 이슈별 전체_어휘일치_근거.csv가 남아 있으면 해당 파일의 text_preview 활용.
원문/문맥 부재 항목은 '근거 부족'으로 숨김. 최초 정책 도입·지역 간 전파·공청회 개최 확정 금지.
"""
from __future__ import annotations
import argparse, csv, json, re, time
from copy import deepcopy
from pathlib import Path
from datetime import datetime

ROOT = Path(r'C:\mini_pj')
ISSUES = ['지역소멸','청년 일자리','저출생','고령화','기후위기']
TERMS = {
 '지역소멸':['지역소멸','지방소멸','인구소멸','인구감소지역','소멸위기','인구유출'],
 '청년 일자리':['청년 일자리','청년일자리','청년 고용','청년고용','청년 취업','청년취업','청년 창업','청년창업','청년 구직','청년구직','취업 지원','취업지원'],
 '저출생':['저출생','저출산','출생률','출산율','출산 지원','출산지원','양육지원'],
 '고령화':['고령화','초고령','고령사회','고령 인구','노인 인구','노인돌봄','고령친화'],
 '기후위기':['기후위기','기후변화','탄소중립','온실가스','기후재난','에너지전환'],
}
NON_POLICY = ['위원님 수고하셨습니다','더 이상 질의하실','질의하실 또 다른','경청해 주셔서 감사합니다',
              '위원님 여러분, 임시회','다음은 전문위원 검토보고','이의 없으십니까','의사진행발언하겠습니다']

def now(): return datetime.now().strftime('%H:%M:%S')
def log(s): print(f'[{now()}] {s}',flush=True)
def safe(s): return re.sub(r'[^0-9a-zA-Z가-힣]+','_',str(s)).strip('_')
def clean(s):
    if s is None:return ''
    s=str(s).strip()
    return '' if s.lower() in ('nan','null','none') else s

def dt(s):
    x=re.sub('[^0-9]','',clean(s))
    if len(x)<8:return ''
    s=f'{x[:4]}-{x[4:6]}-{x[6:8]}'
    try:datetime.strptime(s,'%Y-%m-%d');return s
    except ValueError:return ''

def read_json(path,default=None):
    if not path or not Path(path).is_file():return default
    with open(path,encoding='utf-8-sig') as f:return json.load(f)
def read_csv(path):
    if not path or not Path(path).is_file():return []
    with open(path,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def save_csv(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def latest_file(folder,name):
    if not folder.is_dir():return None
    files=sorted(folder.glob(f'*/{name}'),key=lambda p:p.stat().st_mtime,reverse=True)
    return files[0] if files else None

def find_issue_search(root,issue):
    base=root/'results'/'policy_issue_search'
    if not base.is_dir():return None
    dirs=sorted((p for p in base.iterdir() if p.is_dir() and p.name.endswith('_'+safe(issue))),
                key=lambda p:p.stat().st_mtime,reverse=True)
    return next((p for p in dirs if (p/'전체_어휘일치_근거.csv').is_file() or
                 (p/'relevance_validation'/'전체_정책관련성_검토결과.csv').is_file()),None)

def source_rows(root,issue,search_dir,relevance_override=None):
    if issue=='청년 일자리':
        p=Path(relevance_override) if relevance_override else (search_dir/'relevance_validation'/'전체_정책관련성_검토결과.csv' if search_dir else None)
        data=read_csv(p)
        if data:
            direct=[r for r in data if clean(r.get('review_candidate_label'))=='직접관련_후보']
            if direct:return direct,'직접관련_자동후보_수동미검증',str(p)
    path=search_dir/'전체_어휘일치_근거.csv' if search_dir else None
    return read_csv(path),'어휘일치_수동미검증',str(path) if path else ''

def get_fulltext(row):
    return clean(row.get('text_preview') or row.get('원문') or row.get('text') or row.get('utterance'))

def extract_span(full,issue,max_chars=100):
    """반드시 제공된 원문에서 연속 100자 발췌. 관련 표현이 없으면 추출 실패."""
    if not full:return '', '', '원문 없음'
    terms=TERMS[issue]
    matches=[]
    for term in terms:
        for m in re.finditer(re.escape(term),full,re.I):
            pos=m.start(); left=max(0,pos-35); right=min(len(full),left+max_chars)
            left=max(0,right-max_chars)
            snippet=full[left:right]
            # 회의 진행 구절 중심의 스팬에 낮은 우선순위
            bad=sum(x in snippet for x in NON_POLICY)
            matches.append((bad, -len(term),pos,snippet,term))
    if not matches:return '', '', '핵심 정책 표현이 제공된 원문에 없음'
    matches.sort(key=lambda r:(r[0],r[1],r[2]))
    _,_,_,span,term=matches[0]
    return span,term,'자동발췌_수동미검증'

def candidates_from_rows(rows,issue,scope):
    out=[]; seen=set()
    for r in rows:
        date=dt(r.get('date') or r.get('날짜'))
        assembly=clean(r.get('assembly') or r.get('의회'))
        title=clean(r.get('meeting_title') or r.get('회의록'))
        ident=clean(r.get('chunk_id') or r.get('id'))
        if not date or not assembly:continue
        span,term,status=extract_span(get_fulltext(r),issue)
        if not span:continue
        key=(date,assembly,title,span)
        if key in seen:continue
        seen.add(key)
        out.append({'id':ident,'source_id':clean(r.get('source_id')),'date':date,
                    'assembly':assembly,'meeting_title':title,'excerpt_100':span,
                    'matched_policy_term':term,'excerpt_contiguous_in_available_text':True,
                    'source_scope':scope,'verification_status':status})
    out.sort(key=lambda x:(x['date'],x['assembly'],x['id']))
    return out

def candidates_from_prior(prior,issue):
    """원본 CSV 없이 이전 28 JSON만 이용할 경우 정책 표현이 실제 발췌에 보이는 자료만 유지."""
    out=[];seen=set()
    for p in prior.get('evidence_records',[]):
        span,term,status=extract_span(clean(p.get('excerpt_100')),issue)
        if not span:continue
        date=dt(p.get('date'));assembly=clean(p.get('assembly'))
        if not date or not assembly:continue
        row=deepcopy(p);row['excerpt_100']=span[:100];row['matched_policy_term']=term
        row['verification_status']='기존100자발췌_문맥부족_수동미검증';row['source_scope']='이전28_JSON'
        k=(date,assembly,row['excerpt_100'])
        if k not in seen:seen.add(k);out.append(row)
    return sorted(out,key=lambda r:(r['date'],r['assembly']))

def evidence_select(cands,limit=12):
    chosen=[];seen_meetings=set();seen_assembly=set()
    # 가장 이른 근거 1개 확보 → 다른 지역의 발언 우선
    for phase in (0,1):
        for r in cands:
            k=(r['date'],r['assembly'],r['meeting_title'])
            if k in seen_meetings or (phase==0 and r['assembly'] in seen_assembly):continue
            chosen.append(deepcopy(r));seen_meetings.add(k);seen_assembly.add(r['assembly'])
            if len(chosen)>=limit:break
        if len(chosen)>=limit:break
    for i,r in enumerate(chosen,1):r['display_id']=f'E{i:02d}'
    return chosen

def make_flow(cands,limit=17):
    result=[];seen=set()
    for r in cands:
        if r['assembly'] in seen:continue
        seen.add(r['assembly'])
        result.append({'date':r['date'],'assembly':r['assembly'],
            'activity_label':'논의 관측 후보','activity_status':'정책 표현 포함 원문 자동 탐지; 재논의/전파 미확정',
            'meeting_title':r['meeting_title'],'chunk_id':r['id'],
            'evidence_100':r['excerpt_100'],'matched_policy_term':r['matched_policy_term']})
        if len(result)>=limit:break
    return result

def policy_activities(path,issue,max_rows=30):
    if issue!='청년 일자리':return []
    out=[];seen=set()
    for row in sorted(read_csv(path),key=lambda x:dt(x.get('date'))):
        date=dt(row.get('date'));assembly=clean(row.get('assembly'))
        if not date or not assembly:continue
        label=clean(row.get('의정활동_언급후보'))
        evid=clean(row.get('근거_100자'))
        key=(date,assembly,clean(row.get('chunk_id')),label)
        if key in seen:continue
        seen.add(key)
        # 미검증 의정활동 표현은 별도 검토 후보만 저장; '실시'라고 표시하지 않음
        out.append({'date':date,'assembly':assembly,'activity_cue':label,
            'activity_status':'의정활동 언급 검토 후보 / 실제 실시·의안 동일성 미확인',
            'meeting_title':clean(row.get('meeting_title')),'chunk_id':clean(row.get('chunk_id')),
            'evidence_100':evid[:100]})
        if len(out)>=max_rows:break
    return out

def cross_candidates(path,issue):
    if issue!='청년 일자리':return []
    data=[]
    for row in read_csv(path):
        data.append({'candidate_id':clean(row.get('candidate_id')),
          'earlier_date':dt(row.get('earlier_date')),'earlier_assembly':clean(row.get('earlier_assembly')),
          'later_date':dt(row.get('later_date')),'later_assembly':clean(row.get('later_assembly')),
          'common_policy_mechanisms':clean(row.get('specific_common_mechanisms')),
          'content_judgment':clean(row.get('content_judgment')),
          'status':'자동 정책 유사성 후보 / 인과적 전파·동일 의안 미확인',
          'earlier_chunk_id':clean(row.get('earlier_chunk_id')),
          'later_chunk_id':clean(row.get('later_chunk_id'))})
    return sorted(data,key=lambda x:(x['earlier_date'],x['later_date'],x['candidate_id']))

def main():
    p=argparse.ArgumentParser(description='28 V2: 원문 관련 발췌 + 시계열 후보 연결; LLM 재실행 없음')
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--prior-json',type=Path,default=None,help='기존 28번 all_issues.json (기본: 최신 자동선택)')
    p.add_argument('--timeline-dir',type=Path,default=None,help='25번 timeline_analysis 폴더')
    p.add_argument('--relevance-csv',type=Path,default=None,help='24번 전체_정책관련성_검토결과.csv')
    p.add_argument('--output-dir',type=Path,default=None)
    args=p.parse_args();start=time.perf_counter()
    prior_path=args.prior_json or latest_file(args.root/'results'/'policy_result_integration','all_issues.json')
    if not prior_path or not prior_path.is_file():raise FileNotFoundError('28번 all_issues.json 없음. --prior-json 지정 필요')
    prior=read_json(prior_path,{})
    out=args.output_dir or args.root/'results'/'policy_result_integration_v2'/datetime.now().strftime('%Y%m%d_%H%M%S')
    if out.resolve()==prior_path.parent.resolve():raise ValueError('원본 28번 결과와 출력 경로가 같음: 덮어쓰기 방지')
    out.mkdir(parents=True,exist_ok=True)
    log(f'=== 28 V2 시작 === 기존 JSON: {prior_path}')
    result={};report=[]
    for i,issue in enumerate(ISSUES,1):
        t=time.perf_counter()
        item=deepcopy(prior.get(issue,{}))
        if not item:
            log(f'[{i}/5] {issue}: 기존 JSON에 없음 — 누락 표시')
            item={'schema_version':'2.0','issue':issue,'warnings':['기존 자료 없음']}
            result[issue]=item;save_json(out/f'{i:02d}_{safe(issue)}.json',item)
            continue
        search=find_issue_search(args.root,issue)
        rows,scope,origin=source_rows(args.root,issue,search,args.relevance_csv)
        cands=candidates_from_rows(rows,issue,scope) if rows else candidates_from_prior(item,issue)
        flow=make_flow(cands);ev=evidence_select(cands)
        old_first=item.get('first_observation')
        item['first_observation']=({'date':flow[0]['date'],'assembly':flow[0]['assembly'],
          'meeting_title':flow[0]['meeting_title'],'chunk_id':flow[0]['chunk_id'],
          'verification_status':'정책표현 포함 자동 최초 관측 / 수동 미검증'} if flow else None)
        item['regional_discussion_flow']=flow
        item['evidence_records']=ev
        timeline=args.timeline_dir or (search/'timeline_analysis' if search else None)
        item['policy_activity_cues']=policy_activities(timeline/'의정활동_원문언급_검토후보.csv' if timeline else None,issue)
        item['cross_region_similarity_candidates']=cross_candidates(timeline/'지역간_후속논의_검토후보.csv' if timeline else None,issue)
        item['schema_version']='2.0'
        quality=item.setdefault('data_quality',{})
        quality['observation_scope']=scope if rows else '기존28_JSON_발췌기반_문맥제한'
        quality['manual_verified']=False
        quality['actual_policy_first_adoption_verified']=False
        quality['causal_policy_transfer_verified']=False
        quality['evidence_extraction_method']='정책 표현 위치 기준 원문 연속 100자 추출; 미존재시 제외'
        quality['records_considered']=len(rows) if rows else len(item.get('evidence_records',[]))
        quality['policy_term_matched_records']=len(cands)
        item['previous_first_observation_v1']=old_first
        item['provenance_v2']={'prior_json':str(prior_path),'source_csv':origin,'timeline_dir':str(timeline) if timeline else None}
        warn=item.setdefault('warnings',[])
        if not rows:warn.append('원문 CSV 미발견: 28번의 기존 100자 발췌만 평가, 최초 관측 재확인 불가')
        if not flow:warn.append('관련 정책표현이 확인된 근거가 없어 최초 관측/흐름을 미표시함')
        if issue=='청년 일자리' and not item['cross_region_similarity_candidates']:
            warn.append('20/25번 지역 간 후보 CSV 연결 실패: --timeline-dir 지정 필요')
        if issue!='청년 일자리':warn.append('해당 이슈의 활동유형 분류/지역간 정책 유사성 검증 미수행')
        result[issue]=item
        save_json(out/f'{i:02d}_{safe(issue)}.json',item)
        first=item['first_observation'] or {}
        report.append({'이슈':issue,'구버전_최초관측':(old_first or {}).get('date',''),
          'V2_최초관측':first.get('date',''),'V2_최초의회':first.get('assembly',''),
          '원본검사행':len(rows),'정책표현_발췌후보':len(cands),'지역흐름':len(flow),
          '근거회의록':len(ev),'활동언급후보':len(item['policy_activity_cues']),
          '지역간유사후보':len(item['cross_region_similarity_candidates']),
          '검증수준':'자동 추출 / 수동 미검증','주의사항':'; '.join(warn[-3:])})
        log(f'[{i}/5] {issue}: 후보 {len(cands)}, 흐름 {len(flow)}, 근거 {len(ev)}, '+
            f'활동 {len(item["policy_activity_cues"])}, 지역간 {len(item["cross_region_similarity_candidates"])} / {time.perf_counter()-t:.2f}초')
    save_json(out/'all_issues.json',result)
    save_csv(out/'통합_품질검토표.csv',report,
        ['이슈','구버전_최초관측','V2_최초관측','V2_최초의회','원본검사행','정책표현_발췌후보',
         '지역흐름','근거회의록','활동언급후보','지역간유사후보','검증수준','주의사항'])
    manifest={'created_at':datetime.now().isoformat(timespec='seconds'),
       'original_28_json':str(prior_path),'issues':list(result),'output_dir':str(out),
       'elapsed_seconds':round(time.perf_counter()-start,3),
       'limitations':['자동 표현 검사이며 정책 관련성 정확도 검증 아님',
        '첫 관측은 보유자료 내 최초일 뿐 실제 정책 최초 도입 아님',
        '지역 간 유사성은 인과적 전파가 아님','미검증 공청회/조례/재논의 실시를 단정 금지']}
    save_json(out/'manifest.json',manifest)
    log(f'=== 28 V2 종료 {manifest["elapsed_seconds"]:.2f}초 === {out}')
    log(f'Gradio 연동: {out/"all_issues.json"}')
if __name__=='__main__':main()

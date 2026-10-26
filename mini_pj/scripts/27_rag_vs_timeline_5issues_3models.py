# -*- coding: utf-8 -*-
"""27번 | 5개 이슈 × 3개 Ollama 모델 × 기본 RAG/시계열 강화 RAG 새 실험.

개인 학원 PC C:\\mini_pj 전용. 원본 Chroma/임베딩·원본 CSV 변경 없음.
두 방식은 입력과 프롬프트가 다르므로 '시계열 단독 효과' 실험은 아님.

실행:
  python C:\\mini_pj\\scripts\\27_rag_vs_timeline_5issues_3models.py
  python ... --dry-run   # 전체 전처리 없이 환경과 출력 양식 시험

필수: pip install pandas numpy chromadb sentence-transformers requests xlsxwriter
       Ollama 실행 및 3개 모델 설치

주의: '지역소멸/저출생/고령화/기후위기'는 청년 일자리 전용 관련성 분류기를
      사용하지 않음. 이 네 이슈의 강화 입력은 어휘일치 '미검증 후보' 추이.
      반복 의회 관측은 인과적 정책 전파나 같은 의안의 재논의를 뜻하지 않음.
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, re, subprocess, sys, time
from pathlib import Path
from datetime import datetime

import pandas as pd

DEFAULT_ROOT = Path(r'C:\mini_pj')
ISSUES = [
 ('지역소멸', '지역소멸,지방소멸,인구소멸,인구감소지역'),
 ('청년 일자리', '청년 일자리,청년 고용,청년 취업,청년 창업,취업 지원'),
 ('저출생', '저출생,저출산,출생률,출산 지원'),
 ('고령화', '고령화,초고령,노인 인구,고령사회'),
 ('기후위기', '기후위기,기후변화,탄소중립,온실가스'),
]
MODELS = ['exaone3.5:2.4b','qwen3:8b','gemma3:4b']
FIELDS = ['질문','모델','방식','상태','최초 관측','타 의회 논의 흐름','향후 예측','근거 회의록',
          '근거 수','예측 글자 수','200자 준수','LLM(초)','출력 토큰','토큰/초',
          '질문 임베딩(초)','DB 검색(초)','전체(초)','오류','실행시각','근거 검증 상태']

def now(): return datetime.now().strftime('%H:%M:%S')
def log(s): print(f'[{now()}] {s}',flush=True)
def save_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
def safe(s): return re.sub(r'[^0-9a-zA-Z가-힣]+','_',str(s)).strip('_')
def module_from(path,name):
    sp=importlib.util.spec_from_file_location(name,path)
    if sp is None or sp.loader is None: raise FileNotFoundError(path)
    m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

def check_files(root):
    scripts=root/'scripts'
    for p in ('15_policy_issue_search.py','16_policy_trend_analysis.py',
              '24_policy_relevance_validation.py','26_policy_llm_model_comparison_v4.py'):
        if not (scripts/p).is_file(): raise FileNotFoundError(f'스크립트 없음: {scripts/p}')
    if not (root/'_db'/'chroma_personal').is_dir(): raise FileNotFoundError('개인 Chroma DB 없음')
    if not (root/'results'/'full_topic_modeling'/'assignments').is_dir(): raise FileNotFoundError('15번용 토픽 할당 폴더 없음')

def available(v4,url):
    installed=v4.available_models(url)
    missing=[m for m in MODELS if m not in installed]
    if missing: raise RuntimeError(f'Ollama 모델 누락: {missing}; ollama list 확인')


def evidence_baseline(raw):
    mds=raw.get('metadatas',[[]])[0]; docs=raw.get('documents',[[]])[0]
    ids=raw['ids'][0]; distances=raw.get('distances',[[]])[0]
    ev=[]
    for i,ident in enumerate(ids):
        m=mds[i] or {}
        ds=re.sub(r'\D','',str(m.get('date') or ''))
        dt=f'{ds[:4]}-{ds[4:6]}-{ds[6:8]}' if len(ds)==8 else ''
        ev.append({'id':str(ident),'date':dt,'assembly':str(m.get('assembly') or ''),
                   'meeting':str(m.get('meeting_title') or ''),'agenda':str(m.get('agenda_title') or ''),
                   'text':str(docs[i] or ''),'distance':distances[i] if i<len(distances) else None})
    return ev

def first_flow(evidence):
    val=sorted((x for x in evidence if x['date'] and x['assembly']),key=lambda x:(x['date'],x['assembly']))
    if not val: return '미확인','미확인'
    seen={}
    for e in val: seen.setdefault(e['assembly'],e['date'])
    return f'{val[0]["date"]} / {val[0]["assembly"]}', ' → '.join(f'{d} {a}' for a,d in sorted(seen.items(),key=lambda x:(x[1],x[0])))

def baseline_prompt(issue,evidence):
    blocks=[]
    for i,e in enumerate(evidence,1):
        blocks.append(f'[E{i:02d}] {e["date"]} {e["assembly"]} / {e["meeting"]}\n원문: {e["text"][:950]}')
    return (f'당신은 지방의회 정책 이슈 분석가다. 이슈: {issue}\n'
            '아래 Chroma Top-K 검색 근거만으로 향후 정책 논의 방향을 한국어 200자 이내로 전망하라.\n'
            '근거가 부족하면 조건부로 서술하고, 다른 의회로의 정책 전파나 공청회 실제 실시를 단정하지 말라.\n'
            'JSON, 제목, 목록 없이 전망 문장 하나만 출력한다.\n[근거]\n'+'\n'.join(blocks)+'\n[200자 이내 예측]')

def run_baseline(issue, embedder,col,requests,url,output_dir):
    import numpy as np
    t0=time.perf_counter(); emb=embedder.encode([issue],convert_to_numpy=True); embed_sec=time.perf_counter()-t0
    if emb.shape[1]!=1024: raise ValueError('기존 Chroma 1024차원과 질문 벡터 불일치')
    t=time.perf_counter(); raw=col.query(query_embeddings=emb.astype(np.float32).tolist(),n_results=15,
                                          include=['documents','metadatas','distances']); search_sec=time.perf_counter()-t
    ev=evidence_baseline(raw); first,flow=first_flow(ev)
    prompt=baseline_prompt(issue,ev)
    (output_dir/'기본RAG_입력프롬프트.txt').write_text(prompt,encoding='utf-8')
    (output_dir/'기본RAG_검색근거.json').write_text(json.dumps(ev,ensure_ascii=False,indent=2),encoding='utf-8')
    results=[]
    for model in MODELS:
        log(f'   [기본] {model} 시작'); start=time.perf_counter(); txt='';error='';metrics={}
        try:
            response=requests.post(url+'/api/generate',json={'model':model,'prompt':prompt,'stream':False,'think':False,
                         'options':{'temperature':0,'num_predict':700,'seed':42}},timeout=420)
            response.raise_for_status();obj=response.json();txt=str(obj.get('response') or '').strip()
            metrics=obj
        except Exception as exc: error=f'{type(exc).__name__}: {exc}'
        seconds=round(time.perf_counter()-start,3); token=int(metrics.get('eval_count') or 0)
        token_duration=(metrics.get('eval_duration') or 0)/1e9
        status='ERROR' if error else 'OK' if 0<len(txt)<=200 else 'OVER_200' if txt else 'EMPTY'
        rec={'질문':issue,'모델':model,'방식':'기본 RAG','상태':status,'최초 관측':first,
             '타 의회 논의 흐름':flow,'향후 예측':txt,'근거 회의록':' | '.join(f'{e["date"]} {e["assembly"]} {e["meeting"]}' for e in ev[:5]),
             '근거 수':len(ev),'예측 글자 수':len(txt),'200자 준수':'통과' if status=='OK' else '미준수',
             'LLM(초)':seconds,'출력 토큰':token,'토큰/초':round(token/token_duration,2) if token_duration else None,
             '질문 임베딩(초)':round(embed_sec,3),'DB 검색(초)':round(search_sec,3),
             '전체(초)':round(seconds+embed_sec+search_sec,3),'오류':error,
             '실행시각':datetime.now().isoformat(timespec='seconds'),
             '근거 검증 상태':'Chroma 상위15건 / 정책 관련성·최초 실제 도입·정책 전파 미검증'}
        results.append(rec); (output_dir/f'{safe(model)}_기본_원시응답.txt').write_text(txt,encoding='utf-8')
        log(f'   [기본] {model}: {status}, {len(txt)}자, {seconds:.2f}초')
    return results

def find_15_output(search_root,issue,previous):
    candidates=[p for p in search_root.iterdir() if p.is_dir() and p.name not in previous and
                p.name.endswith('_'+safe(issue)) and (p/'검색_실행요약.json').exists()]
    if not candidates: raise RuntimeError(f'15번 결과 폴더 찾기 실패: {issue}')
    return max(candidates,key=lambda x:x.stat().st_mtime)

def build_enhanced_data(issue,issue_root,evidence_n=12):
    """청년: 24번 직접 후보 / 기타 이슈: 전수 어휘 일치 후보 (별도 명시)."""
    trend=issue_root/'trend_analysis'
    if issue=='청년 일자리':
        path=issue_root/'relevance_validation'/'전체_정책관련성_검토결과.csv'
        df=pd.read_csv(path,encoding='utf-8-sig').fillna('')
        direct=df[df['review_candidate_label'].eq('직접관련_후보')].copy()
        scope='청년 일자리 전용 규칙의 직접 관련 자동후보 / 미검증'
        ev_col='core_evidence_text_100'
    else:
        df=pd.read_csv(issue_root/'전체_어휘일치_근거.csv',encoding='utf-8-sig').fillna('')
        direct=df.copy();scope='어휘일치 자동후보 / 관련성·수동검증 없음';ev_col='text_preview'
    if direct.empty: raise ValueError(f'{issue}: 강화 입력 근거 0개. 성공처럼 저장하지 않음')
    date_df=direct.sort_values(['date','assembly','chunk_id']).copy()
    first=date_df.iloc[0]
    # 최초/후속 의회 목록은 모두 원문 관측시각으로만 해석한다.
    seen=date_df.drop_duplicates('assembly')
    observed=[{'date':str(x.date),'assembly':str(x.assembly)} for x in seen.itertuples(index=False)]
    # 16번 월별 분모: 2024~2025 전 회의록 청크. 정책 관련성 보증은 아님.
    monthly=pd.read_csv(trend/'월별_정책언급추이.csv',encoding='utf-8-sig').fillna(0)
    if 'month' not in monthly or 'all_chunks' not in monthly: raise ValueError('16번 월별 분모 없음')
    direct['month']=direct['date'].astype(str).str[:7]
    counts=direct.groupby('month').size().to_dict()
    timeline=[]
    for r in monthly.sort_values('month').itertuples(index=False):
        month=str(r.month);n=int(counts.get(month,0));total=int(r.all_chunks)
        timeline.append({'월':month,'관측_청크':n,'전체_청크':total,'관측비율_pct':round(100*n/total,4) if total else 0})
    # 소수 지역·서로 다른 연도와 문서 대표성을 확보, 한 문서의 다중 청크를 과도하게 넣지 않음.
    sort_cols=['date','assembly','chunk_id'];selected=[];seen_files=set();seen_regions=set()
    sources=[date_df.head(5),date_df.iloc[len(date_df)//2:].head(10),date_df.tail(20),date_df]
    for pool in sources:
        for r in pool.itertuples(index=False):
            src=str(getattr(r,'source_id','') or getattr(r,'chunk_id',''))
            if src in seen_files: continue
            if len(selected)<evidence_n//2 or r.assembly not in seen_regions or len(selected)<evidence_n:
                seen_files.add(src);seen_regions.add(r.assembly);selected.append(r)
            if len(selected)>=evidence_n:break
        if len(selected)>=evidence_n:break
    ev=[]
    for i,r in enumerate(sorted(selected,key=lambda z:(str(z.date),str(z.assembly))),1):
        excerpt=str(getattr(r,ev_col,'') or '')[:100]
        ev.append({'id':f'E{i:02d}','date':str(r.date),'assembly':str(r.assembly),
                   'meeting':str(getattr(r,'meeting_title','')),'text':excerpt,
                   'chunk_id':str(r.chunk_id)})
    return {'issue':issue,'scope':scope,'first':f'{first.date} / {first.assembly}',
            'flow':' → '.join(f'{x["date"]} {x["assembly"]}' for x in observed[:17]),
            'timeline':timeline,'evidence':ev,'total_lexical':len(df),'candidate_count':len(direct)}

def enhanced_prompt(issue,data,v4):
    compressed={'이슈':issue,'자료기간':'2024~2025','관측집계단위':'청크',
                '분류_검증상태':data['scope'],'최초_관측_자동후보':data['first'],
                '월별_변화':data['timeline'],'지역별_후속관측':[{'날짜':x['date'],'의회':x['assembly']}
                    for x in sorted([{'date':p.split(' ')[0],'assembly':' '.join(p.split(' ')[1:])}
                                     for p in data['flow'].split(' → ') if ' ' in p],key=lambda z:z['date'])[:10]],
                '근거':[{'id':e['id'],'날짜':e['date'],'의회':e['assembly'],'원문발췌':e['text']}
                      for e in data['evidence']],
                '주의':'최초 관측과 논의 흐름은 보유 자료의 자동 검색 관측임. 재논의·공청회 실시·정책 전파 인과관계는 미검증.'}
    system=('당신은 2024~2025 지방의회 회의록을 근거로 2026년 이후 정책 논의 방향을 조건부 전망하는 분석가다. '
            '원문을 복사하거나 12건 근거를 다시 나열하지 말고, 다섯 키의 JSON 한 객체만 생성하라.')
    user=(f'이슈: {issue}. 미래의 정책 논의 방향에 관한 200자 이내 한국어 전망을 forecast에 작성하라. '
          '숫자 증가를 정책 확대나 채택으로 확정하지 마라. 근거 없는 새 정책이나 지역 간 전파를 단정하지 마라. '
          'JSON 키: forecast(미래 조건부 전망), trend_interpretation(과거 월별 추세 설명), '
          'forecast_basis(전망 근거), evidence_ids(E번호 1~3개), uncertainty(검증 한계). '
          '모든 키를 빠짐없이 채우고 출력은 JSON 객체 한 개로 한정한다. '\
          '\n입력자료:'+json.dumps(compressed,ensure_ascii=False,separators=(',',':')))
    return system,user

def run_enhanced(issue,data,v4,url,output_dir,timeout=360):
    system,user=enhanced_prompt(issue,data,v4)
    (output_dir/'시계열강화_입력프롬프트.txt').write_text('SYSTEM\n'+system+'\nUSER\n'+user,encoding='utf-8')
    (output_dir/'시계열강화_근거.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    allowed={e['id'] for e in data['evidence']}; results=[]
    for model in MODELS:
        log(f'   [강화] {model} 시작');st=time.perf_counter(); attempts=[];raw='';parsed=None;err='';metrics={}
        for retry in range(2):
            try:
                messages=([{'role':'system','content':system},{'role':'user','content':user}]
                          if retry==0 else v4.retry_messages(system,user,raw,err))
                rsp=v4.request_json(url,'/api/chat',{'model':model,'stream':False,'messages':messages,
                       'format':v4.OUTPUT_SCHEMA,'think':False,
                       'options':{'temperature':0,'num_predict':700,'seed':42}},timeout)
                raw=str((rsp.get('message') or {}).get('content') or '')
                parsed=v4.parse_model_response(raw)
                metrics=rsp;check=v4.evaluate(parsed,allowed)
                err= ('JSON 파싱 실패' if not isinstance(parsed,dict) else
                      'forecast 빈값' if not parsed.get('forecast') else
                      'forecast 200자 초과' if len(parsed['forecast'])>200 else
                      '필수 필드 누락' if not check['required_fields_present'] else
                      '근거 ID 불일치' if not check['evidence_ids_valid'] else
                      '조건부 전망 표현 누락' if not check['caution_word_present'] else '')
            except Exception as exc: err=f'{type(exc).__name__}: {exc}'
            attempts.append({'round':retry+1,'raw':raw,'error':err})
            if not err:break
        sec=round(time.perf_counter()-st,3);txt=str(parsed.get('forecast') or '') if isinstance(parsed,dict) else ''
        tokens=int(metrics.get('eval_count') or 0); eval_sec=(metrics.get('eval_duration') or 0)/1e9
        status='OK' if not err and 0<len(txt)<=200 else 'ERROR'
        row={'질문':issue,'모델':model,'방식':'시계열 강화 RAG','상태':status,
             '최초 관측':data['first'],'타 의회 논의 흐름':data['flow'],'향후 예측':txt,
             '근거 회의록':' | '.join(f'{e["date"]} {e["assembly"]} {e["meeting"]} [{e["id"]}]' for e in data['evidence'][:5]),
             '근거 수':len(data['evidence']),'예측 글자 수':len(txt),
             '200자 준수':'통과' if status=='OK' else '미준수','LLM(초)':sec,
             '출력 토큰':tokens,'토큰/초':round(tokens/eval_sec,2) if eval_sec else None,
             '질문 임베딩(초)':None,'DB 검색(초)':None,'전체(초)':None,'오류':err,
             '실행시각':datetime.now().isoformat(timespec='seconds'),
             '근거 검증 상태':data['scope']+' / 시계열 사전처리 비용 별도'}
        results.append(row)
        (output_dir/f'{safe(model)}_강화_원시응답.txt').write_text(
            '\n\n'.join(f'== 시도 {x["round"]}; {x["error"]} ==\n{x["raw"]}' for x in attempts),encoding='utf-8')
        log(f'   [강화] {model}: {status}, {len(txt)}자, {sec:.2f}초')
    return results

def diff(a,b,field):
    if not a or not b:return '비교 불가: 양쪽 실행 결과가 필요함'
    aa=str(a.get(field,'') or '').strip();bb=str(b.get(field,'') or '').strip()
    if field=='최초 관측':
        return '표시 날짜·의회 일치(검색 기준은 다름)' if aa==bb else f'미일치: 기본 {aa} / 강화 {bb}. Top-K 표본과 전수 어휘·자동분류 관측의 차이'
    if field=='타 의회 논의 흐름':
        sa=set(x.strip() for x in aa.split('→') if x.strip());sb=set(x.strip() for x in bb.split('→') if x.strip())
        return (f'표시 순서 일치(정책 전파 증거 아님)' if aa==bb else
                f'미일치: 공통 {len(sa&sb)}개 / 기본만 {len(sa-sb)}개 / 강화만 {len(sb-sa)}개. 정책 전파 인과관계 미검증')
    if field=='향후 예측':
        return '예측 문장 완전 동일' if aa==bb else f'미일치: 기본 {len(aa)}자 / 강화 {len(bb)}자. 내용 타당성·근거 충실도는 수동 검증 전 미평가'
    return ('근거 요약 문자열 동일(원문 검증 전)' if aa==bb else
            f'근거 구성 다름: 기본 {a.get("근거 수")}개 / 강화 {b.get("근거 수")}개. 원문 확인 필요')

def make_excel(out, rows, questions):
    """기존 학원 PC의 xlsxwriter 활용. 생성 Excel과 CSV는 최종 5×3×2 결과를 담음."""
    try: import xlsxwriter
    except ImportError:
        log('엑셀 라이브러리 없음: pip install xlsxwriter');return
    look={(r['질문'],r['모델'],r['방식']):r for r in rows}
    path=out/'5질문_3모델_기본RAG_vs_시계열강화RAG_비교.xlsx'
    wb=xlsxwriter.Workbook(str(path))
    title=wb.add_format({'bold':1,'font_size':16,'font_color':'#FFFFFF','bg_color':'#193550','valign':'vcenter'})
    head=wb.add_format({'bold':1,'font_color':'#FFFFFF','bg_color':'#087E86','text_wrap':1,'valign':'vcenter','border':1})
    styles={key:wb.add_format({'bg_color':bg,'text_wrap':1,'valign':'top','border':1})
            for key,bg in [('base','#FFFFFF'),('enh','#F0F5F8'),('diff','#FFF1D9')]}
    num=wb.add_format({'num_format':'0.00','bg_color':'#F0F5F8','border':1})
    def setup(name,headers,widths):
        sh=wb.add_worksheet(name);sh.freeze_panes(2,3);sh.set_row(0,35);sh.merge_range(0,0,0,len(headers)-1,name+' | 5개 이슈 × 3모델',title)
        sh.write_row(1,0,headers,head);sh.set_row(1,36)
        for j,w in enumerate(widths):sh.set_column(j,j,w)
        return sh
    sh=setup('01_질문별3행비교',['질문','모델','구분','최초 등장시점','타 지방의회 흐름','향후 예측 (200자)','근거 회의록','상태'],[19,22,22,44,90,100,90,16])
    sr=2
    for issue in questions:
        for model in MODELS:
            a=look.get((issue,model,'기본 RAG'));b=look.get((issue,model,'시계열 강화 RAG'))
            for kind,r,key in [('기본 RAG',a,'base'),('시계열 강화 RAG',b,'enh')]:
                vals=[issue,model,kind]+([r.get(k,'') for k in ['최초 관측','타 의회 논의 흐름','향후 예측','근거 회의록','상태']] if r else ['미실행']*5)
                sh.write_row(sr,0,vals,styles[key]);sh.set_row(sr,86);sr+=1
            com=[issue,model,'차이·일치 여부']+[diff(a,b,k) for k in ['최초 관측','타 의회 논의 흐름','향후 예측','근거 회의록']]+['검증 전']
            sh.write_row(sr,0,com,styles['diff']);sh.set_row(sr,90);sr+=1
    sh.autofilter(1,0,sr-1,7)
    ph=['질문','모델','방식','상태','예측 글자 수','200자 준수','LLM(초)','출력 토큰','토큰/초','질문 임베딩(초)','DB 검색(초)','전체(초)']
    p=setup('02_실측성능',ph,[18,22,20,14,18,16,15,15,15,18,16,15]);pr=2
    for issue in questions:
        for model in MODELS:
            for kind in ('기본 RAG','시계열 강화 RAG'):
                r=look.get((issue,model,kind))
                if not r:continue
                p.write_row(pr,0,[r.get(k,'') for k in ph],styles['enh'] if kind.startswith('시계열') else styles['base']);pr+=1
    p.autofilter(1,0,max(2,pr-1),len(ph)-1)
    d=setup('03_동일모델_시간차',['질문','모델','기본 LLM 초','강화 LLM 초','차이(강화-기본) 초','기본 200자 준수','강화 200자 준수'],[20,23,19,19,24,22,22]);dr=2
    for issue in questions:
        for model in MODELS:
            a=look.get((issue,model,'기본 RAG'));b=look.get((issue,model,'시계열 강화 RAG'))
            if not a or not b:continue
            d.write_row(dr,0,[issue,model,a.get('LLM(초)'),b.get('LLM(초)'),None,a.get('200자 준수'),b.get('200자 준수')],styles['enh'])
            d.write_formula(dr,4,f'=D{dr+1}-C{dr+1}',num);dr+=1
    h=setup('04_해석과한계',['비교 기준','해석 및 주의사항'],[28,135])
    notes=[('실험 설계','각 이슈별 기본 RAG와 시계열 강화 RAG를 동일한 3모델로 새 실행. 기존 과거 결과와 섞지 않음.'),
           ('관측 범위','2024~2025년 학원 개인 DB. 최초 관측은 실제 법·정책 최초 등장일이 아님.'),
           ('기본 검색','동일 Chroma DB의 의미검색 Top15 표본. 최초 관측도 해당 표본 내 최솟값.'),
           ('강화 검색','기존 15/16번 기반 메타데이터 전수 어휘 검색과 24개월 정규화 월별 시계열. 어휘 검색은 실제 정책 관련 발언 전체가 아님.'),
           ('관련성 분류','청년 일자리: 24번 직접 관련 자동후보. 그 외 4개 질문: 이슈별 분류기 부재로 어휘일치 미검증 후보.'),
           ('의회 흐름','각 의회의 관측 시간순 나열. 공청회, 재논의, 실제 정책 전파의 확인을 뜻하지 않음.'),
           ('비교 유의','두 방식은 입력/프롬프트/JSON 출력 조건이 다름. 시계열 분석만의 인과적 개선 효과로 해석 금지.'),
           ('성능 평가','실측 가능한 글자 수·200자 준수·LLM 시간·출력 토큰·생성 속도만 정량 비교. 내용 정확도/예측 정확도 미평가.'),
           ('처리 시간','강화 RAG 질문 임베딩·DB검색은 미실행/사전처리 재사용. 공백으로 기록, 처리 0초가 아님.'),
           ('모델 성능','단일 실행 시간은 모델 로드/캐시 영향이 있으므로 반복 측정 전에 순위 단정 금지.'),
           ('어휘 사전','검색 표현 확장은 15번 스크립트에서 지정한 후보이며 동의어 선정의 정확도는 수동 검증 전.'),
           ('시연 보안','개인 PC 로컬 Ollama 및 기존 Chroma만 활용. 외부 LLM API 사용 없음.')]
    for j,(a,b) in enumerate(notes,2):h.write_row(j,0,[a,b],styles['base']);h.set_row(j,45)
    s=setup('05_실행정보',['항목','내용'],[38,128])
    metadata=[('실행 시각',datetime.now().isoformat(timespec='seconds')),('모델',', '.join(MODELS)),('질문',', '.join(questions)),
              ('방식','기본 RAG / 시계열 강화 RAG'),('근거 검증','자동 후보. 원문 대조 및 수동 판정 이전'),('실패 결과','삭제하지 않고 오류와 빈 예측 저장')]
    for j,(a,b) in enumerate(metadata,2):s.write_row(j,0,[a,b],styles['base'])
    wb.close();log('Excel 저장: '+str(path))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--ollama-url',default='http://127.0.0.1:11434')
    parser.add_argument('--dry-run',action='store_true',help='모델·전수 검색 실행 없이 예시 엑셀 생성')
    parser.add_argument('--reuse-search',action='store_true',help='이전에 생성한 동일 이슈의 15/16 결과를 재사용(검색 어휘 동일 여부 직접 확인 필요)')
    parser.add_argument('--issues',nargs='+',default=[x[0] for x in ISSUES])
    args=parser.parse_args();root=args.root
    selected=[x for x in ISSUES if x[0] in args.issues]
    if not selected:raise ValueError('실행할 질문 없음')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    out=root/'results'/'rag_vs_timeline_5issues'/stamp;out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();rows=[]
    if args.dry_run:
        example=[]
        for issue,_ in selected:
            for model in MODELS:
                for kind in ('기본 RAG','시계열 강화 RAG'):
                    example.append({'질문':issue,'모델':model,'방식':kind,'상태':'DRY_RUN','예측 글자 수':None,
                                    '200자 준수':'미실행','LLM(초)':None,'근거 검증 상태':'실제 모델 호출 없음'})
        save_csv(out/'5질문_3모델_실험결과.csv',example,FIELDS)
        make_excel(out,example,[x[0] for x in selected]);log(f'DRY RUN: {out}');return
    check_files(root)
    import chromadb, requests, numpy as np
    from sentence_transformers import SentenceTransformer
    v4=module_from(root/'scripts'/'26_policy_llm_model_comparison_v4.py','v4_runner')
    available(v4,args.ollama_url)
    db=chromadb.PersistentClient(path=str(root/'_db'/'chroma_personal'))
    col=db.get_collection('personal_policy_chunks');log(f'Chroma 연결: {col.count():,}건')
    embedder=SentenceTransformer('dragonkue/snowflake-arctic-embed-l-v2.0-ko')
    search_root=root/'results'/'policy_issue_search'
    for i,(issue,terms) in enumerate(selected,1):
        log(f'[{i}/{len(selected)}] === {issue} ===');issue_out=out/safe(issue);issue_out.mkdir(exist_ok=True)
        basic=run_baseline(issue,embedder,col,requests,args.ollama_url,issue_out);rows+=basic
        save_csv(out/'5질문_3모델_실험결과.csv',rows,FIELDS)
        try:
            if args.reuse_search:
                candidates=[p for p in search_root.iterdir() if p.is_dir() and p.name.endswith('_'+safe(issue)) and
                            (p/'검색_실행요약.json').exists()]
                if not candidates:raise FileNotFoundError(f'{issue}에 재사용할 15번 결과 없음')
                search_dir=max(candidates,key=lambda p:p.stat().st_mtime)
                meta=json.loads((search_dir/'검색_실행요약.json').read_text(encoding='utf-8-sig'))
                if set(meta.get('lexical_terms',[]))!=set([issue]+terms.split(',')):
                    raise ValueError('재사용 검색 표현이 이번 실험과 다름. --reuse-search 없이 실행 권장')
            else:
                before={p.name for p in search_root.iterdir() if p.is_dir()}
                command=[sys.executable,str(root/'scripts'/'15_policy_issue_search.py'),'--query',issue,
                         '--terms',terms,'--skip-semantic']
                subprocess.run(command,check=True)
                search_dir=find_15_output(search_root,issue,before)
            log(f'   검색 폴더: {search_dir}')
            if not (search_dir/'trend_analysis'/'월별_정책언급추이.csv').exists():
                subprocess.run([sys.executable,str(root/'scripts'/'16_policy_trend_analysis.py'),
                                '--input-dir',str(search_dir),'--broad-terms',''],check=True)
            if issue=='청년 일자리' and not (search_dir/'relevance_validation'/'전체_정책관련성_검토결과.csv').exists():
                subprocess.run([sys.executable,str(root/'scripts'/'24_policy_relevance_validation.py'),
                                '--input-dir',str(search_dir)],check=True)
            enhanced=build_enhanced_data(issue,search_dir)
            records=run_enhanced(issue,enhanced,v4,args.ollama_url,issue_out)
            for r in records:r['검색 결과 폴더']=str(search_dir)
            rows+=records
        except Exception as exc:
            # 결과 불일치는 빈 예측으로 기록하고 나머지 질문 계속 실행. 허위 성공 방지.
            error=f'{type(exc).__name__}: {exc}';log(f'   [강화 단계 ERROR] {error}')
            for model in MODELS:
                rows.append({'질문':issue,'모델':model,'방식':'시계열 강화 RAG','상태':'ERROR',
                             '오류':error,'200자 준수':'미실행','근거 검증 상태':'강화 단계 실패'})
        save_csv(out/'5질문_3모델_실험결과.csv',rows,FIELDS)
        make_excel(out,rows,[x[0] for x in selected]);log(f'   진행: {len(rows)}/{len(selected)*len(MODELS)*2} 모델 결과')
    log(f'전체 종료 {time.perf_counter()-start:.2f}초 / {out}')
    (out/'실행_요약.json').write_text(json.dumps({'issues':[x[0] for x in selected],
       'models':MODELS,'completed_rows':len(rows),'OK':sum(r.get('상태')=='OK' for r in rows),
       'elapsed_seconds':round(time.perf_counter()-start,2),
       'warning':'검색방법·프롬프트 상이, 결과 개선을 시계열 단독 효과로 단정 금지'},ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()

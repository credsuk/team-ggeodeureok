# -*- coding: utf-8 -*-
"""29번 실시간 Chroma + 시계열 + EXAONE + Gradio.

새 문서는 기존 전처리/임베딩 절차로 Chroma에 추가해야 한다.
모든 정책 분석은 실행 시 DB 전체를 청크 단위로 검사해 최초 관측과 월별/지역별 통계를 생성한다.
컬렉션 청크 수가 바뀌면 이전 검색 인덱스 캐시는 자동 무효화한다.
LLM 전망은 파일에서 읽지 않고 검색 클릭 때마다 Ollama에서 새로 생성한다.

설치: pip install gradio chromadb sentence-transformers requests pandas plotly
실행: python 29_gradio_live_exaone_policy_dashboard.py
점검: python 29_gradio_live_exaone_policy_dashboard.py --dry-run
첫 검색은 DB 전체(수십만 청크) 스캔으로 시간이 걸릴 수 있다. 이후 이슈별 캐시 사용.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import hashlib
import html
import json
import re
import time

ROOT = Path(r'C:\mini_pj')
COLLECTION = 'personal_policy_chunks'
EMBEDDING_MODEL = 'dragonkue/snowflake-arctic-embed-l-v2.0-ko'
MODEL = 'exaone3.5:2.4b'
TERMS = {
    '지역소멸': ['지역소멸','지방소멸','인구소멸','인구감소지역'],
    '청년 일자리': ['청년 일자리','청년일자리','청년 고용','청년고용','청년 취업','청년취업','청년 창업','청년창업','청년 구직','청년구직'],
    '저출생': ['저출생','저출산','초저출생','출생률','출산율'],
    '고령화': ['고령화','초고령','노인 인구','고령사회'],
    '기후위기': ['기후위기','기후 위기','기후변화','탄소중립','온실가스'],
}
REGIONS = {
    '서울특별시의회': (37.5665, 126.9780), '부산광역시의회': (35.1796,129.0756),
    '대구광역시의회': (35.8714,128.6014), '인천광역시의회': (37.4563,126.7052),
    '광주광역시의회': (35.1595,126.8526), '대전광역시의회': (36.3504,127.3845),
    '울산광역시의회': (35.5384,129.3114), '세종특별자치시의회': (36.4800,127.2890),
    '경기도의회': (37.4138,127.5183), '강원특별자치도의회': (37.8228,128.1555),
    '충청북도의회': (36.8000,127.7000), '충청남도의회': (36.5184,126.8000),
    '전북특별자치도의회': (35.7175,127.1530), '전라남도의회': (34.8679,126.9910),
    '경상북도의회': (36.4919,128.8889), '경상남도의회': (35.4606,128.2132),
    '제주특별자치도의회': (33.4996,126.5312),
}

def norm_date(value):
    raw = ''.join(re.findall(r'\d', str(value or '')))
    if len(raw) < 8: return ''
    s = f'{raw[:4]}-{raw[4:6]}-{raw[6:8]}'
    try: datetime.strptime(s, '%Y-%m-%d'); return s
    except ValueError: return ''

def snippet(text, terms, limit=100):
    matches = [(text.find(t),t) for t in terms if t in text]
    if not matches: return '', ''
    position, term = min(matches)
    start = max(0, position-25)
    if start + limit > len(text): start = max(0, len(text)-limit)
    return text[start:start+limit], term

def selection(issue, text):
    return any(t in text for t in TERMS[issue])

def serialize(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)

class LivePolicy:
    def __init__(self, root, url, batch=2000, top_k=15, scan=True):
        import chromadb
        import requests
        self.root = Path(root)
        self.url = url.rstrip('/')
        self.requests = requests
        self.client = chromadb.PersistentClient(path=str(self.root/'_db'/'chroma_personal'))
        self.collection = self.client.get_collection(COLLECTION)
        self.batch = batch
        self.top_k = top_k
        self.scan = scan
        self.embedder = None
        self.cache_dir = self.root/'results'/'policy_live_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def preflight(self):
        count = self.collection.count()
        r = self.requests.get(self.url+'/api/tags', timeout=15)
        r.raise_for_status()
        names = {x.get('name','') for x in r.json().get('models',[])}
        if MODEL not in names:
            raise RuntimeError(f'Ollama 모델 {MODEL} 없음. 현재 설치: {sorted(names)}')
        return count

    def _cache_file(self, issue, count):
        code = hashlib.sha1(('|'.join(TERMS[issue])).encode('utf-8')).hexdigest()[:10]
        return self.cache_dir/f'{issue.replace(" ","_")}_{count}_{code}.json'

    def scan_issue(self, issue):
        """전체 DB를 순회하며 정책어 출현 청크를 검사. 건수 기준 캐시 무효화.
        새 자료가 기존 청크를 교체하되 총 건수가 같다면 --refresh를 사용한다.
        """
        count = self.collection.count()
        target = self._cache_file(issue, count)
        if target.exists():
            return json.loads(target.read_text(encoding='utf-8'))
        old = list(self.cache_dir.glob(issue.replace(' ','_')+'_*.json'))
        for p in old: p.unlink(missing_ok=True)
        monthly_all, monthly_hits = Counter(),Counter()
        per_region = {}
        matched = 0
        for offset in range(0, count, self.batch):
            page = self.collection.get(limit=self.batch, offset=offset, include=['metadatas','documents'])
            for ident, md, doc in zip(page['ids'], page.get('metadatas',[]), page.get('documents',[])):
                md = md or {}; doc = str(doc or '')
                date = norm_date(md.get('date')); area = str(md.get('assembly') or '')
                if date: monthly_all[date[:7]] += 1
                if not date or not area or not selection(issue, doc): continue
                matched += 1
                monthly_hits[date[:7]] += 1
                excerpt, term = snippet(doc, TERMS[issue])
                row = {'id':str(ident),'date':date,'assembly':area,
                       'meeting_title':str(md.get('meeting_title') or ''),
                       'excerpt_100':excerpt,'matched_term':term}
                if area not in per_region or (date, str(ident)) < (per_region[area]['date'],per_region[area]['id']):
                    per_region[area] = row
            if offset % 30000 == 0: print(f'  [DB 스캔] {issue}: {min(offset+self.batch,count):,}/{count:,}',flush=True)
        rows = sorted(per_region.values(), key=lambda x:(x['date'], x['assembly']))
        months = sorted(set(monthly_all)|set(monthly_hits))
        trends=[{'month':m,'matched_chunks':monthly_hits[m],'total_chunks':monthly_all[m],
                'ratio_pct':round(100*monthly_hits[m]/monthly_all[m],4) if monthly_all[m] else 0} for m in months]
        data = {'issue':issue,'db_count':count,'matched_chunks':matched,'regional_events':rows,
                'monthly_timeline':trends,'first_observation':rows[0] if rows else None,
                'scope':'전체 DB 정책 어휘일치 자동 관측(수동 미검증)',
                'generated_at':datetime.now().isoformat(timespec='seconds')}
        target.write_text(serialize(data),encoding='utf-8')
        return data

    def retrieve(self, issue, question):
        import numpy as np
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            self.embedder=SentenceTransformer(EMBEDDING_MODEL)
        emb=self.embedder.encode([question],convert_to_numpy=True).astype(np.float32)
        if emb.shape[1] != 1024: raise ValueError('임베딩이 1024차원이 아닙니다. 기존 Chroma 임베딩 모델 확인 필요.')
        n=min(self.top_k,self.collection.count())
        if n<1: return []
        raw=self.collection.query(query_embeddings=emb.tolist(),n_results=n,
                                  include=['documents','metadatas','distances'])
        results=[]
        for ident,md,doc,dist in zip(raw['ids'][0],raw['metadatas'][0],raw['documents'][0],raw['distances'][0]):
            md=md or {};doc=str(doc or '')
            # 5개 주제의 어휘를 포함하지 않는 의미 검색 결과는 근거로 사용하지 않음
            if not selection(issue,doc): continue
            x,term=snippet(doc,TERMS[issue]); results.append({'id':str(ident),'date':norm_date(md.get('date')),
                'assembly':str(md.get('assembly') or ''),'meeting_title':str(md.get('meeting_title') or ''),
                'text':doc[:900],'excerpt_100':x,'matched_term':term,'distance':float(dist)})
        return results

    def forecast(self, issue, question, summary, evidence):
        if not evidence: return '근거 부족: 해당 이슈의 관련 검색 문서가 없어 전망을 생성하지 않았습니다.',0
        ordered=summary.get('regional_events',[])
        flow=' / '.join(f"{x['date']} {x['assembly']}" for x in ordered[:17])
        months=summary.get('monthly_timeline',[])[-12:]
        trend=', '.join(f"{m['month']} {m['ratio_pct']}%" for m in months)
        items='\n'.join(f"[E{i}] {x['date']} {x['assembly']} / {x['meeting_title']} / {x['id']}\n{x['text'][:450]}"
                        for i,x in enumerate(evidence[:8],1))
        prompt=(f'너는 지방의회 정책 이슈의 신중한 분석가다. 질문: {question}\n주제: {issue}\n'
          '아래 2024~현재 보유 회의록의 자동 검색 근거로만 향후 지방의회에서 논의될 수 있는 정책 방향을 조건부로 전망하라.\n'
          '시간 순서는 정책 전파의 인과 증거가 아니다. 실제 채택/공청회 개최/정책 통과를 단정하지 말라.\n'
          '근거에 없는 기관, 사업, 금액, 도입일을 창작하지 말라. 근거가 모자라면 명시하라.\n'
          '반드시 150자 이내, 한국어 평서문 1~2개로 출력. 제목, 번호, JSON, 설명 문구 금지.\n'
          f'전체 정책어 관측 청크: {summary.get("matched_chunks",0)}; 최초 자동 관측: {ordered[0] if ordered else "없음"}\n'
          f'지역별 첫 관측: {flow}\n월별 관측 비율: {trend}\n근거:\n{items}\n[150자 이내 조건부 전망]')
        t=time.perf_counter()
        payload={'model':MODEL,'prompt':prompt,'stream':False,'think':False,
                 'options':{'temperature':0,'seed':42,'num_predict':220,'num_ctx':8192}}
        r=self.requests.post(self.url+'/api/generate',json=payload,timeout=420)
        r.raise_for_status(); answer=str(r.json().get('response') or '').strip()
        if len(answer)>150:
            correction=(f'다음 문장을 사실관계와 조건부 표현을 유지하며 150자 이내 한국어 평서문으로 재작성하라. '
                        f'새로운 사실 추가 금지. 재작성 결과만 출력.\n{answer}')
            payload['prompt']=correction; payload['options']['num_predict']=160
            r=self.requests.post(self.url+'/api/generate',json=payload,timeout=420)
            r.raise_for_status();answer=str(r.json().get('response') or '').strip()
        if not answer or len(answer)>150:
            answer='EXAONE이 150자 형식을 준수하지 않아 결과를 표시하지 않았습니다.'
        return answer,round(time.perf_counter()-t,2)

    def run(self, issue, question):
        start=time.perf_counter()
        db_count=self.preflight()
        stats=self.scan_issue(issue) if self.scan else {'regional_events':[], 'monthly_timeline':[],
                            'first_observation':None,'matched_chunks':0,'scope':'상위 의미검색 결과만 이용(전체 관측 미실시)'}
        evidence=self.retrieve(issue,question)
        if not self.scan and evidence:
            by_area={}
            for x in sorted(evidence,key=lambda r:(r['date'] or '9999',r['assembly'])):
                if x['date'] and x['assembly']: by_area.setdefault(x['assembly'],x)
            stats['regional_events']=list(by_area.values())
            stats['first_observation']=stats['regional_events'][0] if stats['regional_events'] else None
        forecast, sec=self.forecast(issue,question,stats,evidence)
        return {'issue':issue,'question':question,'model':MODEL,'db_count':db_count,
                'stats':stats,'evidence':evidence,'forecast':forecast,'llm_seconds':sec,
                'elapsed_seconds':round(time.perf_counter()-start,2),
                'limitations':'어휘·검색 근거 자동 추출이며 실제 최초 도입, 정책 전파, 미래 예측 정확도는 미검증',
                'created_at':datetime.now().isoformat(timespec='seconds')}

def plot_map(events):
    import plotly.graph_objects as go
    fig=go.Figure()
    valid=[x for x in events if x.get('assembly') in REGIONS]
    if valid:
        fig.add_trace(go.Scattergeo(lon=[REGIONS[x['assembly']][1] for x in valid],
          lat=[REGIONS[x['assembly']][0] for x in valid],mode='markers+text',
          text=[x['assembly'].replace('의회','') for x in valid],textposition='top center',
          marker={'size':9}, customdata=[x['date'] for x in valid],
          hovertemplate='%{text}<br>최초 자동 관측 %{customdata}<extra></extra>'))
    fig.update_geos(projection_type='mercator',center={'lat':36.3,'lon':127.7},
       lataxis_range=[33,39],lonaxis_range=[125,131],showcoastlines=True,showland=True,fitbounds=False)
    fig.update_layout(height=380,margin={'l':0,'r':0,'t':8,'b':0})
    return fig

def plot_months(months):
    import plotly.graph_objects as go
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=[x['month'] for x in months], y=[x['ratio_pct'] for x in months],
                             mode='lines+markers',name='관측 비율(%)'))
    fig.update_layout(height=260,margin={'l':40,'r':15,'t':15,'b':40},yaxis_title='청크 관측 비율(%)')
    return fig

def run_gradio(engine, host, port, browser):
    import gradio as gr
    import pandas as pd
    def on_run(issue, question):
        q=str(question or '').strip() or f'{issue} 관련 향후 지방의회 정책 논의는 어떻게 전개될까?'
        try:
            r=engine.run(issue,q)
        except Exception as e:
            msg=f'실행 오류: {type(e).__name__}: {e}'
            return msg,'',pd.DataFrame(),plot_months([]),plot_map([]),pd.DataFrame(),None
        stat=r['stats'];ev=stat['regional_events']; f=stat['first_observation'] or {}
        first=(f"{f.get('date','미확인')} / {f.get('assembly','미확인')} — {f.get('meeting_title','')}\n"
               f"{f.get('excerpt_100','')}\n※ {stat.get('scope','자동 관측')}; 실제 정책 최초 도입 아님")
        flows=pd.DataFrame([{'날짜':x['date'],'의회':x['assembly'],'회의록':x['meeting_title'],
                    '근거 100자':x['excerpt_100'],'청크 ID':x['id']} for x in ev])
        sources=pd.DataFrame([{'날짜':x['date'],'의회':x['assembly'],'회의록':x['meeting_title'],
                    '근거 100자':x['excerpt_100'],'청크 ID':x['id']} for x in r['evidence']])
        message=(f"EXAONE 실시간 전망: {r['forecast']}\n\n"
                 f"모델 {MODEL} | Chroma {r['db_count']:,}청크 | 검색 근거 {len(r['evidence'])}건 | "
                 f"LLM {r['llm_seconds']}초 | 전체 {r['elapsed_seconds']}초\n"
                 f"주의: {r['limitations']}")
        export_dir=engine.root/'results'/'policy_live_runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        export_dir.mkdir(parents=True,exist_ok=True)
        path=export_dir/'live_result.json';path.write_text(serialize(r),encoding='utf-8')
        return message,first,flows,plot_months(stat['monthly_timeline']),plot_map(ev),sources,str(path)
    with gr.Blocks(title='광역 지방의회 정책 전망 · EXAONE 실시간 RAG') as demo:
        gr.Markdown('# 지방의회 정책 이슈 · 실시간 EXAONE 전망')
        gr.Markdown('Chroma 최신 검색 + 전체 DB 시계열 관측 + 로컬 Ollama 추론. 이전 JSON의 전망을 재사용하지 않습니다.')
        with gr.Row():
            issue=gr.Dropdown(choices=list(TERMS),value='청년 일자리',label='정책 이슈')
            question=gr.Textbox(label='질문 (선택)',placeholder='예: 청년 일자리 정책의 향후 논의 방향은?')
        search=gr.Button('최신 DB 검색 + EXAONE 새 전망 생성',variant='primary')
        forecast=gr.Textbox(label='③ EXAONE 실시간 조건부 전망',lines=5)
        first=gr.Textbox(label='① 최초 자동 관측 및 원문 근거',lines=5)
        with gr.Row():
            map_component=gr.Plot(label='② 지역별 첫 관측 지도')
            chart=gr.Plot(label='월별 정책어 관측 비율')
        flows=gr.Dataframe(label='② 지역별 관측 흐름(전파 확정 아님)',interactive=False)
        sources=gr.Dataframe(label='④ Chroma 검색 근거(원문 발췌, 자동 미검증)',interactive=False)
        file_output=gr.File(label='이번 실행 결과 JSON 저장 파일')
        search.click(on_run,inputs=[issue,question], outputs=[forecast,first,flows,chart,map_component,sources,file_output])
    demo.queue().launch(server_name=host,server_port=port,inbrowser=browser)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--ollama-url',default='http://127.0.0.1:11434')
    parser.add_argument('--port',type=int,default=7860)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--refresh',action='store_true',help='기존 전체 DB 검색 캐시 강제 삭제')
    parser.add_argument('--fast',action='store_true',help='전체 DB 스캔 생략. 최초 관측/시계열 정확도 저하')
    args=parser.parse_args()
    if args.refresh:
        for f in (args.root/'results'/'policy_live_cache').glob('*.json'): f.unlink(missing_ok=True)
    engine=LivePolicy(args.root,args.ollama_url,scan=not args.fast)
    cnt=engine.preflight()
    print(f'[연결 성공] Chroma 청크 {cnt:,}, Ollama 모델 {MODEL}, 전체 DB 스캔={not args.fast}',flush=True)
    if args.dry_run:
        print('[dry-run] DB 전체 검사 및 LLM 호출 없이 연결 점검만 완료.',flush=True)
        return
    run_gradio(engine,args.host,args.port,not args.no_browser)

if __name__=='__main__': main()

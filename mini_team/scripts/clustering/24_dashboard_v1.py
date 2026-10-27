"""Step 24: Gradio read-only policy viewer. Import does not require Gradio."""
import argparse,importlib,json,html,sys
from pathlib import Path
c=importlib.import_module('20_pipeline_common')
NOTICE='2024~2025년 보유 회의록 기준입니다. 실제 최초 도입 및 지역 간 정책 전파는 미확인입니다. 기존 분석 결과를 조회하며 새 임베딩·LLM 재실행은 하지 않습니다.'
def resolve_issue(query):
 q=''.join(query.split())
 for issue,terms in c.TERMS.items():
  if any(t.replace(' ','') in q for t in [issue]+terms):return issue
 return None
def render(data,query,year='전체',assembly='전체'):
 issue=resolve_issue(query)
 if not issue:return ('지원 범위: '+', '.join(c.ISSUES),'',[],[],[],[],[], '')
 v=data[issue];first=v['first'];e=first['evidence'];esc=lambda s:html.escape(str(s))
 first_text='보유 자료 기준 최초 관측이며 실제 최초 도입은 아닙니다.\n\n'+(f"{e['date']} / {e['assembly']} / {e['meeting']}\n\n청크 {e['chunk_id']}\n\n{e['status']}\n\n{e['original_excerpt']}" if e else '미확인/근거 부족/검증 필요')
 evidence=[x for x in v['evidence'] if (year=='전체' or x['date'].startswith(year)) and (assembly=='전체' or x['assembly']==assembly)]
 timeline=[[x['date'],x['assembly'],x['meeting'],x['chunk_id']] for x in evidence]
 rows=[[x['meeting'],x['date'],x['assembly'],x['original_excerpt'][:180],x['chunk_id'],x['status']] for x in evidence]
 f=v['forecast'];forecast=f"{f['prediction']}\n\n모델: {f['model'] or '미확보'} / {f['status']}\n\n로컬 LLM 전망이며 미래의 실제 결과를 보장하지 않습니다."
 counts={}
 for x in evidence:counts[x['assembly']]=counts.get(x['assembly'],0)+1
 countrows=[[k,n] for k,n in sorted(counts.items())]
 links=[[x['earlier_date'],x['earlier_assembly'],x['later_date'],x['later_assembly'],', '.join(x['shared_purpose']),x['classification'],x['status']] for x in v['links'] if (year=='전체' or x['later_date'].startswith(year)) and (assembly=='전체' or assembly in [x['earlier_assembly'],x['later_assembly']])]
 choices=[x['chunk_id'] for x in evidence]
 return issue,first_text,timeline,countrows,links,rows,choices,forecast
def detail(data,query,cid):
 issue=resolve_issue(query)
 if not issue:return '이슈를 먼저 검색하세요.'
 x=next((x for x in data[issue]['evidence'] if x['chunk_id']==cid),None)
 return json.dumps(x,ensure_ascii=False,indent=2) if x else '근거를 선택하세요.'
def build_app(data):
 try:import gradio as gr
 except ImportError as e:raise RuntimeError('Gradio 미설치. 25_runbook.md의 별도 환경 설치 명령을 실행하세요.') from e
 assemblies=['전체']+sorted({x['assembly'] for v in data.values() for x in v['evidence']})
 with gr.Blocks(title='지방의회 AI 정책 이슈 탐색') as app:
  gr.Markdown('# 지방의회 AI 정책 이슈 탐색\n정책 이슈의 최초 관측 · 타 지방의회의 유사 논의 · 향후 전망 · 회의록 근거')
  with gr.Row():
   query=gr.Textbox(value='지역소멸',label='정책 키워드 또는 질문');button=gr.Button('검색',variant='primary')
  with gr.Row():
   year=gr.Dropdown(['전체','2024','2025'],value='전체',label='연도');assembly=gr.Dropdown(assemblies,value='전체',label='의회')
  status=gr.Markdown();gr.Markdown('## 01 최초 관측');first=gr.Markdown()
  gr.Markdown('## 02 타 지방의회 논의 흐름\n좌표 미검증으로 지역별 건수와 날짜 순 목록을 표시합니다. 연결은 전파를 의미하지 않습니다.')
  timeline=gr.Dataframe(headers=['날짜','의회','회의록명','청크 ID'],interactive=False)
  counts=gr.Dataframe(headers=['의회','검토 후보 관측 수'],interactive=False)
  flows=gr.Dataframe(headers=['선행일','선행의회','후속일','후속의회','공유 목적','분류','검증 상태'],interactive=False)
  gr.Markdown('## 03 AI 향후 예측');forecast=gr.Markdown()
  gr.Markdown('## 04 근거 회의록');evidence=gr.Dataframe(headers=['회의록명','날짜','의회','발췌','청크 ID','검증 상태'],interactive=False)
  pick=gr.Dropdown(choices=[],label='상세 근거 선택');full=gr.Code(language='json',label='원문 상세 근거')
  gr.Markdown(NOTICE)
  def search(q,y,r):
   import pandas as pd
   i,f,t,n,l,e,ch,p=render(data,q,y,r);return i,f,t,pd.DataFrame(n,columns=['??','?? ?? ?? ?']),l,e,gr.Dropdown(choices=ch,value=ch[0] if ch else None),p,detail(data,q,ch[0] if ch else '')
  outputs=[status,first,timeline,counts,flows,evidence,pick,forecast,full]
  button.click(search,[query,year,assembly],outputs);query.submit(search,[query,year,assembly],outputs)
  year.change(search,[query,year,assembly],outputs);assembly.change(search,[query,year,assembly],outputs)
  pick.change(lambda q,i:detail(data,q,i),[query,pick],full);app.load(search,[query,year,assembly],outputs)
 return app
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=c.BASE);p.add_argument('--data',type=Path);p.add_argument('--port',type=int,default=7860);a=p.parse_args()
 path=a.data or c.latest(a.root,23)/'dashboard.json';data=json.loads(path.read_text(encoding='utf-8'))
 app=build_app(data);app.launch(server_name='127.0.0.1',server_port=a.port,share=False,inbrowser=False,theme=__import__('gradio').themes.Soft(primary_hue='teal'),css='body {background:white} .gradio-container {max-width:1200px!important}')

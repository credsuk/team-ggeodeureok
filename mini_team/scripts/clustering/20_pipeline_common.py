"""Shared read-only adapters and exclusive versioned output for steps 20-25."""
import argparse, csv, json, re, sqlite3, sys, time, importlib.util
from pathlib import Path
from datetime import datetime
sys.dont_write_bytecode = True
BASE=Path(__file__).resolve().parent
ISSUES=['지역소멸','청년 일자리','저출생','고령화','기후위기']
TERMS={'지역소멸':['지역소멸','지방소멸','인구감소'], '청년 일자리':['청년 일자리','청년일자리','청년 취업','청년고용'], '저출생':['저출생','저출산','출산 지원','출산지원'], '고령화':['고령화','노인 복지','노인복지','노인돌봄'], '기후위기':['기후위기','기후변화','탄소중립','온실가스']}
SCOPE='2024~2025년 보유 POLICY 자료의 검토 후보 중 최초 관측; 실제 최초 도입 및 전국 최초 미확인'
def read(p,required=()):
 if 'ACTIVE_LOG' in globals(): ACTIVE_LOG['inputs'].append(str(p))
 with Path(p).open(encoding='utf-8-sig',newline='') as f:
  r=csv.DictReader(f)
  if not set(required)<=set(r.fieldnames or []): raise ValueError(f'입력 스키마 오류: {p}, 필요 {required}')
  return list(r)
def dump(p,obj):
 with Path(p).open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2)
def save(p,rows,cols):
 with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
def latest(root,step):
 ps=[f.parent for parent in (root/'results').glob(f'step{step}_*') for f in parent.glob('*/summary.json') if json.loads(f.read_text(encoding='utf-8')).get('status')=='completed']
 if not ps: raise FileNotFoundError(f'{step}단계 완료 결과 없음')
 return max(ps,key=lambda p:p.stat().st_mtime_ns)
def load_step(root,step,name):
 p=latest(root,step)/name
 if 'ACTIVE_LOG' in globals(): ACTIVE_LOG['inputs'].append(str(p))
 return json.loads(p.read_text(encoding='utf-8'))
def helper(root):
 s=importlib.util.spec_from_file_location('readonly_metadata',root/'14_4_2_restore_meeting_metadata.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def score(issue,text):
 text=re.sub(r'\s+',' ',text); compact=text.replace(' ','')
 hits=[t for t in TERMS[issue] if t.replace(' ','') in compact]
 if not hits:return 0,'이슈 표현 없음',''
 at=min(compact.index(t.replace(' ','')) for t in hits);window=compact[max(0,at-120):at+500]
 actions=[t for t in ['지원','추진','조성','확대','운영','개선','시행','도입','구축','양성'] if t in window]
 objects=[t for t in ['사업','정책','계획','센터','주택','교육','수당','일자리','돌봄','에너지','기금','조례'] if t in window]
 penalty=4 if any(t in window for t in ['개의하','성원이','출석확인','의사일정']) else 0
 if len(re.findall(r'\d[\d,]*억|\d[\d,]*만원|\d+쪽',window))>=5:penalty+=4
 n=min(4,len(actions))*2+min(4,len(objects))-penalty
 return n, f'행위 {actions}; 대상/수단 {objects}; 감점 {penalty}; 자동규칙 의미검토 필요',window[:180]
def run(step,fn):
 global ACTIVE_LOG
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=BASE);p.add_argument('--out',type=Path);p.add_argument('--source-root',type=Path,default=BASE.parents[1]/'org2');p.add_argument('--limit',type=int,default=20);a=p.parse_args()
 if not 5<=a.limit<=20:p.error('--limit 5~20')
 parent=(a.out or a.root/'results'/f'step{step}_output').resolve()
 results=(a.root/'results').resolve()
 if not parent.is_relative_to(results) or not parent.name.startswith(f'step{step}_'):p.error('--out은 root/results/step번호_* 신규 경로여야 합니다')
 parent.mkdir(parents=True,exist_ok=True);out=parent/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(exist_ok=False)
 start=time.perf_counter();log={'step':step,'started':datetime.now().isoformat(),'inputs':[],'limitations':[]}
 print(f'{step} 시작: {out}',flush=True)
 ACTIVE_LOG=log
 try:
  fn(a,out,log);log['status']='completed'
 except Exception as e:
  log.update(status='failed',error=repr(e));raise
 finally:
  log.update(ended=datetime.now().isoformat(),elapsed_sec=time.perf_counter()-start);dump(out/'summary.json',log);print(json.dumps(log,ensure_ascii=False),flush=True)

def stream_records(path):
 """Bounded JSON array reader with explicit EOF detection."""
 decoder=json.JSONDecoder();buffer='';pos=0;started=False;eof=False
 with path.open(encoding='utf-8-sig') as f:
  while True:
   if not eof and (pos>=len(buffer) or not buffer[pos:].strip()):
    chunk=f.read(1024*1024);buffer=buffer[pos:]+chunk;pos=0;eof=not chunk
   while pos<len(buffer) and buffer[pos] in ' \r\n\t,':pos+=1
   if not started:
    if pos>=len(buffer) or buffer[pos]!='[':raise ValueError('JSON array required')
    started=True;pos+=1;continue
   if pos<len(buffer) and buffer[pos]==']':return
   if pos>=len(buffer):
    if eof:raise ValueError('Truncated JSON array')
    continue
   try:obj,end=decoder.raw_decode(buffer,pos)
   except json.JSONDecodeError:
    if eof:raise
    chunk=f.read(1024*1024);buffer=buffer[pos:]+chunk;pos=0;eof=not chunk
    continue
   pos=end;yield obj
   if pos>2097152:buffer=buffer[pos:];pos=0

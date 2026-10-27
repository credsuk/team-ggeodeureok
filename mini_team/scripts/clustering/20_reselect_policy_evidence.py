"""Step 20: bounded candidate comparison, readonly cache, original provenance."""
import importlib, sqlite3
from collections import defaultdict
c=importlib.import_module('20_pipeline_common')
def main(a,out,log):
 r=a.root/'results';h=c.helper(a.root);pool={i:{} for i in c.ISSUES}
 for relative in ['competition_submission/competition_submission_audit.csv','competition_submission_validation_17/17_submission_validation_detail.csv','competition_original_evidence_18/18_original_evidence_review.csv','competition_policy_review_19/19_evidence_human_review.csv','competition_policy_review_19/19_issue_human_review.csv']:
  p=r/relative;old=c.read(p);log['inputs'].append(str(p));log.setdefault('prior_review_rows',{})[relative]=len(old)

 files=[r/'policy_issue_trends/issue_first_observed_by_region.csv',r/'policy_issue_trends/issue_evidence_180chars.csv']
 for p in files:
  for row in c.read(p,['issue','chunk_id']):
   if row['issue'] in pool:pool[row['issue']][row['chunk_id']]=row
  log['inputs'].append(str(p))
 links=c.read(r/'policy_discussion_evidence/policy_links_unique.csv',['issue','earlier_chunk_id','later_chunk_id'])
 for row in links:
  if row['issue'] not in pool:continue
  for side in ['earlier','later']:
   cid=row[side+'_chunk_id'];pool[row['issue']].setdefault(cid,{'chunk_id':cid})
 db=r/'all_policy_topics/policy_text_cache.sqlite3';log['inputs'].append(str(db));allrows=[]
 with sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True) as conn:
  conn.execute('PRAGMA query_only=ON')
  if not {'chunk_id','document','assembly','date'}<= {x[1] for x in conn.execute('pragma table_info(policy_docs)')}:raise ValueError('POLICY 캐시 스키마 불일치')
  for issue,items in pool.items():
   ranked=[]
   for cid in items:
    rr=conn.execute('select document,assembly,date,source_key from policy_docs where chunk_id=?',(cid,)).fetchone()
    if not rr:continue
    text,assembly,date,sk=rr;n,reason,excerpt=c.score(issue,text)
    ranked.append(dict(issue=issue,chunk_id=cid,assembly=assembly,date=h.normalize_date(date),source_key=sk,score=n,reason=reason,excerpt=excerpt,document=text))
   # Compare early substantive candidates plus top scoring candidates, bounded at 20.
   eligible=sorted([x for x in ranked if x['score']>=6 and x['date'][:4] in ['2024','2025']],key=lambda x:(x['date'],-x['score']))
   chosen=eligible[:max(5,a.limit//2)]
   for row in sorted(ranked,key=lambda x:-x['score']):
    if len(chosen)>=a.limit:break
    if row not in chosen:chosen.append(row)
   allrows+=chosen
   log.setdefault('candidate_pool',{})[issue]={'pool':len(ranked),'compared':len(chosen)}
 matches=defaultdict(list);targets=defaultdict(list)
 for x in allrows:targets[(h.normalize_assembly(x['assembly']),x['date'])].append(x)
 files_read=records=0
 for p in sorted(a.source_root.rglob('*.json')):
  relevant={k:v for k,v in targets.items() if k[1][:4] in p.stem and k[0] in h.normalize_assembly(p.stem)}
  if not relevant:continue
  print('원문 읽기:',p.name,flush=True);files_read+=1
  for rid,rec in enumerate(c.stream_records(p)):
   records+=1
   group=relevant.get((h.normalize_assembly(rec.get('RASMBLY_NM')),h.normalize_date(rec.get('MTG_DE'))),[])
   if not group:continue
   original=h.html_text(rec.get('MINTS_HTML',''));tight=h.tight(original)
   for x in group:
    anchors=h.get_anchors(x['document'],'');found=[s for s in anchors if s in tight]
    if len(found)<2:continue
    excerpt=x['excerpt'];pos=tight.find(excerpt)
    matches[(x['issue'],x['chunk_id'])].append(dict(meeting=h.normalize(rec.get('MINTS_NM')),source_file=str(p),record_index=rid,original_excerpt=tight[pos:pos+180] if pos>=0 else '',anchors=len(found)))
 selected=[]
 for x in allrows:
  ms=matches[(x['issue'],x['chunk_id'])];x.pop('document')
  x.update(source_match_count=len(ms),meeting='',source_file='',original_excerpt='',status='미확인: 원문 또는 의미 검토 필요')
  if len(ms)==1:
   x.update(ms[0]);x['status']='원문 유일 일치; 정책 적합성 자동후보; 사람 의미검토 미완료'
  elif len(ms)>1:x['status']='복수 원문 후보; 확정 보류'
  x['eligible']=len(ms)==1 and x['score']>=6 and bool(x['original_excerpt']) and bool(x['meeting']) and x['date'][:4] in ['2024','2025']
 for issue in c.ISSUES:
  xs=sorted([x for x in allrows if x['issue']==issue and x['eligible']],key=lambda x:(x['date'],-x['score']))
  selected.append(dict(issue=issue,evidence=xs[0] if xs else None,scope=c.SCOPE,status='자동 선별 최초 관측 후보; 의미검토 필요' if xs else '미확보'))
 c.dump(out/'candidates.json',allrows);c.dump(out/'selected.json',selected)
 c.save(out/'candidates.csv',allrows,list(allrows[0]) if allrows else ['issue','chunk_id','reason'])
 c.save(out/'selected.csv',[dict(issue=x['issue'],status=x['status'],scope=x['scope'],**{k:v for k,v in (x['evidence'] or {}).items() if k not in ['issue','status','scope']}) for x in selected],['issue','status','scope','chunk_id','date','assembly','meeting','original_excerpt','score'])
 log.update(files_read=files_read,records_read=records,candidates=len(allrows),selected=sum(x['evidence'] is not None for x in selected),limitations=['候補集合은 13~14단계 관측/연결 목록이며 전체 캐시 최초를 보장하지 않음','점수는 자동 휴리스틱이며 사람 의미검토 아님'])
if __name__=='__main__':c.run(20,main)

"""Step 21: separate chronology, topic overlap and policy purpose similarity."""
import importlib,itertools,re
c=importlib.import_module('20_pipeline_common')
def main(a,out,log):
 xs=c.load_step(a.root,20,'candidates.json');obs=sorted([x for x in xs if x['eligible']],key=lambda x:(x['issue'],x['date'],x['assembly']))
 old=c.read(a.root/'results/policy_discussion_evidence/policy_links_unique.csv',['issue','earlier_chunk_id','later_chunk_id'])
 pairs={(x['issue'],x['earlier_chunk_id'],x['later_chunk_id']) for x in old};links=[]
 # Only concrete repeated service/purpose expressions can advance beyond topic-only.
 purposes=['청년창업','취업지원','직업훈련','노인돌봄','노인일자리','출산지원','산후조리','보육지원','정주여건','생활인구','탄소중립','온실가스감축','재생에너지','에너지전환']
 for x,y in itertools.combinations(obs,2):
  if x['issue']!=y['issue'] or x['assembly']==y['assembly'] or x['date']>=y['date']:continue
  tx=x['original_excerpt'].replace(' ','');ty=y['original_excerpt'].replace(' ','');shared=[t for t in purposes if t in tx and t in ty]
  links.append(dict(issue=x['issue'],earlier_id=x['chunk_id'],later_id=y['chunk_id'],earlier_date=x['date'],later_date=y['date'],earlier_assembly=x['assembly'],later_assembly=y['assembly'],different_assembly=True,chronology=True,shared_purpose=shared,classification='구체 정책 목적 표현 공유; 동일정책 미확인' if shared else '주제만 동일',status='후속 유사 논의 후보; 확산 여부 미확인',from_step14=(x['issue'],x['chunk_id'],y['chunk_id']) in pairs))
 c.dump(out/'observations.json',obs);c.dump(out/'links.json',links)
 c.save(out/'observations.csv',obs,['issue','date','assembly','meeting','chunk_id','original_excerpt','status'])
 c.save(out/'links.csv',links,['issue','earlier_id','later_id','earlier_date','later_date','earlier_assembly','later_assembly','shared_purpose','classification','status','from_step14'])
 c.dump(out/'regions.json',[{'assembly':v,'geography':'원문 RASMBLY_NM 확인; 좌표 미검증; 표로 표시'} for v in sorted({x['assembly'] for x in obs})])
 c.dump(out/'issue_status.json',[{'issue':i,'status':'후속 유사 논의 후보' if any(x['issue']==i and x['shared_purpose'] for x in links) else '적합한 지역 간 연결 미확보'} for i in c.ISSUES])
 log.update(observations=len(obs),links=len(links),step14_candidates=len(old),limitations=['공유 목적 표현은 정책 동일성 또는 인과 전파 증거가 아님','정확한 지리 좌표 미검증으로 지역 표/차트 사용'])
if __name__=='__main__':c.run(21,main)

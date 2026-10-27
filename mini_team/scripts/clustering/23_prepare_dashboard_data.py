"""Step 23: lightweight dashboard package, no runtime DB dependency."""
import importlib
c=importlib.import_module('20_pipeline_common')
def main(a,out,log):
 selected=c.load_step(a.root,20,'selected.json');obs=c.load_step(a.root,21,'observations.json');links=c.load_step(a.root,21,'links.json');aud=c.load_step(a.root,22,'audit.json');data={}
 monthly=c.read(a.root/'results/policy_issue_trends/issue_monthly_trends.csv',['issue','year_month','assembly','chunk_count']);regions=c.read(a.root/'results/policy_issue_trends/issue_region_summary.csv',['issue','assembly','chunk_count'])
 for s in selected:
  i=s['issue'];audit=next(x for x in aud if x['issue']==i)
  data[i]=dict(first=s,evidence=[x for x in obs if x['issue']==i],links=[x for x in links if x['issue']==i],forecast=audit,monthly=[x for x in monthly if x['issue']==i],region_counts=[x for x in regions if x['issue']==i])
 c.dump(out/'dashboard.json',data);c.dump(out/'spec.json',{'version':1,'scope':c.SCOPE,'geography':'검증된 원문 의회명 표/지역별 건수 차트; 좌표 미검증','runtime':'JSON만 읽음; DB/LLM/API 접근 없음'})
 c.dump(out/'missing.json',[{'issue':i,'first_missing':not v['first']['evidence'],'prediction_missing':not v['forecast']['prediction_available'],'evidence_count':len(v['evidence']),'meaning_review':'미완료'} for i,v in data.items()]);log.update(issues=len(data),monthly_rows=len(monthly),region_rows=len(regions),limitations=['좌표 검증 없이 지도를 만들지 않음; 지역 표 및 차트 대체'])
if __name__=='__main__':c.run(23,main)

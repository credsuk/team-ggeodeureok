"""Step 25: reproducible validation; never labels unavailable Gradio as tested."""
import ast,importlib,json,unittest,subprocess,sys
from pathlib import Path
c=importlib.import_module('20_pipeline_common');dash=importlib.import_module('24_dashboard_v1')
def main(a,out,log):
 data=c.load_step(a.root,23,'dashboard.json');results=[]
 def check(name,fn):
  try:fn();results.append({'check':name,'status':'PASS'})
  except Exception as e:results.append({'check':name,'status':'FAIL','error':repr(e)})
 def require(ok):
  if not ok:raise AssertionError('검증 조건 불충족')
 for p in sorted(a.root.glob('2[0-5]_*.py')):check('syntax '+p.name,lambda p=p:ast.parse(p.read_text(encoding='utf-8-sig')))
 check('5 issues',lambda:require(set(data)==set(c.ISSUES)))
 empty_root=out/'missing_input_root';empty_root.mkdir()
 for step,script in [(20,'20_reselect_policy_evidence.py'),(21,'21_build_issue_flow.py'),(22,'22_generate_final_submission.py'),(23,'23_prepare_dashboard_data.py')]:
  def cli_failure(script=script):
    proc = subprocess.run(
        [sys.executable, '-B', str(a.root / script),
         '--root', str(empty_root)],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=30
    )
    require(proc.returncode!=0)
  check('CLI missing input '+str(step),cli_failure)
 fixture=out/'truncated.json';fixture.write_text('[{"a":',encoding='utf-8')
 def bad_json():
  try:list(c.stream_records(fixture))
  except (ValueError,json.JSONDecodeError):return
  raise AssertionError('?? JSON? ???')
 check('truncated JSON safely fails',bad_json)
 fixture_csv=out/'wrong_schema.csv';fixture_csv.write_text('wrong\nvalue\n',encoding='utf-8')
 def bad_schema():
  try:c.read(fixture_csv,['issue','chunk_id'])
  except ValueError:return
  raise AssertionError('??? ???? ???')
 check('missing columns safely fail',bad_schema)
 def missing_input():
  try:c.read(out/'not_present.csv')
  except FileNotFoundError:return
  raise AssertionError('?? ??? ???')
 check('missing input safely fails',missing_input)
 for relative in ['competition_submission/competition_submission_audit.csv','competition_submission_validation_17/17_submission_validation_detail.csv','competition_original_evidence_18/18_original_evidence_review.csv','competition_policy_review_19/19_evidence_human_review.csv','competition_policy_review_19/19_issue_human_review.csv']:
  rows=c.read(a.root/'results'/relative);log.setdefault('prior_review_inputs',{})[relative]=len(rows)

 for i in c.ISSUES:
  check('search '+i,lambda i=i:require(dash.render(data,i)[0]==i))
  check('forecast length '+i,lambda i=i:require(0<len(data[i]['forecast']['prediction'])<=200))
  check('forecast available '+i,lambda i=i:require(data[i]['forecast']['prediction_available']))
  check('detail '+i,lambda i=i:require(all(json.loads(dash.detail(data,i,x['chunk_id']))['chunk_id']==x['chunk_id'] for x in data[i]['evidence'])))
  check('year filtering '+i,lambda i=i:require(all(row[0].startswith('2024') for row in dash.render(data,i,'2024')[2])))
  check('empty filter '+i,lambda i=i:require(not dash.render(data,i,'전체','존재하지 않는 의회')[2]))
 for q,i in [('청년 취업','청년 일자리'),('출산 지원','저출생'),('노인 복지','고령화')]:check('synonym '+q,lambda q=q,i=i:require(dash.resolve_issue(q)==i))
 check('out of scope',lambda:require(dash.resolve_issue('야구 경기') is None))
 check('procedural rejection',lambda:require(c.score('기후위기','성원이 되었으므로 기후위기 의사일정을 개의하겠습니다')[0]<6))
 check('substantive preference',lambda:require(c.score('청년 일자리','청년일자리 사업 추진 및 취업 지원 센터 운영 확대 계획')[0]>=6))
 sub=c.read(c.latest(a.root,22)/'submission_DRAFT.csv')
 check('five columns/five rows',lambda:require(len(sub)==5 and list(sub[0])==['이슈키워드','등장시점','타 지방의회로의 흐름','향후 예측','근거 회의록']))
 base=a.root/'results/step25_validation_baseline_v1/protected_manifest.json'
 changes=[]
 for name,size,mtime in json.loads(base.read_text(encoding='utf-8')):
  p=Path(name)
  if not p.exists() or (p.stat().st_size,p.stat().st_mtime_ns)!=(size,mtime):changes.append(name)
 check('protected size+mtime unchanged',lambda:require(not changes))
 try:
   import gradio
 except ImportError:results.append({'check':'Gradio build/server/browser integration','status':'BLOCKED','reason':'Gradio 미설치; pip 네트워크 연결 WinError 10013. 보안 제한 우회하지 않음.'})
 else:
  check('Gradio app build',lambda:dash.build_app(data));results.append({'check':'Gradio server/browser integration','status':'NOT_RUN','reason':'별도 실제 브라우저 검증 필요'})
 c.dump(out/'validation.json',results);c.dump(out/'protected_changes.json',changes)
 log.update(checks=len(results),passed=sum(x['status']=='PASS' for x in results),failed=sum(x['status']=='FAIL' for x in results),limitations=[x for x in results if x['status'] in ['BLOCKED','NOT_RUN']])
 report='# 25단계 검증 결과\n\n'+ '\n'.join(f"- {x['status']}: {x['check']} {x.get('error',x.get('reason',''))}" for x in results)+'\n\n원본 보호 검사는 최초 스냅샷의 크기와 수정 시각 비교입니다. 전체 내용 해시 검사는 수행하지 않았습니다.\n'
 (out/'validation_report.md').write_text(report,encoding='utf-8')
 if log['failed']:raise RuntimeError('검증 실패: validation.json 참조')
if __name__=='__main__':c.run(25,main)

"""Step 22: five-column draft with unmodified successful local forecasts."""
import importlib
c=importlib.import_module('20_pipeline_common')
COLS=['이슈키워드','등장시점','타 지방의회로의 흐름','향후 예측','근거 회의록']
def main(a,out,log):
 selected=c.load_step(a.root,20,'selected.json');links=c.load_step(a.root,21,'links.json');preds=c.read(a.root/'results/policy_prediction_comparison/policy_trend_ab_results.csv',['issue_keyword','prediction','model','status'])
 rows=[];audits=[]
 for s in selected:
  issue=s['issue'];e=s['evidence'];ps=[p for p in preds if p['issue_keyword']==issue and p['status'].lower() in ['ok','success','completed'] and not p.get('error') and 0<len(p['prediction'].strip())<=200 and p['model']!='gemma3:12b']
  ps.sort(key=lambda p:(p['model']!='qwen3:8b',not p.get('mode','').startswith('B')));p=ps[0] if ps else None
  ls=[x for x in links if x['issue']==issue and x['shared_purpose']];f=ls[0] if ls else None
  rows.append({'이슈키워드':issue,'등장시점':f"{e['date']} {e['assembly']} (자동 적합 후보 중 최초; 의미 검토 필요)" if e else '미확인/근거 부족/검증 필요','타 지방의회로의 흐름':f"{f['earlier_date']} {f['earlier_assembly']} → {f['later_date']} {f['later_assembly']}: {', '.join(f['shared_purpose'])}; 후속 유사 논의 후보; 전파 미입증" if f else '적합 연결 미확보; 확산 여부 미확인','향후 예측':p['prediction'].strip() if p else '미확인/근거 부족/검증 필요','근거 회의록':f"{e['meeting']} / {e['date']} / {e['assembly']} / {e['original_excerpt']} / 청크 {e['chunk_id']}" if e else '미확인/근거 부족/검증 필요'})
  audits.append(dict(issue=issue,model=p['model'] if p else '',prediction=rows[-1]['향후 예측'],prediction_chars=len(rows[-1]['향후 예측']),prediction_available=bool(p),status='제출 초안; 사람 의미검토 및 예측 근거 정합성 검토 필요',ready_for_submission=False,scope=c.SCOPE,evidence_status=s['status'],forecast_source_evidence_ids=p.get('evidence_ids','') if p else '',forecast_reselected_evidence_alignment='미검증'))
 c.save(out/'submission_DRAFT.csv',rows,COLS);c.dump(out/'submission.json',rows);c.dump(out/'audit.json',audits);c.save(out/'audit.csv',audits,list(audits[0]))
 log.update(rows=len(rows),columns=COLS,predictions_available=sum(x['prediction_available'] for x in audits),limitations=['2024~2025 보유자료 기준; 전국 최초/실제 도입 미확인','기존 모델 출력 재사용; 재선정 근거와 정합성 미검증; AI 모의 점수로 정확성 확정하지 않음'])
if __name__=='__main__':c.run(22,main)

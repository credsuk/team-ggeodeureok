# -*- coding: utf-8 -*-
"""18단계: 17단계 제출 근거 청크를 원본 org2 JSON HTML에서 검증한다.

입력(읽기만 함)
 - results/competition_submission/competition_submission_audit.csv
 - results/competition_submission/competition_submission_draft.csv
 - results/all_policy_topics/policy_text_cache.sqlite3
 - org2/**/*.json (최상위 list, MTG_DE, RASMBLY_NM, MINTS_HTML, MTGNM, MINTS_NM)
출력: results/competition_original_evidence_18/

원본 소스가 확인되지 않은 회의록명/정책 전파는 절대 확정하지 않는다.
14_4_2_restore_meeting_metadata.py의 JSON 스트리밍/정규화 함수를 재사용한다.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

BASE = Path(r'C:\mini_team\scripts\clustering')
SOURCE_ROOT = Path(r'C:\mini_team\org2')
ISSUES = ['지역소멸', '청년 일자리', '저출생', '고령화', '기후위기']
MATCH_COLS = ['이슈키워드','근거역할','chunk_id','캐시존재','캐시메타일치',
              '원본대조상태','원본후보수','강한일치후보수','회의록날짜','의회',
              '회의명_MTGNN','회의록명_MINTS_NM','원본파일','원본레코드번호',
              '매칭앵커수','전체앵커수','원문근거_180자','근거발췌_원본문맥',
              '키워드_단순포함','주제의미검토','정책전파_직접증거','검토메모']
CAND_COLS = ['chunk_id','이슈키워드','근거역할','원본파일','원본레코드번호','회의록날짜',
             '의회','회의명_MTGNN','회의록명_MINTS_NM','매칭앵커수','전체앵커수',
             '강한일치','매칭앵커_미리보기','원본근거_180자','원본문맥_350자']


def log(msg, t0):
    print(f'[{datetime.now():%H:%M:%S}] {msg} | 경과 {time.perf_counter()-t0:.1f}초',flush=True)


def load_csv(p, required):
    if not p.is_file():
        raise FileNotFoundError(f'필수 파일 없음: {p}')
    with p.open('r', encoding='utf-8-sig', newline='') as f:
        reader=csv.DictReader(f)
        missing=set(required)-set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{p.name} 누락 컬럼: {sorted(missing)}; 현재: {reader.fieldnames}')
        return list(reader)


def write_csv(path, rows, names):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        wr=csv.DictWriter(f,fieldnames=names,extrasaction='ignore')
        wr.writeheader();wr.writerows(rows)


def helper_module(script):
    if not script.is_file():
        raise FileNotFoundError(f'14-4-2 원본 비교 코드가 필요합니다: {script}')
    spec=importlib.util.spec_from_file_location('original_html_utils_14_4_2',script)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def snippet(original, anchor, radius=145):
    if not anchor: return ''
    idx=original.find(anchor)
    if idx < 0: return ''
    return original[max(0,idx-radius):min(len(original),idx+len(anchor)+radius)]


def parse_int(value, fallback=0):
    try: return int(value)
    except (ValueError,TypeError): return fallback


def main():
    ap=argparse.ArgumentParser(description='18단계 원본 JSON 회의록 근거 복원/검증')
    ap.add_argument('--root',type=Path,default=BASE,help='스크립트 및 results 위치')
    ap.add_argument('--source-root',type=Path,default=SOURCE_ROOT)
    ap.add_argument('--out',type=Path,default=None)
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--max-files',type=int,default=0,help='진단 시 파일 수 제한(0=전체)')
    args=ap.parse_args()
    t0=time.perf_counter()
    paths={
      'audit':args.root/'results/competition_submission/competition_submission_audit.csv',
      'draft':args.root/'results/competition_submission/competition_submission_draft.csv',
      'cache':args.root/'results/all_policy_topics/policy_text_cache.sqlite3',
      'helpers':args.root/'14_4_2_restore_meeting_metadata.py',
    }
    files=sorted(args.source_root.rglob('*.json')) if args.source_root.is_dir() else []
    log('[1/4] 입력 경로 점검',t0)
    for key,path in paths.items():
        print(f'  {key}: {"있음" if path.is_file() else "없음"} / {path}')
    print(f'  원본 JSON: {len(files)}개 / {args.source_root}')
    if args.check_only: return 0
    absent=[str(v) for v in paths.values() if not v.is_file()]
    if absent or not files: raise FileNotFoundError(f'필수 입력 부족: {absent}; 원본 JSON {len(files)}개')
    if args.max_files>0: files=files[:args.max_files]
    h=helper_module(paths['helpers'])
    audits=load_csv(paths['audit'],['이슈키워드','최초근거청크ID','초기근거청크ID','후속근거청크ID'])
    drafts=load_csv(paths['draft'],['이슈키워드','등장시점','근거 회의록'])
    if len(audits)!=5 or set(r['이슈키워드'] for r in audits)!=set(ISSUES):
        raise ValueError('16단계 감사표에 정확히 5개 이슈가 있어야 합니다.')

    targets={}
    for r in audits:
        issue=r['이슈키워드']
        for role,col in [('최초관측','최초근거청크ID'),('선행','초기근거청크ID'),('후속','후속근거청크ID')]:
            cid=(r.get(col) or '').strip()
            if not cid: continue
            obj=targets.setdefault(cid,{'usages':[]})
            obj['usages'].append((issue,role))
            if role=='최초관측':
                obj['expected_date']=h.normalize_date(r.get('최초관측일_POLICY범위'))
                obj['expected_assembly']=h.normalize_assembly(r.get('최초관측의회'))
                obj['issue']=issue
    log(f'[1/4] 이슈 5개 / 고유 근거 청크 {len(targets)}개 / 역할 기준 {sum(len(v["usages"]) for v in targets.values())}건',t0)

    conn=sqlite3.connect(f'file:{paths["cache"].as_posix()}?mode=ro',uri=True)
    try:
        cids=list(targets)
        for i in range(0,len(cids),400):
            sub=cids[i:i+400]
            placeholders=','.join('?' for _ in sub)
            for cid,doc,assembly,date in conn.execute(
                f'SELECT chunk_id,document,assembly,date FROM policy_docs WHERE chunk_id IN ({placeholders})',sub):
                v=targets[cid]
                v.update(document=doc or '',cache_assembly=h.normalize_assembly(assembly),
                         cache_date=h.normalize_date(date))
    finally: conn.close()
    for cid,v in targets.items():
        v['expected_date']=v.get('expected_date') or v.get('cache_date','')
        v['expected_assembly']=v.get('expected_assembly') or v.get('cache_assembly','')
        v['anchors']=h.get_anchors(v.get('document',''),'') if v.get('document') else []
        v['cache_consistent']=bool(v.get('document') and
                 v.get('expected_date')==v.get('cache_date') and
                 v.get('expected_assembly')==v.get('cache_assembly'))

    groups=defaultdict(list)
    for cid,v in targets.items():
        if v.get('cache_consistent') and v['anchors']:
            groups[(v['expected_date'][:4],v['expected_assembly'],v['expected_date'])].append(cid)
    matches=defaultdict(list)
    file_errors=[]
    stats=Counter()
    log('[2/4] 원본 HTML 회의록 대조 시작',t0)
    for file_num,path in enumerate(files,1):
        file_groups=[(k,ids) for k,ids in groups.items() if k[0] in path.stem and k[1] in h.normalize_assembly(path.stem)]
        if not file_groups:
            stats['files_skipped_irrelevant']+=1
            continue
        lookup=defaultdict(list)
        for (y,assembly,date),ids in file_groups: lookup[(assembly,date)].extend(ids)
        scanned=0
        try:
            for rid,rec in enumerate(h.stream_json_list(path)):
                scanned+=1
                if not isinstance(rec,dict): continue
                day=h.normalize_date(rec.get('MTG_DE'))
                assembly=h.normalize_assembly(rec.get('RASMBLY_NM'))
                ids=lookup.get((assembly,day),[])
                if not ids: continue
                orig=h.tight(h.html_text(rec.get('MINTS_HTML','')))
                if not orig: continue
                for cid in ids:
                    target=targets[cid]
                    found=[a for a in target['anchors'] if a in orig]
                    if not found: continue
                    strong=len(found)>=2 or max(map(len,found))>=72
                    # Extract actual matched HTML text around strongest anchor.
                    anchor=max(found,key=len)
                    context=snippet(orig,anchor,145)
                    item={
                      'chunk_id':cid,'원본파일':str(path),'원본레코드번호':rid,
                      '회의록날짜':day,'의회':h.normalize(rec.get('RASMBLY_NM')),
                      '회의명_MTGNN':h.normalize(rec.get('MTGNM')),
                      '회의록명_MINTS_NM':h.normalize(rec.get('MINTS_NM')),
                      '매칭앵커수':len(found),'전체앵커수':len(target['anchors']),
                      '강한일치':bool(strong),'매칭앵커_미리보기':anchor[:110],
                      '원본근거_180자':context[:180], '원본문맥_350자':context[:350]
                    }
                    matches[cid].append(item)
            stats['files_scanned']+=1;stats['records_read']+=scanned
            log(f'  [{file_num}/{len(files)}] {path.name} / {scanned:,}레코드',t0)
        except Exception as exc:
            file_errors.append({'file':str(path),'error':f'{type(exc).__name__}: {exc}'})
            log(f'  [{file_num}/{len(files)}] 읽기 실패: {path.name} / {exc}',t0)
    for cid, vals in matches.items():
        # A repeated file or minutes record is never automatically collapsed.
        vals.sort(key=lambda r:(r['원본파일'],r['원본레코드번호']))

    rows=[]; all_candidates=[]
    statuses=Counter()
    for cid,v in targets.items():
        candidates=matches.get(cid,[])
        strong=[r for r in candidates if r['강한일치']]
        if not v.get('document'): status='캐시누락'
        elif not v['cache_consistent']: status='캐시메타불일치'
        elif len(strong)==1: status='원본유일강한일치'
        elif len(strong)>1: status='원본강한일치복수'
        elif candidates: status='약한일치_검토필요'
        else: status='원본미발견'
        statuses[status]+=1
        winner=strong[0] if status=='원본유일강한일치' else {}
        for issue,role in v['usages']:
            issue_terms={'지역소멸':['지역소멸','지방소멸','인구감소'],
                         '청년 일자리':['청년','일자리','취업','고용'],
                         '저출생':['저출생','저출산','출산','난임'],
                         '고령화':['고령','노인','어르신','장기요양'],
                         '기후위기':['기후','탄소','온실가스','환경']}[issue]
            has_term=any(x in v.get('document','') for x in issue_terms)
            rows.append({
              '이슈키워드':issue,'근거역할':role,'chunk_id':cid,
              '캐시존재':'확인' if v.get('document') else '누락',
              '캐시메타일치':'일치' if v['cache_consistent'] else '불일치/확인필요',
              '원본대조상태':status,'원본후보수':len(candidates),'강한일치후보수':len(strong),
              '회의록날짜':winner.get('회의록날짜',''),'의회':winner.get('의회',''),
              '회의명_MTGNN':winner.get('회의명_MTGNN',''),
              '회의록명_MINTS_NM':winner.get('회의록명_MINTS_NM',''),
              '원본파일':winner.get('원본파일',''),
              '원본레코드번호':winner.get('원본레코드번호',''),
              '매칭앵커수':winner.get('매칭앵커수',''),'전체앵커수':winner.get('전체앵커수',''),
              '원문근거_180자':winner.get('원본근거_180자',''),
              '근거발췌_원본문맥':winner.get('원본문맥_350자',''),
              '키워드_단순포함':'예' if has_term else '아니오',
              '주제의미검토':'미검증(수동 검토 필요)',
              '정책전파_직접증거':'미검증',
              '검토메모':'원본 일치 여부와 별개로 발언이 이슈의 실질적인 정책 논의인지 확인',
            })
            for item in candidates:
                all_candidates.append({'이슈키워드':issue,'근거역할':role,**item})
    log('[3/4] 복원 결과와 검토용 표 생성',t0)
    out=args.out or args.root/'results/competition_original_evidence_18'
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'18_original_evidence_review.csv',rows,MATCH_COLS)
    write_csv(out/'18_original_match_candidates.csv',all_candidates,CAND_COLS)

    # Copies verified information only; labels are explicit that all items remain a draft.
    by_issue_role={(r['이슈키워드'],r['근거역할']):r for r in rows}
    revised=[]
    for entry in drafts:
        row=dict(entry);issue=row['이슈키워드']
        first=by_issue_role.get((issue,'최초관측'))
        if first and first['원본대조상태']=='원본유일강한일치':
            row['근거 회의록'] += (f" [18단계 최초근거 원본 확인] "
                 f"{first['회의록명_MINTS_NM'] or '(원본 회의록명 공란)'} / "
                 f"{first['회의록날짜']} / {first['의회']} / "
                 f"근거: {first['원문근거_180자']}")
        else:
            row['근거 회의록'] += ' [18단계 최초근거 원본 확인 보류]'
        revised.append(row)
    write_csv(out/'18_submission_enriched_NOT_FINAL.csv',revised,list(drafts[0].keys()))
    totals=Counter(r['원본대조상태'] for r in rows)
    summary={
      'created_at':datetime.now().isoformat(timespec='seconds'),
      'result':'EVIDENCE_REVIEW_NOT_FINAL',
      'unique_chunks':len(targets),'evidence_roles':len(rows),
      'by_role_status':dict(totals),'by_unique_chunk_status':dict(statuses),
      'json_files_found':len(files),'files_scanned':stats['files_scanned'],
      'json_records_read':stats['records_read'],'scanning_errors':file_errors,
      'elapsed_sec':round(time.perf_counter()-t0,2),
      'note':'키워드 단순 포함은 주제 적합성 검증이 아니며, 시간 선후·유사도는 정책 전파 입증이 아닙니다. 원본 HTML과의 부분문자열 강한 일치만 자동 판정합니다.',
    }
    (out/'18_original_evidence_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    log('[4/4] 완료: '+str(out),t0)
    print('  청크:',len(targets),' / 역할:',len(rows),' / 결과:',dict(totals))
    if file_errors: print('  주의: 읽기 실패 JSON',len(file_errors),'개. summary 참고')
    print('  모든 제출 초안은 NOT_FINAL 상태입니다.')
    return 0

if __name__=='__main__':
    try: sys.exit(main())
    except (OSError,ValueError,sqlite3.Error) as exc:
        print(f'[오류] {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)

# -*- coding: utf-8 -*-
"""19단계: 대회 제출 근거 의미 검토표 및 최종 보류/검토 판정.

입력(기존 파일 읽기 전용):
  results/competition_submission/competition_submission_draft.csv
  results/competition_submission/competition_submission_audit.csv
  results/competition_original_evidence_18/18_original_evidence_review.csv
선택:
  results/competition_original_evidence_18/18_original_match_candidates.csv

실행:
  python 19_review_policy_evidence.py --check-only
  python 19_review_policy_evidence.py

원칙:
- 원문 일치, 이슈의 의미 적합성, 지역간 정책 동일성/실제 전파, 전망 타당성은 각각 다른 판단.
- 키워드 포함은 의미 검증이 아님. '유사'는 '전파'가 아님.
- 리뷰 점수/판정은 자동으로 합격 처리하지 않음. CSV에 사람이 직접 판단을 입력하고 재실행.
- 기존의 리뷰 내용을 덮어쓰지 않음; 행 ID가 같은 검토 입력을 보존.
- 최종본 대신 NOT_FINAL 초안과 검토 보고서 생성.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

BASE = Path(r'C:\mini_team\scripts\clustering')
ISSUES = ['지역소멸', '청년 일자리', '저출생', '고령화', '기후위기']
ROLES = ['최초관측', '선행', '후속']
EVIDENCE_COLS = [
    'review_id','이슈키워드','근거역할','chunk_id','원본대조상태','원본후보수',
    '회의록날짜','의회','회의명_MTGNN','회의록명_MINTS_NM','원본파일',
    '원문근거_180자','근거발췌_원본문맥','키워드_단순포함',
    '원문출처_기계검사','주제관련_자동참고','중복정책역할',
    '사람검토_주제적합','사람검토_정책구체성','사람검토_근거문장정확','사람검토_의견',
    '근거검토판정',
]
ISSUE_COLS = [
    '이슈키워드','최초관측근거ID','선행근거ID','후속근거ID','선정예측',
    '예측글자수','예측200자준수','최초관측원본출처','선행원본출처','후속원본출처',
    '참고_단순이슈키워드','사람검토_최초정책발언','사람검토_선후정책동일성',
    '사람검토_지역간전파직접증거','사람검토_예측근거정합성',
    '사람검토_전망논리성','사람검토_비고','최종근거판정','미해결항목',
]
VALID = {'확인','부적합','불확실','미검토'}


def stamp(msg, start):
    print(f'[{datetime.now():%H:%M:%S}] {msg} | 경과 {time.perf_counter()-start:.1f}초', flush=True)


def load(path, needed):
    if not path.is_file():
        raise FileNotFoundError(f'입력 파일 없음: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rdr = csv.DictReader(f)
        miss = set(needed) - set(rdr.fieldnames or [])
        if miss:
            raise ValueError(f'{path.name} 누락 컬럼: {sorted(miss)} / 현재: {rdr.fieldnames}')
        return list(rdr)


def save(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def clean(v):
    return re.sub(r'\s+', ' ', str(v or '')).strip()


def good_judgement(row, field):
    val = clean(row.get(field))
    return val if val in VALID else '미검토'


def read_existing(path, key):
    if not path.exists():
        return {}
    old = load(path, key)
    return {tuple(clean(r.get(k)) for k in key): r for r in old}


def keep_review(row, older, cols):
    for key in cols:
        if clean(older.get(key)):
            row[key] = older[key]


def source_status(record):
    if record is None:
        return '없음'
    st = clean(record.get('원본대조상태'))
    if st == '원본유일강한일치' and clean(record.get('회의록명_MINTS_NM')):
        return '유일원본확인'
    if st == '원본유일강한일치':
        return '유일원본확인_회의록명공란'
    if st == '원본강한일치복수':
        return '원본복수후보'
    return st or '미확인'


def main():
    pa = argparse.ArgumentParser(description='19단계 정책 근거 의미 검토표 생성 및 검토 결과 재집계')
    pa.add_argument('--root', type=Path, default=BASE)
    pa.add_argument('--out', type=Path, default=None)
    pa.add_argument('--check-only', action='store_true')
    args = pa.parse_args()
    start = time.perf_counter()
    root = args.root / 'results'
    inp = {
        '제출 초안': root / 'competition_submission/competition_submission_draft.csv',
        '16단계 감사': root / 'competition_submission/competition_submission_audit.csv',
        '18단계 원문검토': root / 'competition_original_evidence_18/18_original_evidence_review.csv',
    }
    out = args.out or root / 'competition_policy_review_19'
    stamp('[1/4] 입력 경로 확인', start)
    for k, p in inp.items():
        print(f'  {k}: {"있음" if p.is_file() else "없음"} / {p}')
    if args.check_only:
        return 0
    drafts = load(inp['제출 초안'], ['이슈키워드','등장시점','타 지방의회로의 흐름','향후 예측','근거 회의록'])
    audits = load(inp['16단계 감사'], ['이슈키워드','최초근거청크ID','초기근거청크ID','후속근거청크ID'])
    evidence = load(inp['18단계 원문검토'], ['이슈키워드','근거역할','chunk_id','원본대조상태','원문근거_180자'])
    if (len(drafts) != 5 or len(audits) != 5 or
        {r['이슈키워드'] for r in drafts} != set(ISSUES) or
        {r['이슈키워드'] for r in audits} != set(ISSUES)):
        raise ValueError('16단계 CSV는 정확히 5개 고유 정책 이슈가 있어야 합니다.')
    by_draft = {r['이슈키워드']:r for r in drafts}
    by_audit = {r['이슈키워드']:r for r in audits}
    by_evidence = {}
    reused = defaultdict(set)
    for e in evidence:
        identity = (clean(e.get('이슈키워드')), clean(e.get('근거역할')), clean(e.get('chunk_id')))
        if identity in by_evidence:
            raise ValueError(f'18단계 원문검토에 중복 역할 행 발견: {identity}')
        by_evidence[identity] = e
        reused[clean(e.get('chunk_id'))].add(clean(e.get('이슈키워드')))
    stamp(f'[2/4] 제출 5이슈 / 원본대조 역할 {len(evidence)}건 분석', start)

    evidence_path = out / '19_evidence_human_review.csv'
    issue_path = out / '19_issue_human_review.csv'
    old_e = read_existing(evidence_path, ['review_id'])
    old_i = read_existing(issue_path, ['이슈키워드'])
    evidence_out = []
    for r in evidence:
        issue, role, cid = (clean(r.get(k)) for k in ('이슈키워드','근거역할','chunk_id'))
        rid = f'{issue}|{role}|{cid}'
        row = {c: clean(r.get(c)) for c in EVIDENCE_COLS}
        row['review_id'] = rid
        row['원문출처_기계검사'] = source_status(r)
        row['주제관련_자동참고'] = '키워드포함(의미판단불가)' if r.get('키워드_단순포함') == '예' else '키워드없음(의미판단불가)'
        row['중복정책역할'] = ', '.join(sorted(reused[cid])) if len(reused[cid]) > 1 else ''
        for f in ['사람검토_주제적합','사람검토_정책구체성','사람검토_근거문장정확']:
            row[f] = '미검토'
        row['사람검토_의견'] = ''
        if (rid,) in old_e:
            keep_review(row, old_e[(rid,)], ['사람검토_주제적합','사람검토_정책구체성','사람검토_근거문장정확','사람검토_의견'])
        judgements = [good_judgement(row, f) for f in
                      ('사람검토_주제적합','사람검토_정책구체성','사람검토_근거문장정확')]
        if '부적합' in judgements:
            row['근거검토판정'] = '부적합'
        elif row['원문출처_기계검사'] != '유일원본확인':
            row['근거검토판정'] = '검토보류_원본확인'
        elif all(v == '확인' for v in judgements):
            row['근거검토판정'] = '원문근거검토완료'
        else:
            row['근거검토판정'] = '검토보류_의미평가'
        evidence_out.append(row)

    by_new_evidence = {(r['이슈키워드'],r['근거역할'],r['chunk_id']):r for r in evidence_out}
    issue_out = []
    for issue in ISSUES:
        a, d = by_audit[issue], by_draft[issue]
        ids = {
            '최초관측':clean(a.get('최초근거청크ID')),
            '선행':clean(a.get('초기근거청크ID')),
            '후속':clean(a.get('후속근거청크ID')),
        }
        lookup = {role:by_new_evidence.get((issue,role,cid)) if cid else None
                  for role,cid in ids.items()}
        prediction = clean(d.get('향후 예측'))
        row = {c:'' for c in ISSUE_COLS}
        row.update({
            '이슈키워드':issue, '최초관측근거ID':ids['최초관측'],
            '선행근거ID':ids['선행'], '후속근거ID':ids['후속'],
            '선정예측':prediction,'예측글자수':len(prediction),
            '예측200자준수':'통과' if 0<len(prediction)<=200 else '미준수',
            '최초관측원본출처':source_status(lookup['최초관측']),
            '선행원본출처':source_status(lookup['선행']),
            '후속원본출처':source_status(lookup['후속']),
            '참고_단순이슈키워드':'; '.join(f'{role}:{lookup[role]["주제관련_자동참고"]}'
                                     for role in ROLES if lookup[role]),
        })
        judg = ['사람검토_최초정책발언','사람검토_선후정책동일성',
                '사람검토_지역간전파직접증거','사람검토_예측근거정합성','사람검토_전망논리성']
        for f in judg: row[f] = '미검토'
        if (issue,) in old_i:
            keep_review(row, old_i[(issue,)], judg+['사람검토_비고'])

        issues = []
        if row['예측200자준수'] != '통과': issues.append('예측문 200자 제한/빈값')
        first = lookup['최초관측']
        if not first or first['근거검토판정'] != '원문근거검토완료':
            issues.append('최초관측 원문/주제 검토 보류')
        if good_judgement(row, '사람검토_최초정책발언') != '확인':
            issues.append('첫 등장 근거의 정책 논의 적합성 미확정')
        if not lookup['선행'] or not lookup['후속']:
            issues.append('지역 간 선후 근거 없음')
        else:
            for role in ['선행','후속']:
                if lookup[role]['근거검토판정'] != '원문근거검토완료':
                    issues.append(f'{role} 논의 의미 검토 보류')
        if good_judgement(row, '사람검토_선후정책동일성') != '확인':
            issues.append('두 지역 정책의 동일성 미확정')
        # 전파 직접증거가 없어도 '유사 후속 논의'로 제출할 수 있으나, 직접 전파라고 주장 불가.
        if good_judgement(row, '사람검토_지역간전파직접증거') != '확인':
            issues.append('직접 전파 확인 안 됨: 유사 후속 논의로만 표현')
        if good_judgement(row, '사람검토_예측근거정합성') != '확인':
            issues.append('예측문 근거 일치 검토 미완료')
        if good_judgement(row, '사람검토_전망논리성') != '확인':
            issues.append('예측 논리성 검토 미완료')
        issues.append('전국 최초 등장 여부는 POLICY 캐시만으로 증명 불가')
        verdict = '검토보류'
        if any(good_judgement(row,f)=='부적합' for f in judg) or any(
                r and r['근거검토판정']=='부적합' for r in lookup.values()):
            verdict='수정필요'
        row['최종근거판정'] = verdict
        row['미해결항목'] = ' | '.join(dict.fromkeys(issues))
        issue_out.append(row)
        stamp(f'  {issue}: 근거 역할={sum(bool(v) for v in lookup.values())}개 / {verdict}',start)

    stamp('[3/4] 검토용 CSV·보고서 저장',start)
    save(evidence_path,evidence_out,EVIDENCE_COLS)
    save(issue_path,issue_out,ISSUE_COLS)
    review_copy = []
    for d in drafts:
        r = dict(d)
        r['검토상태'] = next(v['최종근거판정'] for v in issue_out if v['이슈키워드'] == d['이슈키워드'])
        r['주의'] = '제출 확정본 아님: 정책 동일성/예측 근거/실제 전파 검토 필요'
        review_copy.append(r)
    save(out/'19_submission_review_NOT_FINAL.csv',review_copy,list(review_copy[0].keys()))
    summary = {
        'created_at':datetime.now().isoformat(timespec='seconds'),
        'status':'NOT_FINAL_MANUAL_SEMANTIC_REVIEW_REQUIRED',
        'issue_count':len(issue_out),'role_count':len(evidence_out),
        'evidence_statuses':dict(Counter(r['근거검토판정'] for r in evidence_out)),
        'issue_statuses':dict(Counter(r['최종근거판정'] for r in issue_out)),
        'source_statuses':dict(Counter(r['원문출처_기계검사'] for r in evidence_out)),
        'review_rule':'키워드/발췌의 형식 검증은 의미 검증이 아님. 임의의 정책 확산 인과성 주장 금지.',
        'instructions':'19_evidence_human_review.csv 와 19_issue_human_review.csv의 사람검토_ 필드에 확인/부적합/불확실/미검토 중 선택하고 재실행. 별도 의견 필드 작성 가능. 기존 입력 보존.',
        'limitations':[ '원본 HTML 부분 일치는 회의 발언의 정책 적합성을 증명하지 않음',
                         '정책 유사성·시간 순서는 실제 지역 간 확산의 인과관계가 아님',
                         '최초관측은 POLICY 선별 데이터 기준이며 전국 전체 회의록 최초가 아님',
                         '모델 예측의 실현 여부는 과거 데이터만으로 평가할 수 없음'],
        'elapsed_sec':round(time.perf_counter()-start,2),
    }
    (out/'19_review_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    report=['# 19단계 정책 근거 정성 검토','',f'- 생성: {summary["created_at"]}',
            '- 판정: **제출 확정 전 수동 정책 내용 검토 필요**','',
            '## 이슈별 검토',
            '| 이슈 | 최초 출처 | 선행 출처 | 후속 출처 | 판정 |',
            '|---|---|---|---|---|']
    for r in issue_out:
        report.append('| '+ ' | '.join(r[k] for k in ['이슈키워드','최초관측원본출처',
                       '선행원본출처','후속원본출처','최종근거판정']) +' |')
    report += ['', '## 검토 방법',
               '1. `19_evidence_human_review.csv`: 발언 원문이 해당 정책을 실제로 다루는지 확인',
               '2. `19_issue_human_review.csv`: 두 지역 정책의 동일성과 향후 전망의 근거·논리성 확인',
               '3. 각 `사람검토_` 판단에는 `확인`, `부적합`, `불확실`, `미검토` 중 하나를 기입',
               '4. CSV 저장 후 동일 명령을 재실행하면 수동 입력값을 유지하며 판정 재집계',
               '', '## 중요 제한',
               '- 이 검토표는 사람이 작성하지 않은 칸을 AI가 임의로 `확인` 처리하지 않습니다.',
               '- 19단계 완료는 최종 제출물의 사실 검증 완료와 다릅니다.',
               '- 전파 직접증거 없으면 반드시 `유사 후속 논의`로 표현하세요.',
               '']
    (out/'19_review_report.md').write_text('\n'.join(report),encoding='utf-8')
    stamp(f'[4/4] 완료: {out} / 정책 {len(issue_out)}개 / 역할 {len(evidence_out)}건',start)
    print('  검토 결과는 NOT_FINAL 상태입니다.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print(f'[오류] {exc}',file=sys.stderr)
        sys.exit(1)

# -*- coding: utf-8 -*-
"""17단계: 지방의회 정책이슈 제출 초안의 구조·캐시 근거 검증.

- 제출 5열 CSV + 16단계 감사표/검증 JSON을 확인
- 로컬 SQLite 정책 원문 캐시의 청크 ID, 날짜, 의회, 근거 일부 대조
- 원본 HTML 대조 및 실제 정책 전파 관계는 수동 검증 필요로 유지
- 원본을 수정하지 않으며, 결과는 새로운 폴더에 저장

실행:
  python 17_validate_submission.py
  python 17_validate_submission.py --check-only
  python 17_validate_submission.py --root "C:\\mini_team\\scripts\\clustering"
  python 17_validate_submission.py --input "...\\competition_submission_draft.csv"
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

DEFAULT_ROOT = Path(r'C:\mini_team\scripts\clustering')
ISSUES = ('지역소멸', '청년 일자리', '저출생', '고령화', '기후위기')
COLS = ('이슈키워드', '등장시점', '타 지방의회로의 흐름', '향후 예측', '근거 회의록')
AUDIT_COLS = ('이슈키워드', '최초근거청크ID', '최초관측일_POLICY범위', '최초관측의회',
              '초기근거청크ID', '후속근거청크ID', '초기회의록명', '후속회의록명', '제출판정')
DETAIL_COLS = ('이슈키워드', '구조검사', '예측글자수', '예측200자', '날짜형식',
               '최초청크검사', '최초청크의회날짜검사', '선행청크검사', '후속청크검사',
               '근거발췌검사', '지역간시간순서', '회의록출처', '정책전파',
               '최종판정', '미해결문제')
CACHE_COLS = ('이슈키워드', '역할', 'chunk_id', '캐시존재', '캐시의회', '캐시날짜',
              '기대의회', '기대날짜', '메타데이터일치', '근거부분문자열일치', '발췌앞부분')


def log(message, start):
    print(f'[{datetime.now():%H:%M:%S}] {message} | 경과 {time.perf_counter()-start:.1f}초', flush=True)


def load_csv(path, must=()):
    if not path.exists():
        raise FileNotFoundError(f'파일을 찾을 수 없습니다: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if missing := set(must) - cols:
            raise ValueError(f'{path}: 필수 컬럼 누락 {sorted(missing)}. 현재 컬럼: {sorted(cols)}')
        return list(reader)


def save_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def canonical(s):
    s = re.sub(r'\s+', '', str(s or '')).replace('전라북도', '전북특별자치도').replace('강원도', '강원특별자치도')
    return s


def datevalue(text):
    m = re.search(r'(20\d{2})[-./]?(\d{2})[-./]?(\d{2})', str(text or ''))
    if not m:
        return ''
    y, mo, d = map(int, m.groups())
    try:
        return datetime(y, mo, d).strftime('%Y-%m-%d')
    except ValueError:
        return ''


def norm_text(s):
    return re.sub(r'\s+', '', str(s or '')).lower()


def check_excerpt(document, excerpt):
    """부분 일치: 증명 범위는 캐시의 청크 내용에 한정."""
    e = norm_text(excerpt)
    d = norm_text(document)
    if not e or not d:
        return '미검증'
    if e in d:
        return '일치'
    if len(e) >= 40 and any(e[k:k+40] in d for k in (0, max(0, len(e)//2-20), len(e)-40)):
        return '부분일치(검토)'
    return '불일치'


def open_cache(path):
    if not path.exists():
        return None
    con = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    found = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='policy_docs'").fetchone()
    if not found:
        con.close()
        raise ValueError(f'policy_docs 테이블이 없습니다: {path}')
    keys = {t[1] for t in con.execute('PRAGMA table_info(policy_docs)')}
    need = {'chunk_id', 'document', 'assembly', 'date'}
    if need - keys:
        con.close()
        raise ValueError(f'policy_docs에서 컬럼 누락: {need - keys}')
    return con


def main():
    start = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--input', type=Path, default=None)
    ap.add_argument('--audit', type=Path, default=None)
    ap.add_argument('--cache', type=Path, default=None)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()
    root = args.root
    r = root / 'results'
    draft_path = args.input or r / 'competition_submission/competition_submission_draft.csv'
    audit_path = args.audit or r / 'competition_submission/competition_submission_audit.csv'
    validation_path = draft_path.parent / 'competition_submission_validation.json'
    cache_path = args.cache or r / 'all_policy_topics/policy_text_cache.sqlite3'
    out = args.out or r / 'competition_submission_validation_17'

    log('[1/4] 입력 경로 확인', start)
    for label, p in [('제출 초안', draft_path), ('16단계 감사표', audit_path),
                     ('16단계 검증 JSON', validation_path), ('정책 원문 SQLite', cache_path)]:
        print(f'  {label}: {"있음" if p.is_file() else "없음"} / {p}')
    if args.check_only:
        return 0

    drafts = load_csv(draft_path, COLS)
    audits = load_csv(audit_path, AUDIT_COLS)
    if validation_path.exists():
        with validation_path.open(encoding='utf-8-sig') as f:
            previous = json.load(f)
    else:
        previous = {}
    audit_by_issue = {}
    for a in audits:
        issue = a.get('이슈키워드', '').strip()
        if issue in audit_by_issue:
            raise ValueError(f'감사표에 중복된 이슈: {issue}')
        audit_by_issue[issue] = a
    count = Counter(x.get('이슈키워드', '').strip() for x in drafts)
    global_errors = []
    for issue in ISSUES:
        if count[issue] != 1:
            global_errors.append(f'{issue}: 제출 초안 행 수 {count[issue]} (필요 1)')
        if issue not in audit_by_issue:
            global_errors.append(f'{issue}: 감사표 행 없음')
    extra = sorted(set(count) - set(ISSUES))
    if extra:
        global_errors.append(f'예상하지 못한 이슈: {extra}')

    cache = open_cache(cache_path)
    if cache is None:
        global_errors.append('SQLite 원문 캐시 없음: 청크 근거 검증 불가')
    cache_rows, detail_rows = [], []
    cache_hits = {}

    def evidence_row(issue, role, chunk_id, assembly, date, snippet=''):
        chunk_id = (chunk_id or '').strip()
        entry = {'이슈키워드': issue, '역할': role, 'chunk_id': chunk_id,
                 '캐시존재': '미확인', '캐시의회': '', '캐시날짜': '',
                 '기대의회': assembly, '기대날짜': date, '메타데이터일치': '미확인',
                 '근거부분문자열일치': '미검증', '발췌앞부분': (snippet or '')[:90]}
        if not chunk_id:
            entry['캐시존재'] = '청크ID없음'
        elif cache:
            if chunk_id not in cache_hits:
                cache_hits[chunk_id] = cache.execute(
                    'SELECT document, assembly, date FROM policy_docs WHERE chunk_id=?', (chunk_id,)).fetchone()
            row = cache_hits[chunk_id]
            if row:
                doc, cached_council, cached_date = row
                entry['캐시존재'] = '확인'
                entry['캐시의회'] = str(cached_council or '')
                entry['캐시날짜'] = str(cached_date or '')
                entry['메타데이터일치'] = ('일치' if canonical(cached_council) == canonical(assembly)
                                                 and datevalue(cached_date) == datevalue(date) else '불일치')
                entry['근거부분문자열일치'] = check_excerpt(doc, snippet)
            else:
                entry['캐시존재'] = '미발견'
        cache_rows.append(entry)
        return entry

    log(f'[2/4] 이슈 {len(drafts)}개 / 감사 {len(audits)}개 검증', start)
    for draft in drafts:
        issue = draft.get('이슈키워드', '').strip()
        a = audit_by_issue.get(issue, {})
        problems = []
        response = str(draft.get('향후 예측') or '').strip()
        n_chars = len(response)
        if not response:
            problems.append('향후 예측문 없음')
        elif n_chars > 200:
            problems.append(f'향후 예측 {n_chars}자: 200자 초과')
        appeared = str(draft.get('등장시점') or '')
        first_date = datevalue(a.get('최초관측일_POLICY범위'))
        first_assembly = a.get('최초관측의회', '').strip()
        first_id = a.get('최초근거청크ID', '').strip()
        displayed_date = datevalue(appeared)
        dates_ok = bool(first_date and displayed_date and first_date == displayed_date)
        if not dates_ok:
            problems.append('최초 관측일 표시/감사표 날짜 불일치 또는 형식 오류')
        if first_assembly and canonical(first_assembly) not in canonical(appeared):
            problems.append('최초 관측 의회 불일치')
        if 'POLICY' not in appeared.upper():
            problems.append('최초 관측 범위(POLICY 선별) 표기 누락')

        first = evidence_row(issue, '최초관측', first_id, first_assembly, first_date)
        if first['캐시존재'] != '확인':
            problems.append('최초 관측 청크 원문 캐시 조회 실패 또는 미확인')
        elif first['메타데이터일치'] != '일치':
            problems.append('최초 관측 청크 메타데이터 불일치')

        flow = str(draft.get('타 지방의회로의 흐름') or '')
        evidence = str(draft.get('근거 회의록') or '')
        early_id, later_id = a.get('초기근거청크ID', '').strip(), a.get('후속근거청크ID', '').strip()
        flow_syntax = '미확보'
        provenance = '미검증'
        excerpt = '미검증'
        prior = after = None
        if early_id or later_id:
            if not early_id or not later_id:
                problems.append('지역 간 연결의 선행/후속 청크 ID 누락')
            # Parse format "대구광역시의회(2024-04-26) → 광주광역시의회(2024-12-03)"
            pair = re.search(r'([^()→]+)\((20\d{2}-\d{2}-\d{2})\)\s*→\s*([^()→]+)\((20\d{2}-\d{2}-\d{2})\)', flow)
            if pair:
                c1, d1, c2, d2 = [v.strip() for v in pair.groups()]
                if datevalue(d1) and datevalue(d2) and datevalue(d1) < datevalue(d2) and canonical(c1) != canonical(c2):
                    flow_syntax = '시간선후관계_일치(전파증명아님)'
                else:
                    flow_syntax = '날짜역전_같은지역_또는오류'
                    problems.append('선후 날짜 또는 지역 구분 오류')
            else:
                c1 = c2 = d1 = d2 = ''
                problems.append('지역 간 흐름 표기에서 날짜·의회 추출 실패')
            prior = evidence_row(issue, '선행', early_id, c1, d1)
            after = evidence_row(issue, '후속', later_id, c2, d2)
            if any(x['캐시존재'] != '확인' for x in [prior, after]):
                problems.append('선행 또는 후속 원문 청크 캐시 조회 실패')
            elif any(x['메타데이터일치'] != '일치' for x in [prior, after]):
                problems.append('선행 또는 후속 청크 의회·날짜 불일치')
            titles = [a.get('초기회의록명','').strip(), a.get('후속회의록명','').strip()]
            if all(titles) and all(title in evidence for title in titles):
                provenance = '복원제목_표시확인(원본HTML미검증)'
            else:
                problems.append('양쪽 회의록명 복원 또는 초안 표시 부족')
                provenance = '부족'
            if '정책 전파 미확인' not in flow:
                problems.append('정책 전파 미확인 주의 문구 누락')
            # Do not assert excerpt verified without raw HTML; cache verifies only if excerpt is unambiguously parsed.
            excerpt = '원본HTML_미검증'
            problems.append('선행·후속 논의의 정책 동일성/실제 전파 수동 검토 필요')
        else:
            if '미확보' not in flow:
                problems.append('유사 논의 청크가 없는데 미확보 미표시')
            problems.append('지역 간 연결 근거 미확보')
        if '회의록명 추가 확인 필요' in evidence:
            problems.append('최초 등장 회의록명 복원 필요')
        if not evidence or '근거 회의록' == evidence:
            problems.append('회의록 근거 빈값')

        problems.append('전체 원본 기준 전국 최초 등장인지 미검증')
        problems.append('LLM 예측과 회의록 주장의 의미적 근거 일치성은 수동 검증 필요')
        # Automated checks cannot certify nationwide priority, true diffusion, causal relationship, or LLM factuality.
        status = '수정필요' if any(('초과' in s or '불일치' in s or '실패' in s or '누락' in s or '없음' in s)
                                    for s in problems) else '검증보류'
        detail_rows.append({
            '이슈키워드': issue,
            '구조검사': '정상' if count[issue] == 1 and issue in audit_by_issue else '오류',
            '예측글자수': n_chars, '예측200자': '통과' if response and n_chars <= 200 else '실패',
            '날짜형식': '일치' if dates_ok else '불일치',
            '최초청크검사': first['캐시존재'],
            '최초청크의회날짜검사': first['메타데이터일치'],
            '선행청크검사': prior['캐시존재'] if prior else '미확보',
            '후속청크검사': after['캐시존재'] if after else '미확보',
            '근거발췌검사': excerpt, '지역간시간순서': flow_syntax,
            '회의록출처': provenance,
            '정책전파': '검증안됨', '최종판정': status,
            '미해결문제': ' | '.join(dict.fromkeys(problems)),
        })
        log(f'  {issue}: {n_chars}자 / 최초청크={first["캐시존재"]} / {status}', start)
    if cache:
        cache.close()
    log('[3/4] 검증 결과 저장', start)
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out / '17_submission_validation_detail.csv', detail_rows, DETAIL_COLS)
    save_csv(out / '17_evidence_cache_checks.csv', cache_rows, CACHE_COLS)
    # A copy is clearly marked DRAFT, and original file is untouched.
    save_csv(out / '17_submission_review_copy_NOT_FINAL.csv', drafts, COLS)
    counts = Counter(row['최종판정'] for row in detail_rows)
    report = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'result': 'NOT_FINAL_HUMAN_REVIEW_REQUIRED',
        'source_draft': str(draft_path),
        'source_audit': str(audit_path),
        'sqlite_cache_used': str(cache_path) if cache_path.is_file() else None,
        'draft_rows': len(drafts),
        'issues_expected': list(ISSUES),
        'counts': dict(counts),
        'global_errors': global_errors,
        'previous_submission_status': previous.get('submission_status', 'unknown'),
        'limitations': [
            'SQLite는 POLICY로 선정된 청크 캐시: 전국 최초 등장 증명 불가',
            '청크의 날짜·의회 일치는 원본 회의록 HTML 직접 대조를 대신하지 않음',
            '코사인 유사도나 시간 선후는 정책 참고/전파 인과관계의 증거가 아님',
            '생성된 200자 예측의 의미적 근거 일치성은 자동 검사로 확정하지 않음',
            '정책 확산을 입증할 수 없으면 유사 후속 논의로만 기술할 것',
        ],
        'elapsed_sec': round(time.perf_counter()-start, 2),
    }
    (out / '17_submission_validation_summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 17단계 대회 제출 검증 보고서', '',
             f'- 작성: {report["created_at"]}',
             f'- 총 {len(drafts)}개 이슈 / 결과: {report["result"]}',
             f'- 원문 캐시: {"확인" if cache_path.is_file() else "없음"}',
             '', '## 항목별 결과', '',
             '| 이슈 | 예측 글자수 | 최초 청크 | 선행·후속 | 판정 |',
             '|---|---:|---|---|---|']
    for rr in detail_rows:
        lines.append(f'| {rr["이슈키워드"]} | {rr["예측글자수"]} | {rr["최초청크검사"]} | '
                     f'{rr["선행청크검사"]}/{rr["후속청크검사"]} | {rr["최종판정"]} |')
    lines.extend(['', '## 판단 원칙', '', *('- ' + x for x in report['limitations']),
                  '', '## 필수 확인', '',
                  '회의록명과 원본 HTML 근거, 정책의 동일성과 지역 간 전파 여부는 검토자 확인 후에만 확정할 수 있음.', ''])
    (out / '17_submission_validation_report.md').write_text('\n'.join(lines), encoding='utf-8')
    log(f'[4/4] 완료: {out} (수정필요={counts["수정필요"]}, 검증보류={counts["검증보류"]})', start)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, sqlite3.Error, json.JSONDecodeError) as e:
        print(f'[오류] {type(e).__name__}: {e}', file=sys.stderr)
        sys.exit(1)

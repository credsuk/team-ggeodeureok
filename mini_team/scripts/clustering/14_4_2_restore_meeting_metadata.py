# -*- coding: utf-8 -*-
"""14-4-2 | 원본 의회 회의록 JSON에서 확인 가능한 회의록명 복원.

집 PC:
  python 14_4_2_restore_meeting_metadata.py
  python 14_4_2_restore_meeting_metadata.py --source-root "C:\\mini_team\\org2"

입력: 14-3 policy_links_unique.csv, 12단계 policy_text_cache.sqlite3,
      원본 JSON list[{'MTG_DE','RASMBLY_NM','MINTS_HTML','MTGNM','MINTS_NM'}]
출력: results/policy_meeting_metadata_v2/ 아래에 원본 매칭/미해결/회의록 복원 CSV와 요약 JSON

안전: 원본 JSON과 ChromaDB를 수정하지 않음. 복수 매칭은 자동 확정하지 않음.
"""
from __future__ import annotations
import argparse
import csv
import html
import json
import re
import sqlite3
import time
from collections import defaultdict, Counter
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

BASE = Path(r"C:\mini_team\scripts\clustering")
INPUT = BASE / "results" / "policy_discussion_evidence" / "policy_links_unique.csv"
TEXT_DB = BASE / "results" / "all_policy_topics" / "policy_text_cache.sqlite3"
DEFAULT_ROOT = Path(r"C:\mini_team\org2")
OUTPUT = BASE / "results" / "policy_meeting_metadata_v2"

# 강한 연속 일치만 '확정 후보'로 인정. 양쪽 시점/의회 일치 필수.
ANCHOR_LENGTH = 75
MIN_ANCHOR = 48
MAX_PER_RECORD = 8

class HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style'}:
            self.skip += 1
        elif tag in {'br', 'p', 'div', 'tr', 'td', 'li', 'section', 'h1', 'h2', 'h3'}:
            self.parts.append(' ')

    def handle_endtag(self, tag):
        if tag in {'script', 'style'} and self.skip:
            self.skip -= 1
        elif tag in {'p', 'div', 'tr', 'td', 'li', 'section', 'h1', 'h2', 'h3'}:
            self.parts.append(' ')

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

    def text(self):
        return ''.join(self.parts)


def stamp(msg, start):
    print(f"[{datetime.now():%H:%M:%S}] {msg} | 경과 {time.perf_counter()-start:.1f}초", flush=True)


def normalize(s):
    return re.sub(r'\s+', ' ', html.unescape(str(s or ''))).strip()


def tight(s):
    # 공백 및 흔한 HTML entity/따옴표 차이만 제거한다. 숫자/한글은 유지.
    return re.sub(r'\s+', '', html.unescape(str(s or '')))


def html_text(raw):
    parser = HTMLText()
    parser.feed(str(raw or ''))
    parser.close()
    return normalize(parser.text())


def normalize_date(v):
    text = str(v or '')
    m = re.search(r'(20\d{2})\D{0,3}(\d{1,2})\D{0,3}(\d{1,2})', text)
    if not m:
        return ''
    y, mo, d = map(int, m.groups())
    try:
        return datetime(y, mo, d).strftime('%Y-%m-%d')
    except ValueError:
        return ''


def normalize_assembly(s):
    return re.sub(r'\s+', '', str(s or '')).replace('전북자치도의회', '전북특별자치도의회')


def get_anchors(doc, evidence):
    """최대 4개, 서로 다른 구간의 정규화된 원문 인용 앵커."""
    text = tight(doc)
    src = tight(evidence)
    anchors = []
    if len(src) >= MIN_ANCHOR:
        p = max(0, (len(src) - ANCHOR_LENGTH)//2)
        anchors.append(src[p:p+ANCHOR_LENGTH])
    if len(text) >= MIN_ANCHOR:
        for f in (0.08, 0.45, 0.80):
            p = int(max(0, len(text)-ANCHOR_LENGTH)*f)
            anchors.append(text[p:p+ANCHOR_LENGTH])
    result = []
    for a in anchors:
        if len(a) >= MIN_ANCHOR and a not in result:
            result.append(a)
    return result


def stream_json_list(path, chunk_size=1024*1024):
    """200MB+ JSON 배열을 항목 단위로 읽음; 외부 ijson 패키지 불필요."""
    decoder = json.JSONDecoder()
    with path.open('r', encoding='utf-8-sig') as f:
        buffer = ''
        index = 0
        eof = False
        started = False
        while True:
            if index >= len(buffer) and not eof:
                buffer = f.read(chunk_size)
                index = 0
                if not buffer:
                    eof = True
            # skip separators and whitespace
            while True:
                while index < len(buffer) and buffer[index] in ' \r\n\t,':
                    index += 1
                if index < len(buffer) or eof:
                    break
                buffer = f.read(chunk_size)
                index = 0
                if not buffer:
                    eof = True
            if not started:
                if index >= len(buffer) or buffer[index] != '[':
                    raise ValueError(f'JSON 최상위가 배열이 아닙니다: {path}')
                index += 1
                started = True
                continue
            if index >= len(buffer):
                if eof:
                    raise ValueError(f'JSON 배열 종료 기호 누락: {path}')
                continue
            if buffer[index] == ']':
                return
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, index)
                    index = end
                    yield value
                    if index > chunk_size * 2:
                        buffer = buffer[index:]
                        index = 0
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    # Keep incomplete record only; large HTML records are buffered individually.
                    buffer = buffer[index:] + f.read(chunk_size)
                    index = 0
                    if not buffer:
                        eof = True
                    else:
                        # detect whether read reached EOF on next loop (sufficient on Windows)
                        if len(buffer) < 2 and f.tell() == path.stat().st_size:
                            eof = True


def csv_save(path, rows, keys):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description='14-4-2 JSON 원본 회의록명 복원')
    ap.add_argument('--source-root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--input-csv', type=Path, default=INPUT)
    args = ap.parse_args()
    start = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not args.source_root.is_dir():
        raise FileNotFoundError(f'원본 폴더 없음: {args.source_root}')
    if not args.input_csv.is_file() or not TEXT_DB.is_file():
        raise FileNotFoundError('14-3 후보 CSV 또는 12단계 원문 캐시를 찾지 못했습니다.')
    with args.input_csv.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError('입력 후보 CSV가 비어 있습니다.')
    details = {}
    for r in rows:
        for side in ('earlier','later'):
            cid = r.get(f'{side}_chunk_id','')
            if cid:
                details[cid] = {'expected_date': normalize_date(r.get(f'{side}_date')),
                                'expected_assembly': normalize_assembly(r.get(f'{side}_assembly')),
                                'evidence': r.get(f'{side}_evidence_180','')}
    conn = sqlite3.connect(f'file:{TEXT_DB.as_posix()}?mode=ro', uri=True)
    ids = list(details)
    for i in range(0,len(ids),200):
        batch = ids[i:i+200]
        marks = ','.join('?' for _ in batch)
        for cid, doc, assembly, date_s in conn.execute(
                f'SELECT chunk_id,document,assembly,date FROM policy_docs WHERE chunk_id IN ({marks})',batch):
            details[cid]['document'] = doc or ''
            details[cid]['cache_date'] = normalize_date(date_s)
            details[cid]['cache_assembly'] = normalize_assembly(assembly)
    conn.close()
    for cid, v in details.items():
        v['anchors'] = get_anchors(v.get('document',''), v['evidence'])
    stamp(f'[1/4] 입력 후보 {len(rows)}쌍 / 고유 청크 {len(details)}개',start)

    # Group by (year, assembly, day) to avoid checking all 80 against every HTML record.
    groups = defaultdict(list)
    for cid, v in details.items():
        date_s = v['expected_date'] or v.get('cache_date','')
        assembly = v['expected_assembly'] or v.get('cache_assembly','')
        if date_s and assembly and v.get('anchors'):
            groups[(date_s[:4], assembly, date_s)].append(cid)
    files = sorted(args.source_root.rglob('*.json'))
    if not files:
        raise FileNotFoundError(f'JSON 파일 없음: {args.source_root}')
    candidates = defaultdict(list)
    stats = Counter()
    file_issues = []
    for i, path in enumerate(files,1):
        # Limit filenames to expected years/councils when possible; duplicates like '(1)' supported.
        matching_groups = [(k,v) for k,v in groups.items()
                           if k[0] in path.stem and k[1] in normalize_assembly(path.stem)]
        if not matching_groups:
            continue
        by_day_assembly = defaultdict(list)
        for (year,assembly,day), chunk_ids in matching_groups:
            by_day_assembly[(assembly,day)].extend(chunk_ids)
        count = 0
        try:
            for index,record in enumerate(stream_json_list(path)):
                count += 1
                if not isinstance(record,dict):
                    continue
                day = normalize_date(record.get('MTG_DE'))
                assembly = normalize_assembly(record.get('RASMBLY_NM'))
                ids_here = by_day_assembly.get((assembly,day),[])
                if not ids_here:
                    continue
                original = tight(html_text(record.get('MINTS_HTML','')))
                if not original:
                    continue
                for cid in ids_here:
                    anchors = details[cid]['anchors']
                    matched = [a for a in anchors if a in original]
                    if not matched:
                        continue
                    # Distinct matching anchors reduce false positives from common boilerplate.
                    strong = len(matched) >= 2 or len(matched[0]) >= 72
                    evidence = tight(details[cid]['evidence'])
                    evidence_match = bool(evidence and len(evidence)>=45 and evidence in original)
                    title = normalize(record.get('MINTS_NM'))
                    meeting = normalize(record.get('MTGNM'))
                    candidates[cid].append({
                        'chunk_id':cid, 'source_json':str(path), 'record_index':index,
                        'meeting_date':day, 'assembly':normalize(record.get('RASMBLY_NM')),
                        'meeting_name':meeting, 'meeting_title':title,
                        'anchor_matches':len(matched), 'anchor_total':len(anchors),
                        'evidence_whole_match':int(evidence_match),
                        'strong_match':int(strong or evidence_match),
                        'matched_anchor_preview':matched[0][:100],
                        'original_html_chars':len(str(record.get('MINTS_HTML',''))),
                    })
            stats['files_scanned'] += 1
            stats['records_read'] += count
        except Exception as exc:
            file_issues.append({'file':str(path),'error':f'{type(exc).__name__}: {exc}'})
        stamp(f'[2/4] 파일 {i}/{len(files)}: {path.name} / 레코드 {count:,}',start)

    matched_rows = [r for items in candidates.values() for r in items]
    csv_save(OUTPUT/'original_match_candidates.csv', matched_rows,
             ['chunk_id','source_json','record_index','meeting_date','assembly','meeting_name',
              'meeting_title','anchor_matches','anchor_total','evidence_whole_match',
              'strong_match','matched_anchor_preview','original_html_chars'])

    resolution = {}
    for cid, meta in details.items():
        items = candidates.get(cid,[])
        strong = [x for x in items if x['strong_match']]
        if len(strong)==1 and strong[0]['meeting_title']:
            status = 'confirmed_unique'
            chosen = strong[0]
        elif len(strong)>1:
            status = 'ambiguous_multiple'
            chosen = None
        elif len(strong)==1:
            status = 'title_missing'
            chosen = None
        elif items:
            status = 'weak_match_review'
            chosen = None
        else:
            status = 'unresolved'
            chosen = None
        resolution[cid] = (status,chosen)
    stamp(f'[3/4] 원본 매칭 검증: {dict(Counter(x[0] for x in resolution.values()))}',start)

    output_rows=[]
    for r in rows:
        row=dict(r)
        for side in ('earlier','later'):
            cid=row.get(f'{side}_chunk_id','')
            state,match=resolution.get(cid,('unresolved',None))
            row[f'{side}_restore_status']=state
            row[f'{side}_original_meeting_name']=match['meeting_name'] if match else ''
            row[f'{side}_original_meeting_title']=match['meeting_title'] if match else ''
            row[f'{side}_original_json_file']=match['source_json'] if match else ''
            row[f'{side}_original_record_index']=match['record_index'] if match else ''
            # Existing title only filled when unique strong match is established.
            if match:
                row[f'{side}_meeting_title']=match['meeting_title']
        row['both_meeting_titles_restored']=int(
            row['earlier_restore_status']=='confirmed_unique' and
            row['later_restore_status']=='confirmed_unique')
        output_rows.append(row)
    additions = []
    for side in ('earlier','later'):
        additions += [f'{side}_{k}' for k in ('restore_status','original_meeting_name',
                        'original_meeting_title','original_json_file','original_record_index')]
    additions += ['both_meeting_titles_restored']
    out_fields = cols + [c for c in additions if c not in cols]
    csv_save(OUTPUT/'policy_links_meeting_metadata_restored.csv',output_rows,out_fields)
    unresolved = []
    for cid,(status,chosen) in resolution.items():
        if status!='confirmed_unique':
            m=details[cid]
            unresolved.append({'chunk_id':cid,'status':status,'assembly':m['expected_assembly'],
                'date':m['expected_date'],'matched_records':len(candidates.get(cid,[])),
                'evidence_180':m['evidence'][:180]})
    csv_save(OUTPUT/'meeting_metadata_unresolved.csv',unresolved,
             ['chunk_id','status','assembly','date','matched_records','evidence_180'])
    summary = {
        'completed_at':datetime.now().isoformat(timespec='seconds'),
        'input_pairs':len(rows),'unique_chunk_ids':len(details),
        'source_root':str(args.source_root),
        'json_files_found':len(files),'json_files_scanned':stats['files_scanned'],
        'json_records_read':stats['records_read'],'scanning_errors':file_issues,
        'candidate_matches':len(matched_rows),
        'restore_status_counts':dict(Counter(x[0] for x in resolution.values())),
        'pairs_with_both_titles':sum(r['both_meeting_titles_restored'] for r in output_rows),
        'elapsed_sec':round(time.perf_counter()-start,2),
        'caution':'회의록명 복원은 동일 의회·날짜와 충분한 원문 앵커가 유일하게 일치한 경우만. 정책 전파 확정은 아님.'}
    (OUTPUT/'meeting_metadata_restore_summary.json').write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    stamp(f'[4/4] 완료. 양쪽 회의록명 확보 {summary["pairs_with_both_titles"]}/{len(rows)}쌍 | {OUTPUT}',start)

if __name__=='__main__':
    main()

"""Read original ingestion TSV rows without editing any source artifact."""
import csv
import sys
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2147483647))

def enrich(evidence, root):
    by_file = {}
    for item in evidence:
        name = item['metadata'].get('source_file')
        rid = item['metadata'].get('row_id')
        if name and Path(name).name == name and isinstance(rid, int):
            by_file.setdefault(name, {}).setdefault(rid, []).append(item)
    for name, wanted in by_file.items():
        files = list((root / 'change_text').glob('*/' + name))
        if len(files) != 1:
            for items in wanted.values():
                for item in items:
                    item['source_check'] = '원본 정제 발언 파일 없음 또는 경로 중복: 별도 대조 미확인'
            continue
        with files[0].open(encoding='utf-8-sig', newline='') as handle:
            for rid, row in enumerate(csv.DictReader(handle, delimiter='\t'), 1):
                if rid not in wanted:
                    continue
                for item in wanted[rid]:
                    meta = item['metadata']
                    checks = [str(meta.get('date')) == row.get('MTG_DE'),
                              meta.get('assembly') == row.get('RASMBLY_NM'),
                              meta.get('agenda') == row.get('agenda_title'),
                              str(meta.get('speaker', '')).strip() == (row.get('speaker_role', '') + ' ' + row.get('speaker_name', '')).strip(),
                              item['text'] in row.get('utterance', ''),
                              item['quote'] in row.get('utterance', ''),
                              all(c['text'] in row.get('utterance', '') and c['quote'] in row.get('utterance', '') for c in item.get('related_chunks', []))]
                    if all(checks):
                        meta['meeting_name'] = row.get('MINTS_NM', '')
                        meta['meeting_type'] = row.get('MTGNM', '')
                        item['source_check'] = '원본 정제 TSV의 행·날짜·의회·안건·발언 원문 일치'
                        item['source_path'] = str(files[0].relative_to(root))
                        item['verification'] = '정책 관련성·Chroma 원문 연속 인용 검증; ' + item['source_check'] + ' (발언자 일치; 수집 JSON 별도 대조 미확인)'
                    else:
                        item['source_check'] = '원본 정제 TSV 대조 불일치'
                        item['source_mismatch'] = True
                if rid >= max(wanted):
                    break
    return [x for x in evidence if not x.get('source_mismatch')]

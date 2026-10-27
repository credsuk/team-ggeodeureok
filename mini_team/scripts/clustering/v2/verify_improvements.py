"""Fresh full-corpus searches; all artifacts remain within V2."""
import json
import argparse
from datetime import datetime
from pathlib import Path
from engine import ReadOnlyChroma, SearchEngine
from policy_guard import matches
from validate_v2 import manifest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    out = Path(__file__).parent / 'validation' / ('improved_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    def write(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out, flush=True)
    before = json.loads(args.baseline.read_text(encoding='utf-8')) if args.baseline else manifest()
    write('protected_before.json', before)
    store = ReadOnlyChroma()
    assert store.count == 2189332
    engine = SearchEngine(store)
    cases = [('unemployment', '실업급여 관련 논의가 있나요?'),
             ('unemployment_synonym', '구직급여 지원은 어떻게 논의되었나요?'),
             ('rental', '전세사기 피해자 지원'),
             ('rental_synonym', '전세 피해 세입자의 주거 지원 논의가 있나요?'),
             ('textbook', '디지털 교과서 도입에 대해 어떤 논의가 있었나요?'),
             ('climate', '기후위기 대응 정책은 어떻게 논의되었나요?'),
             ('unrelated', '김치찌개를 끓이는 레시피를 알려줘'),
             ('insufficient', '해오름ZQX927 주민 교통비 환급사업 논의')]
    results, recordings = [], {}
    try:
        for name, query in cases:
            print('SEARCH', name, flush=True)
            r = engine.analyze(query, k=12)
            write(name + '.json', r)
            recordings[name] = r
            valid = all(matches(query, x['text']) and matches(query, x['quote']) and x['quote'] in x['text'] for x in r['evidence'])
            passed = valid and r['status'] in ('ok', 'limited', 'insufficient')
            if name == 'unrelated':
                passed = r['status'] == 'unrelated' and not r['evidence']
            if name == 'insufficient':
                passed = r['status'] == 'insufficient' and not r['evidence']
            row = {'case': name, 'status': r['status'], 'evidence': len(r['evidence']), 'passed': passed,
                   'seconds': r.get('elapsed_seconds'), 'concept_rejected': len(r.get('concept_rejected', []))}
            results.append(row)
            print(json.dumps(row), flush=True)
    finally:
        consistency = []
        for left, right in [('unemployment', 'unemployment_synonym'), ('rental', 'rental_synonym')]:
            if left in recordings and right in recordings:
                a, b = recordings[left], recordings[right]
                consistency.append({'cases': [left, right], 'same_queries': a.get('search_queries') == b.get('search_queries'),
                                    'both_have_evidence': bool(a['evidence']) and bool(b['evidence']),
                                    'shared_ids': sorted({x['id'] for x in a['evidence']} & {x['id'] for x in b['evidence']})})
        after = manifest()
        write('protected_after.json', after)
        changes = {k: {'before': before.get(k), 'after': after.get(k)} for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
        write('protected_changes.json', changes)
        write('report.json', {'cases': results, 'synonym_consistency': consistency,
                             'protected_changes': len(changes), 'total_chunks': store.count,
                             'all_passed': len(results) == len(cases) and all(r['passed'] for r in results)
                             and all(c['same_queries'] and c['both_have_evidence'] for c in consistency) and not changes})

if __name__ == '__main__':
    main()

"""Actual full-corpus/LLM scenarios plus source preservation checks.

Run once per invocation; results go to an exclusive timestamped V2 directory.
No input, pre-existing result, embedding or model is written.
"""
import hashlib
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from engine import DB, ROOT, LocalLLM, ReadOnlyChroma, SearchEngine, generate_forecast


def manifest():
    """Stat every existing project artifact; hash all source code and Chroma bytes.

    Hashing huge prior intermediate arrays is unnecessary: read-only code never
    opens them. Their sizes/mtimes are nevertheless compared before and after.
    """
    own = Path(__file__).parent.resolve()
    rows = {}
    for path in ROOT.rglob('*'):
        if not path.is_file() or path.is_relative_to(own):
            continue
        stat = path.stat()
        row = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        if path.suffix in ('.py', '.md') or path.is_relative_to(DB):
            h = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(8*1024*1024), b''):
                    h.update(block)
            row['sha256'] = h.hexdigest()
        rows[str(path.relative_to(ROOT))] = row
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, help='동일 작업의 기존 보존 baseline을 재사용')
    parser.add_argument('--reuse-retrievals', type=Path, help='이미 실행한 실제 검색 원문을 재사용하고 전망 단계는 현재 코드로 재검증')
    args = parser.parse_args()
    out = Path(__file__).parent / 'validation' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    write = lambda name, data: (out/name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print('보존 검증 baseline 작성:', out, flush=True)
    before = json.loads(args.baseline.read_text(encoding='utf-8')) if args.baseline else manifest()
    write('protected_before.json', before)
    if args.baseline:
        write('baseline_source.json', {'path': str(args.baseline.resolve())})
    store = ReadOnlyChroma()
    e = SearchEngine(store)
    cases = [
        ('free_keyword', '전세사기 피해자 지원', '전체', '전체'),
        ('synonym_question', '임대차 사기 피해를 입은 세입자를 위한 주거 지원은 어떻게 논의되었나요?', '전체', '전체'),
        ('another_topic', '디지털 교과서 도입에 대해 어떤 논의가 있었나요?', '2025', '전체'),
        ('unrelated', '김치찌개를 맛있게 끓이는 레시피를 알려줘', '전체', '전체'),
        ('insufficient', '해오름ZQX927 주민 교통비 환급사업 예산은 어떻게 논의되었나요?', '전체', '전체'),
    ]
    outcomes = []
    try:
        for name, query, year, assembly in cases:
            print('실제 검색:', name, query, flush=True)
            prior = args.reuse_retrievals / (name+'.json') if args.reuse_retrievals else None
            if prior and prior.is_file():
                result = json.loads(prior.read_text(encoding='utf-8'))
                if result['status'] in ('ok', 'limited') and len(result.get('evidence', [])) >= 2:
                    source_ids = set(result['forecast_evidence_ids'])
                    selected = [x for x in result['evidence'] if x['id'] in source_ids]
                    # Original search and review already ran against the actual corpus.
                    # Explicit provenance distinguishes reuse from a fresh retrieval.
                    result['retrieval_reused_from'] = str(prior.resolve())
                    result['forecast'], result['forecast_citations'] = generate_forecast(e.llm, query, result['scope'], selected)
                else:
                    result = e.analyze(query, year, assembly, k=12)
            else:
                result = e.analyze(query, year, assembly, k=12)
            write(name+'.json', result)
            status = result['status']
            if name in ('free_keyword', 'synonym_question', 'another_topic'):
                passed = status == 'ok' and isinstance(result['forecast'], str) and 30 <= len(result['forecast']) <= 200
            elif name == 'unrelated':
                passed = status == 'unrelated' and not result['evidence']
            else:
                passed = status == 'insufficient' and not result['evidence'] and result['forecast'] is None
            outcomes.append({'case': name, 'status': status, 'passed': passed,
                             'evidence_count': len(result['evidence']), 'seconds': result.get('elapsed_seconds'),
                             'message': result.get('message')})
            print(outcomes[-1], flush=True)
        # Live connection refusal, rather than only a mock failure.
        broken = SearchEngine(store, LocalLLM(url='http://127.0.0.1:1/api/chat', timeout=2))
        result = broken.analyze('도서관 운영 지원')
        write('connection_error.json', result)
        outcomes.append({'case': 'connection_error', 'status': result['status'],
                         'passed': result['status'] == 'error' and result['forecast'] is None})
        keyword = json.loads((out/'free_keyword.json').read_text(encoding='utf-8'))
        synonym = json.loads((out/'synonym_question.json').read_text(encoding='utf-8'))
        # Top-k sets can differ for synonymous phrasings. Check actual semantic
        # evidence, and report the ID overlap as a diagnostic rather than demand it.
        left, right = ({x['id'] for x in r.get('evidence', [])} for r in (keyword, synonym))
        semantic = all(any('전세사기' in x['quote'].replace(' ', '') for x in r.get('evidence', [])) for r in (keyword, synonym))
        outcomes.append({'case': 'synonym_semantic_evidence', 'passed': semantic, 'shared_ids': sorted(left & right),
                         'note': '같은 주제의 인용 근거를 확인함. 검색 상위 후보 ID 일치는 보장하지 않음.'})
    finally:
        after = manifest()
        write('protected_after.json', after)
        changes = {name: {'before': before.get(name), 'after': after.get(name)} for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
        write('protected_changes.json', changes)
        outcomes.append({'case': 'preservation', 'passed': not changes, 'changes': len(changes)})
        report = {'outcomes': outcomes, 'all_passed': all(x['passed'] for x in outcomes),
                  'corpus_count': store.count, 'persisted_vectors': store.base_count, 'wal_vectors': store.queue_count,
                  'assemblies': store.assemblies,
                  'limitations': ['LLM 의미판정은 자동 검토 후보이며 사람 검증을 대체하지 않음',
                                  '최초는 검색 상위 후보 안에서만 산정', '저장 메타데이터는 원본 JSON과 대조 미확인',
                                  '브라우저 시각 검수는 별도 실행 결과 참조']}
        write('report.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0 if report['all_passed'] else 1)


if __name__ == '__main__':
    main()

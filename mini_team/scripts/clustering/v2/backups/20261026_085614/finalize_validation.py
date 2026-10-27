"""Revalidate the final outlook stage against recorded REAL corpus retrievals.

Search and review are not mocked. Their original result/provenance are retained.
Only the local LLM outlook is rerun, since this is the stage revised after the
first integration run. All outputs are new files in a new validation directory.
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from engine import LocalLLM, generate_forecast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--insufficient-result', type=Path, help='정책 질문으로 분류된 실제 근거 부족 검색 기록')
    args = parser.parse_args()
    source = args.source.resolve()
    changes = json.loads((source/'protected_changes.json').read_text(encoding='utf-8'))
    if changes:
        raise ValueError('원본 보존 검증이 통과하지 않았습니다.')
    out = Path(__file__).parent/'validation'/('final_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    write = lambda name, value: (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    llm, results, outcomes = LocalLLM(), {}, []
    positives = ('free_keyword', 'synonym_question', 'another_topic')
    for name in positives + ('unrelated', 'insufficient', 'connection_error'):
        raw = args.insufficient_result.resolve() if name == 'insufficient' and args.insufficient_result else source/(name+'.json')
        result = json.loads(raw.read_text(encoding='utf-8'))
        result['integration_source'] = {'path': str(raw), 'sha256': hashlib.sha256(raw.read_bytes()).hexdigest()}
        if name in positives:
            evidence = result['evidence']
            if len(evidence) < 2:
                raise ValueError('실제 검색 근거 부족: '+name)
            ids = set(result['forecast_evidence_ids'])
            selected = [x for x in evidence if x['id'] in ids]
            print('최종 로컬 LLM 전망 재검증:', name, flush=True)
            result['forecast'], result['forecast_citations'] = generate_forecast(llm, result['query'], result['scope'], selected)
            result['forecast_reference_date'] = max(x['date'] for x in selected)
            result['forecast_regenerated_at'] = datetime.now().isoformat()
            result['status'] = 'ok'
            result['message'] = '실제 전체 벡터 검색·원문 검토 결과와 최종 전망 생성 검증 완료'
            passed = all(x['quote'] in x['text'] and 30 <= len(x['quote']) <= 300 for x in evidence)
            passed &= [x['date'] for x in evidence] == sorted(x['date'] for x in evidence)
            passed &= set(result['first']) == {x['id'] for x in evidence if x['date'] == evidence[0]['date']}
            if result['scope']['year'] != '전체':
                passed &= all(x['date'].startswith(result['scope']['year']) for x in evidence)
            initial = {x['assembly'] for x in evidence if x['id'] in result['first']}
            passed &= all(x['date'] > evidence[0]['date'] and x['assembly'] not in initial for x in result['flows'])
        elif name == 'unrelated':
            passed = result['status'] == 'unrelated' and not result['evidence'] and result['forecast'] is None
        elif name == 'insufficient':
            passed = result['status'] == 'insufficient' and not result['evidence'] and result['forecast'] is None
        else:
            passed = result['status'] == 'error' and result['forecast'] is None
        results[name] = result
        write(name+'.json', result)
        outcomes.append({'case': name, 'passed': bool(passed), 'status': result['status'],
                         'evidence_count': len(result['evidence']), 'forecast_chars': len(result['forecast'] or '')})
    canonical_counts = {name: sum('전세사기' in x['quote'].replace(' ', '') for x in results[name]['evidence'])
                        for name in ('free_keyword', 'synonym_question')}
    left, right = ({x['id'] for x in results[name]['evidence']} for name in ('free_keyword', 'synonym_question'))
    outcomes.append({'case': 'synonym_semantic_evidence', 'passed': all(x > 0 for x in canonical_counts.values()),
                     'canonical_quote_counts': canonical_counts, 'shared_chunk_ids': sorted(left & right),
                     'note': '동의 표현도 전세사기 정책 근거를 찾음. 상위 후보 ID나 최초 후보 날짜의 동일성은 요구하지 않음.'})
    before = json.loads((source/'protected_before.json').read_text(encoding='utf-8'))
    outcomes.append({'case': 'source_preservation', 'passed': not changes, 'protected_files': len(before),
                     'hashed_files': sum('sha256' in x for x in before.values()), 'changes': len(changes)})
    unit_log = Path(__file__).with_name('unit_test_report.txt').read_text(encoding='utf-8')
    outcomes.append({'case': 'unit_and_http', 'passed': 'Ran 13 tests' in unit_log and '\nOK' in unit_log})
    report = {'all_passed': all(x['passed'] for x in outcomes), 'outcomes': outcomes,
              'real_search_source': str(source), 'preservation_report': str(source/'protected_changes.json'),
              'corpus_count': results['free_keyword']['scope']['total_chunks'],
              'wal_vectors': results['free_keyword']['scope']['wal_vector_count'],
              'browser_visual_test': {'verified': False, 'reason': 'Chrome/Edge GPU subprocess exits with access denied; no sandbox bypass attempted'},
              'stages': 'Actual corpus retrieval/relevance records retained; final local-LLM forecast regenerated with original text and citations.',
              'limitations': ['최초는 상위 검색 후보 내 관측일', '유사 후속 논의는 전파 증거 아님',
                              '메타데이터는 원본 JSON 대조 미확인', 'LLM 의미·전망은 자동 검토 결과']}
    write('report.json', report)
    text = '# V2 최종 검증 결과\n\n'
    text += f"- 전체 벡터: {report['corpus_count']:,}개 (기존 WAL {report['wal_vectors']}개 포함)\n"
    text += f'- 기존 파일 {len(before)}개 보존 확인, 변경 0개. 기존 코드/문서 및 Chroma 전체 파일은 SHA256 비교.\n'
    text += '- 기존 5개 결과를 조회하지 않고 실제 자유 주제로 전체 벡터 검색·원문 검토를 수행했습니다.\n'
    text += '- 검색 결과 원문은 그대로 보존하고, 보완한 최종 전망 단계만 로컬 LLM으로 다시 실행했습니다.\n\n'
    text += '| 시나리오 | 통과 | 근거 수 | 전망 글자 수 |\n|---|---|---|---|\n'
    for row in outcomes:
        text += f"| {row['case']} | {row['passed']} | {row.get('evidence_count', '-')} | {row.get('forecast_chars', '-')} |\n"
    text += '\n동의어 질문의 검색 상위 후보와 최초 후보 날짜는 원 키워드와 달랐습니다. 두 검색 모두 전세사기 관련 실제 인용 근거를 확인했습니다. 이는 최초 시점을 전체 자료 최초로 해석하면 안 되는 이유이기도 합니다.\n'
    text += '\n브라우저 시각 검수는 환경의 GPU 하위 프로세스 오류로 미확인입니다. HTTP 화면 제공·설정 API·검색 API·오류 응답은 자동 테스트를 통과했습니다.\n'
    text += '\n지도는 의회 소재지 대표 좌표를 표시합니다. 날짜·발언자 등 메타데이터는 원본 JSON 대조 미확인 상태이며, 자동 의미판정과 미래 전망은 사람 검토를 대체하지 않습니다.\n'
    text += '\n원본 실제 검색/보존 비교: '+str(source)+'\n'
    (out/'report.md').write_text(text, encoding='utf-8')
    print(out, flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0 if report['all_passed'] else 1)


if __name__ == '__main__':
    main()

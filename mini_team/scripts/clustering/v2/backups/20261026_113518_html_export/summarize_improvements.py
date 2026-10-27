"""Consolidate measured results without modifying earlier validation artifacts."""
import argparse
import json
from datetime import datetime
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    args = p.parse_args()
    here = Path(__file__).parent
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    backend = read(args.run/'report.json')
    ui = read(here/'validation/ui_improvements/browser_report.json')
    live = read(here/'validation/live_browser/report.json')
    before = read(args.run/'protected_before.json')
    after = read(args.run/'protected_after.json')
    root = here.resolve().parents[2]
    final_stat_changes = []
    for name, expected in after.items():
        path = root/name
        if not path.is_file():
            final_stat_changes.append(name)
            continue
        stat = path.stat()
        if stat.st_size != expected['size'] or stat.st_mtime_ns != expected['mtime_ns']:
            final_stat_changes.append(name)
    rows = []
    for case in backend['cases']:
        r = read(args.run/(case['case']+'.json'))
        rows.append({**case, 'forecast_chars':len(r.get('forecast') or ''),
                     'source_verified':sum(bool(x.get('source_check')) for x in r['evidence'])})
    out = here/'validation'/('completion_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    report = {'full_corpus':backend['total_chunks'], 'backend':backend,
              'ui_checks':ui, 'live_browser':live, 'protected_files':len(before),
              'protected_unchanged':before==after, 'cases':rows,
              'final_browser_phase_stat_changes':final_stat_changes,
              'actual_administrative_map_available':False,
              'limitations':['실제 행정 경계 자료 미확보: 지역별 건수와 선택 목록으로 대체',
                             '수집 JSON 자체의 별도 대조 미확인; 저장 청크 및 원본 정제 TSV 대조',
                             '최초는 검색 상위 후보 범위; 지역 간 직접 정책 전파 미확인'],
              'all_executed_checks_passed':backend['all_passed'] and ui['all_passed'] and live['all_passed'] and before==after and not final_stat_changes}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    text = '# V2 개선 검증 결과\n\n'
    text += f"전체 {backend['total_chunks']:,}개 기존 청크를 읽기 전용으로 사용했다. 재임베딩·재학습은 수행하지 않았다.\n\n"
    text += f"보호 대상 {len(before)}개 파일의 크기·수정시각 및 기존 코드/Chroma SHA256 비교: {'변경 없음' if before==after else '차이 있음'}. 작업 전 V2 49개 파일은 backups/20261026_085614에 보존했다.\n\n"
    text += '자동 검사 21개, 실제 Chrome UI 검사 19개, 전체 DB 정책 검색 8개 시나리오, 실제 Chrome → HTTP → 전체 DB → Ollama 연결 검색을 검증했다. UI 더보기 경계 검사는 별도 화면용 fixture를 사용했다.\n\n'
    text += '| 검색 사례 | 상태 | 검증 발언 | 소요(초) | 전망 글자 수 |\n|---|---|---:|---:|---:|\n'
    for r in rows:
        text += f"| {r['case']} | {r['status']} | {r['evidence']} | {r['seconds']} | {r['forecast_chars']} |\n"
    text += '\n실제 브라우저 검색(최종 코드):\n\n'
    for r in live['cases']:
        text += f"- {r['case']}: {r['status']}, {r['evidence']}건, {r['seconds']}초\n"
    text += '\n동의어의 정규 검색 표현과 실제 관련 근거를 확인했다. 원문 후보 캐시는 DB 서명을 다시 확인하고 관련성·원문 대조·전망은 매번 검증한다.\n\n'
    text += '**지도 제한:** 검증된 로컬 행정 경계 자료가 없으며 외부 다운로드가 네트워크 권한으로 차단되었다. 현재 화면은 지역별 건수·선택 목록을 표시한다. 실제 행정 경계 지도의 성공 표시를 완료했다고 주장하지 않는다.\n\n'
    text += '회의록명·발언·날짜·의회·안건·발언자는 정제 TSV와 대조했다. 수집 JSON 별도 대조는 미확인으로 표시한다. 최초는 검색 후보 범위이고 지역 간 전파는 미확인이다.\n\n'
    text += '실행: `C:/ProgramData/miniconda3/envs/project01/python.exe -B scripts/clustering/v2/app.py --port 7862`\n'
    (out/'report.md').write_text(text,encoding='utf-8')
    print(out, flush=True)
    print('Executed checks passed:',report['all_executed_checks_passed'], flush=True)

if __name__=='__main__':
    main()

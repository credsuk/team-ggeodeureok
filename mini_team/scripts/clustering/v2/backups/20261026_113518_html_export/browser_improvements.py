"""Actual Chrome CDP checks. Recorded retrieval plus explicit UI boundary fixtures."""
import base64
import argparse
import copy
import json
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen
import websocket
from app import make_server

HERE = Path(__file__).parent
out = HERE / 'validation' / 'ui_improvements'
out.mkdir(exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('--result', type=Path, default=HERE/'validation/final_20261026_072323_479227/free_keyword.json')
args = parser.parse_args()
real = json.loads(args.result.read_text(encoding='utf-8'))
class Replay:
    progress = {'stage': '원문 추출 및 관련성 검증', 'state': 'running'}
    store = type('Store', (), {'assemblies': sorted({x['assembly'] for x in real['evidence']}),
                              'count': 2189332, 'collection': real['scope']['collection']})()
    def analyze(self, *args):
        time.sleep(1.5)
        return copy.deepcopy(real)
server = make_server(Replay(), 0)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f'http://127.0.0.1:{server.server_port}'
with urlopen(Request('http://127.0.0.1:9236/json/new?'+url, method='PUT')) as r:
    tab = json.load(r)
ws = websocket.create_connection(tab['webSocketDebuggerUrl'], timeout=25, suppress_origin=True)
serial = 0
def cdp(method, params=None):
    global serial
    serial += 1
    ws.send(json.dumps({'id': serial, 'method': method, 'params': params or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get('id') == serial:
            if 'error' in r:
                raise RuntimeError(r['error'])
            return r.get('result', {})
def js(expression):
    r = cdp('Runtime.evaluate', {'expression': expression, 'awaitPromise': True, 'returnByValue': True})
    if r.get('exceptionDetails'):
        raise RuntimeError(r['exceptionDetails'])
    return r['result'].get('value')
def wait(expression):
    return js('new Promise((ok,bad)=>{let n=0;const t=setInterval(()=>{if('+expression+'){clearInterval(t);ok(true)}else if(++n>150){clearInterval(t);bad("timeout")}},100)})')
checks = {}
try:
    cdp('Page.enable')
    cdp('Emulation.setDeviceMetricsOverride', {'width': 1440, 'height': 1100, 'deviceScaleFactor': 1, 'mobile': False})
    wait("$('coverage').textContent.includes('2,189,332') && mapData!==null")
    js("$('query').value="+json.dumps(real['query'],ensure_ascii=False)+";$('k').value="+json.dumps(str(real['scope']['top_k']))+";$('search').requestSubmit()")
    wait("$('status').textContent.includes('검색 중 · 원문 추출')")
    checks['live_stage_visible'] = True
    wait("!$('submit').disabled && current?.status!=='searching'")
    checks['real_retrieval_rendered'] = js("current.evidence.length>0 && $('evidence').textContent.includes('실제 원문 발췌')")
    checks['forecast_200'] = js("!current.forecast || [...current.forecast].length<=200")
    checks['raw_id_hidden_by_default'] = js("!document.body.innerText.includes(current.evidence[0].id)")
    checks['map_fallback_reason'] = js("mapData.available || $('map-note').textContent.includes('로컬')")
    checks['region_filters_both_lists'] = js("(()=>{document.querySelector('#regions button').click();return rows().every(x=>province(x.assembly)===region)&&$('timeline-count').textContent.includes('/ '+rows().length+'개')})()")
    js("$('all').click()")
    png=cdp('Page.captureScreenshot', {'format':'png', 'captureBeyondViewport':True})
    (out/'desktop_real.png').write_bytes(base64.b64decode(png['data']))
    fixture = copy.deepcopy(real)
    fixture['evidence'] = []
    for i in range(12):
        x = copy.deepcopy(real['evidence'][i % len(real['evidence'])])
        x.update(id=f'UI_FIXTURE_{i}', date=f'2025-01-{i+1:02d}')
        x['metadata'].update(meeting_name=f'화면 경계 검증용 회의록 {i}', row_id=i)
        fixture['evidence'].append(x)
    fixture['timings'] = {n: {'seconds': 1.23} for n in ['질문 분석 및 검색어 확장','ChromaDB 검색','원문 추출 및 관련성 검증','날짜·의회별 흐름 분석','Ollama LLM 예측']}
    fixture['elapsed_seconds'] = 6.15
    js('render('+json.dumps(fixture, ensure_ascii=False)+')')
    checks['timeline_default_8'] = js("document.querySelectorAll('#timeline>.observation').length===8 && !$('timeline-more').hidden")
    checks['evidence_default_5'] = js("document.querySelectorAll('#evidence>.meeting').length===5 && !$('evidence-more').hidden")
    checks['more_and_collapse'] = js("(()=>{$('timeline-more').click();$('evidence-more').click();const all=document.querySelectorAll('#timeline>.observation').length===12&&document.querySelectorAll('#evidence>.meeting').length===12;$('timeline-more').click();$('evidence-more').click();return all&&document.querySelectorAll('#timeline>.observation').length===8&&document.querySelectorAll('#evidence>.meeting').length===5})()")
    checks['sort_ascending'] = js("(()=>{$('sort').value='asc';$('sort').dispatchEvent(new Event('change'));return document.querySelector('#timeline summary').textContent.includes('2025-01-01')})()")
    checks['source_details'] = js("(()=>{const d=document.querySelector('#evidence>.meeting');d.open=true;d.querySelector('.full-source').open=true;return d.innerText.includes(current.evidence[0].text)||d.innerText.includes(current.evidence[11].text)})()")
    checks['timing_summary'] = js("!$('timing-card').hidden && document.querySelectorAll('.timing').length===6")
    checks['desktop_side_by_side'] = js("Math.abs($('map').getBoundingClientRect().left-$('timeline').getBoundingClientRect().left)>300")
    png=cdp('Page.captureScreenshot', {'format':'png', 'captureBeyondViewport':True})
    (out/'desktop_fixture.png').write_bytes(base64.b64decode(png['data']))
    checks['filter_resets_state'] = js("(()=>{$('timeline-more').click();$('evidence-more').click();$('year').value='2025';$('year').dispatchEvent(new Event('change'));return !timelineAll&&!evidenceAll&&current.evidence.length===0})()")
    checks['small_lists_hide_buttons'] = js('(()=>{const r='+json.dumps(real,ensure_ascii=False)+";r.evidence=r.evidence.slice(0,3);render(r);return $('timeline-more').hidden&&$('evidence-more').hidden})()")
    checks['escaped_html'] = js("(()=>{const r=JSON.parse(JSON.stringify(current));r.evidence[0].quote='<img src=x onerror=alert(1)>';render(r);return !document.querySelector('#evidence img')})()")
    checks['insufficient_clear'] = js("(()=>{render({status:'insufficient',message:'관련 회의록 근거 부족',evidence:[]});return $('evidence').textContent.includes('근거 부족')&&!document.querySelector('#timeline details')})()")
    checks['unrelated_clear'] = js("(()=>{render({status:'unrelated',message:'정책과 무관한 질문',evidence:[]});return $('status').textContent.includes('무관')&&!document.querySelector('#evidence details')})()")
    cdp('Emulation.setDeviceMetricsOverride', {'width':390,'height':844,'deviceScaleFactor':1,'mobile':True})
    checks['mobile_korean'] = js("document.documentElement.scrollWidth<=390 && document.body.innerText.includes('근거 회의록')")
finally:
    (out/'browser_report.json').write_text(json.dumps({'mode':'real Chrome; recorded real retrieval + explicitly synthetic pagination fixtures', 'checks':checks, 'all_passed':all(checks.values())},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    ws.close()
    server.shutdown()

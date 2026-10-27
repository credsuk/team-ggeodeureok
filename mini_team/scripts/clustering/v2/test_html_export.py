"""Real browser search/download and offline report checks. All outputs stay in V2."""
import base64
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
import websocket

HERE = Path(__file__).resolve().parent
OUT = HERE / 'validation' / 'html_export'
OUT.mkdir(exist_ok=True)
with urlopen(Request('http://127.0.0.1:9236/json/new?http://127.0.0.1:7863', method='PUT')) as response:
    tab = json.load(response)
ws = websocket.create_connection(tab['webSocketDebuggerUrl'], timeout=30, suppress_origin=True)
serial = 0
events = []
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
        events.append(r)
def js(expression):
    r = cdp('Runtime.evaluate', {'expression': expression, 'returnByValue': True, 'awaitPromise': True})
    if r.get('exceptionDetails'):
        raise RuntimeError(r['exceptionDetails'])
    return r['result'].get('value')
def wait(expression, seconds=600):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if js(expression):
            return
        time.sleep(.5)
    raise TimeoutError(expression)
checks = {}
cdp('Page.enable')
cdp('Network.enable')
cdp('Browser.setDownloadBehavior', {'behavior': 'allow', 'downloadPath': str(OUT), 'eventsEnabled': True})
wait("typeof buildReport==='function' && mapData!==null && $('assembly').options.length>1", 60)
checks['button_initially_hidden'] = js("$('html-download').hidden")
js("$('query').value='노인 일자리';$('search').requestSubmit()")
print('Actual full-corpus search started', flush=True)
wait("!$('submit').disabled && current?.query==='노인 일자리'", 900)
real = js('current')
(OUT/'search_result.json').write_text(json.dumps(real, ensure_ascii=False, indent=2), encoding='utf-8')
checks['real_search_completed'] = real['status'] in ('ok', 'limited') and bool(real['evidence'])
checks['button_visible_after_completion'] = js("!$('html-download').hidden")
print({'status':real['status'], 'evidence':len(real['evidence']), 'seconds':real.get('elapsed_seconds')}, flush=True)
# Capture state before exporting, including all records hidden by UI pagination/filter.
js("window.beforeExport=JSON.stringify(current);window.exportNetworkStart=performance.getEntriesByType('resource').length;document.querySelector('#regions button')?.click()")
checks['default_limit_present'] = js("groups(current.evidence).length<=5 || document.querySelectorAll('#evidence .meeting').length<=5")
js("$('html-download').click()")
end = time.monotonic()+30
download = None
while time.monotonic()<end:
    files = list(OUT.glob('*.html'))
    if files:
        download = max(files, key=lambda p:p.stat().st_mtime)
        break
    time.sleep(.25)
assert download, 'Browser download did not complete'
checks['no_export_network_calls'] = js("performance.getEntriesByType('resource').length===window.exportNetworkStart")
checks['current_result_unchanged'] = js("JSON.stringify(current)===window.beforeExport")
html = download.read_text('utf-8')
checks['no_internal_ids'] = all(x['id'] not in html for x in real['evidence'])
checks['safe_filename'] = not any(c in download.name for c in '<>:"/\\|?*') and '노인 일자리' in download.name
# Compare parsed report text against the same in-memory values used on screen.
checks['all_result_fields_match'] = js("""(()=>{const d=new DOMParser().parseFromString(buildReport(current),'text/html'),s=d.body.textContent;
 return [current.query,current.forecast,current.model,current.scope.year,current.scope.assembly,...current.evidence.flatMap(x=>[x.date,x.assembly,title(x),x.quote,x.text,x.reason,...(x.related_chunks||[]).map(c=>c.text)])].filter(Boolean).every(v=>s.includes(String(v))) && [...d.querySelectorAll('section')].find(e=>e.querySelector('h2').textContent.startsWith('근거 회의록 전체')).querySelectorAll('article').length===current.evidence.length})()""")
checks['all_timeline_rows'] = js("(()=>{const d=new DOMParser().parseFromString(buildReport(current),'text/html');return [...d.querySelectorAll('section')].find(e=>e.querySelector('h2').textContent.startsWith('날짜별')).querySelectorAll('article').length===current.evidence.length})()")
checks['timings_and_first_and_citations'] = js("(()=>{const s=new DOMParser().parseFromString(buildReport(current),'text/html').body.textContent;return Object.entries(current.timings).every(([n,t])=>s.includes(n)&&s.includes(t.seconds+'초'))&&current.evidence.filter(x=>current.first.includes(x.id)||current.forecast_citations.includes(x.id)).every(x=>s.includes(x.quote))})()")
checks['map_or_explicit_table_fallback'] = js("(()=>{const d=new DOMParser().parseFromString(buildReport(current),'text/html');return !!d.querySelector('table') && (!!d.querySelector('svg path')||d.body.textContent.includes('지도 데이터가 없어 지역별 현황표로 대체'))})()")
checks['empty_and_insufficient_honest'] = js("['insufficient','error','unrelated'].every(status=>{const d=new DOMParser().parseFromString(buildReport({query:'없는 정책',status,message:'검증 근거 없음',evidence:[],model:'test'}),'text/html');return d.body.textContent.includes('근거 부족')&&d.body.textContent.includes('생성을 보류')&&!d.querySelector('article')})")
checks['malicious_html_escaped'] = js("""(()=>{const evil='<img src=x onerror=alert(1)><script>alert(2)</'+'script>',x={...current.evidence[0],quote:evil,text:evil,reason:evil,metadata:{meeting_name:evil}};const d=new DOMParser().parseFromString(buildReport({...current,query:evil,evidence:[x]}),'text/html');return !d.querySelector('script,img,iframe')&&d.body.textContent.includes(evil)&&!/[<>:"/\\\\|?*]/.test(reportFilename(evil,new Date()))})()""")
checks['stale_button_hidden'] = js("$('query').value+=' 변경';$('query').dispatchEvent(new Event('input'));$('html-download').hidden")
# Open the downloaded file with all network connectivity disabled.
cdp('Network.emulateNetworkConditions', {'offline':True,'latency':0,'downloadThroughput':0,'uploadThroughput':0})
events.clear()
cdp('Page.navigate', {'url':download.as_uri()})
wait("document.URL.startsWith('file:') && document.readyState==='complete'", 30)
text = js('document.body.textContent')
checks['download_opened_offline'] = real['query'] in text and all(x['quote'] in text and x['text'] in text for x in real['evidence'])
checks['standalone_no_external_resources'] = js("!document.querySelector('script,link,[src],iframe,object') && getComputedStyle(document.body).color==='rgb(25, 60, 52)'")
checks['no_offline_http_requests'] = not any(e.get('method')=='Network.requestWillBeSent' and e['params']['request']['url'].startswith(('http:','https:')) for e in events)
cdp('Emulation.setDeviceMetricsOverride', {'width':1440,'height':1100,'deviceScaleFactor':1,'mobile':False})
shot=cdp('Page.captureScreenshot', {'format':'png'})
(OUT/'offline_report.png').write_bytes(base64.b64decode(shot['data']))
report={'checks':checks,'all_passed':all(checks.values()),'query':real['query'],'status':real['status'],'evidence_count':len(real['evidence']),'download':str(download),'bytes':download.stat().st_size,'search_seconds':real.get('elapsed_seconds')}
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
ws.close()
assert report['all_passed'], report

"""Real Chrome -> HTTP -> complete read-only corpus -> Ollama (no replay)."""
import argparse
import base64
import json
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen
import websocket
from app import make_server
from engine import ReadOnlyChroma, SearchEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep-serving', action='store_true')
    args = parser.parse_args()
    out = Path(__file__).parent/'validation'/'live_browser'
    out.mkdir(exist_ok=True)
    print('Loading read-only full corpus', flush=True)
    engine = SearchEngine(ReadOnlyChroma())
    server = make_server(engine, 7862)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with urlopen(Request('http://127.0.0.1:9236/json/new?http://127.0.0.1:7862', method='PUT')) as r:
        tab = json.load(r)
    ws = websocket.create_connection(tab['webSocketDebuggerUrl'], timeout=20, suppress_origin=True)
    serial = 0
    def cdp(method, params=None):
        nonlocal serial
        serial += 1
        ws.send(json.dumps({'id':serial,'method':method,'params':params or {}}))
        while True:
            data=json.loads(ws.recv())
            if data.get('id')==serial:
                if 'error' in data:
                    raise RuntimeError(data['error'])
                return data.get('result',{})
    def js(expression):
        r=cdp('Runtime.evaluate',{'expression':expression,'returnByValue':True})
        if r.get('exceptionDetails'):
            raise RuntimeError(r['exceptionDetails'])
        return r['result'].get('value')
    def wait(expression, seconds=420):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(expression):
                return
            time.sleep(.5)
        raise TimeoutError(expression)
    checks={}
    cases=[]
    try:
        cdp('Page.enable')
        cdp('Emulation.setDeviceMetricsOverride',{'width':1440,'height':1100,'deviceScaleFactor':1,'mobile':False})
        wait("document.getElementById('coverage')?.textContent.includes('2,189,332')",30)
        for name, query in [('natural','실업급여 관련 논의가 있나요?'),('keyword','실업급여'),('unrelated','김치찌개 끓이는 레시피를 알려줘'),('insufficient','해오름ZQX927 주민 교통비 환급사업 논의')]:
            print('LIVE BROWSER SEARCH',name,flush=True)
            js("$('query').value="+json.dumps(query,ensure_ascii=False)+";$('k').value='12';$('search').requestSubmit()")
            observed=set()
            end=time.monotonic()+420
            while js("$('submit').disabled"):
                observed.add(js("$('status').textContent"))
                if time.monotonic()>end:
                    raise TimeoutError('live search')
                time.sleep(.5)
            result=js('current')
            (out/(name+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            good = result['status'] in ('ok','limited') if name in ('natural','keyword') else result['status']==name
            if name=='keyword':
                checks['verified_retrieval_cache_hit'] = result.get('retrieval_cache_hit') is True
                checks['keyword_same_policy'] = bool(result['evidence'])
            if name=='natural':
                checks['relevant_originals']=js("current.evidence.length>0&&current.evidence.every(x=>x.quote.includes('실업')||x.quote.includes('구직'))")
                checks['timings_visible']=js("!$('timing-card').hidden&&document.querySelectorAll('.timing').length===6")
                checks['ids_hidden']=js("!document.body.innerText.includes(current.evidence[0].id)")
                checks['source_titles']=js("current.evidence.every(x=>x.metadata.meeting_name)")
                checks['forecast_limit']=js("!current.forecast||[...current.forecast].length<=200")
                checks['progress_observed']=any('검색 중 ·' in s for s in observed)
                js("document.querySelector('#regions button')?.click()")
                checks['region_connects_details']=js("region!==null&&document.querySelector('#timeline details')!==null")
                js("document.querySelector('#timeline details').open=true;document.querySelector('#timeline .full-source').open=true")
                checks['original_visible']=js("$('timeline').innerText.includes(sorted(rows())[0].text)")
                js("$('all').click()")
                shot=cdp('Page.captureScreenshot',{'format':'png','captureBeyondViewport':True})
                (out/'real_search.png').write_bytes(base64.b64decode(shot['data']))
            cases.append({'case':name,'status':result['status'],'evidence':len(result['evidence']),
                          'seconds':result.get('elapsed_seconds'),'passed':good,'observed_stages':sorted(observed)})
            print(cases[-1],flush=True)
        checks['negative_cases_empty']=not result['evidence'] and result['forecast'] is None
    finally:
        report={'mode':'real Chrome and real full-corpus backend; no replay','checks':checks,'cases':cases,
                'all_passed':len(cases)==4 and all(checks.values()) and all(c['passed'] for c in cases)}
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
        ws.close()
        if not args.keep_serving:
            server.shutdown()
    if args.keep_serving:
        print('Dashboard remains available: http://127.0.0.1:7862',flush=True)
        threading.Event().wait()

if __name__=='__main__':
    main()

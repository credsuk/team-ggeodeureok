"""Browser rendering test using a RECORDED real search, avoiding a second LLM run.

Start Chrome/Edge headless on port 9235 with a dedicated v2 profile, then run
this script with --result <validation/free_keyword.json>. Requires websocket-client
already in project01. No browser downloads or third-party assets are used.
"""
import argparse
import base64
import json
import threading
from pathlib import Path
from urllib.request import urlopen, Request

import websocket
from app import make_server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding='utf-8'))
    if result['status'] != 'ok':
        raise ValueError('A successful REAL search recording is required')

    class Replay:
        def analyze(self, *unused):
            return dict(result)

    replay = Replay()
    replay.store = type('Store', (), {'assemblies': sorted({x['assembly'] for x in result['evidence']}),
                                    'count': result['scope']['total_chunks'],
                                    'collection': result['scope']['collection']})()
    server = make_server(replay, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{server.server_port}'
    with urlopen(Request('http://127.0.0.1:9235/json/new?' + url, method='PUT')) as response:
        tab = json.load(response)
    ws = websocket.create_connection(tab['webSocketDebuggerUrl'], timeout=30, suppress_origin=True)
    serial = 0

    def cdp(method, params=None):
        nonlocal serial
        serial += 1
        ws.send(json.dumps({'id': serial, 'method': method, 'params': params or {}}))
        while True:
            message = json.loads(ws.recv())
            if message.get('id') == serial:
                if 'error' in message:
                    raise RuntimeError(message['error'])
                return message.get('result', {})

    def js(expression):
        data = cdp('Runtime.evaluate', {'expression': expression, 'awaitPromise': True, 'returnByValue': True})
        if data.get('exceptionDetails'):
            raise RuntimeError(data['exceptionDetails'])
        return data['result'].get('value')

    checks = {}
    try:
        cdp('Page.enable')
        cdp('Emulation.setDeviceMetricsOverride', {'width': 1440, 'height': 1100, 'deviceScaleFactor': 1, 'mobile': False})
        js("new Promise((resolve,reject)=>{let n=0;const timer=setInterval(()=>{if(document.getElementById('coverage')?.textContent.includes('2,189,332')){clearInterval(timer);resolve(true)}else if(++n>100){clearInterval(timer);reject('config timeout')}},100)})")
        query = json.dumps(result['query'], ensure_ascii=False)
        js("document.getElementById('query').value=" + query + ";document.getElementById('search').requestSubmit()")
        js("new Promise((resolve,reject)=>{let n=0;const timer=setInterval(()=>{if(!document.getElementById('submit').disabled){clearInterval(timer);resolve(true)}else if(++n>100){clearInterval(timer);reject('render timeout')}},100)})")
        checks['evidence_rendered'] = js("document.querySelectorAll('#evidence details').length") == len(result['evidence'])
        checks['first_date'] = result['evidence'][0]['date'] in js("document.getElementById('first').innerText")
        checks['forecast_exact'] = js("document.getElementById('forecast').textContent") == result['forecast']
        checks['map_points'] = js("document.querySelectorAll('#map .point').length") > 0
        checks['timeline_rows'] = js("document.querySelectorAll('#timeline tbody tr').length") == len(result['evidence'])
        js("document.querySelector('#evidence details').open=true")
        checks['full_source_visible'] = result['evidence'][0]['text'] in js("document.querySelector('#evidence details').innerText")
        js("document.querySelector('#map .point').dispatchEvent(new MouseEvent('click',{bubbles:true}))")
        checks['map_filter'] = 0 < js("document.querySelectorAll('#evidence details').length") <= len(result['evidence'])
        js("document.getElementById('all').click();window.scrollTo(0,0)")
        png = cdp('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': True})
        screenshot = args.result.parent / 'dashboard_desktop.png'
        screenshot.write_bytes(base64.b64decode(png['data']))
        # Check escaping through the actual render path. No HTML from a retrieved
        # statement or metadata may become an executable element.
        checks['source_html_escaped'] = js("(()=>{const r=JSON.parse(JSON.stringify(current));r.evidence[0].quote='<img src=x onerror=alert(1)>';render(r);return document.querySelectorAll('#evidence img').length===0})()")
        cdp('Emulation.setDeviceMetricsOverride', {'width': 390, 'height': 844, 'deviceScaleFactor': 1, 'mobile': True})
        checks['mobile_layout'] = js("document.querySelector('.grid').getBoundingClientRect().width<=document.documentElement.clientWidth")
        (args.result.parent/'browser_report.json').write_text(json.dumps({'mode': 'real-result replay through HTTP/UI',
             'checks': checks, 'all_passed': all(checks.values()), 'screenshot': str(screenshot)}, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(checks, ensure_ascii=False, indent=2))
    finally:
        ws.close()
        server.shutdown()
        server.server_close()
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()

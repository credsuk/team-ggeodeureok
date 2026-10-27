"""Local-only dashboard. Run with the existing project01 Python, -B."""
import argparse
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from engine import DB, COLLECTION, LocalLLM, ReadOnlyChroma, SearchEngine, save_result


def make_server(engine, port=7862):
    token = secrets.token_urlsafe(32)
    page = Path(__file__).with_name('dashboard.html').read_text(encoding='utf-8').replace('__TOKEN__', token).encode()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, body, mime='application/json; charset=utf-8'):
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def json_reply(self, code, data):
            self.reply(code, json.dumps(data, ensure_ascii=False).encode())

        def do_GET(self):
            if self.headers.get('Host') not in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'):
                return self.json_reply(403, {'message': '로컬 호스트만 허용합니다.'})
            path = urlsplit(self.path).path
            if path == '/':
                return self.reply(200, page, 'text/html; charset=utf-8')
            if path == '/api/config':
                return self.json_reply(200, {'assemblies': engine.store.assemblies,
                                             'total_chunks': engine.store.count, 'collection': engine.store.collection})
            self.json_reply(404, {'message': '없는 경로'})

        def do_POST(self):
            if (self.path != '/api/search' or self.headers.get('X-V2-Token') != token
                    or self.headers.get('Host') not in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')):
                return self.json_reply(403, {'message': '허용되지 않은 요청'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 1 <= length <= 8192:
                    return self.json_reply(400, {'message': '요청 크기 오류'})
                body = json.loads(self.rfile.read(length))
                result = engine.analyze(body.get('query'), body.get('year', '전체'),
                                        body.get('assembly', '전체'), int(body.get('k', 24)))
                result['run_file'] = save_result(result).name
                self.json_reply(200, result)
            except (ValueError, TypeError, AttributeError) as exc:
                self.json_reply(400, {'status': 'error', 'message': str(exc)[:300]})
            except Exception as exc:
                self.json_reply(500, {'status': 'error', 'message': f'{type(exc).__name__}: {str(exc)[:300]}'})

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description='전체 회의록 자유 검색 V2')
    parser.add_argument('--db', type=Path, default=DB)
    parser.add_argument('--collection', default=COLLECTION)
    parser.add_argument('--model', default='qwen3:8b')
    parser.add_argument('--port', type=int, default=7862)
    parser.add_argument('--query', help='UI 없이 실제 검색을 실행하고 JSON 저장')
    parser.add_argument('--year', default='전체', choices=['전체', '2024', '2025'])
    parser.add_argument('--assembly', default='전체')
    parser.add_argument('--k', default=24, type=int)
    args = parser.parse_args()
    engine = SearchEngine(ReadOnlyChroma(args.db, args.collection), LocalLLM(args.model))
    if args.query:
        result = engine.analyze(args.query, args.year, args.assembly, args.k)
        print(save_result(result), flush=True)
        print(json.dumps({k: v for k, v in result.items() if k not in ('evidence',)}, ensure_ascii=False, indent=2))
        raise SystemExit(1 if result['status'] == 'error' else 0)
    server = make_server(engine, args.port)
    print(f'V2: http://127.0.0.1:{server.server_port} / {engine.store.count:,} chunks (read-only)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

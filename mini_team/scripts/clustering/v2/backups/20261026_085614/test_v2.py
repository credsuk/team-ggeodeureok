"""Deterministic failure/boundary tests; real DB/LLM validation is validate_v2.py."""
import json
import threading
import unittest
import tempfile
import sqlite3
import struct
import pickle
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from app import make_server
from engine import SearchEngine, ReadOnlyChroma, EMBED_MODEL, date_string, identifiers_present


TEXT = '전세사기 피해자의 주거 안정을 위하여 임대주택 공급과 법률 상담 지원 사업을 추진하고 있습니다.'


class Store:
    assemblies = ['서울특별시의회', '부산광역시의회']
    collection = 'full-test-collection'
    count = 1000

    def search(self, vectors, k, year, assembly):
        return [dict(id=str(i), score=.8, text=TEXT,
                     metadata={'assembly': self.assemblies[i % 2], 'date': '20240'+str(i+1)+'01',
                               'source_key': str(i), 'row_id': 1}) for i in range(2)], 1000


class LLM:
    model = 'test'

    def json(self, task, payload):
        if 'candidates' in payload:
            return {'reviews': {x['id']: {'relevant': True, 'quote': TEXT, 'reason': '주거 지원'} for x in payload['candidates']}}
        if 'evidence' in payload:
            return {'forecast': '향후 주거 지원 논의가 이어질 가능성이 있으나 실제 확대 여부는 미확인입니다.',
                    'evidence_ids': [x['id'] for x in payload['evidence']]}
        return {'is_policy': True, 'queries': ['전세 피해 지원']}


def engine(llm=None, store=None):
    value = SearchEngine(store or Store(), llm or LLM())
    value.encode = lambda texts: [[1., 0.] for _ in texts]
    return value


class V2Tests(unittest.TestCase):
    def test_specific_identifier_must_exist_in_original(self):
        self.assertTrue(identifiers_present('RE100 지원', 'RE-100을 지원'))
        self.assertFalse(identifiers_present('해오름ZQX927 교통비 사업', TEXT))
        e = engine()
        result = e.analyze('해오름ZQX927 교통비 사업')
        self.assertEqual(result['status'], 'insufficient')
        self.assertEqual(result['required_identifiers'], ['ZQX927'])
        self.assertEqual(len(result['identifier_rejected']), 2)
        self.assertIsNone(result['forecast'])

    def test_existing_vectors_wal_and_scoped_exact_search(self):
        import numpy as np
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            collection = 'fixture'
            sources = ['2024_서울특별시의회', '2025_부산광역시의회']
            ids = [hashlib.sha256(sources[j].encode()).hexdigest()[:20] + '_' + str(i) + '_1'
                   for i, j in enumerate([0, 0, 1, 0])]
            folder = root/'vector'
            folder.mkdir()
            # Deliberately non-monotonic stored labels; row order != label order.
            labels = [3, 1, 2]
            stride, vo, lo = 148, 132, 140
            (folder/'header.bin').write_bytes(struct.pack('<i6QiI3QdQ', 1, 0, 4, 3, stride, lo, vo, 0, 0, 16, 32, 16, .36, 100))
            raw = bytearray(stride*3)
            for i, vector in enumerate(([1., 0.], [0., 1.], [-1., 0.])):
                struct.pack_into('<2f', raw, i*stride+vo, *vector)
                struct.pack_into('<Q', raw, i*stride+lo, labels[i])
            (folder/'data_level0.bin').write_bytes(raw)
            with (folder/'index_metadata.pickle').open('wb') as f:
                pickle.dump({'label_to_id': dict(zip(labels, ids)), 'id_to_label': dict(zip(ids, labels))}, f)
            conn = sqlite3.connect(root/'chroma.sqlite3')
            conn.executescript('''CREATE TABLE collections(id,name,dimension,schema_str);
                CREATE TABLE segments(scope,id,collection);
                CREATE TABLE embeddings(id,segment_id,embedding_id);
                CREATE TABLE embedding_metadata(id,key,string_value,int_value,float_value,bool_value);
                CREATE TABLE max_seq_id(segment_id,seq_id);
                CREATE TABLE embeddings_queue(seq_id,id,operation,encoding,vector,topic);''')
            schema = {'keys': {'#embedding': {'float_list': {'vector_index': {'config': {'space': 'cosine'}}}}}}
            conn.execute('INSERT INTO collections VALUES (?,?,?,?)', ('c', collection, 2, json.dumps(schema)))
            conn.executemany('INSERT INTO segments VALUES (?,?,?)', [('VECTOR', 'vector', 'c'), ('METADATA', 'meta', 'c')])
            conn.execute('INSERT INTO max_seq_id VALUES (?,?)', ('vector', 3))
            for i, cid in enumerate(ids):
                conn.execute('INSERT INTO embeddings VALUES (?,?,?)', (i, 'meta', cid))
                for key, value in {'chroma:document': TEXT, 'assembly': sources[1 if i==2 else 0].split('_', 1)[1], 'date': '20250101' if i==2 else '20240101'}.items():
                    conn.execute('INSERT INTO embedding_metadata VALUES (?,?,?,NULL,NULL,NULL)', (i, key, value))
            conn.execute('INSERT INTO embeddings_queue VALUES (?,?,?,?,?,?)', (4, ids[3], 2, 'FLOAT32', np.asarray([.9, .1], dtype='<f4').tobytes(), 'persistent://default/default/c'))
            conn.commit()
            conn.close()
            manifest = {'collection': collection, 'model': EMBED_MODEL,
                        'regions': {source: {'chunks': n, 'completed': True} for source, n in zip(sources, [3, 1])}}
            (root/(collection+'_progress.json')).write_text(json.dumps(manifest), encoding='utf-8')
            store = ReadOnlyChroma(root, collection)
            rows, count = store.search([[1., 0.]], 4)
            self.assertEqual(count, 4)
            self.assertEqual([r['id'] for r in rows], [ids[0], ids[3], ids[1], ids[2]])
            self.assertAlmostEqual(rows[1]['score'], .99388, places=4)
            rows, count = store.search([[1., 0.]], 4, '2025', '부산광역시의회')
            self.assertEqual(count, 1)
            self.assertEqual(rows[0]['id'], ids[2])
            self.assertFalse(store.raw.flags.writeable)
            # Release the Windows mmap before TemporaryDirectory cleanup.
            store.raw._mmap.close()

    def test_dates(self):
        self.assertEqual(date_string('2024. 2. 29'), None)  # no ambiguous partial-padding normalization
        self.assertEqual(date_string('20240229'), '2024-02-29')
        self.assertIsNone(date_string('20250229'))

    def test_free_topic_and_forecast(self):
        r = engine().analyze('전세사기 피해자 지원')
        self.assertEqual(r['status'], 'ok')
        self.assertEqual(r['first'], ['0'])
        self.assertEqual(len(r['flows']), 1)
        self.assertLessEqual(len(r['forecast']), 200)
        self.assertTrue(all(x['quote'] in x['text'] for x in r['evidence']))

    def test_invalid_input_and_filter(self):
        e = engine()
        self.assertEqual(e.analyze('')['status'], 'invalid_query')
        self.assertEqual(e.analyze('x'*501)['status'], 'invalid_query')
        self.assertEqual(e.analyze('지원', year='2023')['status'], 'invalid_filter')
        self.assertEqual(e.analyze('지원', k=100)['status'], 'invalid_filter')

    def test_busy(self):
        e = engine()
        e.lock.acquire()
        self.assertEqual(e.analyze('전세 지원')['status'], 'busy')

    def test_unrelated_never_searches(self):
        class Unrelated(LLM):
            def json(self, task, payload):
                return {'is_policy': False, 'queries': [], 'reason': '요리 질문'}
        e = engine(Unrelated())
        e.encode = lambda _: self.fail('무관한 질문은 벡터 검색하지 않아야 합니다.')
        self.assertEqual(e.analyze('김치찌개 끓이는 법')['status'], 'unrelated')

    def test_hallucinated_quote_rejected(self):
        class BadQuote(LLM):
            def json(self, task, payload):
                result = super().json(task, payload)
                for x in result.get('reviews', {}).values():
                    x['quote'] = '원문에 존재하지 않는 조작된 근거를 정책 발언인 것처럼 제시하는 인용문입니다.'
                return result
        r = engine(BadQuote()).analyze('전세 지원')
        self.assertEqual(r['status'], 'insufficient')
        self.assertIsNone(r['forecast'])

    def test_llm_error_and_lock_release(self):
        class Broken(LLM):
            def json(self, task, payload):
                raise TimeoutError('test timeout')
        e = engine(Broken())
        self.assertEqual(e.analyze('전세 지원')['status'], 'error')
        self.assertFalse(e.lock.locked())

    def test_one_evidence_no_forecast(self):
        class Single(Store):
            def search(self, *args):
                return super().search(*args)[0][:1], 1000
        self.assertEqual(engine(store=Single()).analyze('전세 지원')['status'], 'limited')

    def test_oversize_forecast_and_fake_citation(self):
        class BadForecast(LLM):
            def json(self, task, payload):
                if 'evidence' in payload:
                    return {'forecast': '가'*201, 'evidence_ids': ['made-up']}
                return super().json(task, payload)
        r = engine(BadForecast()).analyze('전세 지원')
        self.assertEqual(r['status'], 'error')
        self.assertIsNone(r['forecast'])
        self.assertEqual(len(r['evidence']), 2)

    def test_placeholder_forecast_rejected(self):
        class Placeholder(LLM):
            def json(self, task, payload):
                if 'evidence' in payload:
                    return {'forecast': '전망', 'evidence_ids': ['0']}
                return super().json(task, payload)
        r = engine(Placeholder()).analyze('전세 지원')
        self.assertEqual(r['status'], 'error')
        self.assertIsNone(r['forecast'])

    def test_http_ui_config_and_csrf(self):
        server = make_server(engine(), 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base) as response:
                page = response.read().decode()
                self.assertIn('최초 관측 후보', page)
                self.assertIn('의회별 관측 지도', page)
                self.assertNotIn('__TOKEN__', page)
            with urlopen(base + '/api/config') as response:
                self.assertEqual(json.load(response)['total_chunks'], 1000)
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(base + '/api/search', data=b'{}'))
            self.assertEqual(caught.exception.code, 403)
            import re
            token = re.search("const token='([^']+)'", page)[1]
            request = Request(base + '/api/search', data=json.dumps({'query': '전세 지원'}).encode(),
                              headers={'X-V2-Token': token, 'Content-Type': 'application/json'})
            with urlopen(request) as response:
                self.assertEqual(json.load(response)['status'], 'ok')
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main(verbosity=2)

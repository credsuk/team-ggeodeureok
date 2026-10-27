import unittest
import csv
import tempfile
from pathlib import Path
from source_records import enrich
from map_assets import validate as validate_map
from engine import LocalLLM
from policy_guard import matches, variants, concepts
from test_v2 import engine, LLM, TEXT

class Improvements(unittest.TestCase):
    def test_forecast_grounding_failure_withholds_prediction(self):
        class RejectForecast(LocalLLM):
            def json(self, task, payload):
                if 'forecast_review' in payload:
                    return {'supported':False,'reason':'과거 시한을 미래처럼 표현함'}
                return LLM().json(task, payload)
        r = engine(RejectForecast()).analyze('전세사기')
        self.assertEqual(r['status'], 'error')
        self.assertIsNone(r['forecast'])
        self.assertIn('시제 검증 보류', r['message'])
        self.assertEqual(len(r['evidence']), 2)

    def test_wrong_map_is_not_rendered(self):
        with self.assertRaises(ValueError):
            validate_map({'type':'FeatureCollection','features':[]})
        f={'type':'Feature','properties':{'name':'서울특별시'},'geometry':{'type':'Polygon','coordinates':[]}}
        with self.assertRaises(ValueError):
            validate_map({'type':'FeatureCollection','features':[f]*17})

    def test_retrieval_cache_revalidates_db_and_reviews_original_question(self):
        from test_v2 import Store
        class CountingStore(Store):
            calls = 0
            checks = 0
            def search(self, *args):
                self.calls += 1
                return super().search(*args)
            def check_unchanged(self):
                self.checks += 1
        s = CountingStore()
        e = engine(store=s)
        a = e.analyze('전세사기')
        a['evidence'][0]['text'] = 'changed result only'
        b = e.analyze('전세 피해 주거 지원 논의')
        self.assertEqual(s.calls, 1)
        self.assertEqual(s.checks, 2)
        self.assertTrue(b['retrieval_cache_hit'])
        self.assertEqual(b['query'], '전세 피해 주거 지원 논의')
        self.assertEqual(b['evidence'][0]['text'], TEXT)
        def changed():
            raise RuntimeError('원본 DB 변경 감지')
        s.check_unchanged = changed
        c = e.analyze('전세사기')
        self.assertEqual(c['status'], 'error')
        self.assertFalse(c['evidence'])

    def test_unemployment_must_not_accept_single_household_policy(self):
        r = engine().analyze('실업급여 관련 논의가 있나요?')
        self.assertEqual(r['status'], 'insufficient')
        self.assertFalse(r['evidence'])
        self.assertFalse(r['first'])
        self.assertFalse(r['flows'])
        self.assertIsNone(r['forecast'])
        self.assertEqual(len(r['concept_rejected']), 2)

    def test_synonym_consistency_and_immutable_target(self):
        self.assertTrue(matches('실업급여 논의', '고용보험 구직 급여 지원을 논의합니다'))
        self.assertFalse(matches('구직급여 논의', '1인 가구 생활비 지원 확대'))
        self.assertEqual(list(concepts('실직한 사람에게 주는 급여 논의')), ['실업급여'])
        self.assertEqual(concepts('실직자를 위한 구직 교육 지원'), {})
        self.assertNotIn('1인 가구 지원', variants('실업급여', ['1인 가구 지원']))
        self.assertEqual(variants('실업급여 관련 논의가 있나요?', []), variants('구직급여 지원 논의', ['청년 지원']))
        for query, good, bad in [('전세사기','임대차 사기 피해 지원','청년 월세 지원'),
                                 ('디지털 교과서','AI 교과서 활용','디지털 행정 전환'),
                                 ('기후위기','기후 변화 적응','농촌 지원')]:
            self.assertTrue(matches(query, good))
            self.assertFalse(matches(query, bad))

    def test_all_timing_stages_recorded(self):
        e = engine()
        r = e.analyze('전세사기')
        self.assertEqual(len(r['timings']), 5)
        for t in r['timings'].values():
            self.assertGreaterEqual(t['seconds'], 0)
            self.assertIn('started_at', t)
            self.assertIn('ended_at', t)
        self.assertEqual(e.progress['state'], 'complete')

    def test_quote_must_contain_policy_not_only_surrounding_chunk(self):
        class UnrelatedQuote(LLM):
            def json(self, task, payload):
                r = super().json(task, payload)
                if 'candidates' in payload:
                    for item in r['reviews'].values():
                        item['quote'] = TEXT[6:]
                return r
        self.assertEqual(engine(UnrelatedQuote()).analyze('전세사기')['status'], 'insufficient')

    def test_source_row_verification_and_mismatch_exclusion(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            folder = root/'change_text'/'2024'
            folder.mkdir(parents=True)
            row = {'MTG_DE':'20240101','RASMBLY_NM':'서울특별시의회','MINTS_NM':'제1회 본회의',
                   'MTGNM':'본회의','agenda_title':'주거 지원','speaker_role':'위원','speaker_name':'홍길동','utterance':TEXT}
            with (folder/'sample.txt').open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row),delimiter='\t');w.writeheader();w.writerow(row)
            x={'metadata':{'source_file':'sample.txt','row_id':1,'date':'20240101','assembly':'서울특별시의회',
                           'agenda':'주거 지원','speaker':'위원 홍길동'},'text':TEXT,'quote':TEXT,'verification':'기존 검증'}
            self.assertEqual(enrich([x],root)[0]['metadata']['meeting_name'],'제1회 본회의')
            x['metadata']['date']='20240201'
            self.assertEqual(enrich([x],root),[])

if __name__ == '__main__':
    unittest.main(verbosity=2)

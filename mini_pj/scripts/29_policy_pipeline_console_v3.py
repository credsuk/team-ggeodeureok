# -*- coding: utf-8 -*-
"""29-1: 자유 정책어 → 전체자료 시점 검증 → 후속 논의 후보 → EXAONE 조건부 전망.

학원 PC 전용. 기존 Chroma를 수정하지 않음. 상용 API 호출 없음.

1) 먼저 일회성 검색 인덱스 구축 (전체 DB 1회 순회):
 python 29_policy_pipeline_console_v3.py --build-index
2) 자유 정책 이슈 분석:
 python 29_policy_pipeline_console_v3.py --issue "청년 주거" --aliases "청년주거,청년 월세,청년월세" --cutoff 2025-12-31
3) 빠른 연결 점검:
 python 29_policy_pipeline_console_v3.py --check

인덱스 없이 분석도 가능하지만 이때 최초 관측은 상위 의미검색 결과 내 후보일 뿐이다.
새 회의록이 추가/변경되면 --build-index --rebuild 로 인덱스를 갱신해야 한다.
주의: 날짜 선후 및 텍스트 유사도는 정책 전파의 인과관계 증명이 아니다.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import csv
import json
import re
import sqlite3
import time

ROOT = Path(r'C:\mini_pj')
COLLECTION = 'personal_policy_chunks'
EMBEDDING_MODEL = 'dragonkue/snowflake-arctic-embed-l-v2.0-ko'
OLLAMA_MODEL = 'exaone3.5:2.4b'


def log(msg):
    print(f'[{datetime.now():%H:%M:%S}] {msg}', flush=True)


def date_ok(raw):
    digits = ''.join(re.findall(r'\d', str(raw or '')))
    if len(digits) < 8:
        return ''
    val = f'{digits[:4]}-{digits[4:6]}-{digits[6:8]}'
    try:
        datetime.strptime(val, '%Y-%m-%d')
        return val
    except ValueError:
        return ''


def jsonwrite(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def centered_excerpt(txt, term, limit=180):
    txt = str(txt or '').replace('\x00', ' ')
    pos = txt.casefold().find(term.casefold()) if term else -1
    if pos < 0:
        return txt[:limit]
    start = max(0, pos - 35)
    end = min(len(txt), start + limit)
    return txt[start:end]


def matched_alias(txt, aliases):
    t = str(txt or '').casefold()
    return next((a for a in aliases if a.casefold() in t), '')


def chroma_collection(root):
    import chromadb
    path = root / '_db' / 'chroma_personal'
    if not path.exists():
        raise FileNotFoundError(f'개인 Chroma 폴더 없음: {path}')
    return chromadb.PersistentClient(path=str(path)).get_collection(COLLECTION)


def index_path(root):
    return root / 'results' / 'policy_search_index' / 'meetings_fts.sqlite3'


def connect_index(path):
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def create_index(root, batch=2000, rebuild=False):
    path = index_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    col = chroma_collection(root)
    db_count = col.count()
    if path.exists() and not rebuild:
        with connect_index(path) as conn:
            try:
                meta = dict(conn.execute('SELECT name,value FROM info').fetchall())
                existing = int(meta.get('db_count', '-1'))
                if existing == db_count and meta.get('complete') == 'yes':
                    log(f'기존 인덱스 유지: {path} / {db_count:,}청크')
                    log('※ DB 수정·교체 또는 동일 건수 갱신 시 --rebuild 필요')
                    return path
            except sqlite3.DatabaseError:
                pass
        raise RuntimeError('인덱스가 불완전하거나 DB 건수가 바뀌었습니다. --build-index --rebuild 실행')
    if path.exists():
        path.unlink()
    t0 = time.perf_counter()
    conn = connect_index(path)
    try:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('CREATE TABLE info (name TEXT PRIMARY KEY, value TEXT)')
        conn.execute('CREATE TABLE docs (id TEXT PRIMARY KEY, date TEXT, assembly TEXT, meeting_title TEXT, source_id TEXT, body TEXT)')
        conn.execute("CREATE VIRTUAL TABLE fts USING fts5(id UNINDEXED, body, tokenize='unicode61')")
        for offset in range(0, db_count, batch):
            p = col.get(limit=batch, offset=offset, include=['documents', 'metadatas'])
            documents = []
            fts = []
            for ident, md, txt in zip(p['ids'], p.get('metadatas') or [], p.get('documents') or []):
                md = md or {}
                ident = str(ident)
                txt = str(txt or '')
                documents.append((ident, date_ok(md.get('date')), str(md.get('assembly') or ''),
                                  str(md.get('meeting_title') or ''), str(md.get('source_id') or ''), txt))
                fts.append((ident, txt))
            conn.executemany('INSERT INTO docs VALUES (?,?,?,?,?,?)', documents)
            conn.executemany('INSERT INTO fts(id,body) VALUES (?,?)', fts)
            if offset % 20000 < batch or offset + batch >= db_count:
                conn.commit()
                log(f'인덱싱: {min(offset+batch,db_count):,}/{db_count:,} / {time.perf_counter()-t0:.1f}초')
        conn.execute('CREATE INDEX idx_date_region ON docs(date,assembly)')
        conn.executemany('INSERT INTO info VALUES (?,?)', [('db_count',str(db_count)), ('complete','yes'),
                                                    ('built_at',datetime.now().isoformat())])
        conn.commit()
        log(f'검색 인덱스 완성: {path} / {time.perf_counter()-t0:.1f}초')
        return path
    finally:
        conn.close()


def full_search(path, aliases, cutoff, cap=25000):
    """FTS는 후보 찾기용; Python에서 실제 부분문자열 검증. cap 도달 시 완전성 포기."""
    out, seen = [], set()
    with connect_index(path) as conn:
        meta = dict(conn.execute('SELECT name,value FROM info').fetchall())
        if meta.get('complete') != 'yes':
            raise RuntimeError('미완성 검색 인덱스')
        for alias in aliases:
            # FTS5 검색문자 자체는 인용부호로 둘러싸서 구문 연산자로 해석하지 않음
            fts_query = '"' + alias.replace('"', '""') + '"'
            try:
                cursor = conn.execute('SELECT d.id,d.date,d.assembly,d.meeting_title,d.source_id,d.body '
                                      'FROM fts JOIN docs d ON d.id=fts.id '
                                      'WHERE fts.body MATCH ? AND d.date <= ? ORDER BY d.date LIMIT ?',
                                      (fts_query, cutoff, cap + 1))
                for row in cursor:
                    if row['id'] in seen: continue
                    actual = matched_alias(row['body'], aliases)
                    if not actual or not row['date'] or not row['assembly']: continue
                    seen.add(row['id'])
                    out.append({'id':row['id'], 'date':row['date'], 'assembly':row['assembly'],
                                'meeting_title':row['meeting_title'], 'source_id':row['source_id'],
                                'excerpt_180':centered_excerpt(row['body'],actual), 'matched_term':actual,
                                'full_text':row['body']})
                    if len(seen) >= cap:
                        return out, False, len(seen)
            except sqlite3.OperationalError as e:
                raise RuntimeError(f'FTS 검색 실패 ({alias}): {e}') from e
    return out, True, len(seen)


def semantic_search(col, query, cutoff, top_k, aliases):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    log('임베딩 모델 로딩 및 의미 검색')
    model = SentenceTransformer(EMBEDDING_MODEL)
    vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    if vec.shape[1] != 1024:
        raise ValueError(f'벡터 차원 불일치: {vec.shape[1]} (기대 1024)')
    if col.count() < 1: return []
    raw = col.query(query_embeddings=vec.tolist(), n_results=min(top_k, col.count()),
                    include=['documents','metadatas','distances'])
    results = []
    for ident, txt, md, dist in zip(raw['ids'][0], raw['documents'][0],raw['metadatas'][0],raw['distances'][0]):
        md = md or {}
        date = date_ok(md.get('date'))
        if not date or date > cutoff: continue
        term = matched_alias(txt, aliases)
        results.append({'id':str(ident),'date':date,'assembly':str(md.get('assembly') or ''),
                        'meeting_title':str(md.get('meeting_title') or ''),
                        'source_id':str(md.get('source_id') or ''), 'excerpt_180':centered_excerpt(txt, term or ''),
                        'matched_term':term, 'full_text':str(txt or ''), 'distance':float(dist)})
    return results


def dedup_meetings(rows):
    """같은 회의록의 여러 청크를 근거 수로 과대 계산하지 않음."""
    unique = {}
    for r in sorted(rows, key=lambda x:(x['date'],x['assembly'],x['id'])):
        key = (r['date'],r['assembly'],r['source_id'] or r['meeting_title'] or r['id'])
        unique.setdefault(key, r)
    return list(unique.values())


def ollama_forecast(url, issue, cutoff, evidence, flow, months, model=OLLAMA_MODEL):
    import requests
    if not evidence:
        return '근거 부족: 조건부 전망을 생성하지 않았습니다.', 0.0
    facts = '\n'.join(f"[E{i}] {r['date']} {r['assembly']} {r['meeting_title']} ID={r['id']}\n{r['excerpt_180']}"
                      for i,r in enumerate(evidence[:8],1))
    route = '\n'.join(f"{r['date']} {r['assembly']} {r['meeting_title']}" for r in flow[:12])
    hist = ', '.join(f'{k}: {v}' for k,v in months[-12:])
    prompt = (f'지방의회 회의록 정책 분석가이다. 주제: {issue}; 분석 기준일: {cutoff}.\n'
              '아래 자료 이외의 지식을 사용해 날짜·사업·기관을 만들어내지 마라.\n'
              '검색어를 포함한 것만으로 정책의 동일성이나 전파를 단정하지 마라.\n'
              '제공 자료에 나타난 논의 내용과 시간 변화에 한정하여 향후 논의 가능성을 조건부로 전망하라. '
              '형식은 150자 이내 한국어 평서문 1~2개, 추가 설명 없이 전망문만 출력하라.\n'
              f'자료 내 지역별 최초 후보:\n{route}\n월별 관측 청크 수: {hist}\n근거:\n{facts}')
    t0 = time.perf_counter()
    payload = {'model':model,'prompt':prompt,'stream':False,
               'options':{'temperature':0,'num_ctx':8192,'num_predict':180}}
    resp = requests.post(url.rstrip('/')+'/api/generate', json=payload, timeout=420)
    resp.raise_for_status()
    txt = str(resp.json().get('response') or '').strip()
    if not txt or len(txt)>150:
        return f'150자 기준 미충족(출력 {len(txt)}자): 검토 필요', round(time.perf_counter()-t0,2)
    return txt, round(time.perf_counter()-t0,2)


def main():
    p = argparse.ArgumentParser(description='29-1 지방의회 정책 흐름 콘솔 검증')
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--build-index', action='store_true', help='최초 1회 전체 DB SQLite FTS 인덱스 작성')
    p.add_argument('--rebuild', action='store_true', help='기존 인덱스 삭제 후 전체 재생성')
    p.add_argument('--check', action='store_true')
    p.add_argument('--issue', default='청년 일자리')
    p.add_argument('--aliases', default='', help='추가 동의 표현, 쉼표 구분')
    p.add_argument('--cutoff', default='2025-12-31', help='분석 기준일 YYYY-MM-DD')
    p.add_argument('--top-k', type=int, default=80)
    p.add_argument('--max-lexical', type=int, default=25000)
    p.add_argument('--no-llm', action='store_true')
    p.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    a = p.parse_args()
    t0 = time.perf_counter()
    try:
        cutoff = date_ok(a.cutoff)
        if cutoff != a.cutoff: raise ValueError('--cutoff는 YYYY-MM-DD 형식이어야 합니다')
        col = chroma_collection(a.root)
        log(f'개인 Chroma 연결: {col.count():,} 청크')
        if a.check:
            import requests
            res = requests.get(a.ollama_url+'/api/tags',timeout=15)
            res.raise_for_status()
            names = [x.get('name') for x in res.json().get('models',[])]
            log(f'Ollama 모델: {names}; EXAONE 설치={OLLAMA_MODEL in names}')
            return
        if a.build_index:
            create_index(a.root, rebuild=a.rebuild)
            return
        issue = a.issue.strip()
        if not issue: raise ValueError('--issue를 입력해 주세요')
        aliases = list(dict.fromkeys([issue]+[x.strip() for x in a.aliases.split(',') if x.strip()]))
        log(f'검색 이슈={issue} / 표현={aliases} / 기준일={cutoff}')
        idx = index_path(a.root)
        lexical, complete, n = ([],False,0)
        if idx.exists():
            lexical, complete, n = full_search(idx, aliases, cutoff, a.max_lexical)
            log(f'전체 인덱스 정책어 후보: {n:,}개 / 완전 탐색={complete}')
        else:
            log('전체 검색 인덱스가 없음: 최초 관측은 의미검색 후보 범위에서만 표현됨')
        t1 = time.perf_counter()
        semantic = semantic_search(col, issue, cutoff, a.top_k, aliases)
        log(f'의미검색 반환: {len(semantic)}개 / {time.perf_counter()-t1:.2f}초')
        base = lexical if lexical else [r for r in semantic if r['matched_term']]
        scope = ('전체 DB 색인 정책어 후보' if idx.exists() and complete else '일부 검색 후보(전체 최초 아님)')
        documents = dedup_meetings(base)
        by_region = {}
        for row in sorted(documents, key=lambda r:(r['date'],r['assembly'],r['id'])):
            by_region.setdefault(row['assembly'],row)
        region_flow = sorted(by_region.values(),key=lambda r:(r['date'],r['assembly']))
        first = region_flow[0] if region_flow else None
        counts = Counter(x['date'][:7] for x in documents)
        monthly = sorted(counts.items())
        # LLM 증거는 의미 검색 결과를 우선, 주제가 실제 등장하는 근거 중심으로 합침
        evidence = dedup_meetings([r for r in semantic if r['matched_term']])[:8]
        if len(evidence)<8:
            existing = {r['id'] for r in evidence}
            evidence += [r for r in documents if r['id'] not in existing][:8-len(evidence)]
        # 정책 내용의 동일성 미확정. 추정 전파 확정 대신 날짜순 '후속 논의 후보'로만 출력.
        forecast, llm_sec = ('LLM 실행 생략 (--no-llm)',0) if a.no_llm else ollama_forecast(
            a.ollama_url,issue,cutoff,evidence,region_flow,monthly)
        outdir = a.root/'results'/'policy_console_v3'/datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir.mkdir(parents=True,exist_ok=True)
        safe_evidence = [{k:v for k,v in r.items() if k != 'full_text'} for r in evidence]
        safe_flow = [{k:v for k,v in r.items() if k not in ('full_text','distance')} for r in region_flow]
        result = {'issue':issue,'cutoff':cutoff,'aliases':aliases,'scope':scope,
                  'first_observation_candidate':safe_flow[0] if safe_flow else None,
                  'regional_followup_candidates':safe_flow,'monthly_counts_by_meeting':monthly,
                  'evidence':safe_evidence,'forecast_150':forecast,'model':OLLAMA_MODEL,
                  'llm_seconds':llm_sec,'total_seconds':round(time.perf_counter()-t0,2),
                  'warnings':['어휘 일치 후보이며 수동 검증되지 않음',
                              '날짜 선후·주제 유사성만으로 정책 전파의 인과관계를 확정할 수 없음',
                              '기준일 이후 회의록은 검색·예측 입력에서 제외',
                              '검색어의 동의어가 빠지면 최초 시점과 관측 건수가 누락될 수 있음']}
        jsonwrite(outdir/'analysis.json',result)
        with (outdir/'evidence.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['id','date','assembly','meeting_title','source_id','matched_term','excerpt_180'],extrasaction='ignore')
            writer.writeheader();writer.writerows(safe_evidence)
        log(f'① 최초 관측 후보 ({scope}): {first["date"]} {first["assembly"]} / {first["id"]}' if first else '① 최초 관측 후보 없음')
        log(f'② 타 의회 후속 논의 후보: {len(region_flow)}개 지역 / 회의록 단위 {len(documents)}건')
        log(f'③ EXAONE 전망: {forecast}')
        log(f'④ 원문 근거: {len(evidence)}건 (180자 이내)')
        log(f'⑤ 결과 저장: {outdir} / 총 {time.perf_counter()-t0:.2f}초')
    except Exception as ex:
        log(f'[실패] {type(ex).__name__}: {ex}')
        raise


if __name__=='__main__':
    main()

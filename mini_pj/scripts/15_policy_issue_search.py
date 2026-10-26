# -*- coding: utf-8 -*-
"""학원 PC 개인 프로젝트 | 15. 정책 이슈 탐색 (모델 재학습 없음).

기능
  (1) 684,311개 기존 메타데이터 원문에서 명시한 검색어/동의어를 모두 검사
  (2) 전체 청크 대비 지역·월별 검색어 일치 건수/비율, 표본 내 최초 관측일
  (3) 기존 개인 Chroma에서 의미 기반 상위 근거 발언 별도 검색
주의
  '전체검색' 통계는 지정한 어휘와 일치하는 청크 수이지 정책 관련 발언의 전수 추정이 아님.
  의미검색 상위 K건 역시 전체 관련 발언의 대표 표본이나 확산 증거가 아님.
  Chroma, 기존 임베딩, BERTopic, assignments 입력 파일은 읽기 전용.

사용 예:
  python 15_policy_issue_search.py --query "청년 일자리" --terms "청년 고용,청년 취업,청년 일자리,청년일자리"
  python 15_policy_issue_search.py --query "지역소멸" --terms "지방소멸,인구소멸,지역소멸"
  python 15_policy_issue_search.py --query "기후위기" --skip-semantic
  python 15_policy_issue_search.py --query "청년 일자리" --top-k 100
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\mini_pj")
ASSIGN_DIR = ROOT / "results" / "full_topic_modeling" / "assignments"
DB_DIR = ROOT / "_db" / "chroma_personal"
COLLECTION = "personal_policy_chunks"
MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
OUTPUT_ROOT = ROOT / "results" / "policy_issue_search"
EXPECTED_ROWS = 684_311


def stamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def elapsed(t0):
    t = time.perf_counter() - t0
    return f'{t:,.2f}초 / {t/60:,.2f}분 / {t/3600:,.3f}시간'


def safe_name(text):
    name = re.sub(r'[^\w가-힣-]+', '_', text, flags=re.UNICODE).strip('_')
    return name[:48] or 'issue'


def normalize_date(value):
    digits = re.sub(r'\D', '', str(value or ''))
    if len(digits) != 8:
        return ''
    try:
        return datetime.strptime(digits, '%Y%m%d').strftime('%Y-%m-%d')
    except ValueError:
        return ''


def pattern_for(term):
    # 내부 공백은 0개 이상의 공백으로 인정: '청년 일자리'와 '청년일자리' 매칭
    parts = term.strip().split()
    if not parts:
        raise ValueError('빈 검색어는 허용하지 않습니다.')
    return re.compile(r'\s*'.join(re.escape(p) for p in parts), re.IGNORECASE)


def csv_save(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def scan_corpus(terms, output, max_examples):
    """메타데이터 CSV 원문 전수 검사, assignments와 chunk_id 행 단위 대조."""
    t0 = time.perf_counter()
    assignments = sorted(ASSIGN_DIR.glob('*_토픽할당.csv'))
    if not assignments:
        raise FileNotFoundError(f'토픽 할당 CSV 없음: {ASSIGN_DIR}')
    regexes = [(term, pattern_for(term)) for term in terms]
    total = 0
    match_count = 0
    denominator_region_month = Counter()
    matches_region_month = Counter()
    matches_region = Counter()
    matches_month = Counter()
    matches_topic = Counter()
    first_seen = {}
    examples = []
    # matching texts can be numerous; do not materialize them all in memory
    matched_path = output / '전체_어휘일치_근거.csv'
    matched_fields = ['chunk_id','date','month','assembly','topic_id','matched_terms',
                      'meeting_title','source_file','source_id','text_preview','match_method']
    with matched_path.open('w', encoding='utf-8-sig', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=matched_fields)
        writer.writeheader()
        for i, path in enumerate(assignments, 1):
            s = time.perf_counter()
            assignment = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
            required = {'chunk_id','topic_id','date','assembly','meta_csv'}
            if not required.issubset(assignment.columns):
                raise ValueError(f'{path}: 할당 파일 필수 컬럼 누락 {required-set(assignment.columns)}')
            metas = assignment['meta_csv'].unique().tolist()
            if len(metas) != 1:
                raise ValueError(f'{path}: 단일 할당 파일에 메타데이터 경로가 {len(metas)}개')
            meta_path = Path(metas[0])
            if not meta_path.is_file():
                raise FileNotFoundError(f'원문 메타데이터 없음: {meta_path}')
            meta = pd.read_csv(meta_path, dtype=str, keep_default_na=False,
                               encoding='utf-8-sig', usecols=lambda col: col in {
                                   'chunk_id','text','meeting_title','source_file','source_id'})
            if len(meta) != len(assignment):
                raise ValueError(f'{path}: 원문 {len(meta):,} vs 할당 {len(assignment):,} 행수 불일치')
            if not (meta['chunk_id'].to_numpy() == assignment['chunk_id'].to_numpy()).all():
                raise ValueError(f'{path}: 원문과 토픽 할당 chunk_id 행 순서 불일치. 분석 중단')
            if 'text' not in meta:
                raise ValueError(f'{meta_path}: text 컬럼 없음')
            total += len(assignment)
            for j, (a, text) in enumerate(zip(assignment.itertuples(index=False), meta['text'])):
                # row.Index is not used; same order verified above
                assembly = str(a.assembly).strip()
                date = normalize_date(a.date)
                month = date[:7] if date else ''
                if assembly and month:
                    denominator_region_month[(assembly, month)] += 1
                found = [term for term, rx in regexes if rx.search(str(text))]
                if not found:
                    continue
                match_count += 1
                topic = str(a.topic_id)
                if assembly and month:
                    matches_region_month[(assembly, month)] += 1
                if assembly:
                    matches_region[assembly] += 1
                if month:
                    matches_month[month] += 1
                matches_topic[topic] += 1
                if date and (assembly not in first_seen or date < first_seen[assembly]):
                    first_seen[assembly] = date
                record = {
                    'chunk_id': str(a.chunk_id), 'date': date, 'month': month,
                    'assembly': assembly, 'topic_id': topic, 'matched_terms': '|'.join(found),
                    'meeting_title': str(meta.iloc[j].get('meeting_title', '')),
                    'source_file': str(meta.iloc[j].get('source_file', '')),
                    'source_id': str(meta.iloc[j].get('source_id', '')),
                    'text_preview': str(text)[:600], 'match_method': '어휘일치(전수)'
                }
                writer.writerow(record)
                if len(examples) < max_examples:
                    examples.append(record)
            print(f'[전체 어휘 검색 {i}/{len(assignments)}] '
                  f'누적 {total:,}건 / 일치 {match_count:,}건 / {elapsed(s)}', flush=True)
    if total != EXPECTED_ROWS:
        raise ValueError(f'전체 스캔 청크 {total:,}건 != 예상 {EXPECTED_ROWS:,}건. 결과 해석 금지')
    region_month = []
    for (assembly, month), denom in sorted(denominator_region_month.items()):
        count = matches_region_month[(assembly, month)]
        region_month.append({'assembly': assembly,'month': month, 'matched_chunks': count,
                             'all_chunks': denom, 'match_rate_pct': round(100*count/denom, 4)})
    csv_save(output / '지역_월별_어휘일치_추이.csv', region_month,
             ['assembly','month','matched_chunks','all_chunks','match_rate_pct'])
    by_region = []
    region_total = Counter()
    for (assembly, _month), n in denominator_region_month.items():
        region_total[assembly] += n
    for assembly in sorted(region_total):
        denom = region_total[assembly]
        by_region.append({'assembly': assembly,'matched_chunks': matches_region[assembly],
                          'all_chunks':denom,'match_rate_pct':round(100*matches_region[assembly]/denom,4),
                          'first_seen_in_corpus':first_seen.get(assembly,'')})
    csv_save(output / '지역별_어휘일치_통계.csv', by_region,
             ['assembly','matched_chunks','all_chunks','match_rate_pct','first_seen_in_corpus'])
    month_total = Counter()
    for (_, month), n in denominator_region_month.items():
        month_total[month] += n
    monthly = [{'month': m,'matched_chunks': matches_month[m],'all_chunks': n,
                'match_rate_pct':round(100*matches_month[m]/n,4)}
               for m,n in sorted(month_total.items())]
    csv_save(output/'월별_어휘일치_통계.csv', monthly,
             ['month','matched_chunks','all_chunks','match_rate_pct'])
    topics = [{'topic_id': k, 'matched_chunks': v}
              for k,v in sorted(matches_topic.items(), key=lambda x:-x[1])]
    csv_save(output/'관련_보조토픽_분포.csv',topics,['topic_id','matched_chunks'])
    return {'total_scanned':total,'lexical_matched_chunks':match_count,
            'lexical_matched_fraction_pct':round(100*match_count/total,4),
            'first_observed_date_in_corpus':min(first_seen.values()) if first_seen else '',
            'first_observed_assemblies':sorted(a for a,d in first_seen.items()
                                                 if d==min(first_seen.values())) if first_seen else [],
            'regions_with_matches':sum(v>0 for v in matches_region.values()),
            'processing_time':elapsed(t0), 'sample_rows':len(examples)}


def semantic_search(query, top_k, output):
    """Chroma의 기존 1024차원 임베딩과 같은 모델 사용. 새 DB 생성 금지."""
    t0 = time.perf_counter()
    import chromadb
    import numpy as np
    from sentence_transformers import SentenceTransformer
    if not DB_DIR.is_dir():
        raise FileNotFoundError(f'개인 Chroma 경로 없음: {DB_DIR}')
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION)
    n = collection.count()
    if n != EXPECTED_ROWS:
        print(f'[주의] Chroma {n:,}건 vs 예상 {EXPECTED_ROWS:,}건 (서로 다른 개수)',flush=True)
    model = SentenceTransformer(MODEL)
    vector = model.encode([query],convert_to_numpy=True)
    if vector.ndim != 2 or vector.shape[1] != 1024:
        raise ValueError(f'검색 임베딩 차원이 예상과 다름: {vector.shape}')
    search = collection.query(query_embeddings=vector.astype(np.float32).tolist(),
                              n_results=min(top_k,n),
                              include=['documents','metadatas','distances'])
    fields = ['rank','chunk_id','date','assembly','meeting_title','agenda_title',
              'speaker_name','source_file','distance','text_preview','method']
    rows = []
    ids = search['ids'][0]
    docs = search.get('documents',[[]])[0]
    mds = search.get('metadatas',[[]])[0]
    distances = search.get('distances',[[]])[0]
    for i, chunk_id in enumerate(ids):
        meta = mds[i] or {}
        rows.append({'rank':i+1,'chunk_id':chunk_id,
                     'date':normalize_date(meta.get('date','')),
                     'assembly':meta.get('assembly',''),
                     'meeting_title':meta.get('meeting_title',''),
                     'agenda_title':meta.get('agenda_title',''),
                     'speaker_name':meta.get('speaker_name',''),
                     'source_file':meta.get('source_file',''),
                     'distance':float(distances[i]) if i<len(distances) else '',
                     'text_preview':str(docs[i])[:1000] if i<len(docs) else '',
                     'method':'의미검색_상위K_관련성미검증'})
    csv_save(output/'Chroma_의미검색_근거.csv',rows,fields)
    return {'db_count':n,'semantic_top_k':len(rows), 'processing_time':elapsed(t0),
            'caution':'거리 점수는 품질 보증이나 전체 정책 논의량의 추정치가 아닙니다.'}


def main():
    parser = argparse.ArgumentParser(description='정책 이슈 전체 어휘 검색 + Chroma 의미 검색')
    parser.add_argument('--query',default='청년 일자리',help='검색할 정책 이슈')
    parser.add_argument('--terms',default='',help='쉼표로 구분한 관련 표현; 수동 검증 필요')
    parser.add_argument('--top-k',type=int,default=50,help='Chroma 의미 검색 상위 건수')
    parser.add_argument('--skip-semantic',action='store_true',help='Chroma 의미 검색 건너뛰기')
    parser.add_argument('--max-examples',type=int,default=20,help='콘솔 샘플 수 상한(저장에는 영향 없음)')
    args = parser.parse_args()
    if not args.query.strip():
        parser.error('--query에는 실제 정책 이슈를 입력하세요')
    if args.top_k<1 or args.max_examples<0:
        parser.error('--top-k >= 1, --max-examples >= 0')
    terms = list(dict.fromkeys(t.strip() for t in ([args.query] + args.terms.split(',')) if t.strip()))
    output = OUTPUT_ROOT / f'{datetime.now():%Y%m%d_%H%M%S}_{safe_name(args.query)}'
    output.mkdir(parents=True, exist_ok=False)
    t0=time.perf_counter()
    print(f'=== 정책 이슈 검색 시작: {stamp()} ===',flush=True)
    print(f'이슈: {args.query} | 어휘 검색어: {terms} | 결과: {output}',flush=True)
    report={'query':args.query,'lexical_terms':terms,'started_at':stamp(),
            'search_scope':'2024~2025 학원 PC 개인 프로젝트 회의록 684,311 청크',
            'statistics_scope':'지정 어휘의 전수 텍스트 일치(의미적 관련 발언의 전수 아님)'}
    try:
        report['lexical'] = scan_corpus(terms, output, args.max_examples)
        print(f"[어휘 검색 완료] {report['lexical']['lexical_matched_chunks']:,}건 / {report['lexical']['processing_time']}",flush=True)
        if not args.skip_semantic:
            report['semantic'] = semantic_search(args.query,args.top_k,output)
            print(f"[의미 검색 완료] 상위 {report['semantic']['semantic_top_k']}건 / {report['semantic']['processing_time']}",flush=True)
        else:
            report['semantic']={'skipped':True}
        report['finished_at']=stamp()
        report['total_time']=elapsed(t0)
        (output/'검색_실행요약.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'=== 검색 완료: {stamp()} / {elapsed(t0)} ===\n결과 폴더: {output}',flush=True)
        print('주의: 어휘일치 시점은 보유 코퍼스 내 최초 관측이며 정책의 실제 최초 등장/지역간 확산을 증명하지 않음.')
    except Exception as exc:
        (output/'오류_기록.txt').write_text(f'{stamp()}\n{type(exc).__name__}: {exc}',encoding='utf-8')
        raise


if __name__ == '__main__':
    main()

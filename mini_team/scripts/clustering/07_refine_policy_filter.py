"""07: 정책 발언 필터의 보수적 개선 및 68,000개 균형표본 라벨링.
원본 ChromaDB/NPZ는 읽기만 합니다. AI가 검토한 90개는 독립적인 사람 정답이 아닙니다.
실행: python 07_refine_policy_filter.py
재실행: 이미 출력된 청크 ID를 건너뛰고 이어서 처리합니다.
"""
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

import chromadb
import pandas as pd

ROOT = Path(r'C:\mini_team\scripts\clustering')
RESULTS = ROOT / 'results'
DB_PATH = Path(r'C:\mini_team\_db\chroma_snowflake_regions')
COLLECTION = 'team_regions_6a0a006e92d1e93e'
META_PATH = RESULTS / 'cluster_sample_balanced_metadata.csv'
VALIDATION_PATH = RESULTS / 'policy_validation' / 'policy_chunk_validation_details.csv'
REVIEW_PATH = RESULTS / 'policy_validation' / 'policy_manual_review_90_labeled.csv'
OUT = RESULTS / 'policy_refined'
BATCH_SIZE = 200
LOG_EVERY = 1000

# 정책 단어만 나왔다고 정책 발언으로 분류하지 않는다.
STRONG_POLICY = ('지원사업', '지원대상', '지원금', '예산편성', '예산확보', '예산집행',
    '사업계획', '사업추진', '사업비', '정책수립', '조례개정', '조례제정',
    '개선방안', '지원방안', '대응방안', '대책마련', '확대시행', '확대지원',
    '고용확대', '일자리창출', '창업지원', '주거지원', '돌봄서비스')
POLICY_NOUNS = ('예산', '사업', '지원', '조례', '정책', '대책', '계획', '제도', '집행',
    '개선', '확대', '축소', '의료', '복지', '주거', '교통', '교육', '고용',
    '창업', '저출생', '인구감소', '농업', '탄소중립', '환경', '안전')
SUBSTANCE = ('필요합니다', '추진하', '확대하', '개선하', '지원하', '마련하',
    '검토하', '시행하', '지원해야', '추진해야', '개정하', '제정하',
    '문제가 있', '어떻게 할', '대책이', '어떤 대책', '예산을', '사업을')
PROCEDURAL = ('수고하셨습니다', '질의하실 위원', '질의하실 의원',
    '이의 없으십니까', '가결되었음을 선포', '산회를 선포', '개의를 선포',
    '의석을 정돈', '정회를 선포', '속개하겠습니다', '안건을 상정',
    '표결을 시작', '투표를 시작', '기권의원', '찬성의원', '반대의원',
    '이상으로 회의를', '보고를 마치겠습니다', '감사합니다',
    '검토보고해 주시기 바랍니다', '다음 안건', '원안대로 가결')

FIELDS = ['chunk_id', 'source_key', 'date', 'assembly', 'char_length',
          'old_label', 'refined_label', 'policy_score', 'procedural_hits',
          'policy_hits', 'strong_hits', 'reason', 'is_substantive', 'excerpt_180']


def normalize(text):
    return re.sub(r'\s+', ' ', str(text or '')).strip()


def refine(text, substantive=None):
    """휴리스틱: 높은 확신이 아닐 때 REVIEW로 남겨 정책 발언 유실 방지."""
    s = normalize(text)
    if not s:
        return 'MISSING', 0, 0, 0, 0, '본문 없음'
    proc = sum(w in s for w in PROCEDURAL)
    noun = sum(w in s for w in POLICY_NOUNS)
    strong = sum(w in s for w in STRONG_POLICY)
    acts = sum(w in s for w in SUBSTANCE)
    score = 2 * strong + noun + acts
    # 강한 의사진행 표현이 지배적이며, 실질 정책 설명 단서가 거의 없는 짧은 문장
    if proc >= 1 and len(s) <= 130 and strong == 0 and acts == 0:
        return 'PROCEDURAL', score, proc, noun, strong, '짧은 진행성 문구가 지배적'
    if proc >= 2 and strong == 0 and acts == 0 and len(s) <= 240:
        return 'PROCEDURAL', score, proc, noun, strong, '진행성 문구 반복'
    if strong >= 1 and acts >= 1 and len(s) >= 30:
        return 'POLICY', score, proc, noun, strong, '구체적 정책 결합어 및 실질 내용'
    if noun >= 2 and acts >= 1 and len(s) >= 100:
        return 'POLICY', score, proc, noun, strong, '정책 명사 복수 + 실질 행위'
    if noun >= 3 and len(s) >= 200 and proc == 0:
        return 'POLICY', score, proc, noun, strong, '긴 정책 설명 및 다수 정책 용어'
    return 'REVIEW', score, proc, noun, strong, '보수적 판단 보류'


def review_evaluation():
    if not REVIEW_PATH.exists():
        print('[참고] 90개 검토 파일 없음. AI 검토 비교 생략:', REVIEW_PATH)
        return
    df = pd.read_csv(REVIEW_PATH, dtype=str, encoding='utf-8-sig').fillna('')
    required = {'chunk_id', 'human_label', 'excerpt_350'}
    if not required.issubset(df.columns):
        raise ValueError('검토 CSV 필수 컬럼 누락: ' + str(required - set(df.columns)))
    df = df[df['human_label'].isin(['정책관련', '의사진행', '판단보류'])].copy()
    if df.empty:
        print('[참고] 입력된 AI 검토 라벨 없음')
        return
    # 45/45 분할은 '참고 점검'이며 이 코드의 규칙은 이 데이터로 학습하지 않는다.
    import random
    ids = df.index.tolist()
    random.Random(42).shuffle(ids)
    holdout = set(ids[len(ids)//2:])
    df['review_split'] = ['holdout' if idx in holdout else 'development' for idx in df.index]
    converted = df['excerpt_350'].map(lambda x: refine(x)[0])
    df['refined_label'] = converted
    df['match'] = [({'POLICY':'정책관련','PROCEDURAL':'의사진행','REVIEW':'판단보류','MISSING':'판단보류'}.get(pred) == truth)
                   for pred, truth in zip(df['refined_label'], df['human_label'])]
    df[['review_no','chunk_id','review_split','heuristic_label','human_label',
        'refined_label','match','human_reason']].to_csv(
            OUT / 'review90_rule_comparison.csv', index=False, encoding='utf-8-sig')
    for split, g in df.groupby('review_split'):
        print(f'AI 판정 참고 일치율 [{split}]: {g.match.mean()*100:.1f}% ({g.match.sum()}/{len(g)})')
    print('주의: 이 수치는 AI 발췌 판정과의 일치율이며 실제 인간 정답 기반 정확도가 아닙니다.')


def write_summary(rows_path, elapsed, t0):
    df = pd.read_csv(rows_path, dtype=str, encoding='utf-8-sig').fillna('')
    if df['chunk_id'].duplicated().any():
        raise ValueError('중복 청크 ID 발견: 누적 CSV 확인 필요')
    df['source_key'] = df['source_key'].fillna('')
    count = (df.groupby(['source_key','refined_label']).size()
             .unstack(fill_value=0).reset_index())
    for label in ('POLICY','PROCEDURAL','REVIEW','MISSING'):
        if label not in count: count[label] = 0
    count['total'] = count[['POLICY','PROCEDURAL','REVIEW','MISSING']].sum(axis=1)
    count['policy_pct'] = (count['POLICY'] / count['total'] * 100).round(2)
    count.to_csv(OUT / 'policy_filter_by_region.csv', index=False, encoding='utf-8-sig')
    df.loc[df.refined_label == 'POLICY', ['chunk_id','source_key']].to_csv(
        OUT / 'policy_selected_ids.csv', index=False, encoding='utf-8-sig')
    df.loc[df.refined_label == 'REVIEW', ['chunk_id','source_key']].to_csv(
        OUT / 'policy_review_ids.csv', index=False, encoding='utf-8-sig')
    summary = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'count': int(len(df)), 'counts': {str(k): int(v) for k,v in df.refined_label.value_counts().items()},
        'region_year_groups': int(count.shape[0]), 'last_run_seconds': round(elapsed,2),
        'notes': 'POLICY/PROCEDURAL/REVIEW 규칙 기반 추정. REVIEW 데이터는 삭제 금지. 원본 DB 변경 없음.'
    }
    (OUT / 'policy_filter_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n=== 개선 필터 결과 ===')
    print(df.refined_label.value_counts().to_string())
    print('지역·연도 조합:',len(count))
    print('누적 처리 수:',len(df))
    print('이번 실행 전체 소요:', round(time.perf_counter()-t0,2),'초')


def main():
    t0=time.perf_counter()
    print('시작:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    OUT.mkdir(parents=True,exist_ok=True)
    if not META_PATH.exists():
        raise FileNotFoundError(META_PATH)
    sample=pd.read_csv(META_PATH, dtype=str, encoding='utf-8-sig').fillna('')
    if sample.chunk_id.duplicated().any():
        raise ValueError('68,000개 표본의 ID 중복')
    print('표본:', len(sample), '/ 그룹:',sample.source_key.nunique())
    print('[1/4] 검토 라벨과 개선 규칙 비교')
    review_evaluation()
    old_lookup={}
    if VALIDATION_PATH.exists():
        val=pd.read_csv(VALIDATION_PATH, dtype=str, encoding='utf-8-sig').fillna('')
        old_lookup=dict(zip(val['chunk_id'],val['heuristic_label']))
    output=OUT / 'policy_filter_68000.csv'
    existing=set()
    if output.exists() and output.stat().st_size:
        with output.open(encoding='utf-8-sig',newline='') as fh:
            r=csv.DictReader(fh)
            if r.fieldnames != FIELDS:
                raise ValueError('기존 누적 파일 컬럼이 다름. 백업 후 새로 시작하세요.')
            existing={x['chunk_id'] for x in r}
    todo=sample.loc[~sample.chunk_id.isin(existing)]
    print(f'[2/4] 기존 처리 {len(existing):,} / 이번 조회 {len(todo):,}')
    if len(todo):
        if not (DB_PATH / 'chroma.sqlite3').exists():
            raise FileNotFoundError(DB_PATH / 'chroma.sqlite3')
        connect_t=time.perf_counter()
        client=chromadb.PersistentClient(path=str(DB_PATH))
        coll=client.get_collection(name=COLLECTION)
        print('DB 연결 및 컬렉션:', round(time.perf_counter()-connect_t,2),'초',coll.name)
        with output.open('a',encoding='utf-8' if existing else 'utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS)
            if not existing: writer.writeheader()
            for start in range(0,len(todo),BATCH_SIZE):
                part=todo.iloc[start:start+BATCH_SIZE]
                ids=part['chunk_id'].tolist()
                result=coll.get(ids=ids,include=['documents','metadatas'])
                found={key:(doc,meta or {}) for key,doc,meta in zip(
                    result['ids'],result['documents'] or [],result['metadatas'] or [])}
                batch_rows=[]
                for r in part.to_dict('records'):
                    cid=r['chunk_id']
                    text,meta=found.get(cid,('',{}))
                    label,score,proc,noun,strong,reason=refine(text,meta.get('is_substantive'))
                    batch_rows.append(dict(
                        chunk_id=cid,source_key=r.get('source_key',''),date=r.get('date',''),
                        assembly=r.get('assembly',''),char_length=len(normalize(text)),
                        old_label=old_lookup.get(cid,''),refined_label=label,
                        policy_score=score,procedural_hits=proc,policy_hits=noun,
                        strong_hits=strong,reason=reason,
                        is_substantive=meta.get('is_substantive',''),
                        excerpt_180=normalize(text)[:180]))
                writer.writerows(batch_rows)
                f.flush()
                processed=start+len(ids)
                if processed % LOG_EVERY == 0 or processed == len(todo):
                    elapsed=time.perf_counter()-t0
                    print(f'원문 조회: {processed:,}/{len(todo):,} | {processed/max(elapsed,0.01):.1f}개/초 | 경과 {elapsed:.1f}초')
    print('[3/4] 결과 요약·선정 ID 저장')
    write_summary(output,time.perf_counter()-t0,t0)
    print('[4/4] 파일 저장 위치:',OUT)
    print('종료:',datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('다음 단계 전 정책발언과 REVIEW 표본을 직접 확인하세요.')

if __name__=='__main__':
    main()

# -*- coding: utf-8 -*-
"""학원 PC 개인 작업: 기존 10,000개 표본 그대로 / Kiwi 캐시 / BERTopic 고속 토픽 표현.
실행: mini 환경의 Python으로 scripts 폴더의 이 파일 실행
기존 개인 Chroma DB, 1/2차 평가 CSV, 기존 토픽요약.csv는 수정하지 않음.
"""
from __future__ import annotations
import csv
import json
import re
import time
import hashlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

ROOT = Path(r'C:\mini_pj')
EMBED_DIR = ROOT / 'data' / 'processed' / 'embeddings_by_region'
OLD_SAMPLE = ROOT / 'results' / 'rag_evaluation' / 'bertopic_2차_kiwi' / '표본_청크ID.csv'
OUT = ROOT / 'results' / 'rag_evaluation' / 'bertopic_2차_optimized'
CACHE = OUT / 'cache'
MODEL_DIR = OUT / 'bertopic_model'
SEED = 42
# 기존 군집 기준 고정. UMAP/HDBSCAN 설정은 이전 실험과 동일.
TOP_N_WORDS = 10

# 지나치게 넓은 불용어를 피하고 회의 진행·웹 서식·지역명 등만 제거.
STOPWORDS = {
    '의원','위원','위원장','의장','국장','과장','담당관','실장','본회의','위원회','의회',
    '회의록','회의','발언','질의','질문','답변','보고','말씀','수고','여러분','저희',
    '우리','내용','사항','관련','이번','다음','오늘','지금','현재','해당','부분',
    '정도','때문','생각','감사','진행','설명','사실','자리','안건','상정','가결',
    '투표','재석','찬성','반대','기권','출석','폐회','개의','다운로드','첨부',
    '글자체','줄간격','조회','메뉴','페이지','원본파일','홈페이지','바로가기',
    '충청남도','충청북도','강원특별자치도','경상북도','경상남도','전라북도',
    '전라남도','경기도','서울특별시','울산광역시','대구광역시','부산광역시',
    '인천광역시','광주광역시','대전광역시','세종특별자치시','제주특별자치도',
}
BANNED_SUBSTRINGS = ('navigation','toggle','javascript','http://','https://')


def stamp(): return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def elapsed(t):
    s=time.perf_counter()-t
    return f'{s:,.2f}초 / {s/60:,.2f}분', s

def save_jsonl(path, rows):
    with path.open('w',encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')

def load_jsonl(path):
    with path.open('r',encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]

def sample_signature(df):
    return hashlib.sha256('\n'.join(df['chunk_id'].astype(str)).encode('utf-8')).hexdigest()[:20]


def load_exact_sample():
    """이미 실험에 사용한 청크 ID를 고정하고 원본 배치에서 텍스트·벡터 회수."""
    t=time.perf_counter()
    if not OLD_SAMPLE.exists():
        raise FileNotFoundError(f'기존 표본 목록이 없습니다: {OLD_SAMPLE}')
    sf=pd.read_csv(OLD_SAMPLE, dtype=str, encoding='utf-8-sig', keep_default_na=False)
    if 'chunk_id' not in sf: raise ValueError('기존 표본 CSV에 chunk_id가 없습니다.')
    if sf.chunk_id.duplicated().any(): raise ValueError('표본 청크 ID가 중복됩니다.')
    signature=sample_signature(sf)
    meta_file=CACHE / 'sample_manifest.json'
    text_file=CACHE / 'sample_texts.jsonl'
    vec_file=CACHE / 'sample_vectors.npy'
    if all(p.exists() for p in (meta_file,text_file,vec_file)):
        manifest=json.loads(meta_file.read_text(encoding='utf-8'))
        if manifest.get('signature') == signature:
            rows=load_jsonl(text_file)
            vec=np.load(vec_file,mmap_mode='r')
            if len(rows)==len(sf)==len(vec):
                print(f'[표본 캐시] {len(rows):,}개 / {elapsed(t)[0]}',flush=True)
                return rows,vec,signature
    target=set(sf.chunk_id)
    found={}
    files=sorted(EMBED_DIR.rglob('batch_*.npy'))
    for i, npy in enumerate(files,1):
        meta=npy.with_name(npy.stem+'_metadata.csv')
        if not meta.exists(): raise FileNotFoundError(meta)
        # CSV 원문에는 개행이 있으므로 pandas로 실제 행 기준 읽기
        frame=pd.read_csv(meta,encoding='utf-8-sig',dtype=str,keep_default_na=False)
        if 'chunk_id' not in frame or 'text' not in frame:
            raise ValueError(f'필수 컬럼 없음: {meta}')
        selection=frame['chunk_id'].isin(target)
        if selection.any():
            arr=np.load(npy,mmap_mode='r')
            if len(arr)!=len(frame): raise ValueError(f'벡터·CSV 행 수 불일치: {npy}')
            indices=np.flatnonzero(selection.to_numpy())
            for idx in indices:
                rec=frame.iloc[int(idx)]
                uid=str(rec['chunk_id'])
                if uid in found: raise ValueError(f'원본 배치에 중복 chunk_id: {uid}')
                found[uid]=(str(rec['text']),np.asarray(arr[int(idx)],dtype=np.float32))
        if i % 20 == 0 or i==len(files):
            print(f'  [표본 복원 {i}/{len(files)}] 확보 {len(found):,}/{len(target):,}',flush=True)
        if len(found)==len(target): break
    missing=target-set(found)
    if missing: raise ValueError(f'기존 표본 ID {len(missing)}개를 찾지 못했습니다. 예: {list(missing)[:3]}')
    ordered=[{'chunk_id':u,'text':found[u][0]} for u in sf.chunk_id]
    vectors=np.stack([found[u][1] for u in sf.chunk_id])
    if vectors.shape[1]!=1024 or not np.isfinite(vectors).all():
        raise ValueError(f'벡터 차원/유효성 오류: {vectors.shape}')
    save_jsonl(text_file,ordered)
    np.save(vec_file,vectors)
    meta_file.write_text(json.dumps({'signature':signature,'rows':len(ordered),'dimension':1024},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'[표본 복원 완료] {len(ordered):,}개 / {elapsed(t)[0]}',flush=True)
    return ordered,vectors,signature


def tokenize_and_cache(rows, signature):
    """청크별 Kiwi를 단 한 번만 실행. BERTopic에는 이미 처리된 문자열 입력."""
    t=time.perf_counter()
    token_file=CACHE/f'tokens_{signature}.jsonl'
    if token_file.exists():
        obj=load_jsonl(token_file)
        if len(obj)==len(rows) and all(x.get('chunk_id')==r['chunk_id'] for x,r in zip(obj,rows)):
            print(f'[Kiwi 캐시 재사용] {len(obj):,}개 / {elapsed(t)[0]}',flush=True)
            return [x['tokens'] for x in obj]
    kiwi=Kiwi(num_workers=-1)
    results=[]
    filtered=0
    for idx,row in enumerate(rows,1):
        raw=str(row['text'])
        # 웹 메뉴 라인을 제외하고 발언 원본을 변형하지 않는 별도 표현용 텍스트만 생성
        cleaned='\n'.join(line for line in raw.splitlines()
                          if not any(b in line.lower() for b in BANNED_SUBSTRINGS))
        words=[]
        for tok in kiwi.tokenize(cleaned):
            if tok.tag not in {'NNG','NNP'}: continue
            word=tok.form.strip().lower()
            if len(word)<2 or word in STOPWORDS: continue
            if not re.fullmatch(r'[가-힣]{2,}',word): continue
            words.append(word)
        # 전부 삭제된 청크는 별도 무의미 토큰으로 표시해 빈 사전 오류 방지
        if not words:
            words=['분류불가']
            filtered+=1
        results.append({'chunk_id':row['chunk_id'],'tokens':' '.join(words)})
        if idx % 1000 == 0 or idx==len(rows):
            print(f'  [Kiwi {idx:,}/{len(rows):,}] {elapsed(t)[0]}',flush=True)
    save_jsonl(token_file,results)
    print(f'[Kiwi 완료] {len(results):,}개, 빈 문서 {filtered:,}개 / {elapsed(t)[0]}',flush=True)
    return [r['tokens'] for r in results]


def fit_topics(rows,vectors,token_docs):
    t=time.perf_counter()
    vectorizer=CountVectorizer(
        tokenizer=str.split, token_pattern=None, lowercase=False,
        ngram_range=(1,2), min_df=1, max_df=1.0, max_features=30000)
    # 따로 토큰화한 문자열이 실제 단어 사전으로 만들어지는지 간단 점검
    probe=CountVectorizer(tokenizer=str.split,token_pattern=None,lowercase=False)
    probe.fit(token_docs[:100])
    print(f'[사전 검사] 앞 100개 문서 단어 {len(probe.vocabulary_):,}개',flush=True)
    umap=UMAP(n_neighbors=15,n_components=5,min_dist=0.0,metric='cosine',
              random_state=SEED,low_memory=True)
    cluster=HDBSCAN(min_cluster_size=35,min_samples=10,
                    metric='euclidean',prediction_data=False)
    # 사전 토큰화된 텍스트 전달: 긴 토픽별 합본 문서를 Kiwi로 재분석하지 않음
    topic_model=BERTopic(umap_model=umap,hdbscan_model=cluster,
                        vectorizer_model=vectorizer,verbose=True,
                        language='multilingual',calculate_probabilities=False,
                        low_memory=True,top_n_words=TOP_N_WORDS)
    labels,_=topic_model.fit_transform(token_docs,embeddings=np.asarray(vectors))
    print(f'[모델링 완료] {elapsed(t)[0]}',flush=True)
    return topic_model,np.asarray(labels)


def save_topics(topic_model,labels,rows,signature):
    t=time.perf_counter()
    info=topic_model.get_topic_info().drop(columns=['Representative_Docs'],errors='ignore')
    info.to_csv(OUT/'토픽요약_개선.csv',index=False,encoding='utf-8-sig')
    doc_rows=[]
    examples=[]
    group=defaultdict(list)
    for row,lab in zip(rows,labels):
        lab=int(lab)
        doc_rows.append({'chunk_id':row['chunk_id'],'토픽ID':lab})
        if lab!=-1 and len(group[lab])<3:
            group[lab].append(row)
    for topic_id,docs in sorted(group.items()):
        words=', '.join(x for x,_ in (topic_model.get_topic(topic_id) or [])[:10])
        for j,r in enumerate(docs,1):
            examples.append({'토픽ID':topic_id,'키워드':words,'대표발언순번':j,
                             'chunk_id':r['chunk_id'],'원문 발췌':r['text'][:600],
                             '선정방식':'토픽 내 앞 3개 (대표성 검증 필요)'})
    pd.DataFrame(doc_rows).to_csv(OUT/'표본_토픽할당.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(examples).to_csv(OUT/'토픽대표발언_개선.csv',index=False,encoding='utf-8-sig')
    # 모델 저장: 이후 LLM 실험 시 같은 토픽을 재학습하지 않고 사용.
    # pickle은 신뢰할 수 있는 로컬 파일에만 사용.
    topic_model.save(str(MODEL_DIR),serialization='pickle',save_embedding_model=False)
    np.save(OUT/'토픽라벨.npy',labels)
    (OUT/'실험정보.json').write_text(json.dumps({
        'timestamp':stamp(),'source_sample':str(OLD_SAMPLE),'sample_signature':signature,
        'sample_size':len(rows),'topic_count':int(sum(info['Topic']!=-1)),
        'noise_count':int(sum(labels==-1)), 'embedding_dimension':1024
    },ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'[저장 완료] {elapsed(t)[0]}',flush=True)
    print('[대표 키워드 상위 토픽]')
    for _,r in info[info.Topic!=-1].head(15).iterrows():
        tid=int(r.Topic)
        kw=', '.join(w for w,_ in (topic_model.get_topic(tid) or [])[:7])
        print(f'  토픽 {tid} ({int(r.Count):,}건): {kw}')
    print('노이즈:',int(sum(labels==-1)),'/',len(labels))


def main():
    total=time.perf_counter()
    OUT.mkdir(parents=True,exist_ok=True)
    CACHE.mkdir(parents=True,exist_ok=True)
    print(f'=== 개인 BERTopic 최적화 시작: {stamp()} ===',flush=True)
    print('1차/2차 결과 보존 / 동일 표본 사용 / LLM 실행 없음',flush=True)
    try:
        t=time.perf_counter()
        rows,vectors,signature=load_exact_sample()
        print(f'[1] 표본 준비: {elapsed(t)[0]}',flush=True)
        t=time.perf_counter()
        token_docs=tokenize_and_cache(rows,signature)
        print(f'[2] Kiwi 사전 토큰화: {elapsed(t)[0]}',flush=True)
        t=time.perf_counter()
        model,labels=fit_topics(rows,vectors,token_docs)
        print(f'[3] BERTopic: {elapsed(t)[0]}',flush=True)
        t=time.perf_counter()
        save_topics(model,labels,rows,signature)
        print(f'[4] 결과·모델 저장: {elapsed(t)[0]}',flush=True)
        print('결과 폴더:',OUT,flush=True)
    finally:
        print(f'=== 종료 {stamp()} / 전체 {elapsed(total)[0]} ===',flush=True)

if __name__=='__main__':
    main()

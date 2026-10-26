# -*- coding: utf-8 -*-
"""개인 3차: 2차와 동일한 토픽/Chroma 검색/모델 옵션을 유지하고 프롬프트만 개선한 비교 실험.
BERTopic 재학습 없음. 3차 전용 CSV 누적 및 종료 시 Excel 자동 생성.
"""
import csv
import json
import re
import time
import traceback
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic

ROOT = Path(r'C:\mini_pj')
EMBED_DIR = ROOT / 'data' / 'processed' / 'embeddings_by_region'
DB_DIR = ROOT / '_db' / 'chroma_personal'
COLLECTION = 'personal_policy_chunks'
OUT_DIR = ROOT / 'results' / 'rag_evaluation' / 'bertopic_3차_prompt'
OPT_DIR = ROOT / 'results' / 'rag_evaluation' / 'bertopic_2차_optimized'
MODEL_NAME = 'dragonkue/snowflake-arctic-embed-l-v2.0-ko'
MODELS = ['exaone3.5:2.4b', 'qwen3:8b', 'gemma3:4b']
KEYWORDS = ['지역소멸', '청년 일자리', '저출생', '고령화', '기후위기']
SAMPLE_N = 10000
TOP_K = 15
RANDOM_SEED = 42
NUM_PREDICT = 320
OLLAMA_URL = 'http://127.0.0.1:11434/api/generate'

RESULT_CSV = OUT_DIR / '모델결과_누적.csv'
EVIDENCE_CSV = OUT_DIR / '근거회의록_누적.csv'
RESULT_XLSX = OUT_DIR / '개인_LLM_BERTopic_프롬프트평가누적.xlsx'
TOPIC_CSV = OUT_DIR / '토픽요약.csv'

RESULT_FIELDS = [
    '실험','키워드','모델','상태(200자)','등장시점(검색표본)',
    '타 지방의회 논의순서(검색표본)','향후 예측(앞 200자 표시)',
    '원래 예측 전체','실제 글자 수','200자 준수','근거 회의록(텍스트 요약)',
    '근거 수','질문 임베딩(초)','DB 검색(초)','LLM(초)','전체(초)',
    '모델 로드(초)','출력 토큰','토큰/초','오류','실행시각',
    '검색근거검증','실험ID','BERTopic 관련 토픽','토픽 선택(초)',
    '표본 수','토픽 모델링(초)'
]
EVIDENCE_FIELDS = ['실험ID','이슈','모델','근거 번호','일자','의회',
                   '원본 회의록명','안건','발언자','모델 상태','근거 원문(앞 220자)',
                   '저장된 source_file','row_id','chunk_id','검색 거리','검증 상태']

def timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def clock(sec):
    sec = float(sec)
    return f'{sec:.2f}초 / {sec/60:.2f}분 / {sec/3600:.2f}시간'

def append_csv(path, fields, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = (not path.exists()) or path.stat().st_size == 0
    with path.open('a', newline='', encoding='utf-8-sig' if new else 'utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if new: w.writeheader()
        w.writerows(rows)
        f.flush()

def norm_date(v):
    s = re.sub(r'\D', '', str(v or ''))
    return f'{s[:4]}-{s[4:6]}-{s[6:8]}' if len(s) >= 8 else ''

def collect_evidence(raw):
    result = []
    ids = raw.get('ids', [[]])[0]
    docs = raw.get('documents', [[]])[0]
    mets = raw.get('metadatas', [[]])[0]
    distances = raw.get('distances', [[]])[0]
    for i, uid in enumerate(ids):
        m = mets[i] or {}
        result.append(dict(id=uid, date=norm_date(m.get('date')),
            assembly=str(m.get('assembly') or ''), meeting=str(m.get('meeting_title') or ''),
            agenda=str(m.get('agenda_title') or ''), speaker=str(m.get('speaker_name') or ''),
            source_file=str(m.get('source_file') or ''), row_id=str(m.get('record_index') or ''),
            text=docs[i] or '', distance=distances[i] if i < len(distances) else ''))
    return result

def chronology(evidence):
    rows = sorted([x for x in evidence if x['date'] and x['assembly']],
                  key=lambda x:(x['date'], x['assembly']))
    if not rows: return '검색표본에서 확인 불가', '검색표본에서 확인 불가'
    earliest = f"{rows[0]['date']} / {rows[0]['assembly']}"
    first_by_assembly = {}
    for r in rows:
        first_by_assembly.setdefault(r['assembly'], r['date'])
    flow = ' → '.join(f'{d} {assembly}' for assembly,d in sorted(
        first_by_assembly.items(), key=lambda a:(a[1],a[0])))
    return earliest, flow

def load_saved_topics():
    """최적화 때 저장한 동일한 1만 개 벡터/토픽 번호/키워드만 재사용. 학습하지 않음."""
    import ast
    t0 = time.perf_counter()
    vecfile = OPT_DIR / 'cache' / 'sample_vectors.npy'
    labelfile = OPT_DIR / '토픽라벨.npy'
    topicfile = OPT_DIR / '토픽요약_개선.csv'
    assignment = OPT_DIR / '표본_토픽할당.csv'
    manifestfile = OPT_DIR / 'cache' / 'sample_manifest.json'
    old_sample = ROOT / 'results' / 'rag_evaluation' / 'bertopic_2차_kiwi' / '표본_청크ID.csv'
    for f in (vecfile,labelfile,topicfile,assignment,manifestfile,old_sample):
        if not f.exists(): raise FileNotFoundError(f'최적화 결과를 찾을 수 없습니다: {f}')
    vectors=np.load(vecfile,mmap_mode='r')
    labels=np.load(labelfile)
    df=pd.read_csv(topicfile,encoding='utf-8-sig')
    assigned=pd.read_csv(assignment,dtype=str,encoding='utf-8-sig')
    sample=pd.read_csv(old_sample,dtype=str,encoding='utf-8-sig')
    manifest=json.loads(manifestfile.read_text(encoding='utf-8'))
    import hashlib
    signature=hashlib.sha256('\n'.join(sample['chunk_id'].astype(str)).encode('utf-8')).hexdigest()[:20]
    if signature != manifest.get('signature'):
        raise ValueError('저장된 벡터와 최초 표본 ID의 서명이 다릅니다. 실행 중단.')
    if not (len(vectors)==len(labels)==len(assigned)==len(sample)==10000):
        raise ValueError('표본·벡터·토픽 개수가 일치하지 않습니다.')
    if (assigned['chunk_id'].astype(str).to_numpy()!=sample['chunk_id'].astype(str).to_numpy()).any():
        raise ValueError('표본 청크 ID 순서가 일치하지 않습니다.')
    if not np.array_equal(pd.to_numeric(assigned['토픽ID']).to_numpy(), labels):
        raise ValueError('토픽 할당 정보가 토픽라벨.npy와 일치하지 않습니다.')
    if vectors.shape[1]!=1024: raise ValueError('기존 임베딩 1024차원 불일치')
    class TopicLookup:
        def __init__(self, table):
            self.words={}
            for _,r in table.iterrows():
                tid=int(r['Topic'])
                if tid==-1: continue
                val=r.get('Representation','[]')
                try: wordlist=ast.literal_eval(val) if isinstance(val,str) else val
                except (ValueError,SyntaxError): wordlist=[]
                self.words[tid]=[(str(word),1.0) for word in wordlist]
        def get_topic(self, tid): return self.words.get(int(tid),[])
    print(f'[저장 토픽 로드] 표본 {len(labels):,}개 / 토픽 {len(df[df.Topic!=-1])}개 / {clock(time.perf_counter()-t0)}',flush=True)
    return vectors, labels, TopicLookup(df), time.perf_counter()-t0

def choose_topics(keyword, query_vec, labels, vectors, model):
    t0 = time.perf_counter()
    # query vector와 군집 중심의 코사인 유사도로 상위 2개 토픽 선택
    clusters = sorted(set(labels)-{-1})
    if not clusters: return [], time.perf_counter()-t0
    norm_vec = normalize(vectors)
    centers = []
    for tid in clusters:
        center = norm_vec[np.asarray(labels)==tid].mean(axis=0)
        center = center / max(np.linalg.norm(center), 1e-9)
        centers.append(center)
    scores = np.stack(centers) @ normalize(query_vec.reshape(1,-1))[0]
    best = np.argsort(scores)[::-1][:2]
    chosen=[]
    for i in best:
        tid=clusters[int(i)]
        labels_str=', '.join(word for word,_ in (model.get_topic(tid) or [])[:8])
        chosen.append({'topic_id':tid,'score':round(float(scores[i]),4),'keywords':labels_str})
    return chosen, time.perf_counter()-t0

def prompt_for(keyword, chosen, evidence):
    """2차와 동일한 토픽/원문 맥락을 사용하고 지시문만 개선한다.

    유의: 200자 제한은 '향후 예측' 응답만 적용. 날짜/의회/근거란은
    Chroma 메타데이터로 별도 산출하며 LLM에 추측하게 하지 않는다.
    """
    topic_ctx = '\n'.join(f"토픽 {t['topic_id']}: {t['keywords']} (유사도 {t['score']})" for t in chosen) or '관련 토픽 없음'
    context = '\n\n'.join(
        f"[근거 {i}] 날짜={e['date'] or '미상'}, 의회={e['assembly'] or '미상'}, "
        f"회의록={e['meeting'] or '미상'}, 안건={e['agenda'] or '미상'}\n원문:{e['text'][:950]}"
        for i,e in enumerate(evidence,1))
    return (
        "당신은 지방의회 회의록의 근거를 검증하며 정책 전망을 작성하는 분석가입니다.\n"
        "목표: 아래 검색 결과에서 실제 확인할 수 있는 논의 내용을 바탕으로 '향후 예측'만 작성합니다.\n"
        "반드시 지킬 규칙:\n"
        "1) 한국어 서술형 문장 1~2개, 전체 200자 이내(공백 포함)로 답하세요. 가능하면 120~170자.\n"
        "2) 과거 회의록 요약으로 끝내지 말고 향후 논의될 수 있는 정책 방향과 그 이유를 설명하세요.\n"
        "3) 전망은 가능성으로 표현하세요(예: ~가 논의될 수 있다). 확정된 계획처럼 단정하지 마세요.\n"
        "4) 제공한 원문에 없는 정책명·수치·금액·날짜·기관명·인과관계는 만들어내지 마세요.\n"
        "5) BERTopic 토픽 키워드는 해석 보조자료일 뿐 실제 회의록의 증거가 아닙니다. 원문과 충돌하면 원문을 우선합니다.\n"
        "6) 검색된 지역들의 날짜 순서만으로 정책이 타 지역에 확산했다고 추정하지 마세요.\n"
        "7) 근거가 부족하면 '검색 근거만으로 향후 방향을 단정하기 어렵다'고 명시하세요.\n"
        "8) 제목, 인사말, 번호, 목록, JSON, 설명문, 근거 번호 나열 없이 예측 본문만 출력하세요.\n"
        "출력 직전 200자 이내인지 점검하고 초과하면 스스로 줄이세요.\n\n"
        f"이슈: {keyword}\n[BERTopic 표본 결과]\n{topic_ctx}\n\n"
        f"[Chroma 검색 근거]\n{context}\n\n"
        "향후 예측(200자 이내):"
    )

def ollama(model, prompt):
    started = time.perf_counter()
    res = requests.post(OLLAMA_URL, json={
        'model':model, 'prompt':prompt, 'stream':False, 'think':False,
        'options':{'temperature':0,'num_predict':NUM_PREDICT}
    }, timeout=600)
    res.raise_for_status()
    obj=res.json()
    return ((obj.get('response') or '').strip(), time.perf_counter()-started,
            obj.get('load_duration',0)/1e9, obj.get('eval_count',0), obj.get('eval_duration',0)/1e9)

def excel_export():
    if not RESULT_CSV.exists(): return
    df=pd.read_csv(RESULT_CSV, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    ev=(pd.read_csv(EVIDENCE_CSV, dtype=str, keep_default_na=False, encoding='utf-8-sig')
        if EVIDENCE_CSV.exists() else pd.DataFrame(columns=EVIDENCE_FIELDS))
    rank=df[['키워드','모델','상태(200자)','LLM(초)','200자 준수','BERTopic 관련 토픽']].copy()
    rank.columns=['이슈키워드','모델','상태','LLM 초','200자 준수','관련 토픽']
    rank['예측 타당성(40) 입력']=''
    rank['근거 일치성(30) 입력']=''
    rank['형식(15) 자동']=rank['200자 준수'].eq('통과').astype(int)*15
    rank['속도(15) 자동']=0.0
    for key, group in rank.groupby('이슈키워드'):
        valid=group[(group['200자 준수']=='통과') & (pd.to_numeric(group['LLM 초'],errors='coerce')>0)]
        if len(valid):
            best=pd.to_numeric(valid['LLM 초']).min()
            for idx in valid.index:
                rank.loc[idx,'속도(15) 자동']=round(best/float(rank.loc[idx,'LLM 초'])*15,2)
    rank['총점(100)']='미평가'; rank['키워드 내 순위']='미평가'
    rank['평가 비고']='원본 JSON 검증 및 수동 채점 필요'
    rules=pd.DataFrame([
        ['분석방식','3차: 동일한 저장 토픽 + Chroma RAG + 개선 프롬프트 + 로컬 LLM'],
        ['예측 타당성','수동 0~40점: 근거에 기반한 전망 및 불확실성 표현'],
        ['근거 일치성','수동 0~30점: 원본 JSON 대조 전에 확정 금지'],
        ['형식 준수','자동 0/15점: 생성 원문 200자 이하'],
        ['속도','자동 0~15점: 같은 키워드 내 200자 통과 모델 최단 LLM 시간 비율'],
        ['등장 시점','Chroma Top-15 검색표본 내 가장 이른 날짜; 전체 최초 등장 아님'],
        ['타 지방의회 흐름','검색표본 내 날짜 순서이며 실제 확산의 인과관계 아님'],
        ['BERTopic','연도·의회 균형 표본 1만 개, UMAP+HDBSCAN, Kiwi 사전 토큰화 c-TF-IDF (재학습 없음)'],
        ['비교 주의','2차 대비 프롬프트 지시문만 변경. 같은 검색·토픽·모델 옵션이지만 LLM 생성은 비결정적일 수 있으므로 결과 차이 해석에 주의.'],
        ['질문',' / '.join(KEYWORDS)],
    ],columns=['항목','설명 / 채점 원칙'])
    with pd.ExcelWriter(RESULT_XLSX, engine='xlsxwriter') as writer:
        df.to_excel(writer,sheet_name='모델결과',index=False)
        rules.to_excel(writer,sheet_name='평가기준',index=False)
        rank.to_excel(writer,sheet_name='키워드별순위',index=False)
        ev.to_excel(writer,sheet_name='근거회의록',index=False)
        for ws in writer.sheets.values():
            ws.freeze_panes(1,0)
            ws.set_column(0,3,22)
            ws.set_column(4,22,28)
    print('Excel 저장:',RESULT_XLSX)

def main():
    start=time.perf_counter()
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    print(f'\n=== 3차 개선 프롬프트 + BERTopic + LLM 시작: {timestamp()} ===')
    print('개인 DB:', DB_DIR, '\n기존 1/2차 결과는 수정하지 않음')
    try:
        t=time.perf_counter()
        vectors,topic_labels,topic_model,load_topics_sec=load_saved_topics()
        print('[1] 저장 토픽 로드:',clock(time.perf_counter()-t))
        topic_seconds=0.0
        print('[2] BERTopic 재학습 생략 (0초)')
        t=time.perf_counter()
        db=chromadb.PersistentClient(path=str(DB_DIR)).get_collection(COLLECTION)
        embedder=SentenceTransformer(MODEL_NAME)
        print('[3] DB 연결+질문 임베더 로드:',clock(time.perf_counter()-t),'| 청크:',db.count())
        run_id=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
        for q_idx,keyword in enumerate(KEYWORDS,1):
            t=time.perf_counter()
            qvec=embedder.encode([keyword],convert_to_numpy=True).astype(np.float32)
            embed_sec=time.perf_counter()-t
            t=time.perf_counter()
            raw=db.query(query_embeddings=qvec.tolist(),n_results=TOP_K,
                include=['documents','metadatas','distances'])
            search_sec=time.perf_counter()-t
            evidence=collect_evidence(raw)
            first,flow=chronology(evidence)
            chosen, select_sec=choose_topics(keyword,qvec[0],topic_labels,vectors,topic_model)
            related=' | '.join(f"{c['topic_id']}:{c['keywords']}" for c in chosen)
            prompt=prompt_for(keyword,chosen,evidence)

            print("\n========== 실제 LLM 프롬프트 확인 ==========")
            print("프롬프트 길이:", len(prompt))
            print(
                "개선 프롬프트 적용:",
                "출력 직전 200자 이내인지 점검" in prompt
            )
            print("프롬프트 앞부분:")
            print(prompt[:1000])
            print("==========================================\n")
            
            print(f'\n[{q_idx}/5] {keyword} 임베딩 {embed_sec:.3f}초 / 검색 {search_sec:.3f}초 / 토픽선택 {select_sec:.3f}초',flush=True)
            summary=' | '.join(f"{e['date']} {e['assembly']} {e['meeting']} ({e['id']})" for e in evidence[:5])
            for m_idx,model in enumerate(MODELS,1):
                tick=time.perf_counter()
                text=''; llm_sec=load_sec=0.; ntokens=0; rate=0.; error=''
                try:
                    text,llm_sec,load_sec,ntokens,eval_sec=ollama(model,prompt)
                    rate=ntokens/eval_sec if eval_sec else 0.
                    status='OK' if text and len(text)<=200 else ('OVER_200' if text else 'EMPTY')
                except Exception as exc:
                    status='ERROR'; error=str(exc)[:500]
                total_sec=embed_sec+search_sec+select_sec+(time.perf_counter()-tick)
                rid=f'{run_id}_{q_idx:02d}_{m_idx:02d}'
                row={
                    '실험':'3차_BERTopic_프롬프트개선_LLM_개인','키워드':keyword,'모델':model,
                    '상태(200자)':status,'등장시점(검색표본)':first,
                    '타 지방의회 논의순서(검색표본)':flow,
                    '향후 예측(앞 200자 표시)':text[:200], '원래 예측 전체':text,
                    '실제 글자 수':len(text), '200자 준수':'통과' if status=='OK' else '미준수/오류',
                    '근거 회의록(텍스트 요약)':summary,'근거 수':len(evidence),
                    '질문 임베딩(초)':round(embed_sec,3),'DB 검색(초)':round(search_sec,3),
                    'LLM(초)':round(llm_sec,3),'전체(초)':round(total_sec,3),
                    '모델 로드(초)':round(load_sec,3),'출력 토큰':ntokens,
                    '토큰/초':round(rate,2),'오류':error,'실행시각':timestamp(),
                    '검색근거검증':'Chroma 근거 / 원본 JSON 미검증','실험ID':rid,
                    'BERTopic 관련 토픽':related,'토픽 선택(초)':round(select_sec,3),
                    '표본 수':len(vectors),'토픽 모델링(초)':round(topic_seconds,2),
                }
                append_csv(RESULT_CSV,RESULT_FIELDS,[row])
                ev_rows=[{'실험ID':rid,'이슈':keyword,'모델':model,'근거 번호':j,
                    '일자':e['date'],'의회':e['assembly'],'원본 회의록명':e['meeting'],
                    '안건':e['agenda'],'발언자':e['speaker'],'모델 상태':status,
                    '근거 원문(앞 220자)':e['text'][:220],
                    '저장된 source_file':e['source_file'],'row_id':e['row_id'],
                    'chunk_id':e['id'],'검색 거리':e['distance'],
                    '검증 상태':'원본 JSON 미검증'} for j,e in enumerate(evidence,1)]
                append_csv(EVIDENCE_CSV,EVIDENCE_FIELDS,ev_rows)
                print(f'  [{m_idx}/3] {model} {status} / {len(text)}자 / LLM {llm_sec:.2f}초 / 총 {total_sec:.2f}초',flush=True)
    finally:
        if RESULT_CSV.exists():
            try: excel_export()
            except Exception as exc: print('Excel 생성 오류:',repr(exc))
        print(f'=== 종료 {timestamp()} / 전체 {clock(time.perf_counter()-start)} ===')

if __name__=='__main__':
    main()

# -*- coding: utf-8 -*-
"""학원 PC 개인 프로젝트 / 2차 BERTopic + LLM 실험.
1) 기존 지역별 .npy + metadata.csv에서 연도·의회별 균형 표본 10,000개
2) BERTopic으로 토픽 구성 (임베딩 재계산 없음)
3) 5개 키워드별 관련 토픽 선택 + 기존 개인 Chroma Top-15 검색
4) Ollama 3개 모델 각각 답변 -> 실험별 CSV 즉시 누적 -> 종료시 xlsx 생성
주의: 추출된 토픽은 표본 기반, 등장 시점은 검색 표본 기반, 원본 JSON은 미검증.
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
OUT_DIR = ROOT / 'results' / 'rag_evaluation' / 'bertopic_2차'
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
RESULT_XLSX = OUT_DIR / '개인_LLM_BERTopic_평가누적.xlsx'
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

def balanced_sample():
    """두 번 순회, 모든 메타데이터를 메모리로 읽지 않고 strata별 비례 할당."""
    started = time.perf_counter()
    files = sorted(EMBED_DIR.rglob('batch_*.npy'))
    if not files: raise FileNotFoundError(f'.npy 배치 없음: {EMBED_DIR}')
    groups = defaultdict(list)
    for f in files:
        rel = f.relative_to(EMBED_DIR)
        if len(rel.parts) < 3: raise ValueError(f'예상 경로(연도/의회/배치)가 아님: {f}')
        metadata = f.with_name(f.stem + '_metadata.csv')
        if not metadata.exists(): raise FileNotFoundError(metadata)
        v = np.load(f, mmap_mode='r')

        with open(metadata, "r", encoding="utf-8-sig", newline="") as csvfile:
            n_csv = sum(1 for _ in csv.DictReader(csvfile))


        if len(v) != n_csv: raise ValueError(f'벡터/CSV 수 불일치: {f}')
        groups[(rel.parts[0],rel.parts[1])].append((f, metadata, len(v)))
    print(f'[표본] 지역-연도 그룹 {len(groups)}개, 배치 {len(files)}개')
    n_groups = len(groups)
    n_total = sum(n for group in groups.values() for _,_,n in group)
    # 균형: 그룹별 기본 균등 할당, 그룹 데이터 부족 시 가능한 만큼
    counts = {k:sum(n for _,_,n in fs) for k,fs in groups.items()}
    quotas = {k:min(counts[k], SAMPLE_N // n_groups) for k in groups}
    remaining = SAMPLE_N - sum(quotas.values())
    for k in sorted(groups):
        if remaining <= 0: break
        add = min(remaining, counts[k] - quotas[k])
        quotas[k] += add
        remaining -= add
    rng = np.random.default_rng(RANDOM_SEED)
    texts, vectors, sample_rows = [], [], []
    seen_ids = set()
    for gi, (group, batches) in enumerate(sorted(groups.items()), 1):
        q = quotas[group]
        idx = np.sort(rng.choice(counts[group], size=q, replace=False))
        base = 0
        for npy, meta, n in batches:
            chosen = idx[(idx >= base) & (idx < base+n)] - base
            base += n
            if not len(chosen): continue
            df = pd.read_csv(meta, dtype=str, keep_default_na=False, encoding='utf-8-sig')
            if 'text' not in df or 'chunk_id' not in df: raise ValueError(f'필수 컬럼 누락: {meta}')
            selected = df.iloc[chosen]
            arr = np.load(npy, mmap_mode='r')[chosen]
            for row, vec in zip(selected.to_dict('records'), arr):
                uid = row['chunk_id']
                if not uid or uid in seen_ids: continue
                seen_ids.add(uid)
                if not row['text'].strip(): continue
                texts.append(row['text'])
                vectors.append(np.asarray(vec, dtype=np.float32))
                sample_rows.append({'year':group[0], 'assembly':group[1], 'chunk_id':uid})
        if gi % 5 == 0 or gi == n_groups:
            print(f'  [{gi}/{n_groups}] 표본 누적 {len(texts):,}개', flush=True)
    v = np.stack(vectors)
    if v.shape[1] != 1024 or not np.isfinite(v).all(): raise ValueError(f'벡터 이상: {v.shape}')
    print(f'[표본 완료] {len(texts):,} / 전체 {n_total:,} 청크 | {clock(time.perf_counter()-started)}')
    pd.DataFrame(sample_rows).to_csv(OUT_DIR/'표본_청크ID.csv', index=False, encoding='utf-8-sig')
    return texts, v

def make_topics(texts, vectors):
    started = time.perf_counter()
    # 한국어 별도 형태소 분석 없이 문자 n-gram으로 대표 단어 후보 생성
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2,4), min_df=10, max_df=0.85, max_features=30000)
    reducer = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                   metric='cosine', random_state=RANDOM_SEED, low_memory=True)
    clusterer = HDBSCAN(min_cluster_size=35, min_samples=10,
                        metric='euclidean', prediction_data=False)
    topic_model = BERTopic(umap_model=reducer, hdbscan_model=clusterer,
                          vectorizer_model=vectorizer, calculate_probabilities=False,
                          verbose=True, low_memory=True)
    labels, _ = topic_model.fit_transform(texts, embeddings=vectors)
    info = topic_model.get_topic_info()
    info.to_csv(TOPIC_CSV, index=False, encoding='utf-8-sig')
    print(f'[BERTopic 완료] 토픽(노이즈 제외) {sum(info.Topic != -1)}개 | {clock(time.perf_counter()-started)}')
    print('※ 한글 문자 n-gram 토픽명은 중간 결과이며 품질 점검이 필요합니다.')
    return topic_model, labels, time.perf_counter()-started

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
    topic_ctx = '\n'.join(f"토픽 {t['topic_id']}: {t['keywords']} (유사도 {t['score']})" for t in chosen) or '관련 토픽 없음'
    context = '\n\n'.join(
        f"[근거 {i}] 날짜={e['date'] or '미상'}, 의회={e['assembly'] or '미상'}, "
        f"회의록={e['meeting'] or '미상'}, 안건={e['agenda'] or '미상'}\n원문:{e['text'][:950]}"
        for i,e in enumerate(evidence,1))
    return ("당신은 지방의회 정책 분석가입니다. 아래 검색 근거와 BERTopic 표본 토픽을 참고하여 "
            "향후 논의 방향을 200자 이내 한국어 서술문으로 작성하세요. "
            "토픽 키워드는 해석 보조자료일 뿐 검증된 사실이 아닙니다. "
            "타 지역으로 정책이 실제 전파됐다고 단정하지 마세요. "
            "근거에 없는 세부 정책·수치·일정은 쓰지 말고 불확실성을 표현하세요. "
            "제목, JSON, 서론, 목록 없이 전망 문장만 출력하세요.\n\n"
            f"이슈: {keyword}\n[BERTopic 표본 결과]\n{topic_ctx}\n\n"
            f"[Chroma 검색 근거]\n{context}\n\n향후 예측(200자 이내):")

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
        ['분석방식','2차: 기존 개인 임베딩 + BERTopic(균형 표본) + Chroma RAG + 로컬 LLM'],
        ['예측 타당성','수동 0~40점: 근거에 기반한 전망 및 불확실성 표현'],
        ['근거 일치성','수동 0~30점: 원본 JSON 대조 전에 확정 금지'],
        ['형식 준수','자동 0/15점: 생성 원문 200자 이하'],
        ['속도','자동 0~15점: 같은 키워드 내 200자 통과 모델 최단 LLM 시간 비율'],
        ['등장 시점','Chroma Top-15 검색표본 내 가장 이른 날짜; 전체 최초 등장 아님'],
        ['타 지방의회 흐름','검색표본 내 날짜 순서이며 실제 확산의 인과관계 아님'],
        ['BERTopic','연도·의회별 균형 표본 1만 개, UMAP+HDBSCAN, 문자 n-gram c-TF-IDF'],
        ['비교 주의','1차와 프롬프트가 같지 않음. 2차는 토픽 추가 효과+프롬프트 차이가 혼합됨'],
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
    print(f'\n=== 2차 BERTopic + LLM 시작: {timestamp()} ===')
    print('개인 DB:', DB_DIR, '\n기존 1차 결과는 수정하지 않음')
    try:
        t=time.perf_counter()
        texts,vectors=balanced_sample()
        print('[1] 균형 표본 준비:',clock(time.perf_counter()-t))
        t=time.perf_counter()
        topic_model, topic_labels, topic_seconds=make_topics(texts,vectors)
        print('[2] 군집/토픽 모델링:',clock(time.perf_counter()-t))
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
                    '실험':'2차_BERTopic_LLM_개인','키워드':keyword,'모델':model,
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
                    '표본 수':len(texts),'토픽 모델링(초)':round(topic_seconds,2),
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

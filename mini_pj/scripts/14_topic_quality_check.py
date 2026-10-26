# -*- coding: utf-8 -*-
"""개인 프로젝트 전체 토픽 80개 품질 점검 (모델 재학습 없음).

입력:
  C:\\mini_pj\\results\\full_topic_modeling\\전체_토픽요약.csv
  C:\\mini_pj\\results\\full_topic_modeling\\최종_토픽분포.csv
  C:\\mini_pj\\results\\full_topic_modeling\\웹_지역별_토픽집계.csv
  C:\\mini_pj\\results\\full_topic_modeling\\웹_월별_토픽집계.csv
  C:\\mini_pj\\results\\full_topic_modeling\\assignments\\*_토픽할당.csv
출력: full_topic_modeling\\quality_check\\ (새 파일만 생성)

사용: python 14_topic_quality_check.py
      python 14_topic_quality_check.py --top-n 3
      python 14_topic_quality_check.py --skip-full-text  (원문을 읽지 않고 preview만 이용)

주의: 이 스크립트의 위험 플래그는 규칙 기반 '검토 후보'일 뿐, 검증된
      품질 점수나 인명 판정, 정책 확산·태동의 증거가 아닙니다.
"""
from __future__ import annotations
import argparse
import ast
import heapq
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT = Path(r'C:\mini_pj\results\full_topic_modeling')
OUT = ROOT / 'quality_check'
EXPECTED = 684_311
TOPICS_EXPECTED = 80
STOPWORDS = {'사업','지원','계획','추진','관련','위원회','의원','위원','사항', '대한',
             '조례안','회의','의회','보고','질의','질문','답변','내용','말씀','부분',
             '안건','행정','제출','심사','운영','경우','방안','의견'}
PROCEDURAL = {'투표','표결','찬성','반대','의사일정','회의록','산회','개의','의결','재석','출석','이의','위원장'}


def stamp(): return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def seconds(t):
    s=time.perf_counter()-t
    return f'{s:,.2f}초 / {s/60:,.2f}분 / {s/3600:,.2f}시간'


def load_required(name, needed):
    p=ROOT/name
    if not p.is_file(): raise FileNotFoundError(f'필수 파일 없음: {p}')
    df=pd.read_csv(p, encoding='utf-8-sig', low_memory=False)
    if not set(needed).issubset(df.columns):
        raise ValueError(f'{name}: 필요한 컬럼 {needed}, 실제 컬럼 {df.columns.tolist()}')
    print(f'[입력] {name}: {len(df):,}행', flush=True)
    return df


def parse_keywords(value):
    try:
        parsed=ast.literal_eval(str(value))
        if isinstance(parsed,(list,tuple)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (SyntaxError,ValueError,TypeError):
        pass
    return [x.strip() for x in str(value).split(',') if x.strip()]


def risk_signals(words):
    single=[x.lower() for x in words if ' ' not in x]
    not_specific=sum(w in STOPWORDS for w in single)
    proced=sum(w in PROCEDURAL for w in single)
    repeated=[w for w in single if len(w)>=2 and sum(w in phrase.lower().split() for phrase in words)>=3]
    # 사람 이름처럼 보이는 형태에 대한 자동 판정은 정확하지 않아 사용하지 않는다.
    flags=[]
    if proced>=2: flags.append('의사진행 중심 가능성')
    if not_specific>=3: flags.append('일반·행정 용어 비중 높음')
    if repeated: flags.append('특정 단어 반복: '+', '.join(repeated[:3]))
    if len(words)<4: flags.append('대표 키워드 부족')
    return flags


def score_preview(preview, words):
    # 대표발언 검색 목적의 키워드 중복 점수; 토픽 품질·근거 일치 점수가 아님.
    candidates=[w for w in words[:8] if len(w)>=2 and w not in STOPWORDS]
    return sum(1 for w in candidates if w in preview)


def select_candidates(files, keywords, top_n):
    """전체 할당을 다시 학습하지 않고 훑어 대표발언 후보를 선택한다.

    전체 파일을 순서대로 보고 키워드 일치도가 높은 발언을 모은다.
    순서와 중복을 완화하기 위해 토픽별 다양한 의회를 우선 선택한다.
    """
    candidates=defaultdict(list)
    max_keep=max(24, top_n*9)
    total=0
    for i,p in enumerate(files,1):
        t=time.perf_counter()
        for df in pd.read_csv(p, chunksize=5000, dtype=str,
                              keep_default_na=False, encoding='utf-8-sig', low_memory=False):
            needed=['topic_id','chunk_id','assembly','date','text_preview','meta_csv','row_index']
            missing=[c for c in needed if c not in df.columns]
            if missing: raise ValueError(f'{p.name} 누락 컬럼: {missing}')
            total+=len(df)
            for r in df.itertuples(index=False):
                # itertuples에서 기본 영문 컬럼명에 접근
                tid=int(r.topic_id)
                text=r.text_preview
                value=score_preview(text, keywords.get(tid,[]))
                # 키워드 빈약한 토픽도 예시 확보: 최대 개수 이상의 낮은 점수는 건너뜀
                arr=candidates[tid]
                key=(value, min(len(text),180), r.chunk_id)
                entry=(key, {'topic_id':tid,'chunk_id':r.chunk_id,'assembly':r.assembly,
                             'date':r.date,'preview':text,'meta_csv':r.meta_csv,
                             'row_index':int(r.row_index),'keyword_hits':value})
                # 유지목록이 작아 list 정렬로 안전하고 단순하게 처리
                if len(arr)<max_keep:
                    arr.append(entry)
                elif key>arr[0][0]:
                    arr[0]=entry
                if len(arr)>=max_keep:
                    arr.sort(key=lambda x:x[0])
        if i % 20==0 or i==len(files):
            print(f'[대표발언 후보 {i}/{len(files)}] {total:,}건 검사 / 마지막 파일 {seconds(t)}',flush=True)
    result=[]
    for tid in sorted(keywords):
        ranked=sorted(candidates.get(tid,[]),key=lambda x:x[0],reverse=True)
        chosen=[]; used_assembly=set(); used_id=set()
        for _,row in ranked:
            if row['assembly'] not in used_assembly and row['chunk_id'] not in used_id:
                chosen.append(row);used_assembly.add(row['assembly']);used_id.add(row['chunk_id'])
                if len(chosen)>=top_n:break
        if len(chosen)<top_n:
            for _,row in ranked:
                if row['chunk_id'] not in used_id:
                    chosen.append(row);used_id.add(row['chunk_id'])
                    if len(chosen)>=top_n:break
        for rank,row in enumerate(chosen,1):
            result.append({**row,'representative_rank':rank})
    if total!=EXPECTED:
        raise ValueError(f'전체 할당 데이터 개수 불일치: {total:,} / 예상 {EXPECTED:,}')
    return pd.DataFrame(result),total


def enrich_full_text(reps):
    """원본 메타 CSV의 row_index를 이용해 선택된 대표 발언 원문만 복원한다."""
    reps['text_excerpt']=reps['preview']
    reps['source_verified']='미확인(원문 조회 전)'
    if reps.empty:return reps
    grouped=defaultdict(list)
    for idx,row in reps.iterrows():
        grouped[str(row['meta_csv'])].append((idx,int(row['row_index']),str(row['chunk_id'])))
    print(f'[원문 확인] 필요한 메타 CSV {len(grouped)}개',flush=True)
    for i,(filename,targets) in enumerate(grouped.items(),1):
        p=Path(filename)
        if not p.exists():
            for idx,_,_ in targets:reps.at[idx,'source_verified']='원본 CSV 파일 없음'
            continue
        byrow=defaultdict(list)
        for idx,j,cid in targets:byrow[j].append((idx,cid))
        offset=0
        try:
            for part in pd.read_csv(p, chunksize=5000, dtype=str, keep_default_na=False,
                                    encoding='utf-8-sig',usecols=['chunk_id','text']):
                for j,requests in byrow.items():
                    if offset<=j<offset+len(part):
                        row=part.iloc[j-offset]
                        for idx,expected_id in requests:
                            if str(row['chunk_id'])==expected_id:
                                reps.at[idx,'text_excerpt']=str(row['text'])[:1000]
                                reps.at[idx,'source_verified']='ID 일치'
                            else:
                                reps.at[idx,'source_verified']='ID 불일치'
                offset+=len(part)
        except Exception as e:
            for idx,_,_ in targets:
                if reps.at[idx,'source_verified']=='미확인(원문 조회 전)':
                    reps.at[idx,'source_verified']=f'조회 오류: {str(e)[:100]}'
        if i%20==0 or i==len(grouped):
            print(f'[원문 확인 {i}/{len(grouped)}] 완료',flush=True)
    return reps


def main():
    parser=argparse.ArgumentParser(description='전체 토픽 품질점검; 기존 모델·CSV 보존')
    parser.add_argument('--top-n',type=int,default=3,help='토픽별 대표 발언 건수 (기본 3)')
    parser.add_argument('--skip-full-text',action='store_true',help='원본 메타 CSV 조회 생략')
    args=parser.parse_args()
    if not 1<=args.top_n<=10:parser.error('--top-n은 1~10')
    t_all=time.perf_counter()
    print(f'=== 전체 토픽 품질 검사 시작: {stamp()} ===',flush=True)
    topic=load_required('전체_토픽요약.csv',['Topic','Count','Representation'])
    final=load_required('최종_토픽분포.csv',['topic_id','count'])
    region=load_required('웹_지역별_토픽집계.csv',['assembly','topic_id','count'])
    monthly=load_required('웹_월별_토픽집계.csv',['month','topic_id','count'])
    if len(topic)!=TOPICS_EXPECTED or len(final)!=TOPICS_EXPECTED:
        print(f'[주의] 설정 80토픽과 실제 토픽 수 다름: 요약 {len(topic)}, 최종 {len(final)}',flush=True)
    if topic['Topic'].duplicated().any() or final['topic_id'].duplicated().any():
        raise ValueError('토픽 ID 중복')
    if int(final['count'].sum())!=EXPECTED:raise ValueError('최종_토픽분포 총합 불일치')
    if int(region['count'].sum())!=EXPECTED:raise ValueError('지역별 집계 총합 불일치')
    if int(monthly['count'].sum())!=EXPECTED:raise ValueError('월별 집계 총합 불일치: 월별 복구 파일 확인')
    if region.duplicated(['assembly','topic_id']).any():raise ValueError('지역별 토픽 조합 중복')
    if monthly.duplicated(['month','topic_id']).any():raise ValueError('월별 토픽 조합 중복')
    # 각 토픽별 집계끼리 서로 맞는지 추가 검증
    refs=final.set_index('topic_id')['count'].astype(int)
    reg=region.groupby('topic_id')['count'].sum().astype(int)
    mon=monthly.groupby('topic_id')['count'].sum().astype(int)
    if not refs.equals(reg.reindex(refs.index)) or not refs.equals(mon.reindex(refs.index)):
        raise ValueError('토픽별 최종 분포/지역/월 집계 불일치')
    print(f'[집계 검증] {EXPECTED:,}건 / {len(final)}개 토픽 / '
          f'{region.assembly.nunique()}개 지역 / {monthly.month.nunique()}개 월',flush=True)

    topic=topic.rename(columns={'Topic':'topic_id','Count':'online_count','Name':'online_name'})
    topic['topic_id']=topic['topic_id'].astype(int)
    topic['keywords_list']=topic['Representation'].map(parse_keywords)
    words=dict(zip(topic.topic_id,topic.keywords_list))
    merged=topic.merge(final,on='topic_id',how='outer',validate='one_to_one')
    if merged['Representation'].isna().any():raise ValueError('최종 할당 토픽 중 온라인 토픽 요약 누락')
    rows=[]
    for _,r in merged.iterrows():
        tid=int(r['topic_id']); k=words[tid]; flags=risk_signals(k)
        row=region.loc[region.topic_id==tid].sort_values('count',ascending=False)
        topregs=' / '.join(f'{a}({int(n):,})' for a,n in row[['assembly','count']].head(3).itertuples(index=False,name=None))
        m=monthly.loc[monthly.topic_id==tid].sort_values('month')
        rows.append({'topic_id':tid,'최종_할당건수':int(r['count']),
                     '전체대비비율(%)':round(int(r['count'])/EXPECTED*100,3),
                     '온라인학습중_건수':int(r['online_count']),
                     '온라인-최종건수차이':int(r['count'])-int(r['online_count']),
                     '토픽명_임시':r['online_name'],'키워드':', '.join(k),
                     '대표키워드_개수':len(k),'자동검토표시':' / '.join(flags),
                     '검토필요': '예' if flags else '아니오',
                     '논의지역수':len(row),'논의월수':len(m),
                     '관측시작월':str(m.month.iloc[0]) if len(m) else '',
                     '관측종료월':str(m.month.iloc[-1]) if len(m) else '',
                     '상위3지역':topregs,
                     '사람검토_토픽명':'','사람검토_판정':'',
                     '사람검토_메모':''})
    out=pd.DataFrame(rows).sort_values('topic_id').reset_index(drop=True)
    files=sorted((ROOT/'assignments').glob('*_토픽할당.csv'))
    if not files:raise FileNotFoundError(f'할당 CSV 없음: {ROOT/"assignments"}')
    print(f'[할당 파일] {len(files)}개',flush=True)
    t_rep=time.perf_counter()
    reps,processed=select_candidates(files,words,args.top_n)
    print(f'[대표 후보 추출 완료] {len(reps):,}개 / {processed:,}건 검사 / {seconds(t_rep)}',flush=True)
    if not args.skip_full_text:
        reps=enrich_full_text(reps)
    else:
        reps['text_excerpt']=reps['preview'];reps['source_verified']='미확인(--skip-full-text)'
    reps=reps.rename(columns={'keyword_hits':'키워드_일치개수(참고)',
                              'representative_rank':'대표순위','source_verified':'원문_ID검증',
                              'text_excerpt':'원문발췌_1000자'})
    keep=['topic_id','대표순위','chunk_id','assembly','date',
          '키워드_일치개수(참고)','원문_ID검증','원문발췌_1000자',
          'meta_csv','row_index']
    OUT.mkdir(parents=True,exist_ok=True)
    out.to_csv(OUT/'토픽별_품질검토.csv',index=False,encoding='utf-8-sig')
    out[out['검토필요']=='예'].to_csv(OUT/'자동_검토후보.csv',index=False,encoding='utf-8-sig')
    reps[keep].to_csv(OUT/'토픽별_대표발언.csv',index=False,encoding='utf-8-sig')
    counts={'총할당건수':EXPECTED,'토픽수':len(out),'지역수':int(region.assembly.nunique()),
            '월수':int(monthly.month.nunique()),'대표발언수':len(reps),
            '자동검토후보수':int((out['검토필요']=='예').sum()),
            '원문ID일치건수':int((reps['원문_ID검증']=='ID 일치').sum()),
            '원문ID불일치건수':int((reps['원문_ID검증']=='ID 불일치').sum()),
            '시작시각':stamp(),'전체소요초':round(time.perf_counter()-t_all,2),
            '설명':'자동검토후보는 휴리스틱 의심 표시이며 객관적 토픽 품질점수나 원문 근거 검증 판정이 아님'}
    (OUT/'품질검토_요약.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n=== 전체 품질검토 종료 ===')
    print(f'전체 건수 {EXPECTED:,} / 토픽 {len(out)} / 지역 {counts["지역수"]} / 월 {counts["월수"]}')
    print(f'자동 검토 후보 {counts["자동검토후보수"]}개 (사람 검토 필요)')
    print(f'대표발언 {len(reps)}개 / 원문 ID 일치 {counts["원문ID일치건수"]}개')
    print(f'총 소요 {seconds(t_all)}')
    print('결과 폴더:',OUT)
    print('기존 모델·Chroma·토픽 할당·웹 집계 파일 변경 없음')

if __name__=='__main__':
    main()

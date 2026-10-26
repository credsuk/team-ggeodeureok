# -*- coding: utf-8 -*-
"""25단계: 정책 관련성 자동 후보를 활용한 지역·월별 시계열 및 의정활동 *후보* 탐색.

필수 입력: <검색 폴더>/relevance_validation/전체_정책관련성_검토결과.csv
분모: <검색 폴더>/trend_analysis/지역월별_정책언급추이.csv,
      <검색 폴더>/trend_analysis/월별_정책언급추이.csv
선택 입력: <검색 폴더>/flow_validation/정책내용_고유후보_검토결과.csv
기존 데이터는 절대 변경하지 않음. 월별 통계에서 '자동후보'를 검증완료 건수로 혼동 금지.

실행: python 25_policy_timeline_analysis.py --input-dir "C:\\mini_pj\\results\\policy_issue_search\\20261025_000801_청년_일자리"
"""
from __future__ import annotations
import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path(r'C:\mini_pj\results\policy_issue_search\20261025_000801_청년_일자리')
DIRECT = '직접관련_후보'
INDIRECT = '간접관련_후보'
HOLD = '판단보류'
NO = '무관가능성_검토'

# 표결·공청회·심사 등은 우선 후보만 생성. 문맥과 실제 실시 여부는 별도 검증해야 함.
ACTIVITY_PATTERNS = {
    '공청회_언급': r'공청회|주민공청회',
    '토론회_언급': r'토론회|정책토론',
    '조례안_언급': r'조례안|조례\s*(?:제정|개정|심사)',
    '예산심의_언급': r'예산안|추경|추가경정예산|예산\s*심사',
    '의결표결_언급': r'가결|부결|표결|의결',
    '질의응답_언급': r'질의|답변|질문',
}


def log(msg):
    print(f'[{datetime.now():%H:%M:%S}] {msg}', flush=True)


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f'필수 입력 없음: {path}')
    return pd.read_csv(path, encoding='utf-8-sig', low_memory=False)


def write_csv(df, path):
    df.to_csv(path, index=False, encoding='utf-8-sig')
    log(f'저장 {path.name}: {len(df):,}행')


def normalize_dates(df):
    v = df['date'].astype('string').str.strip()
    ymd = v.str.replace(r'[^0-9]', '', regex=True)
    dates = pd.to_datetime(ymd, format='%Y%m%d', errors='coerce')
    if dates.isna().any():
        raise ValueError(f'date 파싱 실패: {int(dates.isna().sum()):,}행, 예시 {v[dates.isna()].head(3).tolist()}')
    df['date'] = dates.dt.strftime('%Y-%m-%d')
    df['month'] = dates.dt.strftime('%Y-%m')
    return df


def split_mechanisms(s):
    if pd.isna(s) or not str(s).strip():
        return []
    return [x.strip() for x in str(s).split('|') if x.strip()]


def snippet(text, pat, max_chars=100):
    t = str(text) if pd.notna(text) else ''
    match = re.search(pat, t, re.I)
    if not match:
        return t[:max_chars]
    start = max(0, match.start()-25)
    start = min(start, max(0, len(t)-max_chars))
    return t[start:start+max_chars]


def main():
    ap=argparse.ArgumentParser(description='25: 정책 시계열 및 의정활동 근거 후보 (수동 검토 없이)')
    ap.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    ap.add_argument('--output-dir', type=Path, default=None)
    ap.add_argument('--max-evidence', type=int, default=300, help='의정활동 근거 후보 최대 행수')
    args=ap.parse_args()
    start=time.perf_counter()
    input_dir=args.input_dir
    out=args.output_dir or (input_dir/'timeline_analysis')
    out.mkdir(parents=True,exist_ok=True)
    log(f'=== 25 정책 시계열 시작 === 입력: {input_dir}')

    t=time.perf_counter()
    df=read_csv(input_dir/'relevance_validation'/'전체_정책관련성_검토결과.csv')
    required={'chunk_id','date','assembly','source_id','meeting_title','text_preview','review_candidate_label','mechanisms_detected'}
    missing=required-set(df.columns)
    if missing:
        raise ValueError(f'24번 필수 컬럼 없음: {sorted(missing)}')
    if df['chunk_id'].duplicated().any():
        raise ValueError('중복 chunk_id 발견. 데이터 정렬/유일성 확인 후 실행하세요.')
    df=normalize_dates(df)
    for c in ['chunk_id','assembly','source_id','meeting_title','text_preview']:
        df[c]=df[c].fillna('').astype(str)
    df['is_auto_direct']=df['review_candidate_label'].eq(DIRECT)
    df['is_auto_indirect']=df['review_candidate_label'].eq(INDIRECT)
    df['is_lexical_match']=True
    log(f'[1/5 입력 검증] {len(df):,}건 / 직접후보 {df.is_auto_direct.sum():,}건 / {time.perf_counter()-t:.2f}초')

    t=time.perf_counter()
    # 원본 16번 전체 문서 수: 검색 문서 수(분자)가 아니라 반드시 해당 월의 전체 회의록 청크 수(분모).
    denom=read_csv(input_dir/'trend_analysis'/'지역월별_정책언급추이.csv')
    required_d={'assembly','month','all_chunks'}
    if required_d-set(denom.columns):
        raise ValueError('16번 지역월별 분모 파일에 컬럼 누락')
    denom=denom[['assembly','month','all_chunks']].copy()
    if denom.duplicated(['assembly','month']).any():
        raise ValueError('지역×월 분모 중복 발견')
    denom['all_chunks']=pd.to_numeric(denom['all_chunks'],errors='raise').astype(int)
    if (denom['all_chunks']<0).any(): raise ValueError('음수 분모 발견')
    all_m=read_csv(input_dir/'trend_analysis'/'월별_정책언급추이.csv')[['month','all_chunks']].copy()
    if all_m.duplicated('month').any(): raise ValueError('월별 분모 중복 발견')
    # 지역x월 집계: 없는 지역x월도 0으로 유지
    agg=df.groupby(['assembly','month'],as_index=False).agg(
        어휘일치_청크=('is_lexical_match','sum'),
        직접관련_자동후보_청크=('is_auto_direct','sum'),
        간접관련_자동후보_청크=('is_auto_indirect','sum'),
    )
    rm=denom.merge(agg,on=['assembly','month'],how='left',validate='one_to_one')
    for c in ['어휘일치_청크','직접관련_자동후보_청크','간접관련_자동후보_청크']:
        rm[c]=rm[c].fillna(0).astype(int)
        rm[c.replace('_청크','_비율_pct')]=np.where(rm.all_chunks>0,100*rm[c]/rm.all_chunks,0).round(4)
    if rm['어휘일치_청크'].sum()!=len(df):
        raise ValueError('지역×월 집계 합계 불일치: 전체 지역월 분모와 assembly/month 정합성 검사 필요')
    write_csv(rm.sort_values(['month','assembly']),out/'지역월별_정책관련성_추이.csv')
    mm=rm.groupby('month',as_index=False)[['어휘일치_청크','직접관련_자동후보_청크','간접관련_자동후보_청크']].sum()
    mm=all_m.merge(mm,on='month',how='left',validate='one_to_one').sort_values('month')
    for c in ['어휘일치_청크','직접관련_자동후보_청크','간접관련_자동후보_청크']:
        mm[c]=mm[c].fillna(0).astype(int)
        mm[c.replace('_청크','_비율_pct')]=np.where(mm.all_chunks>0,100*mm[c]/mm.all_chunks,np.nan).round(4)
    mm['전년동월_직접후보비율_pct']=mm['직접관련_자동후보_비율_pct'].shift(12)
    mm['전년동월_증감_pp']=(mm['직접관련_자동후보_비율_pct']-mm['전년동월_직접후보비율_pct']).round(4)
    write_csv(mm,out/'월별_정책관련성_추이.csv')
    # 회의록 단위 통계: source_id는 분석 편의상의 회의록 소스 식별자이며, 정책 채택/발언 수가 아님
    meeting=df[df.is_auto_direct].groupby(['assembly','source_id','date','meeting_title'],as_index=False).agg(
        직접관련_자동후보_청크=('chunk_id','size'),
        근거ID_수=('chunk_id','nunique'))
    write_csv(meeting.sort_values(['date','assembly']),out/'회의록단위_직접후보_통계.csv')
    log(f'[2/5 시계열·회의록 집계] 월 {len(mm)} / 지역월 {len(rm)} / 회의록 {len(meeting)} / {time.perf_counter()-t:.2f}초')

    t=time.perf_counter()
    # 최초 관측: 어휘일치와 자동직접후보 별도. 수동 검증 전이라는 라벨 보존.
    first_rows=[]
    for scope,sub in [('전체_어휘일치',df),('직접관련_자동후보',df[df.is_auto_direct])]:
        if sub.empty: continue
        s=sub.sort_values(['date','assembly','chunk_id']).drop_duplicates('assembly',keep='first')
        for r in s.itertuples(index=False):
            first_rows.append({'관측범위':scope,'assembly':r.assembly,'date':r.date,
                               'meeting_title':r.meeting_title,'source_id':r.source_id,'chunk_id':r.chunk_id,
                               'core_evidence_text_100':str(getattr(r,'core_evidence_text_100','') or '')[:100],
                               '검증상태':'자동후보_수동미검증'})
    ff=pd.DataFrame(first_rows).sort_values(['관측범위','date','assembly'])
    write_csv(ff,out/'지역별_최초관측_구분.csv')
    log(f'[3/5 최초 관측] {len(ff)}행 / {time.perf_counter()-t:.2f}초')

    t=time.perf_counter()
    # 의정활동: 단어 검출은 '공청회 실시' 증거 아님. 개최/의결 등 확정 라벨 금지.
    ev=[]
    direct=df[df.is_auto_direct].sort_values(['date','assembly','chunk_id'])
    for r in direct.itertuples(index=False):
        full=str(r.text_preview)
        title=str(r.meeting_title)
        for kind,pattern in ACTIVITY_PATTERNS.items():
            match=re.search(pattern,full,re.I) or re.search(pattern,title,re.I)
            if match:
                ev.append({'date':r.date,'assembly':r.assembly,'의정활동_언급후보':kind,
                    'meeting_title':title,'source_id':r.source_id,'chunk_id':r.chunk_id,
                    '근거_100자':snippet(full,pattern,100),
                    '검증상태':'표현_자동검출_행위실시미확인'})
    evdf=pd.DataFrame(ev,columns=['date','assembly','의정활동_언급후보','meeting_title','source_id','chunk_id','근거_100자','검증상태'])
    # 한 종류가 너무 많이 쏠리지 않도록 날짜/지역 기준으로 대표 후보 표본만 상한 적용
    evdf=evdf.drop_duplicates(['chunk_id','의정활동_언급후보']).sort_values(['date','assembly','chunk_id'])
    write_csv(evdf.head(max(0,args.max_evidence)),out/'의정활동_원문언급_검토후보.csv')
    # 같은 의회에서 같은 정책수단이 서로 다른 source_id에서 확인됨: 재논의 *후보*일 뿐
    mechanism_events=[]
    for r in direct.itertuples(index=False):
        for mechanism in split_mechanisms(r.mechanisms_detected):
            mechanism_events.append({'assembly':r.assembly,'date':r.date,'source_id':r.source_id,
                         'meeting_title':r.meeting_title,'chunk_id':r.chunk_id,'정책수단':mechanism,
                         '근거_100자':str(getattr(r,'core_evidence_text_100','') or '')[:100]})
    medf=pd.DataFrame(mechanism_events)
    repeats=[]
    if not medf.empty:
        medf=medf.sort_values(['assembly','정책수단','date','source_id','chunk_id'])
        unique_meetings=medf.drop_duplicates(['assembly','정책수단','source_id'])
        for (assembly,mechanism),group in unique_meetings.groupby(['assembly','정책수단'],sort=False):
            if group.source_id.nunique()<2: continue
            for prev,cur in zip(group.iloc[:-1].itertuples(index=False),group.iloc[1:].itertuples(index=False)):
                if cur.date<=prev.date: continue
                repeats.append({'assembly':assembly,'정책수단':mechanism,
                         '이전관측일':prev.date,'후속관측일':cur.date,
                         '이전_source_id':prev.source_id,'후속_source_id':cur.source_id,
                         '이전_chunk_id':prev.chunk_id,'후속_chunk_id':cur.chunk_id,
                         '이전_근거100자':prev.근거_100자,'후속_근거100자':cur.근거_100자,
                         '검증상태':'동일정책수단_반복관측_재논의미확정'})
    re_df=pd.DataFrame(repeats,columns=['assembly','정책수단','이전관측일','후속관측일','이전_source_id','후속_source_id','이전_chunk_id','후속_chunk_id','이전_근거100자','후속_근거100자','검증상태'])
    write_csv(re_df,out/'동일의회_재논의_검토후보.csv')
    log(f'[4/5 의정활동 단서·반복관측] 활동 {len(evdf)} / 재논의후보 {len(re_df)} / {time.perf_counter()-t:.2f}초')

    t=time.perf_counter()
    optional=input_dir/'flow_validation'/'정책내용_고유후보_검토결과.csv'
    if optional.exists():
        cross=read_csv(optional)
        cols=[c for c in ['candidate_id','earlier_date','earlier_assembly','later_date','later_assembly','specific_common_mechanisms','content_judgment','transfer_status','earlier_chunk_id','later_chunk_id'] if c in cross.columns]
        cross=cross[cols].copy()
        cross['검증상태']='자동후보_정책내용및전파수동미검증'
    else:
        cross=pd.DataFrame(columns=['candidate_id','earlier_date','earlier_assembly','later_date','later_assembly','specific_common_mechanisms','content_judgment','transfer_status','검증상태'])
    write_csv(cross,out/'지역간_후속논의_검토후보.csv')
    if int(mm['직접관련_자동후보_청크'].sum()) != int(df.is_auto_direct.sum()):
        raise RuntimeError('직접후보 시계열 합계 검증 실패')
    report={
        'started_at':datetime.now().isoformat(timespec='seconds'),
        'query':'청년 일자리 (입력 24번 검색 결과 기준)',
        'input_dir':str(input_dir),
        'full_lexical_matches':len(df),
        'auto_direct_candidates':int(df.is_auto_direct.sum()),
        'auto_indirect_candidates':int(df.is_auto_indirect.sum()),
        'monthly_points':len(mm), 'regional_month_points':len(rm),
        'direct_candidate_source_groups':len(meeting),
        'activity_expression_rows':len(evdf),
        'same_assembly_repeated_mechanism_candidates':len(re_df),
        'cross_region_unverified_pairs':len(cross),
        'manual_verified':0,
        'earliest_auto_direct':(df.loc[df.is_auto_direct,'date'].min() if df.is_auto_direct.any() else None),
        'limitations':[
            '시계열의 단위는 청크이며 회의/정책 채택 건수가 아님',
            '직접관련_자동후보는 수동 확인되지 않았으며 판단보류 청크를 누락하므로 실제 정책 언급량의 추정치가 아님',
            '월별 분모는 16번 전체 회의록 청크량이며 집계 변화에 회기·수집량 영향 존재',
            '의정활동 종류는 단어 언급 탐지일 뿐 공청회 실시·의결 완료 등을 확인하지 않음',
            '동일 정책 수단 반복 관측은 실제 같은 안건 재논의나 정책 전파를 입증하지 않음',
            '지역 간 후보는 이전 20번 결과를 가져온 것이며 실제 정책 전파가 아님',
            '최초 등장 시점은 보유한 2024~2025년 자료 내 자동 후보 관측일',
        ],
        'elapsed_seconds':round(time.perf_counter()-start,3),
        'finished_at':datetime.now().isoformat(timespec='seconds'),
    }
    (out/'시계열_분석요약.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8-sig')
    log(f'[5/5 요약 저장] {time.perf_counter()-t:.2f}초')
    log(f'=== 25 종료: {time.perf_counter()-start:.2f}초 / 결과: {out} ===')

if __name__=='__main__':
    main()

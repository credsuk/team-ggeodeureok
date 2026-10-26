# -*- coding: utf-8 -*-
"""개인 1/2/3차 RAG 평가 통합. LLM/BERTopic 실행 없이 기존 결과만 분석.
실행: python evaluation_01_compare.py
필요: pandas, XlsxWriter, openpyxl (xlsx 읽기 전용)
"""
from pathlib import Path
from datetime import datetime
import time
import pandas as pd

ROOT = Path(r'C:\mini_pj')
BASE = ROOT / 'results' / 'rag_evaluation'
OUT = BASE / 'evaluation_comparison'
KEYWORDS = ['지역소멸','청년 일자리','저출생','고령화','기후위기']
MODELS = ['exaone3.5:2.4b','qwen3:8b','gemma3:4b']
# 2번은 최적화 후 LLM 평가를 기준으로 한다. 첫 2차 실험은 포함하지 않는다.
SOURCES = [
    ('1차_기본RAG', BASE, '개인_LLM_RAG_평가누적.xlsx'),
    ('2차_BERTopic', BASE / 'bertopic_2차_optimized_llm', '개인_LLM_BERTopic_최적화평가누적.xlsx'),
    ('3차_프롬프트', BASE / 'bertopic_3차_prompt', '개인_LLM_BERTopic_프롬프트평가_중복정리.xlsx'),
]
RESULT_CSV_NAME = '모델결과_누적.csv'
EVIDENCE_CSV_NAME = '근거회의록_누적.csv'

def load_data(label, folder, workbook_name):
    # 3차의 중복 정리본을 최우선으로 사용한다.
    if label.startswith('3차'):
        candidates = [
            (folder/'모델결과_중복정리.csv',folder/'근거회의록_중복정리.csv'),
            (BASE/'모델결과_중복정리.csv',BASE/'근거회의록_중복정리.csv'),
        ]
        for result, ev in candidates:
            if result.exists() and ev.exists():
                print(f'[{label}] 정리 CSV 사용: {result}')
                return pd.read_csv(result,dtype=str,keep_default_na=False,encoding='utf-8-sig'),pd.read_csv(ev,dtype=str,keep_default_na=False,encoding='utf-8-sig')
    result, ev = folder / RESULT_CSV_NAME, folder / EVIDENCE_CSV_NAME
    if result.exists() and ev.exists():
        print(f'[{label}] CSV 사용: {result}')
        return pd.read_csv(result,dtype=str,keep_default_na=False,encoding='utf-8-sig'),pd.read_csv(ev,dtype=str,keep_default_na=False,encoding='utf-8-sig')
    xlsx = folder / workbook_name
    if not xlsx.exists():
        # 다운로드한 통합 작업용 별도 파일이 있는 경우
        xlsx = ROOT / workbook_name
    if not xlsx.exists():
        raise FileNotFoundError(f'{label}: CSV 또는 Excel을 찾지 못함: {folder}\n다운로드한 정리본을 {folder}에 복사하세요.')
    print(f'[{label}] Excel 사용: {xlsx}')
    return (pd.read_excel(xlsx,sheet_name='모델결과',dtype=str,keep_default_na=False),
            pd.read_excel(xlsx,sheet_name='근거회의록',dtype=str,keep_default_na=False))

def main():
    tick=time.perf_counter()
    print('통합 평가 시작:',datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    groups=[]; evidences=[]; warnings=[]
    for label,folder,book in SOURCES:
        df,ev=load_data(label,folder,book)
        for col in ['키워드','모델','실험ID']:
            if col not in df.columns: raise ValueError(f'{label}: 필수 컬럼 없음: {col}')
        raw_n=len(df)
        df=df.loc[df['키워드'].isin(KEYWORDS)&df['모델'].isin(MODELS)].copy()
        df=df.drop_duplicates(subset=['키워드','모델'],keep='first').copy()
        if len(df)!=15: warnings.append(f'{label}: 고유 조합 {len(df)}건 (예상 15건), 원본 {raw_n}건')
        df.insert(0,'분석방식',label)
        df['예측 타당성(40) 수동']=''
        df['근거 일치성(30) 수동']=''
        df['근거 원문 검증']='미검증'
        groups.append(df)
        ev=ev.copy()
        if '실험ID' in ev.columns:
            ev=ev.loc[ev['실험ID'].isin(set(df['실험ID']))].copy()
        ev.insert(0,'분석방식',label)
        evidences.append(ev)
        print(f'  {label}: 모델결과 {len(df)}건 / 근거 {len(ev)}건')
    all_df=pd.concat(groups,ignore_index=True,sort=False).fillna('')
    ev_df=pd.concat(evidences,ignore_index=True,sort=False).fillna('')
    for col in ['질문 임베딩(초)','DB 검색(초)','토픽 선택(초)','LLM(초)','전체(초)','실제 글자 수']:
        all_df[col]=pd.to_numeric(all_df[col] if col in all_df else pd.Series([0]*len(all_df)), errors='coerce')
    all_df['형식 통과']=all_df['200자 준수'].astype(str).eq('통과')
    # 오직 형식만 객관적으로 점수화. 속도는 평균/최솟값을 따로 표시.
    all_df['형식 점수(15)']=all_df['형식 통과'].astype(int)*15
    all_df['수동평가 상태']='미평가'
    all_df['최종점수(100)']='미평가'

    def summarise(cols):
        out=(all_df.groupby(cols,sort=False).agg(
            실험수=('모델','size'),
            평균_질문임베딩_초=('질문 임베딩(초)','mean'),
            평균_DB검색_초=('DB 검색(초)','mean'),
            평균_토픽선택_초=('토픽 선택(초)','mean'),
            평균_LLM_초=('LLM(초)','mean'),
            평균_전체_초=('전체(초)','mean'),
            평균_글자수=('실제 글자 수','mean'),
            글자수_준수건수=('형식 통과','sum'),
            글자수_준수율=('형식 통과','mean'),
        ).reset_index())
        out['글자수_준수율(%)']=(out.pop('글자수_준수율')*100).round(1)
        for c in out.select_dtypes(include='number').columns:
            if c not in ['실험수','글자수_준수건수']:
                out[c]=out[c].round(3)
        return out

    model_df=summarise(['분석방식','모델'])
    approach_df=summarise(['분석방식'])
    rules=pd.DataFrame([
        ['실험 구성','1·2·3차 × 모델 3개 × 질문 5개 = 45건. 2차는 최적화 모델 재사용 실험.'],
        ['예측 타당성','0~40점 수동 채점. 미래 전망의 논리·근거·불확실성 표현 평가.'],
        ['근거 일치성','0~30점 수동 채점. 원본 JSON과 근거 구간을 대조한 후 입력.'],
        ['형식 준수','생성 원문 200자 이내이면 15점, 아니면 0점. 자동 채점.'],
        ['속도 평가','최대 15점. 모델 로드 포함 여부와 콜드/웜 상태를 통제한 후 별도 점수 산정. 현재 자동 점수 부여하지 않음.'],
        ['최종점수','수동 평가 미완료. 정확성과 타당성이 검증되기 전에는 총점·순위를 계산하지 않음.'],
        ['등장시점','Chroma Top-15 검색표본 내 최초 관찰 시점. 실제 전체 DB 최초 등장과 다름.'],
        ['타 의회 흐름','검색 근거를 날짜별 나열한 것. 정책 확산의 인과관계를 의미하지 않음.'],
        ['BERTopic','1만 개 균형 표본에서 학습한 47개 토픽을 활용. 전체 684,311개 군집화 결과가 아님.'],
        ['비교주의','2차/3차 텍스트가 동일하면 실제 프롬프트 변화 적용 여부를 추가 검증.'],
        ['중복 방지','방식·키워드·모델당 첫 실행 1건만 사용. 재실험은 별도 실행 ID를 검토.'],
        ['Excel 보존','새 타임스탬프 이름으로 출력. 기존 파일과 수동 평가 점수를 덮어쓰지 않음.'],
    ],columns=['항목','평가기준 및 유의사항'])
    OUT.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path=OUT / f'전체결과_통합_{stamp}.csv'
    excel_path=OUT / f'개인_LLM_3방식_성능비교_{stamp}.xlsx'
    all_df.to_csv(csv_path,index=False,encoding='utf-8-sig')
    with pd.ExcelWriter(excel_path,engine='xlsxwriter') as writer:
        for sheet, data in [('전체결과',all_df),('모델별비교',model_df),('방식별비교',approach_df),('평가기준',rules),('근거회의록',ev_df)]:
            data.to_excel(writer,sheet_name=sheet,index=False)
            ws=writer.sheets[sheet]
            ws.freeze_panes(1,0)
            ws.autofilter(0,0,len(data),len(data.columns)-1)
            ws.set_row(0,26)
            ws.set_column(0,len(data.columns)-1,22)
            if sheet in ('전체결과','근거회의록'): ws.set_column(5,9,35)
    print('통합 결과:',len(all_df),'건 / 근거:',len(ev_df),'건')
    for w in warnings: print('주의:',w)
    print('CSV:',csv_path)
    print('Excel:',excel_path)
    print(f'총 소요 시간: {time.perf_counter()-tick:.2f}초')

if __name__=='__main__':
    main()

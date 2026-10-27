"""06: 정책발언 규칙 분류 결과 3그룹에서 각 30개 수동 검토용 추출.
실행: python 06_review_policy_labels.py
원본 파일은 읽기만 하며, 판정 입력 파일을 재실행해도 덮어쓰지 않습니다.
"""
import csv
import random
import time
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

INPUT = Path(r'C:\mini_team\scripts\clustering\results\policy_validation\policy_chunk_validation_details.csv')
OUT_DIR = Path(r'C:\mini_team\scripts\clustering\results\policy_validation')
OUT_CSV = OUT_DIR / 'policy_manual_review_90.csv'
OUT_XLSX = OUT_DIR / 'policy_manual_review_90.xlsx'
LABELS = ['정책관련_추정', '검토필요', '의사진행_추정']
N_PER_LABEL = 30
SEED = 42


def draw_balanced(candidates, n, rng):
    """지역·연도(source_key)를 순환하며 추출. 가능한 한 지역 분산."""
    buckets = defaultdict(list)
    for row in candidates:
        key = row.get('source_key') or f"{row.get('date','')[:4]}_{row.get('assembly','')}"
        buckets[key].append(row)
    for v in buckets.values():
        rng.shuffle(v)
    keys = sorted(buckets)
    rng.shuffle(keys)
    selected = []
    while len(selected) < n:
        progress = False
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop())
                progress = True
                if len(selected) == n:
                    break
        if not progress:
            break
    return selected


def main():
    t0 = time.perf_counter()
    print('시작:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    if OUT_CSV.exists() or OUT_XLSX.exists():
        raise FileExistsError('기존 수동 판정 파일이 있습니다. 판정 결과 보호를 위해 덮어쓰지 않습니다: ' + str(OUT_DIR))
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)
    with INPUT.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # 무작위 추출 후보는 균형표본 3,400개만 이용. 대표 발언을 섞지 않음.
    samples = [r for r in rows if '균형표본' in r.get('sample_type', '')]
    print('균형표본:', len(samples), '/ 전체 상세:', len(rows))
    rng = random.Random(SEED)
    selected = []
    for label in LABELS:
        group = [r for r in samples if r.get('heuristic_label') == label]
        draw = draw_balanced(group, N_PER_LABEL, rng)
        if len(draw) < N_PER_LABEL:
            raise RuntimeError(f'{label}: {len(draw)}개만 추출, 목표 {N_PER_LABEL}개 부족')
        selected.extend(draw)
        print(f'{label}: 후보 {len(group):,}개 → 추출 {len(draw)}개 / 지역·연도 {len(set(x.get("source_key") for x in draw))}조합')
    rng.shuffle(selected)

    columns = [
        'review_no','heuristic_label','source_key','date','assembly',
        'speaker','agenda','char_length','is_substantive',
        'excerpt_350','label_reason','chunk_id',
        'human_label','human_reason','reviewer','reviewed_at'
    ]
    review_rows = []
    for i, r in enumerate(selected, 1):
        review_rows.append({
            'review_no': i,
            **{key: r.get(key,'') for key in columns[1:12]},
            'human_label':'', 'human_reason':'', 'reviewer':'', 'reviewed_at':''
        })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns)
        w.writeheader(); w.writerows(review_rows)

    # Excel은 별도로 작성하고 싶지 않으면 CSV만 검토해도 됨
    try:
        import xlsxwriter
        wb=xlsxwriter.Workbook(str(OUT_XLSX))
        ws=wb.add_worksheet('수동판정_90개')
        guide=wb.add_worksheet('판정기준')
        header=wb.add_format({'bold':True,'font_color':'white','bg_color':'#174A67','text_wrap':True,'valign':'vcenter','border':1})
        textfmt=wb.add_format({'text_wrap':True,'valign':'top','border':1,'font_size':10})
        inputfmt=wb.add_format({'text_wrap':True,'valign':'top','border':1,'bg_color':'#FFF2C6'})
        for col,key in enumerate(columns): ws.write(0,col,key,header)
        for ri,r in enumerate(review_rows,1):
            for ci,key in enumerate(columns): ws.write(ri,ci,r.get(key,''),inputfmt if ci>=12 else textfmt)
        ws.freeze_panes(1,2)
        ws.autofilter(0,0,len(review_rows),len(columns)-1)
        ws.set_column('A:A',11)
        ws.set_column('B:C',22)
        ws.set_column('D:E',18)
        ws.set_column('F:F',23)
        ws.set_column('G:G',36)
        ws.set_column('H:I',15)
        ws.set_column('J:J',75)
        ws.set_column('K:K',34)
        ws.set_column('L:L',38)
        ws.set_column('M:M',20)
        ws.set_column('N:N',45)
        ws.set_column('O:P',20)
        ws.set_default_row(47)
        ws.data_validation(1,12,len(review_rows),12,{'validate':'list','source':['정책관련','의사진행','판단보류']})
        notes=[
            ('정책관련','실질적인 정책·사업·예산·조례·행정 문제나 개선 방안을 제시/질의하는 발언. 짧아도 해당하면 포함'),
            ('의사진행','회의 개회/산회/표결/발언 순서 안내 등 정책 자체의 실질 내용이 없는 발언'),
            ('판단보류','발언 일부만으로 정책/의사진행 구분 불가. 앞뒤 문맥이 필요한 경우'),
            ('인간 판정 위치','수동판정_90개 시트의 노란색 human_label, human_reason, reviewer, reviewed_at'),
            ('표본 설계','3,400개 지역·연도 균형표본 중 규칙 추정 3종에서 각 30개 추출, 각 종류 안에서 지역·연도 분산'),
            ('주의','원문은 excerpt_350으로 최대 350자 발췌. 애매하면 판단보류 후 ChromaDB 원문 확인'),
            ('보존','수정한 Excel은 별도 저장하고 원본 CSV·DB는 변경하지 않음')
        ]
        for i,(name,desc) in enumerate([('분류','판정 방법')]+notes):
            guide.write(i,0,name,header if i==0 else textfmt)
            guide.write(i,1,desc,header if i==0 else textfmt)
        guide.set_column('A:A',23);guide.set_column('B:B',90)
        guide.freeze_panes(1,0)
        wb.close()
    except ImportError:
        print('xlsxwriter 미설치: CSV만 저장. 설치: python -m pip install xlsxwriter')
    print('CSV:', OUT_CSV)
    if OUT_XLSX.exists(): print('Excel:', OUT_XLSX)
    print('총 판정 대상:',len(review_rows))
    print('총 소요시간:',round(time.perf_counter()-t0,2),'초')
    print('종료:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

if __name__=='__main__':
    main()

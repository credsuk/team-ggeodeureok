"""05_validate_policy_chunks.py
68,000개 균형 표본 중 지역·연도별 재표본을 뽑아 정책 내용 품질 점검.
원본 ChromaDB 읽기만 수행하며, 분류는 휴리스틱 기반 '검토용' 분류다.

설치: pip install numpy pandas chromadb
실행: python 05_validate_policy_chunks.py
"""
from collections import Counter
from datetime import datetime
from pathlib import Path
import re
import time

import chromadb
import numpy as np
import pandas as pd

ROOT = Path(r'C:\mini_team\scripts\clustering')
RESULTS = ROOT / 'results'
DB_PATH = Path(r'C:\mini_team\_db\chroma_snowflake_regions')
COLLECTION = 'team_regions_6a0a006e92d1e93e'
METADATA_PATH = RESULTS / 'cluster_sample_balanced_metadata.csv'
REPRESENTATIVE_PATH = RESULTS / 'cluster_inspection' / 'cluster_representative_utterances_k100.csv'
OUT_DIR = RESULTS / 'policy_validation'
PER_GROUP = 100            # 34개 조합 x 100개 = 3,400개
BATCH_SIZE = 100           # ChromaDB ID 조회 배치
RANDOM_SEED = 42

# 확인 가능한 의사진행 표현: 정책 발언과 혼합될 수 있으므로 '강한 근거' 조건으로만 판정
PROCEDURAL = [
    '수고하셨습니다', '질의하실 위원', '질의하실 의원', '이의 없으십니까',
    '가결되었음을 선포', '산회를 선포', '개의를 선포', '의석을 정돈',
    '정회를 선포', '속개하겠습니다', '안건을 상정', '표결을 시작',
    '투표를 시작', '기권 의원', '기권의원', '찬성 의원', '반대 의원',
    '이상으로 회의를', '보고를 마치겠습니다', '감사합니다',
]
POLICY = [
    '지원', '조례', '예산', '사업', '정책', '계획', '개선', '확대',
    '시행', '추진', '대책', '육성', '개정', '제정', '보조금',
    '고용', '취업', '창업', '인구', '돌봄', '복지', '교육',
    '환경', '교통', '주거', '기후', '산업', '농업', '안전',
    '의료', '시설', '시민', '도민', '청년', '지역', '탄소',
]


def classify(text, substantive):
    """정책 적합성의 확정 판정이 아닌 보수적 우선 검토용 규칙."""
    clean = re.sub(r'\s+', ' ', str(text or '')).strip()
    plen = len(clean)
    pc = sum(term in clean for term in PROCEDURAL)
    kc = sum(term in clean for term in POLICY)
    if not clean:
        tag, reason = '본문없음', '원문 조회 불가'
    elif plen <= 65 and pc >= 1 and kc == 0:
        tag, reason = '의사진행_추정', '짧은 의사진행 표현'
    elif plen <= 100 and pc >= 2 and kc == 0:
        tag, reason = '의사진행_추정', '의사진행 표현 반복'
    elif kc >= 2 and plen >= 80:
        tag, reason = '정책관련_추정', '정책 단어 복수 및 본문 길이 충족'
    elif kc >= 1 and plen >= 180:
        tag, reason = '정책관련_추정', '정책 단어 포함 및 충분한 본문 길이'
    else:
        tag, reason = '검토필요', '규칙만으로 판별 어려움'
    return tag, reason, plen, kc, pc


def read_sample():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(METADATA_PATH)
    df = pd.read_csv(METADATA_PATH, encoding='utf-8-sig', dtype=str).fillna('')
    for col in ('chunk_id', 'source_key', 'assembly', 'date'):
        if col not in df.columns:
            raise ValueError(f'표본 메타데이터에 {col} 컬럼이 없습니다.')
    if df['chunk_id'].duplicated().any():
        raise ValueError('68,000 표본에 중복 청크 ID가 있습니다.')
    rng = np.random.default_rng(RANDOM_SEED)
    parts = []
    for key, group in df.groupby('source_key', sort=True):
        n = min(PER_GROUP, len(group))
        indexes = rng.choice(len(group), size=n, replace=False)
        parts.append(group.iloc[indexes])
    return pd.concat(parts, ignore_index=True), df


def get_representative_ids():
    """기존 대표 발언 CSV의 ID 열이 있으면 함께 검증한다."""
    if not REPRESENTATIVE_PATH.exists():
        return [], None
    df = pd.read_csv(REPRESENTATIVE_PATH, encoding='utf-8-sig', dtype=str).fillna('')
    id_col = next((col for col in ('chunk_id', 'id', 'embedding_id') if col in df.columns), None)
    if id_col is None:
        print('[참고] 대표 발언 CSV에 chunk_id/id 열이 없어 해당 파일 합산은 생략합니다.')
        return [], df
    return df[id_col].loc[lambda s: s.ne('')].drop_duplicates().tolist(), df


def main():
    t_start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected, all_meta = read_sample()
    rep_ids, _ = get_representative_ids()
    sample_ids = selected['chunk_id'].tolist()
    ids = list(dict.fromkeys(sample_ids + rep_ids))
    rep_set = set(rep_ids)
    print(f'전체 표본: {len(all_meta):,}개 / 선택 표본: {len(sample_ids):,}개 / 지역·연도: {selected.source_key.nunique()}')
    print(f'대표 발언 추가 ID: {len(rep_ids):,}개 / 중복 제외 조회: {len(ids):,}개')
    print('DB 연결 중...')
    client = chromadb.PersistentClient(path=str(DB_PATH))
    coll = client.get_collection(name=COLLECTION)
    print(f'컬렉션: {coll.name} / 청크 수: {coll.count():,}')
    requested_meta = selected.set_index('chunk_id').to_dict('index')
    rows = []
    for start in range(0, len(ids), BATCH_SIZE):
        batch_ids = ids[start:start + BATCH_SIZE]
        result = coll.get(ids=batch_ids, include=['documents', 'metadatas'])
        lookup = {cid: (doc, meta or {}) for cid, doc, meta in zip(
            result['ids'], result.get('documents') or [], result.get('metadatas') or []
        )}
        for cid in batch_ids:
            text, meta = lookup.get(cid, ('', {}))
            label, reason, nchar, pcount, proc_count = classify(text, meta.get('is_substantive'))
            base = requested_meta.get(cid, {})
            rows.append({
                'chunk_id': cid,
                'source_key': meta.get('source_key') or base.get('source_key', ''),
                'assembly': meta.get('assembly') or base.get('assembly', ''),
                'date': meta.get('date') or base.get('date', ''),
                'speaker': meta.get('speaker') or base.get('speaker', ''),
                'agenda': meta.get('agenda') or base.get('agenda', ''),
                'is_substantive': meta.get('is_substantive', ''),
                'sample_type': ('균형표본+대표발언' if cid in rep_set else '균형표본') if cid in requested_meta else '대표발언',
                'heuristic_label': label,
                'label_reason': reason,
                'char_length': nchar,
                'policy_terms_hit': pcount,
                'procedural_terms_hit': proc_count,
                'excerpt_350': re.sub(r'\s+', ' ', text or '').strip()[:350],
            })
        processed = min(start + BATCH_SIZE, len(ids))
        if processed % 500 == 0 or processed == len(ids):
            print(f'원문 조회: {processed:,}/{len(ids):,}')

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DIR / 'policy_chunk_validation_details.csv', index=False, encoding='utf-8-sig')
    base_only = detail[detail['chunk_id'].isin(sample_ids)]
    summary = (base_only.groupby(['source_key', 'heuristic_label'])
               .size().unstack(fill_value=0).reset_index())
    for k in ['정책관련_추정', '의사진행_추정', '검토필요', '본문없음']:
        if k not in summary.columns:
            summary[k] = 0
    summary['표본수'] = summary[['정책관련_추정', '의사진행_추정', '검토필요', '본문없음']].sum(axis=1)
    summary['정책관련_추정비율_pct'] = (100 * summary['정책관련_추정'] / summary['표본수']).round(1)
    summary.to_csv(OUT_DIR / 'policy_chunk_validation_by_region.csv', index=False, encoding='utf-8-sig')
    overall = base_only['heuristic_label'].value_counts()
    print('\n=== 34개 지역·연도 균형표본 판정(참고용) ===')
    for key in ('정책관련_추정', '의사진행_추정', '검토필요', '본문없음'):
        print(f'{key}: {overall.get(key, 0):,}개 / {overall.get(key, 0) / max(len(base_only),1)*100:.1f}%')
    if rep_set:
        reps = detail[detail['chunk_id'].isin(rep_set)]
        print('\n=== K=100 대표 발언 판정(참고용) ===')
        print(reps['heuristic_label'].value_counts().to_string())
    print('\n결과 파일:')
    print(OUT_DIR / 'policy_chunk_validation_details.csv')
    print(OUT_DIR / 'policy_chunk_validation_by_region.csv')
    print('총 소요 시간:', round(time.perf_counter() - t_start, 2), '초')
    print('주의: 키워드 규칙에 의한 추정 분류이며 정답/정확도가 아닙니다.')
    print('정책관련_추정·의사진행_추정·검토필요 각각에서 원문을 수작업으로 확인하세요.')


if __name__ == '__main__':
    main()

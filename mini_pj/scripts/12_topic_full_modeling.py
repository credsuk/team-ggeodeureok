# -*- coding: utf-8 -*-
"""개인 프로젝트: 684,311개 전체 임베딩의 온라인 BERTopic 학습 + 최종 토픽 할당.

기존 Chroma 및 1~3차 평가 결과를 수정하지 않습니다.
실행 전 검증: python 12_topic_full_modeling.py --inspect
전체 실행:  python 12_topic_full_modeling.py --run
중단 재개:  같은 --run 명령 재실행

방법: 저장된 1024차원 임베딩 -> 고정 SVD(64차원) -> MiniBatchKMeans
     -> BERTopic.partial_fit + OnlineCountVectorizer(Kiwi 명사).
주의: HDBSCAN 기반 기존 1만 표본 모델과 달리 KMeans는 토픽 수를 설정하며
     노이즈(-1)를 자동 탐지하지 않습니다. 토픽은 분석 후보이지 검증된 정책명이 아닙니다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD

ROOT = Path(r"C:\mini_pj")
EMBED_DIR = ROOT / "data" / "processed" / "embeddings_by_region"
OUT = ROOT / "results" / "full_topic_modeling"
CHECKPOINT = OUT / "online_checkpoint.joblib"
SVD_PATH = OUT / "svd_고정축.joblib"
MANIFEST_PATH = OUT / "input_manifest.json"
EXPECTED_ROWS = 684_311
EXPECTED_DIM = 1024
BATCH_SIZE = 2048
N_TOPICS = 80        # 초기 탐색용; 확정된 최적 토픽 수가 아님
SVD_DIM = 64
SVD_SAMPLE_PER_FILE = 64   # 155개 파일일 때 대략 1만 개로 SVD 학습
SEED = 42
TOP_WORDS = 12

STOPWORDS = {
    '의원','위원','위원장','의장','국장','과장','담당관','실장','본회의','위원회','의회',
    '회의록','회의','발언','질의','질문','답변','보고','말씀','수고','여러분','저희',
    '우리','내용','사항','관련','이번','다음','오늘','지금','현재','해당','부분',
    '정도','때문','생각','감사','진행','설명','사실','자리','안건','상정','가결',
    '출석','폐회','개의','다운로드','첨부','조회','메뉴','페이지','홈페이지',
    '서울특별시','경기도','강원특별자치도','충청북도','충청남도','전북특별자치도',
    '전라남도','경상북도','경상남도','부산광역시','대구광역시','광주광역시',
    '인천광역시','대전광역시','울산광역시','세종특별자치시','제주특별자치도',
}
BANNED = ('javascript','http://','https://','navigation','toggle')


class FixedSVD:
    """BERTopic 온라인 학습 내내 좌표계가 변하지 않는 차원 축소기."""
    def __init__(self, svd):
        self.svd = svd
    def fit(self, x, y=None):
        return self
    def partial_fit(self, x, y=None):
        return self
    def transform(self, x):
        return self.svd.transform(x).astype(np.float32)
    def fit_transform(self, x, y=None):
        return self.transform(x)


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def elapsed(start):
    s = time.perf_counter() - start
    return f'{s:,.2f}초 / {s/60:,.2f}분 / {s/3600:,.2f}시간'


def atomic_json(path, data):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def atomic_joblib(path, data):
    tmp = path.with_name(path.name + '.tmp')
    joblib.dump(data, tmp, compress=1)
    os.replace(tmp, path)


def discover():
    files = sorted(EMBED_DIR.rglob('batch_*.npy'))
    if not files:
        raise FileNotFoundError(f'배치 임베딩을 찾지 못함: {EMBED_DIR}\\**\\batch_*.npy')
    items = []
    total = 0
    for i, npy in enumerate(files, 1):
        meta = npy.with_name(npy.stem + '_metadata.csv')
        if not meta.is_file():
            raise FileNotFoundError(f'짝 메타데이터 없음: {meta}')
        arr = np.load(npy, mmap_mode='r', allow_pickle=False)
        if arr.ndim != 2 or arr.shape[1] != EXPECTED_DIM:
            raise ValueError(f'임베딩 차원 오류: {npy} {arr.shape}')
        with meta.open('r', encoding='utf-8-sig', newline='') as f:
            columns = csv.DictReader(f).fieldnames or []
        if not {'chunk_id','text'}.issubset(columns):
            raise ValueError(f'메타데이터 필수 열(chunk_id, text) 없음: {meta} / {columns}')
        n = int(arr.shape[0]); total += n
        items.append({'npy': str(npy), 'meta': str(meta), 'count': n,
                      'bytes_npy': npy.stat().st_size, 'bytes_meta': meta.stat().st_size,
                      'mtime_npy_ns': npy.stat().st_mtime_ns,
                      'mtime_meta_ns': meta.stat().st_mtime_ns})
        if i % 20 == 0 or i == len(files):
            print(f'[검사 {i}/{len(files)}] 벡터 {total:,}개', flush=True)
    if total != EXPECTED_ROWS:
        raise ValueError(f'전체 벡터 개수가 예상과 다름: {total:,} / 예상 {EXPECTED_ROWS:,}. '
                         '서로 다른 데이터에 모델을 덮어쓰지 않도록 중단함.')
    signature = hashlib.sha256(json.dumps(items, ensure_ascii=False, sort_keys=True)
                               .encode('utf-8')).hexdigest()
    return items, signature


def load_frame(item):
    t = time.perf_counter()
    frame = pd.read_csv(item['meta'], dtype=str, keep_default_na=False,
                        encoding='utf-8-sig', low_memory=False)
    if len(frame) != item['count']:
        raise ValueError(f'행 수 불일치 {item["meta"]}: CSV {len(frame):,} vs NPY {item["count"]:,}')
    if frame['chunk_id'].eq('').any() or frame['chunk_id'].duplicated().any():
        raise ValueError(f'빈/중복 chunk_id (단일 배치): {item["meta"]}')
    print(f'   메타 로드 {elapsed(t)}', flush=True)
    return frame


def fit_svd(items, signature):
    if SVD_PATH.exists():
        state = joblib.load(SVD_PATH)
        if state.get('signature') != signature:
            raise RuntimeError(f'입력 파일이 바뀌었음. 기존 SVD를 보존하고 별도 OUT 경로 사용: {SVD_PATH}')
        print('[SVD] 이전 고정축 재사용', flush=True)
        return state['svd']
    rng = np.random.default_rng(SEED)
    blocks = []
    t = time.perf_counter()
    for i, item in enumerate(items, 1):
        arr = np.load(item['npy'], mmap_mode='r', allow_pickle=False)
        take = min(SVD_SAMPLE_PER_FILE, len(arr))
        ix = rng.choice(len(arr), size=take, replace=False)
        x = np.asarray(arr[ix], dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f'임베딩 NaN/Inf: {item["npy"]}')
        blocks.append(x)
        if i % 30 == 0 or i == len(items):
            print(f'[SVD 표본 {i}/{len(items)}] {sum(map(len, blocks)):,}건', flush=True)
    sample = np.concatenate(blocks)
    svd = TruncatedSVD(n_components=min(SVD_DIM, EXPECTED_DIM-1, len(sample)-1),
                       n_iter=5, random_state=SEED)
    svd.fit(sample)
    atomic_joblib(SVD_PATH, {'signature': signature, 'svd': svd})
    print(f'[SVD 고정축 학습] {len(sample):,}건 / {elapsed(t)}', flush=True)
    return svd


def tokens_for_texts(texts, kiwi):
    docs = []
    for raw in texts:
        cleaned = '\n'.join(line for line in str(raw).splitlines()
                            if not any(tag in line.lower() for tag in BANNED))
        words = []
        for tok in kiwi.tokenize(cleaned):
            if tok.tag not in {'NNG','NNP'}:
                continue
            w = tok.form.strip().lower()
            if len(w) >= 2 and w not in STOPWORDS and re.fullmatch(r'[가-힣]{2,}', w):
                words.append(w)
        docs.append(' '.join(words) if words else '분류불가')
    return docs


def new_topic_model(svd):
    from bertopic import BERTopic
    from bertopic.vectorizers import OnlineCountVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = OnlineCountVectorizer(
        tokenizer=str.split, token_pattern=None, lowercase=False,
        ngram_range=(1, 2), max_features=30_000,
        decay=0.0, delete_min_df=2)
    clustering = MiniBatchKMeans(n_clusters=N_TOPICS, random_state=SEED,
                                 batch_size=BATCH_SIZE, n_init=3,
                                 reassignment_ratio=0.01)
    return BERTopic(umap_model=FixedSVD(svd), hdbscan_model=clustering,
                    vectorizer_model=vectorizer,
                    language='multilingual', top_n_words=TOP_WORDS,
                    calculate_probabilities=False, verbose=False)


def train(items, signature, svd):
    from kiwipiepy import Kiwi
    kiwi = Kiwi(num_workers=0)
    if CHECKPOINT.exists():
        state = joblib.load(CHECKPOINT)
        if state['signature'] != signature or state['topics'] != N_TOPICS:
            raise RuntimeError('기존 체크포인트 입력/토픽 수 불일치. 기존 파일을 보존하고 출력 경로 변경 필요')
        model, start_file = state['model'], int(state['next_file'])
        print(f'[체크포인트 재개] 완료 {start_file}/{len(items)}개 배치파일', flush=True)
    else:
        model, start_file = new_topic_model(svd), 0
    t_all = time.perf_counter()
    for file_i in range(start_file, len(items)):
        item = items[file_i]; t = time.perf_counter()
        frame = load_frame(item)
        arr = np.load(item['npy'], mmap_mode='r', allow_pickle=False)
        for start in range(0, len(frame), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(frame))
            v = np.asarray(arr[start:stop], dtype=np.float32)
            if not np.isfinite(v).all():
                raise ValueError(f'비정상 임베딩: {item["npy"]} offset={start}')
            docs = tokens_for_texts(frame['text'].iloc[start:stop], kiwi)
            # 첫 배치가 N_TOPICS 미만인 경우도 명시적으로 차단
            if file_i == 0 and start == 0 and len(v) < N_TOPICS:
                raise ValueError(f'첫 배치가 N_TOPICS({N_TOPICS})보다 작음')
            model.partial_fit(docs, embeddings=v)
            if (stop // BATCH_SIZE) % 5 == 0 or stop == len(frame):
                print(f'    파일 내 {stop:,}/{len(frame):,}건 / {elapsed(t)}', flush=True)
        # 완료된 '파일'까지만 체크포인트에 기록하여 중단 재개 가능.
        atomic_joblib(CHECKPOINT, {'signature': signature, 'topics': N_TOPICS,
                                   'next_file': file_i + 1, 'model': model})
        print(f'[학습 {file_i+1}/{len(items)}] {Path(item["npy"]).name} '
              f'{item["count"]:,}건 / {elapsed(t)} / 전체 경과 {elapsed(t_all)}', flush=True)
    info = model.get_topic_info().drop(columns=['Representative_Docs'], errors='ignore')
    info.to_csv(OUT / '전체_토픽요약.csv', index=False, encoding='utf-8-sig')
    joblib.dump(model.hdbscan_model, OUT / '최종_군집모델.joblib', compress=1)
    print(f'[온라인 토픽 학습 완료] {elapsed(t_all)} / 토픽 요약 {len(info)}행', flush=True)
    return model


def final_assign(items, signature, model):
    """학습 중 변경된 군집 중심이 아닌, 최종 모델로 전 청크를 일관되게 재할당."""
    assign_dir = OUT / 'assignments'
    assign_dir.mkdir(parents=True, exist_ok=True)
    all_counts = Counter()
    t_all = time.perf_counter()
    for i, item in enumerate(items, 1):
        # 파일이 끝나야 최종 CSV 이름으로 변경됨; 이미 완료한 파일은 안전하게 재사용
        path = assign_dir / f'{i:03d}_토픽할당.csv'
        done = assign_dir / f'{i:03d}_토픽할당.json'
        if path.exists() and done.exists():
            state = json.loads(done.read_text(encoding='utf-8'))
            if state.get('signature') == signature and state.get('count') == item['count']:
                all_counts.update({int(k): int(v) for k,v in state['topic_counts'].items()})
                print(f'[할당 {i}/{len(items)}] 기존 결과 재사용 {item["count"]:,}건', flush=True)
                continue
        frame = load_frame(item)
        arr = np.load(item['npy'], mmap_mode='r', allow_pickle=False)
        tmp = path.with_name(path.name + '.tmp')
        counter = Counter()
        t = time.perf_counter()
        # 원문 전체는 원본 메타데이터 CSV와 ChromaDB에서 조회할 수 있도록 ID를 보존.
        # 행 위치를 기록해서 메타데이터 파일 및 벡터와 1:1로 재연결할 수 있음.
        fields = ['chunk_id','topic_id','date','assembly','year','source_file',
                  'meeting_title','source_id','meta_csv','row_index','text_preview']
        with tmp.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for start in range(0, len(frame), BATCH_SIZE):
                stop = min(start + BATCH_SIZE, len(frame))
                v = np.asarray(arr[start:stop], dtype=np.float32)
                # 온라인 BERTopic의 topic ID 매핑을 보존하기 위해 transform 사용.
                labels, _ = model.transform(['분류용'] * len(v), embeddings=v)
                if len(labels) != len(v):
                    raise RuntimeError(f'토픽 할당 수 불일치: {item["npy"]}')
                counter.update(map(int, labels))
                for j, lab in enumerate(labels, start):
                    r = frame.iloc[j]
                    date = str(r.get('date', ''))
                    assembly = str(r.get('assembly', ''))
                    year = date[:4] if re.match(r'^\d{4}', date) else ''
                    writer.writerow({'chunk_id': r['chunk_id'], 'topic_id': int(lab),
                                     'date': date, 'assembly': assembly, 'year': year,
                                     'source_file': str(r.get('source_file', '')),
                                     'meeting_title': str(r.get('meeting_title', r.get('meeting', ''))),
                                     'source_id': str(r.get('source_id', '')),
                                     'meta_csv': str(item['meta']), 'row_index': j,
                                     'text_preview': str(r['text'])[:180]})
        os.replace(tmp, path)
        atomic_json(done, {'signature': signature, 'count': item['count'],
                           'topic_counts': dict(counter)})
        all_counts.update(counter)
        print(f'[최종 할당 {i}/{len(items)}] {sum(counter.values()):,}건 / {elapsed(t)}', flush=True)
    if sum(all_counts.values()) != EXPECTED_ROWS:
        raise ValueError(f'전체 토픽 할당 건수 오류: {sum(all_counts.values())} / {EXPECTED_ROWS}')
    pd.DataFrame([{'topic_id': k, 'count': v} for k,v in sorted(all_counts.items())])\
        .to_csv(OUT / '최종_토픽분포.csv', index=False, encoding='utf-8-sig')
    print(f'[전체 토픽 할당 완료] {sum(all_counts.values()):,}건 / {elapsed(t_all)}', flush=True)


def build_dashboard_aggregates(items):
    """웹 시각화용 집계. 전체 청크 단위 CSV는 assignments/ 폴더에 보존."""
    from collections import defaultdict
    region = defaultdict(int); monthly = defaultdict(int)
    for i in range(1, len(items)+1):
        path = OUT / 'assignments' / f'{i:03d}_토픽할당.csv'
        for frame in pd.read_csv(path, chunksize=20_000, encoding='utf-8-sig', dtype=str,
                                 keep_default_na=False):
            for (assembly, topic), n in frame.groupby(['assembly','topic_id']).size().items():
                region[(assembly, topic)] += int(n)
            month = frame['date'].str.slice(0,7)
            valid = month.str.match(r'^\d{4}-\d{2}$')
            for (m, topic), n in frame.loc[valid].assign(month=month[valid])\
                                 .groupby(['month','topic_id']).size().items():
                monthly[(m, topic)] += int(n)
    pd.DataFrame([{'assembly':a,'topic_id':t,'count':n} for (a,t),n in region.items()])\
        .to_csv(OUT/'웹_지역별_토픽집계.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'month':m,'topic_id':t,'count':n} for (m,t),n in monthly.items()])\
        .to_csv(OUT/'웹_월별_토픽집계.csv',index=False,encoding='utf-8-sig')
    print('[웹 집계 완료] 지역×토픽 / 월×토픽 저장', flush=True)


def main():
    parser = argparse.ArgumentParser(description='전체 684,311개 BERTopic 온라인 학습')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--inspect', action='store_true', help='기존 파일 155개 개수/구조만 확인')
    mode.add_argument('--run', action='store_true', help='전체 학습 + 최종 할당 + 웹 집계')
    args = parser.parse_args()
    print(f'=== 전체 토픽 모델링 시작: {now()} ===', flush=True)
    t = time.perf_counter()
    items, signature = discover()
    print(f'검증: 파일 {len(items)}개 / 벡터 {sum(x["count"] for x in items):,}개 '
          f'/ {EXPECTED_DIM}차원 / 시그니처 {signature[:16]}', flush=True)
    if args.inspect:
        print('검증만 수행. 모델 학습·원본 변경 없음.')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        old = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
        if old.get('signature') != signature:
            raise RuntimeError('입력 원본이 이전 실행과 달라짐. 기존 OUT 폴더를 보존하고 확인 필요')
    else:
        atomic_json(MANIFEST_PATH, {'signature': signature, 'items': items,
                                    'expected_rows': EXPECTED_ROWS, 'dim': EXPECTED_DIM,
                                    'method': 'BERTopic online / fixed SVD / MiniBatchKMeans / Kiwi',
                                    'n_topics': N_TOPICS, 'created_at': now()})
    svd = fit_svd(items, signature)
    model = train(items, signature, svd)
    final_assign(items, signature, model)
    build_dashboard_aggregates(items)
    atomic_json(OUT/'완료정보.json', {'completed_at': now(), 'signature': signature,
                                   'rows': EXPECTED_ROWS, 'topics_configured': N_TOPICS,
                                   'runtime': elapsed(t)})
    print(f'=== 완료: {now()} / {elapsed(t)} ===', flush=True)
    print(f'결과 폴더: {OUT}', flush=True)


if __name__ == '__main__':

    if len(sys.argv) == 1:
        sys.argv.append("--run")

    main()



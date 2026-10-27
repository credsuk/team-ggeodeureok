# TXT·JSONL 연결 검증 → 청킹 → Snowflake 모델 임베딩 → ChromaDB 저장 → 검색 테스트

# 준비사항 
# 가상환경 : project01
# 설치파일 : python -m pip install sentence-transformers chromadb
# GPU & 파이토치 설치 
# CPU용 PyTorch 제거
# python -m pip uninstall torch -y
# CUDA 12.8용 PyTorch 설치
# python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
# PyTorch: 2.11.0+cu128
# GPU 사용 가능: True
# GPU: NVIDIA GeForce RTX 4060 Laptop GPU


"""지역별 TXT/JSONL 연결 → 5,000청크 단위 처리 → Snowflake → ChromaDB.
설치: python -m pip install sentence-transformers chromadb
실행: python embed_regions_chroma.py
TXT_DIR, META_DIR, DB_PATH만 본인 경로로 수정하세요.
"""
import csv
import hashlib
import json
import re
from pathlib import Path

# ===== 사용자가 수정할 경로 =====
TXT_DIR = Path(r"C:/mini_team/change_text")
META_DIR = Path(r"C:/mini_team/matadata_jsonl_python")
DB_PATH = Path(r"C:/mini_team/_db/chroma_snowflake_regions")

MODEL_NAME = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
CHUNK_TOKENS = 450
OVERLAP_TOKENS = 50
BLOCK_SIZE = 5000       # 진행 상황을 기록하는 단위
SAVE_BATCH = 128        # Chroma에 실제로 저장하는 단위
BATCH_SIZE = 8          # GPU 임베딩 단위. 메모리 부족 시 4로 변경
YEARS = {"2024", "2025"}
csv.field_size_limit(100_000_000)


def norm(value):
    return re.sub(r"\s+", "", str(value or ""))


def file_key(path):
    # cleaned_2025_광주광역시의회(1).jsonl 같은 다운로드 접미사도 허용
    name = re.sub(r"(?:\(\d+\)|_\d+)$", "", path.stem)
    match = re.fullmatch(r"cleaned_(\d{4})_(.+)", name)
    return match.groups() if match else None


def excluded(key):
    year, region = key
    return year in YEARS and "강원" in region


def discover(folder, extension):
    if not folder.is_dir():
        raise FileNotFoundError(f"폴더를 확인하세요: {folder}")
    found = {}
    for path in sorted(folder.rglob('*' + extension)):
        key = file_key(path)
        if not key or key[0] not in YEARS:
            continue
        if excluded(key):
            print("강원 제외:", path.name) # 파일 수정 중
            continue
        if key in found:
            raise ValueError(f"같은 연도·지역 파일이 2개입니다. 사용할 파일 하나만 남기세요:\n{found[key]}\n{path}")
        found[key] = path
    return found


def find_pairs(txt_dir, meta_dir):
    texts = discover(txt_dir, '.txt')
    metas = discover(meta_dir, '.jsonl')
    if set(texts) != set(metas):
        raise ValueError(f"파일 쌍 누락. JSONL 없는 TXT: {sorted(set(texts)-set(metas))}; TXT 없는 JSONL: {sorted(set(metas)-set(texts))}")
    if not texts:
        raise ValueError("처리할 파일이 없습니다. 폴더와 cleaned_연도_지역 파일명을 확인하세요.")
    return [(key, texts[key], metas[key]) for key in sorted(texts)]


def load_pair(txt_path, meta_path, key):
    with txt_path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')  # 첫 줄 컬럼명 자동 제외
        required = {'MTG_DE', 'RASMBLY_NM', 'speaker_role', 'speaker_name', 'agenda_title', 'utterance'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"TXT 컬럼 확인: {txt_path.name}")
        rows = list(reader)
    metadata = {}
    with meta_path.open(encoding='utf-8-sig') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or not {'row_id','date','assembly','speaker','agenda'}.issubset(item):
                raise ValueError(f"JSONL 필수 항목 확인: {meta_path.name}:{line_no}")
            rid = item['row_id']
            if type(rid) is not int or rid in metadata:
                raise ValueError(f"row_id 자료형/중복 확인: {meta_path.name}:{line_no}")
            metadata[rid] = item
    if not rows or len(rows) != len(metadata) or set(metadata) != set(range(1, len(rows)+1)):
        raise ValueError(f"행 수/ID 불일치: {txt_path.name}, TXT={len(rows)}, JSONL={len(metadata)}")
    result = []
    for rid, row in enumerate(rows, 1):
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f"TXT 컬럼 수 오류: {txt_path.name}, 데이터 {rid}행")
        meta = metadata[rid]
        checks = [(row['MTG_DE'],meta['date']), (row['RASMBLY_NM'],meta['assembly']),
                  (row['speaker_role']+row['speaker_name'],meta['speaker']),
                  (row['agenda_title'],meta['agenda'])]
        if not all(norm(a)==norm(b) for a,b in checks):
            raise ValueError(f"날짜·지역·발언자·안건 불일치: {txt_path.name}, row_id={rid}")
        if not str(meta['date']).startswith(key[0]) or norm(meta['assembly']) != norm(key[1]):
            raise ValueError(f"파일명과 연도·지역 불일치: {txt_path.name}, row_id={rid}")
        if not row['utterance'].strip():
            raise ValueError(f"본문 없음: {txt_path.name}, row_id={rid}")
        result.append((row['utterance'], meta))
    return result


def digest_files(txt, meta):
    h = hashlib.sha256()
    for path in (txt, meta):
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def chunk_records(matched, tokenizer, source_key, fingerprint, source_file):
    namespace = hashlib.sha256(source_key.encode()).hexdigest()[:20]
    for text, meta in matched:
        offsets = tokenizer(text, add_special_tokens=False, truncation=False,
                            return_offsets_mapping=True)['offset_mapping']
        if not offsets:
            raise ValueError(f"토큰화 결과 없음: {source_key}, row_id={meta['row_id']}")
        for number, start in enumerate(range(0,len(offsets),CHUNK_TOKENS-OVERLAP_TOKENS),1):
            end = min(start+CHUNK_TOKENS,len(offsets))
            begin_char, end_char = offsets[start][0], offsets[end-1][1]
            md = {k:(v if isinstance(v,(str,int,float,bool)) else json.dumps(v,ensure_ascii=False))
                  for k,v in meta.items() if v is not None}
            md.update(source_key=source_key, source_fingerprint=fingerprint,
                      source_file=source_file, chunk_no=number,
                      char_start=begin_char, char_end=end_char)
            yield {'id':f'{namespace}_{meta["row_id"]}_{number}',
                   'text':text[begin_char:end_char], 'metadata':md}
            if end == len(offsets):
                break


def save_block(block, collection, model):
    added = 0
    for start in range(0,len(block),SAVE_BATCH):
        batch = block[start:start+SAVE_BATCH]
        old = set(collection.get(ids=[r['id'] for r in batch],include=[])['ids'])
        new = [r for r in batch if r['id'] not in old]
        if new:
            vectors = model.encode([r['text'] for r in new], prompt='',
                                   batch_size=BATCH_SIZE, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
            collection.upsert(ids=[r['id'] for r in new], documents=[r['text'] for r in new],
                              metadatas=[r['metadata'] for r in new], embeddings=vectors.tolist())
            added += len(new)
        if (start//SAVE_BATCH+1)%10 == 0 or start+SAVE_BATCH >= len(block):
            print(f"  블록 내 처리: {min(start+SAVE_BATCH,len(block)):,}/{len(block):,}",flush=True)
    return added


def record_progress(path, progress):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def main():
    if not (0 <= OVERLAP_TOKENS < CHUNK_TOKENS and BLOCK_SIZE > 0 and SAVE_BATCH > 0 and BATCH_SIZE > 0):
        raise ValueError('청킹·배치 설정을 확인하세요.')
    pairs = find_pairs(TXT_DIR,META_DIR)
    print(f"강원 제외 후 처리할 연도·지역: {len(pairs)}개")
    # 전체 파일의 연결을 먼저 검사합니다. 오류가 있으면 모델 실행 전 중단합니다.
    for key,txt,meta in pairs:
        count = len(load_pair(txt,meta,key))
        print(f"검증 완료: {'_'.join(key)} / {count:,}개 발언",flush=True)
    import chromadb
    import torch
    from sentence_transformers import SentenceTransformer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('실행 장치:',device)
    model = SentenceTransformer(MODEL_NAME,device=device)
    model.max_seq_length = CHUNK_TOKENS + 2
    if not model.tokenizer.is_fast:
        raise ValueError('fast tokenizer가 필요합니다.')
    settings = f'{MODEL_NAME}|{CHUNK_TOKENS}|{OVERLAP_TOKENS}|cosine|v1'
    collection_name = 'team_regions_' + hashlib.sha256(settings.encode()).hexdigest()[:16]
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(name=collection_name,embedding_function=None,
        metadata={'hnsw:space':'cosine','embedding_model':MODEL_NAME,
                  'chunk_tokens':CHUNK_TOKENS,'overlap_tokens':OVERLAP_TOKENS})
    progress_path = DB_PATH / (collection_name + '_progress.json')
    progress = {'collection':collection_name,'model':MODEL_NAME,'regions':{}}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding='utf-8'))
    for index,(key,txt,meta) in enumerate(pairs,1):
        source_key = '_'.join(key)
        fingerprint = digest_files(txt,meta)
        previous = progress['regions'].get(source_key,{})
        # 바뀐 원본에 예전 벡터를 그대로 붙이는 것을 방지합니다.
        sample = collection.get(where={'source_key':source_key},limit=1,include=['metadatas'])
        if (previous and previous.get('fingerprint') != fingerprint) or any(
                x.get('source_fingerprint') != fingerprint for x in sample['metadatas']):
            raise ValueError(f'{source_key}: 이전 입력 파일과 다릅니다. 수정본은 새 DB_PATH로 실행하세요.')
        if previous.get('completed'):
            stored = collection.get(where={'source_key':source_key},include=[])['ids']
            if len(stored) == previous.get('chunks'):
                print(f'[{index}/{len(pairs)}] 완료 지역 건너뜀: {source_key}',flush=True)
                continue
        print(f'[{index}/{len(pairs)}] 시작: {source_key}',flush=True)
        matched = load_pair(txt,meta,key)
        block, total, block_no = [], 0, 0
        for record in chunk_records(matched,model.tokenizer,source_key,fingerprint,txt.name):
            block.append(record)
            if len(block)==BLOCK_SIZE:
                block_no += 1
                print(f'  {block_no}번째 블록 처리 시작: {len(block):,}청크',flush=True)
                added = save_block(block,collection,model)
                total += len(block)
                progress['regions'][source_key] = {'fingerprint':fingerprint,'completed':False,
                    'chunks':total,'completed_blocks':block_no}
                record_progress(progress_path,progress)
                print(f'  블록 완료 / 누적 {total:,} / 신규 저장 {added:,}',flush=True)
                block.clear()
        if block:
            block_no += 1
            print(f'  마지막 블록 처리: {len(block):,}청크',flush=True)
            save_block(block,collection,model)
            total += len(block)
        stored = collection.get(where={'source_key':source_key},include=[])['ids']
        if len(stored)!=total:
            raise ValueError(f'{source_key}: 예상 청크 {total}, DB 청크 {len(stored)} 불일치')
        progress['regions'][source_key] = {'fingerprint':fingerprint,'completed':True,
            'chunks':total,'completed_blocks':block_no}
        record_progress(progress_path,progress)
        print(f'지역 완료: {source_key} / {total:,}청크',flush=True)
    print('전체 완료. DB 경로:',DB_PATH)
    print('컬렉션:',collection_name)
    print('총 저장 청크:',collection.count())
    print('강원 수정본을 추가할 때는 excluded()의 강원 제외 조건을 해제하세요.')


if __name__=='__main__':
    main()

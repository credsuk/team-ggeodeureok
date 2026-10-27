"""09: POLICY K=100 군집의 대표·무작위 발언과 주제 일관성 진단.
기존 ChromaDB는 조회만 수행. 결과 CSV는 별도 폴더에 보존.
실행: python 09_inspect_policy_clusters.py
"""
from pathlib import Path
from datetime import datetime
from collections import Counter
import json
import time
import re
import numpy as np
import pandas as pd
import joblib
import chromadb

ROOT = Path(r"C:\mini_team\scripts\clustering\results")
NPZ = ROOT / "cluster_sample_balanced.npz"
META = ROOT / "cluster_sample_balanced_metadata.csv"
FILTER = ROOT / "policy_refined" / "policy_filter_68000.csv"
MODEL = ROOT / "policy_kmeans" / "policy_minibatch_kmeans_k100.joblib"
OUT = ROOT / "policy_cluster_inspection"
DB_PATH = Path(r"C:\mini_team\_db\chroma_snowflake_regions")
COLLECTION_NAME = "team_regions_6a0a006e92d1e93e"
REP_N = 5
RANDOM_N = 5
GET_BATCH = 100
SEED = 42

# 토픽명이 아닌 '참고 키워드'만 추출. 정식 c-TF-IDF는 10단계에 진행.
STOPWORDS = set("저희 제가 우리 그리고 그런데 그래서 그런 이렇게 대해서 있는 있습니다 합니다 것입니다 위원 의원 말씀 감사합니다 수고 오늘 지금 이번 해당 관련 대한 통해 또한 하는 있는지 합니다만 생각 그러면 이것 부분 경우 지금까지 정말 이걸 이게 그게 있는 것으로".split())


def keywords(docs, top=12):
    counts = Counter()
    for doc in docs:
        # 한국어 형태소분석기는 아직 사용하지 않음. 공백 단위 후보어만 추출.
        for token in re.findall(r"[가-힣A-Za-z]{2,}", doc):
            if token not in STOPWORDS and len(token) >= 2:
                counts[token] += 1
    return ", ".join(word for word, _ in counts.most_common(top))


def cosine_pair_mean(vectors):
    if len(vectors) < 2:
        return np.nan
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norms, 1e-12)
    sim = arr @ arr.T
    ii = np.triu_indices(len(arr), 1)
    return float(np.mean(sim[ii]))


def main():
    t_all = time.perf_counter()
    print("시작:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("[1/5] 표본·POLICY 필터·K=100 모델 로드", flush=True)
    start = time.perf_counter()
    with np.load(NPZ, allow_pickle=False) as z:
        x_all = z["embeddings"].astype(np.float32)
        ids_all = z["ids"].astype(str)
    md = pd.read_csv(META, dtype={"chunk_id": str}, low_memory=False).fillna("")
    fl = pd.read_csv(FILTER, usecols=["chunk_id", "refined_label"], dtype=str)
    model = joblib.load(MODEL)
    if len(x_all) != len(ids_all) or len(md) != len(ids_all):
        raise ValueError("NPZ, 표본 메타데이터 행 수가 일치하지 않습니다.")
    if not np.array_equal(md["chunk_id"].to_numpy(dtype=str), ids_all):
        raise ValueError("NPZ와 CSV의 청크 ID 순서가 다릅니다. 진행 중단.")
    selected = set(fl.loc[fl["refined_label"] == "POLICY", "chunk_id"])
    mask = np.fromiter((i in selected for i in ids_all), dtype=bool, count=len(ids_all))
    x = x_all[mask]
    ids = ids_all[mask]
    meta = md.loc[mask].reset_index(drop=True)
    if len(x) != len(selected):
        raise ValueError(f"선별 표본 불일치: NPZ {len(x):,}, 필터 {len(selected):,}")
    print(f"POLICY {len(x):,}개 / K={model.n_clusters} / 로드 {time.perf_counter()-start:.2f}초",flush=True)

    print("[2/5] 군집별 중심 근접 5개 + 무작위 5개 선정",flush=True)
    start = time.perf_counter()
    labels = model.predict(x)
    centers = model.cluster_centers_
    rows = []
    group_info = {}
    for k in range(model.n_clusters):
        pos = np.flatnonzero(labels == k)
        if not len(pos):
            continue
        distance = np.sum((x[pos] - centers[k]) ** 2, axis=1)
        order = np.argsort(distance)
        rep_idx = pos[order[:min(REP_N,len(pos))]]
        pool = np.setdiff1d(pos, rep_idx, assume_unique=True)
        random_idx = rng.choice(pool,size=min(RANDOM_N,len(pool)),replace=False) if len(pool) else np.array([],dtype=int)
        group_info[k] = {"size":len(pos),"indices":np.r_[rep_idx,random_idx]}
        for typ, part in (("대표",rep_idx),("무작위",random_idx)):
            for rank, ix in enumerate(part,1):
                m = meta.iloc[ix]
                rows.append({"cluster_id":k,"sample_size":len(pos),"sampling_type":typ,"rank":rank,
                             "chunk_id":ids[ix],"date":m.get("date",""),"assembly":m.get("assembly",""),
                             "speaker":m.get("speaker",""),"agenda":m.get("agenda",""),
                             "distance_to_center":round(float(np.sum((x[ix]-centers[k])**2)),5),
                             "_ix":int(ix)})
    print(f"선정 {len(rows):,}개 / {time.perf_counter()-start:.2f}초",flush=True)

    print("[3/5] ChromaDB에서 원문 조회",flush=True)
    start = time.perf_counter()
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    fetched = {}
    all_ids = [r["chunk_id"] for r in rows]
    for offset in range(0,len(all_ids),GET_BATCH):
        chunk = all_ids[offset:offset+GET_BATCH]
        result = collection.get(ids=chunk,include=["documents"])
        fetched.update(zip(result["ids"],result["documents"] or []))
        print(f"  조회 {min(offset+GET_BATCH,len(all_ids)):,}/{len(all_ids):,}",flush=True)
    missing = 0
    for r in rows:
        full_text = fetched.get(r["chunk_id"]) or ""
        if not full_text: missing += 1
        r["excerpt_350"] = re.sub(r"\s+"," ",full_text).strip()[:350]
        r["human_topic"] = ""  # 사람이 검토할 주제명
        r["topic_match"] = ""  # 일치 / 부분일치 / 불일치 / 판단보류
        r["review_note"] = ""
    print(f"원문 누락 {missing}개 / 조회 {time.perf_counter()-start:.2f}초",flush=True)

    print("[4/5] 군집 요약·일관성 참고 지표",flush=True)
    start = time.perf_counter()
    review = pd.DataFrame(rows)
    summary = []
    for k, info in group_info.items():
        part = review[review.cluster_id==k]
        indices = info["indices"]
        docs = [fetched.get(ids[ix],"") or "" for ix in indices]
        centroid_norm = float(np.linalg.norm(centers[k]))
        summary.append({
            "cluster_id":k,
            "policy_sample_size":info["size"],
            "representative_count":int((part.sampling_type=="대표").sum()),
            "random_count":int((part.sampling_type=="무작위").sum()),
            "mean_pairwise_cosine_10":round(cosine_pair_mean(x[indices]),4),
            "center_norm":round(centroid_norm,4),
            "candidate_keywords":keywords(docs),
            "distinct_assemblies_in_10":len(set(part.assembly)),
            "proposed_topic":"",   # 대표·무작위 발언 읽고 입력
            "consistency_rating":"", # 높음/보통/낮음/판단보류
            "review_note":""
        })
    summary_df = pd.DataFrame(summary).sort_values("cluster_id")
    review = review.drop(columns=["_ix"])
    detail_file = OUT / "policy_cluster_utterances_k100.csv"
    summary_file = OUT / "policy_cluster_topic_review_k100.csv"
    review.to_csv(detail_file,index=False,encoding="utf-8-sig")
    summary_df.to_csv(summary_file,index=False,encoding="utf-8-sig")
    print(f"요약 {len(summary_df)}군집 / 발언 {len(review)}개 / 저장 {time.perf_counter()-start:.2f}초",flush=True)

    print("[5/5] 검토 안내 저장",flush=True)
    instructions = {
        "purpose":"K=100 POLICY 군집의 정책 주제 일관성 점검",
        "sample_size":len(x),"clusters":len(summary_df),"inspected_utterances":len(review),
        "missing_documents":missing,
        "meaning":"mean_pairwise_cosine_10은 참고 지표이며 주제 일관성 점수가 아닙니다",
        "human_review":"두 CSV에서 대표/무작위 발언을 확인하고 proposed_topic 및 consistency_rating을 기록하세요",
        "limits":"임베딩 유사도와 자동 키워드는 정책적 동일성을 보장하지 않습니다. REVIEW 청크는 이번 학습에 포함되지 않습니다.",
        "total_sec":round(time.perf_counter()-t_all,2),
        "created_at":datetime.now().isoformat(timespec="seconds")
    }
    with (OUT/"inspection_summary.json").open("w",encoding="utf-8") as f:
        json.dump(instructions,f,ensure_ascii=False,indent=2)
    print("요약:",summary_file,flush=True)
    print("발언:",detail_file,flush=True)
    print(f"전체 소요 {instructions['total_sec']}초 / 종료 {instructions['created_at']}",flush=True)

if __name__ == "__main__":
    main()

"""K=100 MiniBatchKMeans 군집 대표 발언 및 키워드 후보 점검.
기존 ChromaDB를 수정하지 않습니다. 이 단계는 토픽 확정/c-TF-IDF가 아닙니다.
"""
import ast
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

import chromadb
import joblib
import numpy as np

ROOT = Path(r"C:\mini_team\scripts\clustering\results")
NPZ_FILE = ROOT / "cluster_sample_balanced.npz"
META_FILE = ROOT / "cluster_sample_balanced_metadata.csv"
MODEL_FILE = ROOT / "kmeans_models" / "minibatch_kmeans_k100.joblib"
DB_PATH = Path(r"C:\mini_team\_db\chroma_snowflake_regions")
COLLECTION_NAME = "team_regions_6a0a006e92d1e93e"
OUT_DIR = ROOT / "cluster_inspection"
REPRESENTATIVES_PER_CLUSTER = 5
KEYWORDS_PER_CLUSTER = 8
FETCH_BATCH = 100
STOPWORDS = {
    "그리고", "그런데", "그러나", "따라서", "있습니다", "없습니다", "합니다", "됩니다",
    "대한", "관련", "이번", "저희", "우리", "위원", "위원님", "의원", "의원님",
    "회의", "발언", "말씀", "감사", "감사합니다", "생각합니다", "것입니다",
    "위해서", "대해서", "하도록", "이러한", "것으로", "있도록", "있으며", "부분",
    "사항", "경우", "지금", "오늘", "어떤", "통해서", "이렇게", "많은", "대하여",
    "입니다", "이라고", "이런", "있는", "하는", "것을", "정말", "정도", "때문에",
}


def parse_keywords(value):
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parsed = None
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
        return [x.strip() for x in value.split(",") if x.strip()]
    return []


def clean_text(text, max_chars=280):
    return re.sub(r"\s+", " ", text or "").strip()[:max_chars]


def words_from_text(text):
    # 형태소 분석이 아닌 단순 후보 추출: 실제 토픽 명명 전 검토 필수.
    for token in re.findall(r"[가-힣A-Za-z]{2,}", text or ""):
        token = token.lower()
        if token not in STOPWORDS and len(token) >= 2:
            yield token


def main():
    start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in (NPZ_FILE, META_FILE, MODEL_FILE, DB_PATH / "chroma.sqlite3"):
        if not p.is_file():
            raise FileNotFoundError(f"필요 파일이 없습니다: {p}")

    print("[1/5] 임베딩 표본·모델 로드")
    with np.load(NPZ_FILE, allow_pickle=False) as npz:
        X = np.asarray(npz["embeddings"], dtype=np.float32)
        ids = [str(x) for x in npz["ids"]]
    with META_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(ids) != len(rows) or len(ids) != len(X):
        raise ValueError("NPZ와 메타데이터 CSV 행 개수가 서로 다릅니다.")
    if any(a != b["chunk_id"] for a, b in zip(ids, rows)):
        raise ValueError("NPZ와 메타데이터 CSV의 chunk_id 순서가 다릅니다.")
    model = joblib.load(MODEL_FILE)  # 직접 생성한 신뢰할 수 있는 모델 파일만 로드
    if X.shape[1] != model.cluster_centers_.shape[1]:
        raise ValueError("모델과 표본 임베딩의 차원이 일치하지 않습니다.")
    labels = model.predict(X)
    k = int(model.n_clusters)
    print(f"표본: {len(X):,}개 / 차원: {X.shape[1]} / 군집: {k}")

    print("[2/5] 군집 중심과 가까운 대표 청크 선택")
    picks = []
    cluster_counts = Counter(labels.tolist())
    for cluster_id in range(k):
        indices = np.flatnonzero(labels == cluster_id)
        if len(indices) == 0:
            continue
        center = model.cluster_centers_[cluster_id]
        # MiniBatchKMeans와 동일한 유클리드 거리 기준으로 대표를 선정
        distances = np.sum((X[indices] - center) ** 2, axis=1)
        take = min(REPRESENTATIVES_PER_CLUSTER, len(indices))
        near = np.argsort(distances)[:take]
        for rank, pos in enumerate(near, 1):
            idx = int(indices[pos])
            picks.append({
                "cluster_id": cluster_id,
                "rank": rank,
                "chunk_id": ids[idx],
                "squared_distance": float(distances[pos]),
                "sample_info": rows[idx],
            })

    print(f"[3/5] ChromaDB에서 대표 발언 {len(picks):,}개 조회")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    found = {}
    wanted = [p["chunk_id"] for p in picks]
    for i in range(0, len(wanted), FETCH_BATCH):
        batch = wanted[i:i + FETCH_BATCH]
        result = collection.get(ids=batch, include=["documents", "metadatas"])
        for chunk_id, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
            found[str(chunk_id)] = (doc or "", meta or {})
        print(f"  조회: {min(i + FETCH_BATCH, len(wanted)):,}/{len(wanted):,}")

    print("[4/5] 군집별 요약과 대표 발언 저장")
    details = []
    tokens_by_cluster = {i: Counter() for i in range(k)}
    places_by_cluster = {i: set() for i in range(k)}
    for p in picks:
        cid = p["cluster_id"]
        doc, meta = found.get(p["chunk_id"], ("", {}))
        source = p["sample_info"]
        places_by_cluster[cid].add((str(meta.get("source_key") or source.get("source_key") or "")))
        kw_list = parse_keywords(meta.get("keywords"))
        for kw in kw_list:
            if kw and kw.lower() not in STOPWORDS:
                tokens_by_cluster[cid][kw] += 2
        tokens_by_cluster[cid].update(words_from_text(doc))
        details.append({
            "cluster_id": cid,
            "representative_rank": p["rank"],
            "cluster_sample_size": cluster_counts[cid],
            "chunk_id": p["chunk_id"],
            "squared_distance_to_center": round(p["squared_distance"], 6),
            "date": meta.get("date", source.get("date", "")),
            "assembly": meta.get("assembly", source.get("assembly", "")),
            "speaker": meta.get("speaker", source.get("speaker", "")),
            "agenda": meta.get("agenda", source.get("agenda", "")),
            "source_file": meta.get("source_file", source.get("source_file", "")),
            "source_key": meta.get("source_key", source.get("source_key", "")),
            "excerpt_280": clean_text(doc),
            "text_missing": not bool(doc),
        })

    details_path = OUT_DIR / "cluster_representative_utterances_k100.csv"
    with details_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)

    summary = []
    for cid in range(k):
        kw = [w for w, _ in tokens_by_cluster[cid].most_common(KEYWORDS_PER_CLUSTER)]
        summary.append({
            "cluster_id": cid,
            "sample_chunk_count": cluster_counts[cid],
            "representative_count": sum(p["cluster_id"] == cid for p in picks),
            "representative_region_year_count": len(places_by_cluster[cid]),
            "keyword_candidates": ", ".join(kw),
            "topic_name_review": "",  # 사람이 대표 발언을 읽고 입력
            "quality_review": "",  # 예: 명확 / 혼합 / 불명확
        })
    summary_path = OUT_DIR / "cluster_topic_review_k100.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    missing = sum(d["text_missing"] for d in details)
    print("[5/5] 완료")
    print("군집 요약:", summary_path)
    print("대표 발언:", details_path)
    print("원문 누락:", missing)
    print("전체 소요 시간:", round(time.perf_counter() - start, 2), "초")
    print("주의: 키워드는 형태소 분석/정식 c-TF-IDF 결과가 아닌 대표 발언 기반 후보입니다.")


if __name__ == "__main__":
    main()

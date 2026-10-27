"""34개 지역·연도 균형 표본(68,000개) MiniBatchKMeans 군집 수 비교.
기존 ChromaDB 읽기/쓰기 없음. NPZ/CSV만 읽어 결과 저장.
실행: python 03_train_minibatch_kmeans.py
설치: python -m pip install scikit-learn joblib pandas threadpoolctl
"""
from pathlib import Path
from datetime import datetime
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

BASE = Path(r"C:\mini_team\scripts\clustering\results")
NPZ = BASE / "cluster_sample_balanced.npz"
META = BASE / "cluster_sample_balanced_metadata.csv"
OUT = BASE / "kmeans_models"
K_VALUES = (100, 200, 300)
RANDOM_STATE = 42
SIL_SAMPLE = 1200  # 계산 부담을 줄이기 위한 고정 표본
BATCH_SIZE = 2048


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start_all = time.perf_counter()
    print("시작:", datetime.now())
    with np.load(NPZ) as data:
        X = data["embeddings"].astype(np.float32, copy=False)
        ids = data["ids"].astype(str)
    meta = pd.read_csv(META, encoding="utf-8-sig", low_memory=False)
    assert len(X) == len(ids) == len(meta), "임베딩/메타데이터 개수가 다릅니다."
    assert np.array_equal(ids, meta["chunk_id"].astype(str).to_numpy()), "청크 ID 순서가 다릅니다."
    assert X.shape[1] == 1024 and np.isfinite(X).all(), "임베딩 차원 또는 수치 오류"
    print("데이터:", X.shape, "| 그룹:", meta["source_key"].nunique())
    print("참고: 군집 수 선정은 실루엣 점수뿐 아니라 정책 주제의 해석 가능성도 검토하세요.")

    rng = np.random.default_rng(RANDOM_STATE)
    eval_idx = rng.choice(len(X), size=min(SIL_SAMPLE, len(X)), replace=False)
    results = []

    for k in K_VALUES:
        print(f"\n=== K={k} 학습 시작 ===", flush=True)
        tic = time.perf_counter()
        model = MiniBatchKMeans(
            n_clusters=k,
            batch_size=BATCH_SIZE,
            n_init=3,
            max_iter=100,
            random_state=RANDOM_STATE,
            reassignment_ratio=0.01,
        )
        # Windows에서 과도한 BLAS 스레드 경쟁을 방지
        with threadpool_limits(limits=4, user_api="blas"):
            labels = model.fit_predict(X)
            train_sec = time.perf_counter() - tic
            counts = np.bincount(labels, minlength=k)
            # 정규화된 임베딩에는 코사인 실루엣을 참고 지표로 사용
            score = silhouette_score(
                X[eval_idx], labels[eval_idx], metric="cosine"
            ) if len(np.unique(labels[eval_idx])) > 1 else float("nan")

        assigned = meta[["chunk_id", "source_key", "date", "assembly"]].copy()
        assigned["cluster_id"] = labels
        assigned.to_csv(OUT / f"sample_clusters_k{k}.csv", index=False, encoding="utf-8-sig")
        joblib.dump(model, OUT / f"minibatch_kmeans_k{k}.joblib")
        pd.DataFrame({"cluster_id": np.arange(k), "sample_count": counts}).to_csv(
            OUT / f"cluster_sizes_k{k}.csv", index=False, encoding="utf-8-sig"
        )
        row = {
            "k": k,
            "sample_count": len(X),
            "used_clusters": int((counts > 0).sum()),
            "min_cluster_size": int(counts.min()),
            "max_cluster_size": int(counts.max()),
            "clusters_under_10": int((counts < 10).sum()),
            "silhouette_cosine_1200": round(float(score), 5),
            "inertia": round(float(model.inertia_), 3),
            "train_and_evaluation_sec": round(time.perf_counter() - tic, 2),
            "model_path": str(OUT / f"minibatch_kmeans_k{k}.joblib"),
        }
        results.append(row)
        pd.DataFrame(results).to_csv(OUT / "kmeans_comparison.csv", index=False, encoding="utf-8-sig")
        print(row, flush=True)

    print("\n완료:", OUT / "kmeans_comparison.csv")
    print("전체 소요 시간:", round(time.perf_counter() - start_all, 2), "초")
    print("종료:", datetime.now())


if __name__ == "__main__":
    main()

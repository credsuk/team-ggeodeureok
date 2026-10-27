"""08단계: POLICY 표본 MiniBatchKMeans 재학습 및 기존 모델과 공통 표본 비교.
입력: cluster_sample_balanced.npz, policy_filter_68000.csv, 기존 KMeans 모델(선택)
출력: clustering/results/policy_kmeans/ 아래 CSV/모델
기존 ChromaDB나 기존 모델 파일은 수정하지 않습니다.
"""
from datetime import datetime
from pathlib import Path
import csv
import json
import time

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

ROOT = Path(r"C:\mini_team\scripts\clustering\results")
NPZ_PATH = ROOT / "cluster_sample_balanced.npz"
FILTER_CSV = ROOT / "policy_refined" / "policy_filter_68000.csv"
OLD_MODEL_DIR = ROOT / "kmeans_models"
OUT = ROOT / "policy_kmeans"
KS = (100, 200, 300)
EVAL_COUNT = 1200
SEED = 42
BATCH_SIZE = 2048
# PC 과부하 방지. 필요하면 4~8로 변경
THREADS = 4


def save_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def metric(x, labels):
    sizes = np.bincount(labels, minlength=int(labels.max()) + 1)
    # 유효 군집이 2개 이상, 각 데이터 수보다 적어야 계산 가능
    if 2 <= np.unique(labels).size < len(labels):
        score = float(silhouette_score(x, labels, metric="cosine"))
    else:
        score = float("nan")
    return score, sizes


def main():
    started = time.perf_counter()
    print("시작:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (NPZ_PATH, FILTER_CSV):
        if not path.is_file():
            raise FileNotFoundError(f"입력 파일 없음: {path}")

    t = time.perf_counter()
    with np.load(NPZ_PATH) as data:
        vectors = np.asarray(data["embeddings"], dtype=np.float32)
        ids = data["ids"].astype(str)
    if vectors.ndim != 2 or vectors.shape[1] != 1024 or len(ids) != len(vectors):
        raise ValueError("임베딩 차원 또는 ID 개수 불일치")
    if len(set(ids)) != len(ids) or not np.isfinite(vectors).all():
        raise ValueError("중복 ID 또는 비정상 임베딩 값 발견")
    print(f"[1/4] 입력 벡터 {vectors.shape}, 로딩 {time.perf_counter()-t:.2f}초", flush=True)

    # 원본 NPZ의 ID 순서를 기준으로 필터와 정확히 매칭 (CSV 행 순서 가정 금지)
    t = time.perf_counter()
    labels_by_id = {}
    with FILTER_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = row["chunk_id"]
            if key in labels_by_id:
                raise ValueError(f"필터 CSV 중복 chunk_id: {key}")
            labels_by_id[key] = row["refined_label"]
    missing = [key for key in ids if key not in labels_by_id]
    if missing:
        raise ValueError(f"필터에 누락된 ID가 {len(missing)}개 있습니다. 예: {missing[:3]}")
    selected_indices = np.fromiter(
        (i for i, key in enumerate(ids) if labels_by_id[key] == "POLICY"), dtype=np.int64
    )
    x = np.ascontiguousarray(vectors[selected_indices], dtype=np.float32)
    policy_ids = ids[selected_indices]
    print(f"[2/4] POLICY {len(x):,}개 선택 ({len(x)/len(ids):.1%}), 매칭 {time.perf_counter()-t:.2f}초", flush=True)
    if len(x) <= max(KS):
        raise ValueError("선정 정책 청크 수가 군집 수보다 적습니다")
    if not np.isfinite(x).all():
        raise ValueError("POLICY 표본에 NaN/Inf 포함")

    rng = np.random.default_rng(SEED)
    eval_indices = rng.choice(len(x), size=min(EVAL_COUNT, len(x)), replace=False)
    eval_x = x[eval_indices]
    save_csv(OUT / "evaluation_sample_ids.csv",
             [{"chunk_id": str(policy_ids[i])} for i in eval_indices], ["chunk_id"])
    rows = []

    with threadpool_limits(limits=THREADS):
        for k in KS:
            print(f"\n[3/4] K={k} 정책 군집 학습", flush=True)
            t = time.perf_counter()
            model = MiniBatchKMeans(
                n_clusters=k, batch_size=BATCH_SIZE, random_state=SEED,
                n_init=3, max_iter=100, reassignment_ratio=0.01,
            )
            model.fit(x)
            fit_sec = time.perf_counter() - t
            all_labels = model.predict(x)
            new_score, _ = metric(eval_x, model.predict(eval_x))
            sizes = np.bincount(all_labels, minlength=k)
            model_file = OUT / f"policy_minibatch_kmeans_k{k}.joblib"
            joblib.dump(model, model_file)
            save_csv(OUT / f"policy_cluster_sizes_k{k}.csv",
                     [{"cluster_id": i, "sample_count": int(n)} for i, n in enumerate(sizes)],
                     ["cluster_id", "sample_count"])
            save_csv(OUT / f"policy_sample_clusters_k{k}.csv",
                     [{"chunk_id": str(cid), "cluster_id": int(label)}
                      for cid, label in zip(policy_ids, all_labels)],
                     ["chunk_id", "cluster_id"])
            row = dict(
                k=k, policy_count=len(x), used_clusters=int((sizes > 0).sum()),
                min_cluster_size=int(sizes.min()), max_cluster_size=int(sizes.max()),
                clusters_under_10=int((sizes < 10).sum()),
                silhouette_cosine_same_policy_1200=round(new_score, 5),
                fit_sec=round(fit_sec, 3), model_path=str(model_file),
                baseline_silhouette_same_policy_1200="", silhouette_delta="",
            )
            # 기존 모델과 동일한 POLICY 평가 청크로 계산: 직접 비교 가능
            old_path = OLD_MODEL_DIR / f"minibatch_kmeans_k{k}.joblib"
            if old_path.is_file():
                old_model = joblib.load(old_path)
                old_score, _ = metric(eval_x, old_model.predict(eval_x))
                row["baseline_silhouette_same_policy_1200"] = round(old_score, 5)
                row["silhouette_delta"] = round(new_score - old_score, 5)
            else:
                print(f"  기존 모델 없음: {old_path} (기준 비교 생략)", flush=True)
            rows.append(row)
            print(f"  {row} / 단계 {time.perf_counter()-t:.2f}초", flush=True)
            save_csv(OUT / "policy_kmeans_comparison.csv", rows, list(row))

    elapsed = time.perf_counter() - started
    summary = {
        "started_on": datetime.now().isoformat(timespec="seconds"),
        "original_sample_count": len(ids), "policy_count": len(x),
        "eval_count": len(eval_x), "seed": SEED, "batch_size": BATCH_SIZE,
        "total_sec": round(elapsed, 3), "results": rows,
        "note": "규칙기반 POLICY만 학습. 동일 POLICY 평가 표본에서 기존/신규 모델을 비교."
    }
    (OUT / "policy_kmeans_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[4/4] 완료: {OUT / 'policy_kmeans_comparison.csv'}")
    print(f"총 소요 시간: {elapsed:.2f}초 / 종료: {datetime.now()}")
    print("주의: 실루엣 개선이 정책 주제 해석 품질 향상을 보장하지는 않습니다.")


if __name__ == "__main__":
    main()

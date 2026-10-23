"""Return ranking metrics for the top-k retrieved source IDs.

Precision@k uses k as its denominator even when fewer than k candidates are returned.
"""


def ranking_metrics(ranked_ids, relevant_ids, k=10):
    """Return precision@k, recall@k, reciprocal rank, and hit@k."""
    if k < 1:
        raise ValueError("k must be at least 1")
    ranked = list(ranked_ids)[:k]
    relevant = set(relevant_ids)
    hits = [index for index, item in enumerate(ranked, 1) if item in relevant]
    precision = len(hits) / k
    recall = len({item for item in ranked if item in relevant}) / max(len(relevant), 1)
    return {
        "k": int(k), "precision_at_k": precision, "recall_at_k": recall,
        "reciprocal_rank": 1.0 / hits[0] if hits else 0.0,
        "hit_at_k": int(bool(hits)),
    }

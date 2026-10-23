"""Cross-Encoder ranking for read-only Chroma retrieval candidates."""
from sentence_transformers import CrossEncoder
from _constant import RERANKER_MODEL

# Initial cutoff for the multilingual reranker; calibrate with reviewed examples.
RERANKER_MIN_SCORE = 0.0


class DocumentReranker:
    def __init__(self, model_name=RERANKER_MODEL):
        self.model = CrossEncoder(model_name, max_length=512)
        self.model_name = model_name

    def rank_and_filter(self, query, chunks, top_n=100, min_score=RERANKER_MIN_SCORE):
        if not chunks:
            return []
        pairs = [[query, row["text"]] for row in chunks]
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)
        ranked = []
        for row, score in zip(chunks, scores):
            row["rerank_score"] = float(score)
            if min_score is None or row["rerank_score"] >= min_score:
                ranked.append(row)
        ranked.sort(key=lambda row: row["rerank_score"], reverse=True)
        return ranked if top_n is None else ranked[:top_n]

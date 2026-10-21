"""Optional cross-encoder ranking; keeps separate meetings from one council."""
from sentence_transformers import CrossEncoder
from _constant import RERANKER_MODEL


class DocumentReranker:
    def __init__(self, model_name=RERANKER_MODEL):
        self.model = CrossEncoder(model_name)

    def rank_and_filter(self, query, chunks, top_n=100):
        if not chunks:
            return []
        scores = self.model.predict([[query, row["text"]] for row in chunks])
        best_by_meeting = {}
        for row, score in zip(chunks, scores):
            row["rerank_score"] = float(score)
            meta = row.get("metadata", {})
            key = (meta.get("dataset_key"), meta.get("assembly"), meta.get("date"), meta.get("agenda_title"))
            if not meta.get("date"):
                key = (row.get("id"),)
            if key not in best_by_meeting or row["rerank_score"] > best_by_meeting[key]["rerank_score"]:
                best_by_meeting[key] = row
        ranked = sorted(best_by_meeting.values(), key=lambda row: row["rerank_score"], reverse=True)
        if top_n is not None:
            ranked = ranked[:top_n]
        return sorted(ranked, key=lambda row: (str(row.get("metadata", {}).get("date", "99999999")),
                                               str(row.get("metadata", {}).get("assembly", ""))))

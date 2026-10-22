"""Orchestrate read-only Chroma retrieval, ranking, evidence, and dashboard output."""
import json
import re
from collections import Counter

from llm_engine import LLMEngine
from retriever import HybridRetriever
from reranker import DocumentReranker
from schema import FinalOutput, Forecast, IssueResult, MeetingReference, SupplementaryMetrics


def _date(meta):
    return str(meta.get("date") or meta.get("MTG_DE") or "")


def _council(meta):
    return str(meta.get("assembly") or meta.get("RASMBLY_NM") or "미상")


def _agenda(meta):
    return str(meta.get("agenda_title") or meta.get("agenda") or meta.get("MTGNM") or "")


def _ref(row):
    meta = row["metadata"]
    text = row.get("text", "").strip()
    excerpt = text[:1200] + ("…" if len(text) > 1200 else "")
    return MeetingReference(
        sourceId=row["id"], date=_date(meta), council=_council(meta),
        agendaTitle=_agenda(meta), excerpt=excerpt,
        chunkIds=row.get("chunk_ids", []),
        relevanceScore=row.get("rerank_score"),
    )


def _day_key(row):
    meta = row["metadata"]
    council = re.sub(r"[\W_]+", "", _council(meta).lower(), flags=re.UNICODE)
    return council, _date(meta)


def _daily_representatives(rows):
    best = {}
    for row in rows:
        key = _day_key(row)
        if not key[1]:
            continue
        score = float(row.get("rerank_score", float("-inf")))
        if key not in best or score > float(best[key].get("rerank_score", float("-inf"))):
            best[key] = row
    return sorted(best.values(), key=lambda row: (_date(row["metadata"]), _council(row["metadata"])))


def _metrics(rows):
    years = Counter()
    councils = Counter()
    for row in rows:
        date = _date(row["metadata"])
        year = re.search(r"20\d{2}", date)
        if year:
            years[year.group(0)] += 1
        councils[_council(row["metadata"])] += 1
    total = len(rows)
    top = councils.most_common(1)[0] if councils else (None, 0)
    return SupplementaryMetrics(
        label="검색 및 관련성 검증을 통과한 결과 범위의 참고값",
        retrievedCouncilDays=total,
        councilsDiscussing=len(councils),
        yearBreakdown=dict(sorted(years.items())),
        topCouncil=top[0],
        topCouncilSharePercent=round(top[1] * 100 / total, 1) if total else None,
    )


class LocalRAGPipeline:
    def __init__(self):
        self.llm = LLMEngine()
        self.retriever = HybridRetriever()
        self.reranker = DocumentReranker()

    def run(self, user_query):
        params = self.llm.extract_search_params(user_query)
        candidates = self.retriever.search(params, top_k=100)
        reranked = self.reranker.rank_and_filter(params["query"], candidates, top_n=100, min_score=0.0)
        expanded = self.retriever.expand_rows(reranked)
        expanded.sort(key=lambda row: float(row.get("rerank_score", float("-inf"))), reverse=True)
        daily_rows = _daily_representatives(expanded)
        metrics = _metrics(daily_rows)

        # Keep the language model input bounded while retaining the best evidence from each council-day.
        prompt_rows = sorted(daily_rows, key=lambda row: float(row.get("rerank_score", 0)), reverse=True)[:30]
        evidence_payload = [
            {"source_id": row["id"], "date": _date(row["metadata"]),
             "council": _council(row["metadata"]), "agenda_title": _agenda(row["metadata"]),
             "excerpt": row.get("text", "")[:1200]}
            for row in prompt_rows
        ]
        analysis = self.llm.synthesize_analysis(params, evidence_payload, metrics.model_dump())

        # Scan all explicit topic-term hits in Chroma for the earliest observed date.
        historical_chunks = self.retriever.scan_all_lexical(params)
        historical_rows = self.retriever.expand_rows(historical_chunks)
        historical_ranked = self.reranker.rank_and_filter(
            params["query"], historical_rows, top_n=None, min_score=0.0
        )
        timeline_rows = _daily_representatives([*expanded, *historical_ranked])
        dates = [_date(row["metadata"]) for row in timeline_rows if _date(row["metadata"])]
        first_date = min(dates) if dates else None
        first_rows = [row for row in timeline_rows if _date(row["metadata"]) == first_date] if first_date else []
        later_rows = [row for row in timeline_rows if first_date and _date(row["metadata"]) > first_date][:100]
        all_refs = [_ref(row) for row in daily_rows]
        # Keep forecasts qualitative and visibly uncertain when the retrieved evidence is sparse.
        if len(daily_rows) < 3:
            forecast_text = "검색된 근거 자료가 적어 향후 논의 의제를 구체적으로 전망하기 어렵습니다."
            forecast_status, forecast_confidence = "insufficient_evidence", "low"
            forecast_ids = []
        else:
            forecast_text = analysis["forecast"]
            forecast_ids = analysis["forecast_evidence_ids"]
            forecast_status = "supported" if forecast_ids and analysis["forecast_status"] == "supported" else "insufficient_evidence"
            forecast_confidence = "medium" if len({row["metadata"].get("assembly", row["metadata"].get("RASMBLY_NM")) for row in daily_rows}) >= 3 else "low"
            if not forecast_ids:
                forecast_text = "검색 근거와 연결되는 향후 의제를 제시하기 어려워 전망 신뢰도가 낮습니다."
                forecast_confidence = "low"

        result = IssueResult(
            issueKeyword=params["topic"],
            summary=analysis["summary"],
            summaryEvidenceIds=analysis["summary_evidence_ids"],
            firstObserved=[_ref(row) for row in first_rows],
            subsequentDiscussions=[_ref(row) for row in later_rows],
            supplementaryMetrics=metrics,
            forecast=Forecast(text=forecast_text, status=forecast_status,
                              confidence=forecast_confidence, evidenceIds=forecast_ids),
            evidence=all_refs,
        )
        output = FinalOutput(
            dataPeriod="2024–2025 (벡터 DB 자료 기준)",
            dataScope=("벡터 DB에 저장된 특별시·광역시·도 의회 회의록만 사용했습니다. "
                       "최초 관측은 명시적 주제어가 포함된 DB 자료를 전수 검색하고 Cross-Encoder로 검증한 범위입니다. " +
                       "보조 통계는 상위 검색 결과 범위의 참고값입니다. "
                       "시간순 논의는 영향 관계를 뜻하지 않습니다."),
            results=[result],
        )
        return output.model_dump()


if __name__ == "__main__":
    print(json.dumps(LocalRAGPipeline().run("지역 소멸에 대한 의회 논의 현황 분석해줘"),
                     ensure_ascii=False, indent=2))






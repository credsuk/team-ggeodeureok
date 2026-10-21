"""Produce grounded issue/appearance/regional-flow/evidence output."""
import json

from llm_engine import LLMEngine
from retriever import HybridRetriever
from schema import FinalOutput


def _ref(event):
    return {"date": str(event.get("date", "")), "council": str(event.get("assembly", "미상")),
            "agendaTitle": str(event.get("agenda_title", "")), "excerpt": str(event.get("evidence", ""))}


class LocalRAGPipeline:
    def __init__(self):
        self.llm = LLMEngine()
        self.retriever = HybridRetriever()

    def run(self, user_query):
        params = self.llm.extract_search_params(user_query)
        events = self.retriever.matching_events(params["keyword"], params["start_year"], params["end_year"])
        events = self.retriever.evidence_for_events(params["keyword"], events)
        # Grouping is a date/council/agenda proxy because source TSV has no meeting identifier.
        valid = [e for e in events if e["evidence"]]
        first_date = valid[0]["date"] if valid else None
        first_events = [e for e in valid if e["date"] == first_date]
        first_councils = {e["assembly"] for e in first_events}
        appearance = [_ref(e) for e in first_events]
        # Do not infer an intraday order; flow contains later observations in other councils.
        flow = [_ref(e) for e in valid if e["date"] != first_date and e["assembly"] not in first_councils]
        councils = {e["assembly"] for e in valid}
        dates = {e["date"] for e in valid}
        if len(valid) < 3 or len(councils) < 2 or len(dates) < 2:
            forecast = "자료에서 확인된 논의 회의가 적거나 지역·시점 범위가 좁아 향후 확산 여부를 예측하기 어렵습니다."
        else:
            forecast = ("자료에서는 여러 회의·지역에서 논의가 관측됩니다. 이는 과거 자료의 분포를 요약한 것이며, "
                        "향후 지속 또는 확산을 확정하는 예측은 아닙니다.")
        result = FinalOutput(results=[{
            "issueKeyword": params["keyword"], "appearance": appearance,
            "regionalFlow": flow, "forecast": forecast,
            "evidence": [_ref(e) for e in valid],
        }], dataScope=("설정된 데이터셋의 JSONL 메타데이터에서 검색어가 core_issue/summary/agenda/keywords에 "
                       "포함된 레코드를 날짜·의회·안건 단위로 묶었습니다. "
                       "샘플 파일만 설정된 경우 전체 의회 기록을 대표하지 않습니다."))
        return result.model_dump()


if __name__ == "__main__":
    print(json.dumps(LocalRAGPipeline().run("최근 의회에서 논의된 지역 소멸 관련 현황을 분석해줘"), ensure_ascii=False, indent=2))

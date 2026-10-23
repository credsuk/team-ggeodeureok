"""Orchestrate read-only Chroma retrieval, ranking, evidence, and dashboard output."""
import json
import os
import re
import time
from collections import Counter

from llm_engine import LLMEngine
from retriever import HybridRetriever
from reranker import DocumentReranker
from schema import FinalOutput, Forecast, IssueResult, MeetingReference, SupplementaryMetrics
from retrieval_utils import normalize_date_key
from analysis_scope import apply_analysis_period, ANALYSIS_START_YEAR, ANALYSIS_END_YEAR
from source_reference import reference_metadata
from output_validation import validate_output_evidence


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
    citation = reference_metadata(meta)
    row_id = citation["originalRowId"]
    return MeetingReference(
        sourceId=citation["sourceId"] or row["id"], date=_date(meta), council=_council(meta),
        meetingTitle=citation["meetingTitle"], agendaTitle=citation["agendaTitle"],
        datasetKey=citation["datasetKey"], originalRowId=row_id, excerpt=excerpt,
        chunkIds=row.get("chunk_ids", []), relevanceScore=row.get("rerank_score"),
    )


def _day_key(row):
    meta = row["metadata"]
    council = re.sub(r"[\W_]+", "", _council(meta).lower(), flags=re.UNICODE)
    return council, normalize_date_key(_date(meta))


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
        self.last_error = None
        self.last_output = None

    def run(self, user_query):
        started = time.perf_counter()
        params = {}
        timings = {}
        self.last_timings = timings
        self.last_params = params
        self.last_error = None
        self.last_output = None
        if not str(user_query or "").strip():
            self.last_error = "검색 질문이 비어 있습니다."
            raise ValueError(self.last_error)

        try:
            return self._run_query(user_query, started, timings, params)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            timings["전체 처리"] = time.perf_counter() - started
            self.last_timings = timings
            raise

    def _run_query(self, user_query, started, timings, params):
        step = time.perf_counter()
        params.update(apply_analysis_period(self.llm.extract_search_params(user_query)))
        self.last_params = dict(params)
        timings["질문 분석 LLM"] = time.perf_counter() - step

        step = time.perf_counter()
        candidates = self.retriever.search(params, top_k=100)
        timings["Vector DB 검색"] = time.perf_counter() - step

        step = time.perf_counter()
        reranked = self.reranker.rank_and_filter(params["query"], candidates, top_n=100, min_score=0.0)
        timings["검색 결과 재정렬"] = time.perf_counter() - step

        step = time.perf_counter()
        expanded = self.retriever.expand_rows(reranked)
        expanded.sort(key=lambda row: float(row.get("rerank_score", float("-inf"))), reverse=True)
        daily_rows = _daily_representatives(expanded)
        metrics = _metrics(daily_rows)
        timings["근거 확장 및 통계"] = time.perf_counter() - step

        # Keep the language model input bounded while retaining the best evidence from each council-day.
        prompt_rows = sorted(daily_rows, key=lambda row: float(row.get("rerank_score", 0)), reverse=True)[:30]
        evidence_payload = [
            {"source_id": reference_metadata(row["metadata"])["sourceId"] or row["id"], "date": _date(row["metadata"]),
             "council": _council(row["metadata"]), "meeting_title": reference_metadata(row["metadata"])["meetingTitle"],
             "agenda_title": _agenda(row["metadata"]),
             "excerpt": row.get("text", "")[:1200]}
            for row in prompt_rows
        ]
        step = time.perf_counter()
        analysis = self.llm.synthesize_analysis(params, evidence_payload, metrics.model_dump())
        timings["요약 및 전망 LLM"] = time.perf_counter() - step

        # Search explicit topic-term hits chronologically and stop once an earliest relevant day is found.
        step = time.perf_counter()
        first_date, first_rows, later_rows = self.retriever.first_observed_and_followup_rows(
            params, self.reranker, followup_limit=100
        )
        timings["Vector DB 주제어 검색 및 최초 관측"] = time.perf_counter() - step
        step = time.perf_counter()
        first_rows = _daily_representatives(first_rows)
        later_rows = _daily_representatives(later_rows)[:100]
        if first_date:
            later_rows = [
                row for row in later_rows
                if normalize_date_key(_date(row["metadata"])) > first_date
            ]
        reference_rows_by_id = {row["id"]: row for row in [*daily_rows, *first_rows, *later_rows]}
        all_refs = [_ref(row) for row in reference_rows_by_id.values()]
        timings["최초 관측 후속 논의 정리"] = time.perf_counter() - step

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
            dataPeriod=(f"요청 기간이 분석 범위({ANALYSIS_START_YEAR}–{ANALYSIS_END_YEAR})와 겹치지 않음"
                        if params["start_year"] > params["end_year"] else
                        f"{params['start_year']}–{params['end_year']} (고정 분석 범위 {ANALYSIS_START_YEAR}–{ANALYSIS_END_YEAR}, 색인 DB 내 자료 기준)"),
            dataScope=("특별시·광역시·도 의회 자료만 사용합니다. "
                       "요청한 기간은 분석 범위와 교집합으로 제한하며, 현재 출력은 색인 DB에서 검색된 기록에 한정됩니다. "
                       "최초 관측은 등록된 명시적 검색 표현을 포함하고 Cross-Encoder 기준을 통과한 DB 자료 중 가장 이른 기록입니다. "
                       "이는 제공 전체 데이터 또는 의미상 모든 동의어의 최초를 보장하지 않습니다. "
                       "보조 통계는 검색 결과 범위의 참고값이며 시간순 논의는 영향 관계를 뜻하지 않습니다."),
            results=[result],
        )
        timings["전체 처리"] = time.perf_counter() - started
        self.last_timings = timings
        serialized = output.model_dump()
        validate_output_evidence(serialized)
        self.last_output = serialized
        return serialized


def _print_result(output, timings):
    """Print the dashboard response as readable Korean text instead of raw JSON."""
    print("\n" + "=" * 72)
    for result in output.get("results", []):
        print(f"\n[분석 주제] {result.get('issueKeyword', '미상')}")
        print(f"\n[요약]\n{result.get('summary', '내용 없음')}")
        print(f"\n[최초 관측] {len(result.get('firstObserved', []))}건")
        for ref in result.get("firstObserved", [])[:5]:
            print(f"  • {ref.get('date', '')} | {ref.get('council', '')} | {ref.get('meetingTitle', '')} | {ref.get('agendaTitle', '')}")
        if len(result.get("firstObserved", [])) > 5:
            print(f"  … 외 {len(result['firstObserved']) - 5}건")

        discussions = result.get("subsequentDiscussions", [])
        print(f"\n[후속 논의] {len(discussions)}건")
        for ref in discussions[:8]:
            print(f"  • {ref.get('date', '')} | {ref.get('council', '')} | {ref.get('meetingTitle', '')} | {ref.get('agendaTitle', '')}")
        if len(discussions) > 8:
            print(f"  … 외 {len(discussions) - 8}건")

        metrics = result.get("supplementaryMetrics", {})
        print("\n[참고 통계]")
        print(f"  검색 결과 범위: {metrics.get('retrievedCouncilDays', 0)}건(의회·일) / 의회 {metrics.get('councilsDiscussing', 0)}곳")
        if metrics.get("yearBreakdown"):
            print("  연도별: " + ", ".join(f"{year}년 {count}건" for year, count in metrics["yearBreakdown"].items()))
        if metrics.get("topCouncil"):
            print(f"  가장 많이 논의한 의회: {metrics['topCouncil']} ({metrics.get('topCouncilSharePercent')}%)")

        forecast = result.get("forecast", {})
        print(f"\n[향후 논의 전망: {forecast.get('status', 'unknown')} / 신뢰도 {forecast.get('confidence', 'unknown')}]\n{forecast.get('text', '내용 없음')}")
        print(f"\n[근거 자료] {len(result.get('evidence', []))}건")

    print("\n[처리 시간]")
    for name, seconds in timings.items():
        print(f"  {name:<30} {seconds:8.2f}초")
    print("=" * 72)


def _strip_unavailable_source_ids(value):
    """Remove source ID fields that cannot be trusted/present in the current dataset."""
    if isinstance(value, dict):
        return {
            key: _strip_unavailable_source_ids(item)
            for key, item in value.items()
            if key not in {"sourceId", "source_id", "chunkIds", "chunk_ids"}
        }
    if isinstance(value, list):
        return [_strip_unavailable_source_ids(item) for item in value]
    return value


def _append_history(query, output, timings, params=None, error=None):
    """Append one run's exact input, extracted search parameters, output, and timing data."""
    history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_history.json")
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input": query,
        "extracted_params": params or {},
        "status": "error" if error else "success",
        "error": error,
        "timings_seconds": {name: round(seconds, 4) for name, seconds in timings.items()},
        "output": _strip_unavailable_source_ids(output),
    }
    try:
        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as history_file:
                history = json.load(history_file)
            if not isinstance(history, list):
                raise ValueError("테스트 이력 파일 형식이 배열이 아닙니다")
        else:
            history = []
        history.append(entry)
        temporary_path = history_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as history_file:
            json.dump(history, history_file, ensure_ascii=False, indent=2)
            history_file.write("\n")
        os.replace(temporary_path, history_path)
        print(f"\n이번 질문의 기록을 저장했습니다: {history_path}")
        return True
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"\n기록 파일 저장에 실패했습니다: {exc}")
        return False


def _interactive():
    pipeline = LocalRAGPipeline()
    print("의회 회의록 분석 테스트")
    print("질문을 입력하면 분석합니다. 여러 번 입력할 수 있고, 종료하려면 '종료' 또는 'quit'을 입력하세요.")
    while True:
        try:
            query = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n테스트를 종료합니다.")
            break
        if query.lower() in {"종료", "끝", "quit", "exit", "q"}:
            print("테스트를 종료합니다.")
            break
        if not query:
            print("질문을 입력해 주세요.")
            continue
        try:
            output = pipeline.run(query)
        except Exception as exc:
            print(f"\n분석 중 오류가 발생했습니다: {exc}")
            print("프로그램은 계속 실행됩니다. 다른 질문을 입력하거나 종료할 수 있습니다.")
            _append_history(
                query, None, pipeline.last_timings,
                params=getattr(pipeline, "last_params", {}),
                error=getattr(pipeline, "last_error", None) or f"{type(exc).__name__}: {exc}",
            )
            continue
        _append_history(query, output, pipeline.last_timings, params=pipeline.last_params)
        _print_result(output, pipeline.last_timings)


if __name__ == "__main__":
    _interactive()


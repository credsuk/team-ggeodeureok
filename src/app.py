from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache import AnalysisCache, SemanticQuestionCache
from src.config import Settings
from src.intent import IntentExtractor
from src.llm import OllamaClient
from src.map_view import build_map_points, build_numbered_map_deck
from src.models import AnalysisResult
from src.pipeline import AnalysisPipeline, AnalysisWriter
from src.regional_spread import build_regional_spread
from src.reranking import CrossEncoderReranker
from src.retrieval import load_retriever
from src.snippets import extract_relevant_excerpt
from src.source_index import ensure_source_index

QUESTION_CACHE_COLLECTION = "analysis_question_cache"
APP_STYLE = """<style>
[data-testid="stAppViewContainer"] { background: #ffffff; }
h1, h2, h3 { color: #111111; font-weight: 500; }
[data-testid="stChatMessage"] { border: 1px solid #e5e5e5; border-radius: 12px; }
div.stButton > button, div[data-testid="stChatInput"] textarea { border-radius: 9999px; }
[data-testid="stCaptionContainer"] { color: #737373; }
</style>"""


def create_pipeline(settings: Settings) -> AnalysisPipeline:
    if not settings.allow_model_download:
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("chromadb가 없습니다. src/requirements-app.txt를 설치하세요.") from exc
    if not settings.chroma_dir.is_dir():
        raise FileNotFoundError(f"Chroma DB 폴더가 없습니다: {settings.chroma_dir}")
    if not settings.bm25_db.is_file():
        raise FileNotFoundError(f"BM25 색인이 없습니다: {settings.bm25_db}")
    ensure_source_index(settings.data_dir / "to_text", settings.source_metadata_db)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    retriever = load_retriever(settings, client=client)
    query_collection = client.get_or_create_collection(
        name=QUESTION_CACHE_COLLECTION,
        metadata={"hnsw:space": "cosine", "embedding_model": settings.embedding_model},
    )
    cache_model = (query_collection.metadata or {}).get("embedding_model")
    if cache_model != settings.embedding_model:
        raise RuntimeError(
            f"질문 캐시 임베딩 모델 불일치: 캐시={cache_model!r}, 설정={settings.embedding_model!r}. "
            "캐시 컬렉션을 새 이름으로 분리해야 합니다."
        )
    llm = OllamaClient(settings.ollama_url, settings.ollama_model, keep_alive=settings.ollama_keep_alive)
    cache = AnalysisCache(settings.cache_db)
    semantic_cache = SemanticQuestionCache(
        query_collection,
        similarity_threshold=settings.cache_similarity,
        data_version=settings.data_version,
        prompt_version=settings.prompt_version,
        model_id=settings.ollama_model,
    )
    reranker = CrossEncoderReranker(settings.reranker_model, device=settings.reranker_device)
    writer = AnalysisWriter(llm, evidence_limit=settings.answer_evidence_limit)
    return AnalysisPipeline(
        intent_extractor=IntentExtractor(llm),
        cache=cache,
        semantic_cache=semantic_cache,
        retriever=retriever,
        reranker=reranker,
        writer=writer,
        data_version=settings.data_version,
        prompt_version=settings.prompt_version,
        model_id=settings.ollama_model,
        rerank_limit=settings.rerank_limit,
    )


def _details_section(st: Any, label: str, *, expanded: bool = False) -> Any:
    return st.expander(label, expanded=expanded)


def _render_evidence(
    st: Any,
    evidence: Any,
    label: str,
    query_terms: list[str] | tuple[str, ...] = (),
    *,
    show_full_text: bool = True,
) -> None:
    title = evidence.meeting_title or evidence.minutes_title or "회의 제목 확인 필요"
    st.markdown(f"**{label} · {evidence.date or '날짜 미상'} · {evidence.assembly or '의회 미상'}**")
    st.caption(f"{evidence.minutes_title} / {title} · 안건: {evidence.agenda or '미상'} · 출처 ID: `{evidence.id}`")
    st.markdown("**질문 관련 구절**")
    st.text(extract_relevant_excerpt(evidence.text, query_terms))
    if show_full_text:
        with st.expander("원문 전체 보기", expanded=False):
            st.text(evidence.text)


def render_result(st: Any, result: AnalysisResult, *, inside_question: bool = False) -> None:
    if result.cache_hit:
        st.caption("저장된 분석을 재사용했습니다.")
    evidence_by_id = {
        item.id: item for item in [*result.flow, *result.regional_spread_evidence]
    }

    with st.container(border=True):
        st.subheader(f"향후 전망 · 기준일 {result.forecast_as_of or '미상'}")
        st.caption(f"신뢰도: {result.confidence}")
        forecast = (result.forecast or "").strip()
        if forecast:
            st.markdown(forecast)
        else:
            st.warning("이 분석 결과에는 전망이 없습니다. 새 질문으로 다시 분석해 주세요.")
        st.subheader("향후 논의 가능 주제")
        if result.likely_discussion_topics:
            st.markdown("\n".join(
                f"- {row['topic']} · 근거: " + ", ".join(f"`{source_id}`" for source_id in row["source_ids"])
                for row in result.likely_discussion_topics
            ))
        else:
            st.caption("근거 요약에서 구체적인 후속 논의 주제를 확인하지 못했습니다.")
        if result.forecast_citations:
            with _details_section(st, "전망 근거 보기", expanded=False):
                for source_id in result.forecast_citations:
                    evidence = evidence_by_id.get(source_id)
                    if evidence:
                        _render_evidence(
                            st, evidence, "전망 근거", [result.intent.issue, *result.intent.keywords],
                            show_full_text=False,
                        )
                    else:
                        st.write(source_id)

        regional_spread = result.regional_spread or build_regional_spread(
            result.intent, result.flow, set(result.forecast_citations)
        )
        st.subheader("다른 시·도 확산 가능성")
        st.markdown(regional_spread.get("outlook", "현재 근거로 확산 가능성을 판단할 수 없습니다."))
        st.caption("확산 판단은 주 검색 결과와 최대 8개의 타 의회 비교 후보에서 요약된 근거에 한정되며, 전체 corpus를 훑은 결과가 아닙니다. 유사 논의가 여러 의회에 있어도 정책 전파를 증명하지 않으며, 확산 확률은 산출하지 않습니다.")
        origin_councils = regional_spread.get("origin_councils", [])
        observed_councils = regional_spread.get("observed_councils", [])
        for label, councils in (("기준 의회", origin_councils), ("비교 의회", observed_councils)):
            for council in councils:
                date_range = council.get("date_range") or {}
                period = f"{date_range.get('start', '날짜 미상')}–{date_range.get('end', '날짜 미상')}" if date_range else "날짜 미상"
                st.markdown(f"- {label}: {council.get('assembly', '의회 미상')} · {council.get('record_count', 0)}건 · {period}")
        spread_source_ids = list(dict.fromkeys(
            source_id
            for council in [*origin_councils, *observed_councils]
            for source_id in council.get("source_ids", [])
        ))
        if spread_source_ids:
            with _details_section(st, "확산 판단 근거 보기", expanded=False):
                for source_id in spread_source_ids:
                    evidence = evidence_by_id.get(source_id)
                    if evidence:
                        _render_evidence(
                            st, evidence, "관련 의회 논의", [result.intent.issue, *result.intent.keywords],
                            show_full_text=False,
                        )

        st.subheader("핵심 사실")
        if result.core_facts:
            st.markdown("\n".join(f"- {fact}" for fact in result.core_facts))
        else:
            st.info("이 분석 결과에는 표시할 핵심 사실이 없습니다.")

        if result.answer.strip():
            with _details_section(st, "요약 설명", expanded=False):
                st.markdown(result.answer)

    if result.first_seen:
        st.subheader("최초 관측 근거")
        st.caption("Cross-Encoder가 최종 채택한 근거 중 가장 이른 기록입니다. 전체 자료의 절대 최초를 보증하지 않습니다.")
        _render_evidence(
            st, result.first_seen, "최초 관측", [result.intent.issue, *result.intent.keywords],
            show_full_text=not inside_question,
        )
    else:
        st.info("이 질문과 맞는 근거 발언을 찾지 못했습니다.")

    points = build_map_points(result.flow)
    if points:
        with _details_section(st, "의회별 위치와 논의 순서", expanded=False):
            st.caption("의회마다 첫 발언을 나타내는 원형 표식 하나만 표시합니다. 숫자는 각 의회의 첫 발언을 시간순으로 정렬한 지도 순번(1, 2, 3…)입니다. 아래 표의 ‘발언 순서’는 모든 발언의 시간순이고, ‘지도 번호’는 각 의회의 첫 발언 행에만 표시됩니다. 마우스 오버 설명은 없습니다. 파란색은 가장 먼저 관측된 의회, 주황색은 그다음 의회입니다. 점은 의회별 대표 위치이며 일부는 시·도 중심 좌표로 근사했습니다.")
            try:
                st.pydeck_chart(
                    build_numbered_map_deck(points),
                    width="stretch",
                    alt="의회별 위치와 숫자로 표시한 논의 순서 지도",
                )
            except Exception as exc:
                st.warning(f"지도 렌더링을 사용할 수 없습니다: {exc}")
            map_number_by_source = {
                str(point["source_id"]): str(point["order_label"])
                for point in points
            }
            st.dataframe(
                [{"발언 순서": i, "지도 번호": map_number_by_source.get(item.id, ""),
                  "날짜": item.date, "의회": item.assembly,
                  "회의": item.meeting_title, "안건": item.agenda, "출처 ID": item.id}
                 for i, item in enumerate(result.flow, start=1)],
                width="stretch",
                hide_index=True,
            )
    elif result.flow:
        st.info("검색 근거의 의회 위치를 좌표표에서 찾지 못했습니다.")

    if result.clusters:
        with _details_section(st, "쟁점별 근거 묶음", expanded=False):
            st.dataframe(result.clusters, width="stretch", hide_index=True)

    if result.citations:
        with _details_section(st, "출처와 인용", expanded=False):
            for source_id in result.citations:
                evidence = evidence_by_id.get(source_id)
                if evidence:
                    _render_evidence(
                        st, evidence, "인용", [result.intent.issue, *result.intent.keywords],
                        show_full_text=False,
                    )
                else:
                    st.write(source_id)
    if result.warnings:
        with _details_section(st, "주의 및 한계", expanded=False):
            for warning in result.warnings:
                st.warning(warning)


def render_chat_history(st: Any, messages: list[dict[str, Any]]) -> None:
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") != "user":
            if message.get("role") == "assistant" and not message.get("result"):
                st.error(message.get("content", "오류"))
            index += 1
            continue

        question = str(message.get("content", "")).strip()
        response = messages[index + 1] if index + 1 < len(messages) else None
        if response and response.get("role") == "assistant":
            result_data = response.get("result")
            result = AnalysisResult.from_dict(result_data) if result_data else None
            topic = result.intent.issue if result else question
        else:
            result = None
            topic = question
        short_question = question if len(question) <= 54 else question[:51] + "…"
        label = f"🔎 {topic} · {short_question}"
        with st.expander(label, expanded=False):
            st.caption(f"질문: {question}")
            if result is not None:
                render_result(st, result, inside_question=True)
            elif response and response.get("role") == "assistant":
                st.error(response.get("content", "오류"))
            else:
                st.caption("분석 결과를 기다리는 중입니다.")
        index += 2 if response and response.get("role") == "assistant" else 1


def render_page_header(st: Any, *, disabled: bool) -> bool:
    title_column, action_column = st.columns([6, 1], vertical_alignment="center")
    with title_column:
        st.title("지방의회 회의록 분석")
    with action_column:
        return st.button(
            "캐시 초기화",
            key="clear_analysis_cache",
            disabled=disabled,
            help="저장된 답변과 유사 질문 캐시만 삭제합니다. 채팅 기록과 검색 인덱스는 유지됩니다.",
            width="content",
        )


def chat_input_disabled(session_state: Any) -> bool:
    return bool(session_state.get("processing", False))


def queue_chat_prompt(session_state: Any, prompt: str) -> bool:
    if chat_input_disabled(session_state):
        return False
    messages = session_state.get("messages")
    if messages is None:
        messages = []
        session_state["messages"] = messages
    session_state["pending_history"] = list(messages)[-8:]
    messages.append({"role": "user", "content": prompt})
    session_state["pending_prompt"] = prompt
    session_state["processing"] = True
    return True


def finish_chat_prompt(session_state: Any) -> None:
    session_state["pending_prompt"] = None
    session_state["pending_history"] = []
    session_state["processing"] = False


def clear_pipeline_caches(pipeline: Any) -> tuple[int, int]:
    answer_count = pipeline.cache.clear()
    semantic_count = pipeline.semantic_cache.clear() if pipeline.semantic_cache is not None else 0
    return answer_count, semantic_count


def main() -> None:
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit이 없습니다. 프로젝트 Python 환경에 src/requirements-app.txt를 설치하세요.")
        return

    st.set_page_config(page_title="지방의회 회의록 분석", page_icon="▦", layout="wide", menu_items={})
    st.markdown(APP_STYLE, unsafe_allow_html=True)
    clear_cache_clicked = render_page_header(
        st, disabled=chat_input_disabled(st.session_state)
    )
    st.caption("질문을 입력하면 로컬 LLM과 로컬 검색 인덱스로 쟁점의 최초 관측·확산 흐름·향후 전망을 분석합니다.")

    settings = Settings.from_env()
    with st.sidebar:
        st.subheader("현재 설정")
        st.write(f"로컬 LLM: `{settings.ollama_model}`")
        st.write(f"임베딩: `{settings.embedding_model}` ({settings.embedding_device})")
        st.write(f"재순위 모델: `{settings.reranker_model}` ({settings.reranker_device})")
        st.write(f"자료 버전: `{settings.data_version}`")
        st.caption("모델 다운로드는 기본 차단됩니다. 이미 내려받은 가중치만 사용합니다.")

    @st.cache_resource(show_spinner=False)
    def runtime() -> AnalysisPipeline:
        return create_pipeline(Settings.from_env())

    source_index_needs_build = (
        not settings.source_metadata_db.exists()
        and settings.chroma_dir.is_dir()
        and settings.bm25_db.is_file()
    )
    try:
        if source_index_needs_build:
            with st.status("첫 실행: 원본 TXT 인덱스를 생성합니다", expanded=True) as startup_status:
                startup_status.write("TXT 메타데이터를 SQLite로 구성합니다. 완료 후 다음 실행부터는 재사용합니다.")
                pipeline = runtime()
                startup_status.update(label="앱 준비 완료", state="complete", expanded=False)
        else:
            with st.spinner("검색 구성요소를 준비합니다..."):
                pipeline = runtime()
    except Exception as exc:
        st.error(f"앱 준비가 완료되지 않았습니다: {exc}")
        st.info("Chroma와 BM25 색인을 준비하세요. TXT 메타데이터 색인은 없을 때 앱이 자동 생성하며, 기존 파일이 있지만 유효하지 않으면 자동 덮어쓰지 않습니다.")
        st.stop()

    if clear_cache_clicked:
        try:
            answer_count, semantic_count = clear_pipeline_caches(pipeline)
            st.sidebar.success(
                f"캐시 초기화 완료: 답변 {answer_count:,}건, 유사 질문 {semantic_count:,}건 삭제"
            )
        except Exception as exc:
            st.sidebar.error(f"캐시 초기화에 실패했습니다: {exc}")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    render_chat_history(st, st.session_state.messages)

    prompt = st.chat_input(
        "예: 학교 급식 예산 논의가 어느 의회에서 먼저 나오고 어떻게 퍼졌어?",
        disabled=chat_input_disabled(st.session_state),
    )
    if prompt and queue_chat_prompt(st.session_state, prompt):
        st.rerun()

    pending_prompt = st.session_state.get("pending_prompt")
    if pending_prompt:
        history = st.session_state.get("pending_history", [])
        with st.chat_message("assistant"):
            with st.status("분석을 준비합니다", expanded=True) as status:
                progress = st.progress(0)

                def on_stage(label: str, value: int) -> None:
                    status.write(label)
                    progress.progress(value)

                try:
                    result = pipeline.ask(pending_prompt, history=history, on_stage=on_stage)
                    status.update(label="분석 완료", state="complete", expanded=False)
                    render_result(st, result)
                    st.session_state.messages.append({
                        "role": "assistant", "content": result.answer, "result": result.to_dict()
                    })
                except Exception as exc:
                    status.update(label="분석 중 오류", state="error", expanded=True)
                    st.error(str(exc))
                    st.session_state.messages.append({"role": "assistant", "content": str(exc)})
                finally:
                    finish_chat_prompt(st.session_state)
        st.rerun()


def run_entrypoint(st: Any, script_path: Path | str, ui_main: Any) -> None:
    """Run the UI in Streamlit, starting its server when invoked with plain Python."""
    runtime = getattr(st, "runtime", None)
    runtime_exists = getattr(runtime, "exists", None)
    if callable(runtime_exists) and runtime_exists():
        ui_main()
        return

    app_factory = getattr(st, "App", None)
    if not callable(app_factory):
        raise RuntimeError("직접 실행에는 Streamlit 1.66 이상이 필요합니다. streamlit run src/app.py를 사용하세요.")
    app_factory(str(Path(script_path).resolve())).run()


if __name__ == "__main__":
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit이 없습니다. 같은 Python 환경에 src/requirements-app.txt를 설치하세요.")
        raise SystemExit(1)
    run_entrypoint(st, Path(__file__), main)

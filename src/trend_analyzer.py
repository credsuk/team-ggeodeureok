"""
지방의회 정책 이슈 시계열 확산 분석 및 미래 예측 엔진 (PolicyTrendAnalyzer)
========================================================================
[경진대회 제출 규격 부합 모델]
- 주요 이슈 키워드 분석
- 제공 데이터 범위 내 최초 관측 시점 (등장시점)
- 타 지방의회에서의 후속 논의 흐름 (시계열 확산)
- 로컬 LLM 기반 향후 예측 (서술형)
- 공식 회의록명(MINTS_NM), 회의일자 및 판단에 사용한 원문 구간 정밀 매핑
- 최종 JSON 포맷 및 제출 양식 표(Markdown Table) 자동 생성
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.config import (
    COLLECTION_NAME,
    DEFAULT_DB_TYPE,
    EMBEDDING_MODEL_NAME,
    get_pg_connection_string,
    resolve_embedding_model,
)
from src.indexer import ChromaIndexer
from src.pg_indexer import PGVectorIndexer

logger = logging.getLogger(__name__)


TREND_ANALYSIS_SYSTEM_PROMPT = """당신은 대한민국 지방의회 정책 분석 및 정책 시계열 확산 예측 전문가입니다.
주어진 지방의회 회의록 시계열 데이터만을 바탕으로, 특정 정책 이슈의 확산 흐름을 분석하고 향후 전개를 객관적으로 예측하세요.

[⚠️ 준수 규칙]
1. 반드시 제공된 회의록 근거에 나타난 사실만을 바탕으로 논의 흐름을 정리하세요.
2. 특정 지자체에서 시작된 논의가 다른 지방의회로 어떻게 전파되고 심화되었는지 관점의 변화(조례 제정, 예산 확보, 공청회, 정부 건의 등)를 명확히 기술하세요.
3. [향후 예측]은 제공된 논의 흐름을 바탕으로 향후 타 지자체 및 중앙정부와의 연계, 지속 가능성에 대해 객관적이고 설득력 있는 서술형으로 1~3문장 작성하세요.
4. 순수 JSON 형식으로만 응답하세요. (마크다운 코드블록 제외)

[응답 JSON 스키마]
{
  "trend_summary": [
    {
      "date": "2024년 2월 27일",
      "assembly": "광주광역시의회",
      "point": "청년 인구 유출 방지 및 일자리 창출 관점에서 정책 대안 논의"
    },
    {
      "date": "2024년 6월 22일",
      "assembly": "강원특별자치도의회",
      "point": "지역소멸 대응 특별 조례안 제정을 위한 공청회 실시 및 재정 지원 협의"
    }
  ],
  "future_prediction": "향후 농어촌 비율이 높은 지방의회를 중심으로 정주여건 개선 및 맞춤형 지원 조례 제정이 심화될 것으로 예측되며, 인접 지자체 간 초광역 협력 사업으로 확장될 가능성이 높음."
}
"""


def normalize_date_str(raw_date: Any) -> Tuple[str, str]:
    """날짜 문자열을 (YYYY-MM-DD, YYYY년 M월 D일) 튜플로 정규화"""
    if not raw_date:
        return "9999-12-31", "미상"

    s = re.sub(r"[^0-9]", "", str(raw_date))
    if len(s) >= 8:
        y, m, d = s[:4], s[4:6], s[6:8]
        iso = f"{y}-{m}-{d}"
        kor = f"{int(y)}년 {int(m)}월 {int(d)}일"
        return iso, kor
    elif len(s) == 4:
        return f"{s}-01-01", f"{s}년"

    return str(raw_date), str(raw_date)


def extract_context_window(text: str, keyword: str, max_chars: int = 180) -> str:
    """
    원문 텍스트에서 키워드가 등장하는 핵심 문맥 구간(Context Window)을 정밀 추출
    """
    if not text:
        return ""

    # 접두사([원문 발언], 메타 정보 등) 제거
    clean = text
    if "[원문 발언]" in clean:
        clean = clean.split("[원문 발언]", 1)[-1].strip()

    clean = re.sub(r"\[[^\]]*\]", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    idx = clean.find(keyword)
    if idx != -1:
        half = max_chars // 2
        start = max(0, idx - half)
        end = min(len(clean), idx + len(keyword) + half)
        span = clean[start:end].strip()
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(clean) else ""
        return f"{prefix}{span}{suffix}"

    if len(clean) > max_chars:
        return clean[:max_chars].strip() + "..."
    return clean


class PolicyTrendAnalyzer:
    """
    지방의회 정책 이슈 타임라인 및 미래 예측 모델
    """

    def __init__(
        self,
        db_type: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        llm_model: str = "gemma4:e2b",
        ollama_base_url: str = "http://localhost:11434",
        pg_connection_string: Optional[str] = None,
        cutoff_date: Optional[str] = None,  # 분석 기준일 (YYYY-MM-DD, 이후 데이터 배제)
    ):
        self.db_type = (db_type or DEFAULT_DB_TYPE).lower()
        self.collection_name = collection_name or COLLECTION_NAME
        self.embedding_model_name = resolve_embedding_model(embedding_model_name or EMBEDDING_MODEL_NAME)
        self.llm_model = llm_model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.pg_connection_string = pg_connection_string or get_pg_connection_string()
        self.cutoff_date = cutoff_date  # 주최 측 지정 분석 기준일

        # 벡터 데이터베이스 인덱서 연결
        if self.db_type == "pgvector":
            self.indexer = PGVectorIndexer(
                connection_string=self.pg_connection_string,
                collection_name=self.collection_name,
                embedding_model_name=self.embedding_model_name,
            )
        else:
            self.db_type = "chroma"
            self.indexer = ChromaIndexer(
                collection_name=self.collection_name,
                embedding_model_name=self.embedding_model_name,
            )

    def _call_ollama(self, prompt: str, system_prompt: str = "", temperature: float = 0.2) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.llm_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        try:
            resp = requests.post(f"{self.ollama_base_url}/api/chat", json=payload, timeout=90)
            if resp.status_code == 200:
                return resp.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.error(f"Ollama 호출 실패: {e}")
        return ""

    def _parse_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        if not raw_text:
            return None
        text = raw_text.strip()
        if "```" in text:
            text = re.sub(r'```(?:json)?\s*', '', text)
            text = text.replace('```', '').strip()
        match = re.search(r'(\{[\s\S]*\})', text)
        candidate = match.group(1).strip() if match else text
        try:
            return json.loads(candidate)
        except Exception:
            try:
                fixed = re.sub(r',\s*\}', '}', candidate)
                fixed = re.sub(r',\s*\]', ']', fixed)
                return json.loads(fixed)
            except Exception:
                return None

    def analyze_issue(
        self,
        issue_keyword: str,
        retrieval_k: int = 30,
        max_flow_items: int = 4,
    ) -> Dict[str, Any]:
        """
        주요 이슈 키워드를 기반으로 최초 등장 시점, 타 의회 확산 흐름, 향후 예측 분석 수행
        """
        start_time = time.time()

        # 1. 이슈 키워드 기반 시맨틱 검색 수행
        search_query = f"{issue_keyword} 정책 조례 지원 추진 방안"
        raw_results = self.indexer.query(
            query_text=search_query,
            n_results=retrieval_k,
            where=None,
        )

        if not raw_results:
            return {
                "status": "not_found",
                "issue_keyword": issue_keyword,
                "message": f"'{issue_keyword}'에 대한 관련 회의록을 찾지 못했습니다.",
            }

        # 2. 결과 문서 정제 및 시계열(날짜순) 정렬
        sorted_docs = []
        for doc, score in raw_results:
            meta = doc.metadata
            iso_date, kor_date = normalize_date_str(meta.get("date"))

            # 분석 기준일(cutoff_date) 이후 정보는 배제 (경진대회 규정 준수)
            if self.cutoff_date and iso_date > self.cutoff_date:
                continue

            content = doc.page_content.strip()
            if len(content) < 40:
                continue

            # 공식 회의록명 (mints_nm)
            mints_nm = meta.get("mints_nm") or meta.get("agenda_title") or f"{meta.get('assembly', '')} 회의록"
            agenda = meta.get("agenda_title", "회의 안건")
            speaker = f"{meta.get('speaker_role', meta.get('speaker_position', ''))} {meta.get('speaker_name', '')}".strip()

            sorted_docs.append({
                "iso_date": iso_date,
                "kor_date": kor_date,
                "assembly": meta.get("assembly", "지방의회"),
                "mints_nm": mints_nm,
                "agenda_title": agenda,
                "speaker": speaker if speaker else "발언자 미상",
                "policy_category": meta.get("policy_category", ""),
                "core_issue": meta.get("core_issue", ""),
                "keywords": meta.get("keywords", ""),
                "summary": meta.get("summary", ""),
                "content": content,
                "score": float(score),
            })

        # 날짜 오름차순 정렬 (과거 -> 최근)
        sorted_docs.sort(key=lambda x: (x["iso_date"], x["assembly"]))

        if not sorted_docs:
            return {
                "status": "not_found",
                "issue_keyword": issue_keyword,
                "message": "기준일 이전의 유효한 회의록 데이터가 없습니다.",
            }

        # 3. 최초 관측 시점 (등장시점) 도출
        first_doc = sorted_docs[0]
        first_snippet = extract_context_window(first_doc["content"], issue_keyword, max_chars=180)
        first_observed = {
            "date": first_doc["kor_date"],
            "iso_date": first_doc["iso_date"],
            "assembly": first_doc["assembly"],
            "mints_nm": first_doc["mints_nm"],
            "display": f"{first_doc['kor_date']}, {first_doc['assembly']}",
            "speaker": first_doc["speaker"],
            "agenda_title": first_doc["agenda_title"],
            "used_text_span": first_snippet,
        }

        # 4. 타 지방의회로의 후속 논의 흐름 추출
        subsequent_docs = []
        seen_assemblies = {first_doc["assembly"]}

        for doc in sorted_docs[1:]:
            assembly = doc["assembly"]
            # 새로운 의회의 최초 논의거나, 기존 의회라도 날짜가 확연히 다른 주요 안건
            if assembly not in seen_assemblies or len(seen_assemblies) < 3:
                subsequent_docs.append(doc)
                seen_assemblies.add(assembly)
            if len(subsequent_docs) >= 8:
                break

        # LLM에게 타임라인 요약 및 향후 예측 요청
        timeline_prompt_blocks = []
        for idx, doc in enumerate([first_doc] + subsequent_docs[:6], 1):
            snippet = extract_context_window(doc["content"], issue_keyword, max_chars=160)
            timeline_prompt_blocks.append(
                f"[{idx}] 일시: {doc['kor_date']} | 의회: {doc['assembly']} | 회의록명: {doc['mints_nm']}\n"
                f"발언자: {doc['speaker']} | 안건: {doc['agenda_title']}\n"
                f"발언 요약/맥락: {doc['summary'] or snippet}"
            )

        llm_input = f"""[분석 이슈 키워드]
{issue_keyword}

[수집된 시계열 지방의회 회의록 발언 목록]
{chr(10).join(timeline_prompt_blocks)}

위 시계열 목록을 바탕으로, 최초 등장 이후 타 지방의회로 전파된 논의 흐름(최대 {max_flow_items}개)과 향후 예측을 JSON으로 작성하세요."""

        llm_response = self._call_ollama(llm_input, system_prompt=TREND_ANALYSIS_SYSTEM_PROMPT)
        parsed_llm = self._parse_json(llm_response)

        # 5. 후속 논의 흐름 목록 구성
        trend_flow_list = []
        if parsed_llm and "trend_summary" in parsed_llm and isinstance(parsed_llm["trend_summary"], list):
            for item in parsed_llm["trend_summary"]:
                date_str = item.get("date", "")
                assembly_str = item.get("assembly", "")
                point_str = item.get("point", "")
                trend_flow_list.append(f"- {date_str}, {assembly_str}에서 {point_str}")
        else:
            # Fallback 규칙 기반 구성
            for doc in subsequent_docs[:max_flow_items]:
                trend_flow_list.append(
                    f"- {doc['kor_date']}, {doc['assembly']}에서 '{doc['agenda_title']}' 관련 논의 진행"
                )

        # 6. 향후 예측 생성
        if parsed_llm and parsed_llm.get("future_prediction"):
            future_prediction = parsed_llm["future_prediction"]
        else:
            future_prediction = f"향후 {issue_keyword} 관련 논의는 조례 제정 및 국비/지방비 확보를 중심으로 타 광역·기초 의회로 지속 확산될 것으로 예측됨."

        # 7. 근거 회의록 목록 (공식 회의록명, 회의일자, 사용 원문 구간)
        evidences = []
        for doc in [first_doc] + subsequent_docs[:max_flow_items]:
            span = extract_context_window(doc["content"], issue_keyword, max_chars=180)
            evidences.append({
                "assembly": doc["assembly"],
                "date": doc["kor_date"],
                "iso_date": doc["iso_date"],
                "mints_nm": doc["mints_nm"],
                "agenda_title": doc["agenda_title"],
                "speaker": doc["speaker"],
                "used_text_span": span,
            })

        elapsed_sec = round(time.time() - start_time, 2)

        # 8. 제출 형식 텍스트 요약 (Markdown Table 호환)
        flow_formatted = "\n".join(trend_flow_list)
        evidence_summary_list = [
            f"• [{ev['date']}, {ev['assembly']}] {ev['mints_nm']} ({ev['speaker']}): \"{ev['used_text_span']}\""
            for ev in evidences[:3]
        ]
        evidence_formatted = "\n".join(evidence_summary_list)

        result = {
            "status": "success",
            "issue_keyword": issue_keyword,
            "first_observed": first_observed,
            "cross_council_trend": trend_flow_list,
            "future_prediction": future_prediction,
            "evidences": evidences,
            "submission_table_row": {
                "이슈키워드": issue_keyword,
                "등장시점": first_observed["display"],
                "타 지방의회로의 흐름": flow_formatted,
                "향후 예측": future_prediction,
                "근거 회의록": evidence_formatted,
            },
            "metadata": {
                "vector_db": self.db_type,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model_name,
                "llm_model": self.llm_model,
                "analyzed_chunks": len(sorted_docs),
                "elapsed_time_sec": elapsed_sec,
            },
        }

        return result

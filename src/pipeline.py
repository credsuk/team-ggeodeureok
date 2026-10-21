"""
지방의회 회의록 RAG 모델 파이프라인 (CouncilRAGPipeline)
======================================================
팀 회의 결정 규격:
1. 사용자 쿼리 (User Query)
2. 키워드 및 메타데이터 필터 추출 (Keyword & Intent Extraction)
3. DB 메타데이터 필터링 및 벡터 유사도 검색 (ChromaDB Metadata & Vector Search)
4. Re-ranking (재순위화: 로컬 LLM 또는 CrossEncoder 기반)
5. 최종 표준 JSON 아웃풋 생성 (LLM 답변 + 근거 데이터 반환)
"""

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

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


# =====================================================================
# 1. 키워드 및 검색 필터 추출 프롬프트 정의
# =====================================================================
KEYWORD_EXTRACTION_SYSTEM_PROMPT = """당신은 지방의회 회의록 데이터베이스 검색을 위한 질의 분석 전문가입니다.
사용자의 질문(자연어 질의)을 분석하여 검색에 가장 적합한 키워드와 메타데이터 필터 조건을 추출해 순수 JSON으로만 응답하세요.

[추출 규칙]
1. keywords:
   - 검색 정확도를 높이기 위한 핵심 키워드 리스트 (2~5개).
   - 흔한 일반 명사('회의', '의견', '관련', '내용', '질문', '알려줘' 등)는 제거하고 실질적 정책/이슈/고유명사 중심 추출.
2. filters:
   - assembly: 지자체/의회명 ("강원특별자치도의회", "경기도의회", "충청남도의회", "충청북도의회", "경상남도의회" 등). 질의에 명시되지 않았으면 null.
   - year: 연도 (정수형, 예: 2024, 2025). 질의에 명시되지 않았으면 null.
   - speaker_name: 특정 발언자/의원 성명 (예: "김진태", "김동연" 등). 없으면 null.
   - policy_category: 정책분야 ("경제/산업", "농림/수산", "보건/복지", "교육/문화", "환경/안전", "도시/교통", "자치/행정", "기타" 중 명확할 때만 선택, 애매하면 null).
3. refined_query:
   - 벡터 임베딩 유사도 검색에 가장 효과적인 핵심 의미 문장 (불필요한 조사/어미 정제).

[응답 JSON 포맷 - 마크다운 코드블록 없이 순수 JSON만 출력]
{
  "keywords": ["농지 규제", "청년농", "농가 소득"],
  "filters": {
    "assembly": "강원특별자치도의회",
    "year": 2024,
    "speaker_name": null,
    "policy_category": "농림/수산"
  },
  "refined_query": "농지 소유 규제 완화 및 청년 농업인 유입 지원 정책"
}
"""

# =====================================================================
# 2. 로컬 LLM 기반 최종 답변 및 근거 요약 생성 프롬프트
# =====================================================================
ANSWER_GENERATION_SYSTEM_PROMPT = """당신은 대한민국 지방의회 회의록 분석 및 정책 전문 AI 어시스턴트입니다.
제공된 [회의록 근거 문서]만을 바탕으로 사용자의 질문에 대해 명확하고 논리적인 답변을 작성하세요.

[작성 원칙]
1. 반드시 제공된 근거 문서에 기반하여 사실만을 작성하고, 문서에 없는 내용은 추측하지 마세요.
2. 발언자(의원/지사 등), 회의 일시, 해당 의회명을 인용하여 답변의 신뢰도를 높이세요.
3. 근거가 부족하거나 질문과 일치하지 않는 경우 솔직하게 근거가 부족함을 밝히세요.
4. 요약 및 핵심 논점을 명확히 정리하세요.
"""


class CouncilRAGPipeline:
    """
    회의록 검색·재순위화·답변 생성 엔드투엔드 모델 파이프라인
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        llm_model: str = "gemma4:e2b",
        ollama_base_url: str = "http://localhost:11434",
        reranker_type: str = "llm",  # 'llm', 'cross-encoder', 'none'
        cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
        timeout: int = 60,
        db_type: Optional[str] = None,  # 'chroma' 또는 'pgvector'
        pg_connection_string: Optional[str] = None,
    ):
        self.collection_name = collection_name or COLLECTION_NAME
        self.embedding_model_name = resolve_embedding_model(embedding_model_name or EMBEDDING_MODEL_NAME)
        self.llm_model = llm_model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.reranker_type = reranker_type.lower()
        self.cross_encoder_model = cross_encoder_model
        self.timeout = timeout
        self.db_type = (db_type or DEFAULT_DB_TYPE).lower()
        self.pg_connection_string = pg_connection_string or get_pg_connection_string()

        # 1. 벡터 데이터베이스 인덱서 초기화 (ChromaDB 또는 PostgreSQL pgvector)
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

        # 2. Cross-Encoder 초기화 (선택적)
        self._cross_encoder = None
        if self.reranker_type == "cross-encoder":
            self._init_cross_encoder()

    def _init_cross_encoder(self):
        """Cross-Encoder 모델 지연 로딩"""
        try:
            from sentence_transformers import CrossEncoder
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"CrossEncoder '{self.cross_encoder_model}' 모델 로드 중 (디바이스: {device})...")
            self._cross_encoder = CrossEncoder(self.cross_encoder_model, device=device)
        except Exception as e:
            logger.warning(f"CrossEncoder 로드 실패, LLM 기반 재순위화로 대체합니다: {e}")
            self.reranker_type = "llm"

    # =================================================================
    # Step 1: Ollama LLM 통신 헬퍼
    # =================================================================
    def _call_ollama(
        self,
        prompt: str,
        system_prompt: str = "",
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> str:
        """로컬 Ollama API 호출"""
        target_model = model or self.llm_model
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        try:
            resp = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("message", {}).get("content", "").strip()
            else:
                logger.error(f"Ollama 호출 실패 (HTTP {resp.status_code}): {resp.text}")
                return ""
        except Exception as e:
            logger.error(f"Ollama 호출 예외: {e}")
            return ""

    def _parse_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """LLM 응답 텍스트에서 안전하게 JSON 파싱"""
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

    # =================================================================
    # Step 2: 키워드 및 메타데이터 필터 추출
    # =================================================================
    def extract_keywords_and_filters(self, user_query: str) -> Dict[str, Any]:
        """
        사용자 질의로부터 검색 키워드, 메타데이터 필터, 정제 질의 추출
        """
        prompt = f"사용자 질의: \"{user_query}\"\n위 질의를 분석하여 지정된 JSON 포맷으로 키워드와 필터를 추출하세요."
        response_text = self._call_ollama(
            prompt=prompt,
            system_prompt=KEYWORD_EXTRACTION_SYSTEM_PROMPT,
            temperature=0.0,
        )

        parsed = self._parse_json(response_text)
        if not parsed or not isinstance(parsed, dict):
            # Fallback 룰베이스 추출
            fallback_keywords = [w for w in re.findall(r'[가-힣a-zA-Z0-9]{2,}', user_query) if w not in ["어떤", "어떻게", "대한", "관련", "알려줘", "회의", "의회"]]
            return {
                "keywords": fallback_keywords[:5],
                "filters": {
                    "assembly": None,
                    "year": None,
                    "speaker_name": None,
                    "policy_category": None,
                },
                "refined_query": user_query,
            }

        # 필터 값 정규화
        filters = parsed.get("filters", {})
        if not isinstance(filters, dict):
            filters = {}

        # 연도 정수 변환
        if filters.get("year"):
            try:
                filters["year"] = int(filters["year"])
            except (ValueError, TypeError):
                filters["year"] = None

        return {
            "keywords": parsed.get("keywords", []),
            "filters": {
                "assembly": filters.get("assembly"),
                "year": filters.get("year"),
                "speaker_name": filters.get("speaker_name"),
                "policy_category": filters.get("policy_category"),
            },
            "refined_query": parsed.get("refined_query", user_query) or user_query,
        }

    # =================================================================
    # Step 3: DB 메타데이터 및 벡터 유사도 검색
    # =================================================================
    def search_db(
        self,
        query_text: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 15,
    ) -> List[Tuple[Any, float]]:
        """
        ChromaDB에서 메타데이터 필터 적용 및 벡터 유사도 검색 수행
        """
        conditions = []
        if filters:
            if filters.get("assembly"):
                conditions.append({"assembly": str(filters["assembly"])})
            if filters.get("year") and int(filters["year"]) > 0:
                conditions.append({"year": int(filters["year"])})
            if filters.get("speaker_name"):
                conditions.append({"speaker_name": str(filters["speaker_name"])})
            if filters.get("policy_category"):
                conditions.append({"policy_category": str(filters["policy_category"])})

        where_clause = None
        if len(conditions) == 1:
            where_clause = conditions[0]
        elif len(conditions) > 1:
            where_clause = {"$and": conditions}

        # 1차 검색 실행
        results = self.indexer.query(
            query_text=query_text,
            n_results=top_k,
            where=where_clause,
        )

        # 만약 엄격한 필터로 인해 결과가 너무 적을 경우 필터 완화 재검색
        if len(results) == 0 and where_clause is not None:
            logger.info("필터 적용 결과가 없어 필터 없이 재검색합니다.")
            results = self.indexer.query(
                query_text=query_text,
                n_results=top_k,
                where=None,
            )

        return results

    # =================================================================
    # Step 4: Re-ranking (재순위화)
    # =================================================================
    def rerank_documents(
        self,
        query: str,
        candidate_docs: List[Tuple[Any, float]],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        1차 검색된 후보 문서를 질의 관련성에 따라 재순위화
        """
        if not candidate_docs:
            return []

        if len(candidate_docs) <= top_n and self.reranker_type == "none":
            # 단순 거리순 반환
            reranked = []
            for rank, (doc, dist) in enumerate(candidate_docs, 1):
                # distance to score 변환 (cosine distance 기준)
                sim_score = max(0.0, 1.0 - float(dist))
                reranked.append({
                    "rank": rank,
                    "rerank_score": round(sim_score, 4),
                    "initial_distance": round(float(dist), 4),
                    "document": doc,
                })
            return reranked

        # 전략 A: Cross-Encoder 모델 기반 재순위화
        if self.reranker_type == "cross-encoder" and self._cross_encoder is not None:
            return self._rerank_by_cross_encoder(query, candidate_docs, top_n)

        # 전략 B: 로컬 LLM 기반 관련도 점수 산출 재순위화
        return self._rerank_by_llm(query, candidate_docs, top_n)

    def _rerank_by_cross_encoder(
        self,
        query: str,
        candidate_docs: List[Tuple[Any, float]],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        pairs = [[query, doc.page_content] for doc, _ in candidate_docs]
        scores = self._cross_encoder.predict(pairs)

        scored_docs = []
        for i, (doc, dist) in enumerate(candidate_docs):
            scored_docs.append({
                "initial_distance": round(float(dist), 4),
                "rerank_score": round(float(scores[i]), 4),
                "document": doc,
            })

        # 점수 높은 순 정렬
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        for rank, item in enumerate(scored_docs[:top_n], 1):
            item["rank"] = rank

        return scored_docs[:top_n]

    def _rerank_by_llm(
        self,
        query: str,
        candidate_docs: List[Tuple[Any, float]],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        로컬 LLM을 통한 정밀 관련도 평가
        """
        # 상위 후보 최대 8건으로 압축하여 평가
        eval_candidates = candidate_docs[:min(len(candidate_docs), 8)]

        doc_snippets = []
        for i, (doc, dist) in enumerate(eval_candidates, 1):
            meta = doc.metadata
            speaker = f"{meta.get('speaker_position', '')} {meta.get('speaker_name', '')}".strip()
            title = meta.get('agenda_title', '안건')
            preview = doc.page_content[:200].replace("\n", " ")
            doc_snippets.append(f"[{i}] 안건: {title} | 발언자: {speaker}\n내용: {preview}")

        prompt = f"""사용자 질문: "{query}"

다음 회의록 후보 문서 중 질문에 답하는 데 가장 관련성이 높고 유용한 문서의 번호를 관련도 순서대로 최대 {top_n}개 나열하고 각 0~100점 점수를 매기세요.

[후보 문서 목록]
{chr(10).join(doc_snippets)}

[응답 포맷 JSON만 출력]
{{
  "rankings": [
    {{"doc_index": 1, "relevance_score": 95, "reason": "직접적인 조례 개정 발언"}},
    {{"doc_index": 3, "relevance_score": 82, "reason": "관련 예산 질의 내용"}}
  ]
}}
"""
        response = self._call_ollama(prompt, temperature=0.0)
        parsed = self._parse_json(response)

        ranked_results = []
        used_indices = set()

        if parsed and "rankings" in parsed and isinstance(parsed["rankings"], list):
            rank = 1
            for item in parsed["rankings"]:
                idx = item.get("doc_index")
                if isinstance(idx, int) and 1 <= idx <= len(eval_candidates) and idx not in used_indices:
                    used_indices.add(idx)
                    doc, dist = eval_candidates[idx - 1]
                    score = float(item.get("relevance_score", 80)) / 100.0
                    ranked_results.append({
                        "rank": rank,
                        "rerank_score": round(score, 4),
                        "initial_distance": round(float(dist), 4),
                        "rerank_reason": item.get("reason", ""),
                        "document": doc,
                    })
                    rank += 1
                    if len(ranked_results) >= top_n:
                        break

        # LLM 파싱 실패나 부족한 건수는 기존 벡터 유사도(Distance) 순으로 보충
        if len(ranked_results) < top_n:
            current_rank = len(ranked_results) + 1
            for idx, (doc, dist) in enumerate(candidate_docs, 1):
                if idx not in used_indices:
                    used_indices.add(idx)
                    sim_score = max(0.0, 1.0 - float(dist))
                    ranked_results.append({
                        "rank": current_rank,
                        "rerank_score": round(sim_score, 4),
                        "initial_distance": round(float(dist), 4),
                        "rerank_reason": "벡터 유사도 기반 기본 순위",
                        "document": doc,
                    })
                    current_rank += 1
                    if len(ranked_results) >= top_n:
                        break

        return ranked_results

    # =================================================================
    # Step 5: 최종 답변 생성 및 JSON 포맷 빌드
    # =================================================================
    def generate_answer(
        self,
        query: str,
        ranked_docs: List[Dict[str, Any]],
    ) -> str:
        """
        재순위화된 근거 문서를 기반으로 최종 답변 생성
        """
        if not ranked_docs:
            return "질문과 관련된 지방의회 회의록 문서를 찾을 수 없습니다."

        context_blocks = []
        for item in ranked_docs:
            doc = item["document"]
            meta = doc.metadata
            header = f"[근거 {item['rank']}] {meta.get('assembly', '')} ({meta.get('date', '')}) - {meta.get('speaker_position', '')} {meta.get('speaker_name', '')}"
            if meta.get("agenda_title"):
                header += f" | 안건: {meta.get('agenda_title')}"
            context_blocks.append(f"{header}\n발언내용:\n{doc.page_content}\n")

        context_str = "\n".join(context_blocks)
        prompt = f"""[질문]
{query}

[회의록 근거 문서]
{context_str}

위 회의록 근거만을 참고하여 사용자의 질문에 객관적이고 정확하게 답변하세요."""

        answer = self._call_ollama(
            prompt=prompt,
            system_prompt=ANSWER_GENERATION_SYSTEM_PROMPT,
            temperature=0.2,
        )
        return answer

    # =================================================================
    # 전체 엔드투엔드 파이프라인 실행
    # =================================================================
    def run(
        self,
        user_query: str,
        top_k: int = 15,
        top_n: int = 4,
        generate_answer: bool = True,
    ) -> Dict[str, Any]:
        """
        사용자 질의 -> 키워드 추출 -> DB 검색 -> Re-ranking -> 최종 JSON 반환
        """
        start_time = time.time()

        # 1. 키워드 및 메타데이터 필터 추출
        extracted = self.extract_keywords_and_filters(user_query)
        keywords = extracted.get("keywords", [])
        filters = extracted.get("filters", {})
        refined_query = extracted.get("refined_query", user_query)

        # 검색 쿼리 구성 (정제 쿼리 + 핵심 키워드 조합)
        search_query = refined_query
        if keywords and not any(kw in search_query for kw in keywords):
            search_query = f"{refined_query} {' '.join(keywords)}"

        # 2. DB 메타데이터/벡터 검색
        candidate_docs = self.search_db(
            query_text=search_query,
            filters=filters,
            top_k=top_k,
        )

        # 3. Re-ranking (재순위화)
        ranked_docs = self.rerank_documents(
            query=user_query,
            candidate_docs=candidate_docs,
            top_n=top_n,
        )

        # 4. 최종 답변 생성 (선택)
        answer_text = ""
        if generate_answer:
            answer_text = self.generate_answer(user_query, ranked_docs)

        elapsed_sec = round(time.time() - start_time, 3)

        # 5. 최종 표준 JSON 아웃풋 포맷팅
        evidences = []
        for item in ranked_docs:
            doc = item["document"]
            meta = doc.metadata
            evidences.append({
                "rank": item["rank"],
                "score": item["rerank_score"],
                "initial_distance": item.get("initial_distance"),
                "reason": item.get("rerank_reason", ""),
                "assembly": meta.get("assembly"),
                "date": meta.get("date"),
                "agenda_title": meta.get("agenda_title"),
                "speaker": f"{meta.get('speaker_position', '')} {meta.get('speaker_name', '')}".strip(),
                "speaker_type": meta.get("speaker_type"),
                "policy_category": meta.get("policy_category"),
                "core_issue": meta.get("core_issue"),
                "keywords": meta.get("keywords"),
                "chunk_info": f"{meta.get('chunk_index', 0)+1}/{meta.get('total_chunks', 1)}",
                "content": doc.page_content,
            })

        output_json = {
            "status": "success",
            "query": user_query,
            "pipeline_flow": "query -> keyword_extraction -> vector_db_search -> reranking -> json_output",
            "extracted": {
                "keywords": keywords,
                "filters": filters,
                "refined_query": refined_query,
            },
            "answer": answer_text,
            "evidences": evidences,
            "metadata": {
                "vector_db": self.db_type,
                "total_retrieved": len(candidate_docs),
                "total_reranked": len(evidences),
                "reranker_type": self.reranker_type,
                "llm_model": self.llm_model,
                "embedding_model": self.embedding_model_name,
                "collection_name": self.collection_name,
                "elapsed_time_sec": elapsed_sec,
            },
        }

        return output_json

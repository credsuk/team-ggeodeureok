"""
PostgreSQL pgvector 기반 VectorStore 및 배치 적재/검색 관리자
(LangChain Community PGVector 연동, GPU 임베딩 및 메타데이터 필터링 지원)
"""

import logging
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document

from src.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    get_pg_connection_string,
    resolve_embedding_model,
)

# LangChain 및 psycopg 경고 숨김
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", message=".*LangChainPendingDeprecationWarning.*")

logger = logging.getLogger(__name__)


class PGVectorIndexer:
    """
    PostgreSQL pgvector 기반 벡터스토어 관리 클래스
    - ChromaIndexer와 호환되는 동일한 인터페이스 제공 (add_batch, query, count)
    - HuggingFace 한국어 임베딩 연동 (CUDA GPU 지원)
    - 메타데이터 필터링 및 배치 적재 지원
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        pre_delete_collection: bool = False,
    ):
        self.connection_string = connection_string or get_pg_connection_string()
        self.collection_name = collection_name or COLLECTION_NAME
        self.model_name = resolve_embedding_model(embedding_model_name or EMBEDDING_MODEL_NAME)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 마스킹된 연결 정보 표시 (비밀번호 숨김)
        safe_conn = self._mask_connection_string(self.connection_string)
        print(f"[임베딩] HuggingFace '{self.model_name}' 모델 로드 중 (디바이스: {self.device.upper()})...")
        print(f"[PostgreSQL pgvector] 컬렉션: '{self.collection_name}' | 연결: '{safe_conn}'")

        # 1. HuggingFaceEmbeddings 초기화
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": False},
        )

        # 2. PGVector 인스턴스 초기화 (JSONB 메타데이터 활성화)
        try:
            self.vectorstore = PGVector(
                connection_string=self.connection_string,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                pre_delete_collection=pre_delete_collection,
                use_jsonb=True,
            )
        except Exception as e:
            self._handle_connection_error(e)

    def _mask_connection_string(self, conn_str: str) -> str:
        """연결 문자열에서 비밀번호 마스킹"""
        import re
        return re.sub(r':([^@]+)@', ':****@', conn_str)

    def _handle_connection_error(self, exc: Exception):
        """연결 실패 시 친절한 원인 및 트러블슈팅 가이드 제공"""
        safe_conn = self._mask_connection_string(self.connection_string)
        guide_msg = (
            "\n" + "=" * 70 + "\n"
            f"[오류] PostgreSQL pgvector 연결에 실패했습니다.\n"
            f"▶ 시도한 연결 주소: {safe_conn}\n"
            f"▶ 원인: {exc}\n\n"
            "💡 [해결 방법 가이드]:\n"
            "1. PostgreSQL 서비스가 실행 중인지 확인하세요.\n"
            "   (로컬 또는 원격 호스트/포트 확인: 기본 localhost:5432)\n"
            "2. 환경 변수로 접속 정보를 설정할 수 있습니다:\n"
            "   $env:PG_HOST = \"localhost\"\n"
            "   $env:PG_PORT = \"5432\"\n"
            "   $env:PG_DATABASE = \"council_db\"\n"
            "   $env:PG_USER = \"postgres\"\n"
            "   $env:PG_PASSWORD = \"your_password\"\n"
            "   또는\n"
            "   $env:PG_CONNECTION_STRING = \"postgresql+psycopg2://user:password@host:port/dbname\"\n\n"
            "3. 대상 DB에 pgvector 확장이 설치되어 있는지 확인하세요:\n"
            "   psql -U postgres -d council_db -c \"CREATE EXTENSION IF NOT EXISTS vector;\"\n"
            + "=" * 70
        )
        raise ConnectionError(guide_msg) from exc

    def add_batch(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        """배치 단위 문서 및 메타데이터 적재"""
        if not documents:
            return 0

        # LangChain Core Document 객체 목록 생성
        docs = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(documents, metadatas)
        ]

        self.vectorstore.add_documents(documents=docs, ids=ids)
        return len(documents)

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        시맨틱 유사도 검색 및 메타데이터 필터링 수행
        반환: List[Tuple[Document, score]]
        """
        try:
            # PGVector filter 형식 변환
            # Chroma의 $and 형태를 PGVector 호환 필터로 단순화
            pg_filter = self._convert_where_to_pg_filter(where)

            results = self.vectorstore.similarity_search_with_score(
                query=query_text,
                k=n_results,
                filter=pg_filter,
            )
            return results
        except Exception as e:
            logger.error(f"[PGVector] 검색 중 예외 발생: {e}")
            return []

    def _convert_where_to_pg_filter(self, where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Chroma 스타일의 where 절을 PGVector 호환 필터로 변환"""
        if not where:
            return None

        # 단일 조건인 경우
        if "$and" not in where:
            return where

        # $and: [{"key1": val1}, {"key2": val2}] 형태 평탄화
        flattened = {}
        for condition in where.get("$and", []):
            if isinstance(condition, dict):
                flattened.update(condition)
        return flattened if flattened else None

    def delete_by_file_stem(self, file_stem: str) -> int:
        """특정 파일 스템에 해당하는 기존 청크 삭제 (재적재 시 중복 방지)"""
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(self.connection_string)
            with engine.connect() as conn:
                res = conn.execute(
                    text("""
                        DELETE FROM langchain_pg_embedding 
                        WHERE collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = :name)
                          AND cmetadata->>'file_stem' = :file_stem
                    """),
                    {"name": self.collection_name, "file_stem": file_stem}
                )
                conn.commit()
                return res.rowcount if hasattr(res, 'rowcount') else 0
        except Exception as e:
            logger.debug(f"[PGVector] delete_by_file_stem 예외: {e}")
            return 0

    def count(self) -> int:
        """현재 컬렉션에 적재된 문서 수 확인"""
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(self.connection_string)
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT count(*) FROM langchain_pg_embedding WHERE collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = :name)"),
                    {"name": self.collection_name}
                )
                row = result.fetchone()
                return int(row[0]) if row else 0
        except Exception:
            return 0

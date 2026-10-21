"""
LangChain Community Chroma 기반 시맨틱 검색 및 메타데이터 필터링 테스트 CLI
(임베딩 모델, 컬렉션 지정 및 LLM 정책 메타데이터 표시/필터링 지원)
"""
import argparse
import sys
import warnings
from typing import Optional

from src.config import (
    COLLECTION_NAME,
    DEFAULT_DB_TYPE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PRESETS,
    get_pg_connection_string,
    resolve_embedding_model,
)
from src.indexer import ChromaIndexer
from src.pg_indexer import PGVectorIndexer

# LangChain 경고 숨김
warnings.filterwarnings("ignore")


def search(
    query_text: str,
    assembly: str = "",
    year: int = 0,
    speaker: str = "",
    category: str = "",
    n_results: int = 5,
    model_name: Optional[str] = None,
    collection_name: Optional[str] = None,
    db_type: str = "chroma",
    pg_conn: Optional[str] = None,
):
    resolved_model = resolve_embedding_model(model_name)
    target_collection = collection_name or COLLECTION_NAME
    selected_db = db_type.lower()

    print("=" * 65)
    print(f" [LangChain {selected_db.upper()} 회의록 시맨틱 검색] ")
    print("=" * 65)
    print(f" 벡터 DB     : {selected_db.upper()}")
    print(f" 검색 질의어 : '{query_text}'")
    print(f" 임베딩 모델 : {resolved_model}")
    print(f" 대상 컬렉션 : {target_collection}")

    # 메타데이터 조건식($and) 생성
    conditions = []
    if assembly:
        conditions.append({"assembly": assembly})
        print(f" 의회 필터   : {assembly}")
    if year > 0:
        conditions.append({"year": year})
        print(f" 연도 필터   : {year}")
    if speaker:
        conditions.append({"speaker_name": speaker})
        print(f" 발언자 필터 : {speaker}")
    if category:
        conditions.append({"policy_category": category})
        print(f" 정책분야 필터: {category}")

    where_clause = None
    if len(conditions) == 1:
        where_clause = conditions[0]
    elif len(conditions) > 1:
        where_clause = {"$and": conditions}

    print("=" * 65)

    if selected_db == "pgvector":
        indexer = PGVectorIndexer(
            connection_string=pg_conn or get_pg_connection_string(),
            collection_name=target_collection,
            embedding_model_name=resolved_model,
        )
    else:
        indexer = ChromaIndexer(
            collection_name=target_collection,
            embedding_model_name=resolved_model,
        )
    results = indexer.query(
        query_text=query_text,
        n_results=n_results,
        where=where_clause,
    )

    if not results:
        print("\n검색 결과가 없습니다.")
        return

    print(f"\n총 {len(results)}건의 결과 발견:\n")
    for idx, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata
        print(f"[{idx}] 유사도 거리(Distance): {score:.4f}")
        print(f"    - 일시/의회: {meta.get('date')} ({meta.get('assembly')})")
        print(f"    - 안건명: {meta.get('agenda_title')}")
        print(f"    - 발언자: {meta.get('speaker_position')} {meta.get('speaker_name')} (구분: {meta.get('speaker_type')})")
        
        # LLM 추출 정책 메타데이터가 존재하는 경우 출력
        if meta.get("core_issue"):
            print(f"    - [정책분야] : {meta.get('policy_category', '기타')}")
            print(f"    - [핵심쟁점] : {meta.get('core_issue')}")
            print(f"    - [정책키워드] : {meta.get('keywords')}")

        print(f"    - 청크 정보: {meta.get('chunk_index', 0)+1}/{meta.get('total_chunks', 1)} (길이: {meta.get('char_length')}자)")
        print(f"    - 발언 본문:")
        preview = doc.page_content.strip().replace("\n", " ")
        if len(preview) > 180:
            preview = preview[:180] + "..."
        print(f"      \"{preview}\"")
        print("-" * 65)


if __name__ == "__main__":
    preset_help = ", ".join([f"'{k}' ({v})" for k, v in EMBEDDING_PRESETS.items()])
    parser = argparse.ArgumentParser(description="지방의회 회의록 시맨틱 검색 도구 (ChromaDB / pgvector)")
    parser.add_argument("query", type=str, help="검색할 질의어 (예: '농지 소유 규제 완화')")
    parser.add_argument(
        "--db",
        type=str,
        choices=["chroma", "pgvector"],
        default=DEFAULT_DB_TYPE,
        help="사용할 벡터 DB ('chroma' 또는 'pgvector')",
    )
    parser.add_argument(
        "--pg-conn",
        type=str,
        default=None,
        help="PostgreSQL 연결 문자열 (미지정 시 환경변수 또는 config 기본값 사용)",
    )
    parser.add_argument("--assembly", type=str, default="", help="특정 의회명 필터 (예: '강원특별자치도의회')")
    parser.add_argument("--year", type=int, default=0, help="연도 필터 (예: 2024)")
    parser.add_argument("--speaker", type=str, default="", help="발언자 이름 필터 (예: '최종수')")
    parser.add_argument("--category", type=str, default="", help="정책 분야 필터 (예: '농림/수산', '보건/복지')")
    parser.add_argument("-k", "--limit", type=int, default=3, help="조회할 결과 개수 (기본: 3)")
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=EMBEDDING_MODEL_NAME,
        help=f"임베딩 모델명 또는 프리셋 별칭. (기본: {EMBEDDING_MODEL_NAME})",
    )
    parser.add_argument(
        "-c", "--collection",
        type=str,
        default=COLLECTION_NAME,
        help=f"검색 대상 ChromaDB 컬렉션 이름 (기본: {COLLECTION_NAME})",
    )

    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    search(
        query_text=args.query,
        assembly=args.assembly,
        year=args.year,
        speaker=args.speaker,
        category=args.category,
        n_results=args.limit,
        model_name=args.model,
        collection_name=args.collection,
        db_type=args.db,
        pg_conn=args.pg_conn,
    )

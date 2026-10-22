"""
지방의회 회의록 하이브리드 적재 파이프라인 (run_ingest.py)
======================================================
1. data/ (JSONL: 정책 카테고리, 쟁점, 키워드, 요약, 실질논의 여부)
2. change_text/ (TXT: 원문 발언, 공식 회의록명, 회의명, 상세 직함/구분)
두 데이터 소스를 row_id 기준으로 1:1 실시간 스트리밍 조인하여,
ChromaDB 또는 PostgreSQL pgvector에 고품질 벡터 및 메타데이터를 일괄 색인합니다.
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from src.config import (
    BASE_DIR,
    BATCH_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    DEFAULT_DB_TYPE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PRESETS,
    MIN_CHAR_LEN,
    get_pg_connection_string,
    resolve_embedding_model,
)
from src.chunker import CouncilMinuteChunker
from src.hybrid_loader import HybridCouncilLoader, HybridCouncilRecord, create_augmented_text
from src.indexer import ChromaIndexer
from src.pg_indexer import PGVectorIndexer


def format_seconds(seconds: float) -> str:
    """초 단위를 시:분:초 형식으로 변환"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}시간 {m:02d}분 {s:02d}초"
    elif m > 0:
        return f"{m}분 {s:02d}초"
    return f"{s}초"


def run_hybrid_ingest(
    data_dir: Path,
    change_text_dir: Path,
    target_year: Optional[str] = None,
    sample_lines: int = 0,
    batch_size: int = BATCH_SIZE,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_char_len: int = MIN_CHAR_LEN,
    substantive_only: bool = True,
    augment_text: bool = True,
    model_name: Optional[str] = None,
    collection_name: Optional[str] = None,
    reset: bool = False,
    db_type: str = "chroma",
    pg_conn: Optional[str] = None,
):
    start_time = time.time()
    resolved_model = resolve_embedding_model(model_name)
    target_collection = collection_name or COLLECTION_NAME
    selected_db = db_type.lower()

    # 하이브리드 로더 초기화
    loader = HybridCouncilLoader(
        data_dir=data_dir,
        change_text_dir=change_text_dir,
        substantive_only=substantive_only,
        min_char_len=min_char_len,
    )

    matched_pairs = loader.discover_matched_pairs(target_year=target_year)
    if not matched_pairs:
        print(f"[오류] '{data_dir}'와 '{change_text_dir}' 간 매칭되는 데이터 쌍을 찾지 못했습니다.")
        return

    print("=" * 70)
    print(f" 🚀 [지방의회 회의록 {selected_db.upper()} 하이브리드 고도화 적재 파이프라인] ")
    print("=" * 70)
    print(f" * 메타데이터 경로 (data)       : {data_dir}")
    print(f" * 원문 텍스트 경로 (change_text): {change_text_dir}")
    print(f" * 대상 파일 쌍                : 총 {len(matched_pairs)}개 파일 쌍 매칭 완료")
    print(f" * 벡터 데이터베이스           : {selected_db.upper()}")
    print(f" * 임베딩 모델                 : {resolved_model}")
    print(f" * 대상 컬렉션                 : {target_collection}")
    print(f" * 실질 정책 발언만 선별 색인 : {'활성화 (단순 의전 멘트 제외)' if substantive_only else '비활성화 (전체 적재)'}")
    print(f" * 의미 보강 텍스트 임베딩     : {'활성화 (메타데이터 쟁점/키워드 주입)' if augment_text else '비활성화'}")
    print(f" * 청크 크기 / 중첩            : {chunk_size}자 / {chunk_overlap}자")
    print(f" * 배치 크기                   : {batch_size}")
    if sample_lines > 0:
        print(f" * [테스트 모드]                : 파일당 최대 {sample_lines:,}행만 적재")
    print("=" * 70)

    # 1. 인덱서 초기화
    if selected_db == "pgvector":
        indexer = PGVectorIndexer(
            connection_string=pg_conn or get_pg_connection_string(),
            collection_name=target_collection,
            embedding_model_name=resolved_model,
            pre_delete_collection=reset,
        )
    else:
        indexer = ChromaIndexer(
            collection_name=target_collection,
            embedding_model_name=resolved_model,
            reset=reset,
        )

    # 2. 청커 초기화
    chunker = CouncilMinuteChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_char_len=min_char_len,
    )

    total_chunks_added = 0
    total_processed_records = 0

    start_dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "-" * 70)
    print(f" [적재 진행 상황] (시작 시각: {start_dt_str})")
    print("-" * 70)

    with tqdm(
        total=len(matched_pairs),
        desc="[전체 파일 진행률]",
        unit="files",
        bar_format="{desc}: {percentage:3.0f}%|{bar:22}| {n_fmt}/{total_fmt} 파일 [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        dynamic_ncols=True,
    ) as file_pbar:

        for jsonl_path, txt_path in matched_pairs:
            short_name = jsonl_path.stem.replace("cleaned_", "")[:18]
            batch_ids: List[str] = []
            batch_docs: List[str] = []
            batch_metas: List[Dict[str, Any]] = []

            # 단일 파일 스트리밍 순회
            record_stream = loader.stream_hybrid_records(
                jsonl_path=jsonl_path,
                txt_path=txt_path,
                max_rows=sample_lines,
            )

            file_chunks = 0
            for record in record_stream:
                total_processed_records += 1

                # 장문 서브청킹
                sub_chunks = chunker.split_text(record.utterance)
                total_c = len(sub_chunks)

                for c_idx, raw_chunk in enumerate(sub_chunks):
                    chunk_id = f"{record.file_stem}_r{record.row_id}_c{c_idx}"

                    # 의미 보강 텍스트 생성 (메타 쟁점 + 키워드 결합)
                    if augment_text:
                        embedding_text = create_augmented_text(record, raw_chunk)
                    else:
                        embedding_text = raw_chunk

                    meta = {
                        "row_id": record.row_id,
                        "date": record.date,
                        "year": record.year,
                        "assembly": record.assembly,
                        "mints_nm": record.mints_nm,
                        "mtg_nm": record.mtg_nm,
                        "agenda_title": record.agenda_title,
                        "speaker_name": record.speaker_name,
                        "speaker_role": record.speaker_role,
                        "speaker_type": record.speaker_type,
                        "policy_category": record.policy_category,
                        "core_issue": record.core_issue,
                        "keywords": ", ".join(record.keywords),
                        "summary": record.summary,
                        "is_substantive": record.is_substantive,
                        "chunk_index": c_idx,
                        "total_chunks": total_c,
                        "file_stem": record.file_stem,
                    }

                    batch_ids.append(chunk_id)
                    batch_docs.append(embedding_text)
                    batch_metas.append(meta)

                    if len(batch_ids) >= batch_size:
                        indexer.add_batch(batch_ids, batch_docs, batch_metas)
                        count_b = len(batch_ids)
                        file_chunks += count_b
                        total_chunks_added += count_b
                        batch_ids.clear()
                        batch_docs.clear()
                        batch_metas.clear()

            # 파일 잔여 청크 적재
            if batch_ids:
                indexer.add_batch(batch_ids, batch_docs, batch_metas)
                count_b = len(batch_ids)
                file_chunks += count_b
                total_chunks_added += count_b

            file_pbar.set_postfix({
                "현재파일": short_name,
                "누적청크": f"{total_chunks_added:,}",
            })
            file_pbar.update(1)

    elapsed_total = time.time() - start_time
    end_dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    avg_chunks_per_sec = total_chunks_added / elapsed_total if elapsed_total > 0 else 0

    print("-" * 70)
    print(" 🏁 [최종 적재 결과 보고서]")
    print(f" * 벡터 데이터베이스   : {selected_db.upper()}")
    print(f" * 시작 / 종료 시각    : {start_dt_str}  ~  {end_dt_str}")
    print(f" * 총 소요 시간         : {format_seconds(elapsed_total)} ({elapsed_total:.1f}초)")
    print(f" * 처리된 유효 발언 건수: {total_processed_records:,}건")
    print(f" * 생성/적재된 청크     : {total_chunks_added:,}개 (평균 속도: 초당 {avg_chunks_per_sec:.1f}개 청크)")
    print(f" * {selected_db.upper()} 전체 청크     : {indexer.count():,}개 (컬렉션: {target_collection})")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="지방의회 회의록 하이브리드 고도화 적재 도구 (data/ 메타 + change_text/ 원문 결합)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(BASE_DIR / "data"),
        help="정제 메타데이터(JSONL) 디렉터리 경로",
    )
    parser.add_argument(
        "--text-dir",
        type=str,
        default=str(BASE_DIR / "change_text"),
        help="원문 텍스트(TXT) 디렉터리 경로",
    )
    parser.add_argument(
        "--year",
        type=str,
        default=None,
        help="특정 연도 파일만 적재 (예: 2024, 2025)",
    )
    parser.add_argument(
        "--sample-lines",
        type=int,
        default=0,
        help="테스트용: 각 파일당 읽을 최대 행 수 (0이면 전체 처리)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="배치 적재 단위 크기",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="장문 서브청킹 기준 크기 (문자 수)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help="서브청킹 중첩 크기 (문자 수)",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=MIN_CHAR_LEN,
        help="최소 발언 길이 필터링 (문자 수)",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="단순 의전/개의 멘트(is_substantive=False)까지 모두 포함하여 적재",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="메타데이터 요약 접두사 없이 순수 원문만 임베딩",
    )
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
        help="PostgreSQL 연결 문자열",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=EMBEDDING_MODEL_NAME,
        help="임베딩 모델명 또는 프리셋 별칭",
    )
    parser.add_argument(
        "-c", "--collection",
        type=str,
        default=COLLECTION_NAME,
        help="적재 대상 컬렉션 이름",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="적재 전 대상 컬렉션 초기화 여부",
    )

    args = parser.parse_args()
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    run_hybrid_ingest(
        data_dir=Path(args.data_dir),
        change_text_dir=Path(args.text_dir),
        target_year=args.year,
        sample_lines=args.sample_lines,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_char_len=args.min_len,
        substantive_only=not args.include_all,
        augment_text=not args.no_augment,
        model_name=args.model,
        collection_name=args.collection,
        reset=args.reset,
        db_type=args.db,
        pg_conn=args.pg_conn,
    )


if __name__ == "__main__":
    main()

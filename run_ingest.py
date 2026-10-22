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
import warnings
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

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


CHECKPOINT_FILE = BASE_DIR / ".ingest_checkpoint.json"


def load_checkpoint(db_type: str, collection: str) -> Dict[str, Any]:
    """저장된 체크포인트 파일 불러오기"""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("db_type") == db_type and data.get("collection") == collection:
                    return data
        except Exception:
            pass
    return {
        "completed_files": [],
        "file_progress": {},
        "total_chunks": 0,
        "total_records": 0,
    }


def save_checkpoint(
    completed_files: List[str],
    file_progress: Dict[str, Dict[str, Any]],
    total_chunks: int,
    total_records: int,
    db_type: str,
    collection: str,
):
    """체크포인트 파일 저장 (행 단위 및 파일 단위 실시간 이어하기 보장)"""
    data = {
        "collection": collection,
        "db_type": db_type,
        "completed_files": sorted(list(completed_files)),
        "file_progress": file_progress,
        "total_chunks": total_chunks,
        "total_records": total_records,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp_file = CHECKPOINT_FILE.with_suffix(".tmp")
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp_file.replace(CHECKPOINT_FILE)
    except Exception:
        try:
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def clear_checkpoint():
    """체크포인트 파일 삭제"""
    for f in (CHECKPOINT_FILE, CHECKPOINT_FILE.with_suffix(".tmp")):
        if f.exists():
            try:
                f.unlink()
            except Exception:
                pass


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
    no_resume: bool = False,
    db_type: str = "chroma",
    pg_conn: Optional[str] = None,
):
    start_time = time.time()
    resolved_model = resolve_embedding_model(model_name)
    target_collection = collection_name or COLLECTION_NAME
    selected_db = db_type.lower()

    # 0. 체크포인트 상태 로드 및 이어하기(Resume) 처리
    if reset or no_resume:
        if reset:
            clear_checkpoint()
        completed_files = set()
        file_progress = {}
        total_chunks_added = 0
        total_processed_records = 0
    else:
        chk = load_checkpoint(selected_db, target_collection)
        completed_files = set(chk.get("completed_files", []))
        file_progress = chk.get("file_progress", {})
        total_chunks_added = chk.get("total_chunks", 0)
        total_processed_records = chk.get("total_records", 0)

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

    # 현재 대상 파일 중 완료된 파일만 필터링
    matched_stems = {jsonl_path.stem for jsonl_path, _ in matched_pairs}
    completed_in_matched = completed_files.intersection(matched_stems)
    active_in_progress = {s: p for s, p in file_progress.items() if s in matched_stems}

    print("=" * 70)
    print(f" 🚀 [지방의회 회의록 {selected_db.upper()} 하이브리드 고도화 적재 파이프라인] ")
    print("=" * 70)
    print(f" * 메타데이터 경로 (data)       : {data_dir}")
    print(f" * 원문 텍스트 경로 (change_text): {change_text_dir}")
    print(f" * 대상 파일 쌍                : 총 {len(matched_pairs)}개 파일 쌍 매칭 완료")
    if completed_in_matched or active_in_progress:
        print(f" * [이어하기 모드 감지]        : 완료 파일 {len(completed_in_matched)}개 제외 (남은 파일: {len(matched_pairs) - len(completed_in_matched)}개)")
        for f_stem, p_info in active_in_progress.items():
            s_name = f_stem.replace("cleaned_", "")[:18]
            print(f"   └ [행 단위 이어하기 예정]   : '{s_name}' -> {p_info.get('last_row_id', 0):,}행부터 재개")
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

    start_dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "-" * 70)
    print(f" [적재 진행 상황] (시작 시각: {start_dt_str})")
    print("-" * 70)

    with tqdm(
        total=len(matched_pairs),
        initial=len(completed_in_matched),
        desc="[전체 파일 진행률]",
        unit="files",
        bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} 파일 [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        dynamic_ncols=True,
    ) as file_pbar:
        try:
            for jsonl_path, txt_path in matched_pairs:
                file_stem = jsonl_path.stem
                short_name = file_stem.replace("cleaned_", "")[:18]

                # 이미 완료된 파일이면 건너뛰기
                if file_stem in completed_files:
                    continue

                file_start_time = time.time()

                # 파일 전체 행 수 사전 집계 (행 단위 진행률 표시용)
                try:
                    with open(jsonl_path, "r", encoding="utf-8") as f:
                        file_total_lines = sum(1 for _ in f)
                except Exception:
                    file_total_lines = 0

                target_lines = min(sample_lines, file_total_lines) if sample_lines > 0 else file_total_lines

                # 행 단위 이어하기 확인
                file_prog = file_progress.get(file_stem, {})
                start_row = file_prog.get("last_row_id", 0)
                file_chunks = file_prog.get("chunks", 0)
                file_records = file_prog.get("records", 0)

                # 만약 이전 테스트などで 이미 target_lines 이상 도달했으면 완료 처리
                if target_lines > 0 and start_row >= target_lines:
                    completed_files.add(file_stem)
                    file_progress.pop(file_stem, None)
                    save_checkpoint(
                        completed_files=list(completed_files),
                        file_progress=file_progress,
                        total_chunks=total_chunks_added,
                        total_records=total_processed_records,
                        db_type=selected_db,
                        collection=target_collection,
                    )
                    file_pbar.update(1)
                    continue

                if start_row > 0:
                    tqdm.write(f" ℹ️  [{short_name}] 이어하기: {start_row:,}행까지 완료됨 -> {start_row + 1:,}행부터 재개 (기존 청크: {file_chunks:,}개)")
                    # 이전 불완전 실행 시 start_row 초과 번호로 들어간 청크가 있다면 정리
                    indexer.delete_chunks_after_row(file_stem, start_row)
                else:
                    # 새로운 파일 시작: 이전 불완전 실행 시 남아있을 수 있는 청크 사전 정리 (중복 방지)
                    indexer.delete_by_file_stem(file_stem)
                    file_chunks = 0
                    file_records = 0

                batch_ids: List[str] = []
                batch_docs: List[str] = []
                batch_metas: List[Dict[str, Any]] = []

                # 단일 파일 스트리밍 순회
                record_stream = loader.stream_hybrid_records(
                    jsonl_path=jsonl_path,
                    txt_path=txt_path,
                    max_rows=sample_lines,
                    start_row=start_row,
                )

                last_committed_row = start_row
                current_record_row = start_row

                with tqdm(
                    total=target_lines,
                    initial=start_row,
                    desc=f" └ [{short_name}]",
                    unit="행",
                    leave=False,
                    dynamic_ncols=True,
                    bar_format="{desc}: {percentage:3.0f}%|{bar:16}| {n_fmt}/{total_fmt}행 [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                ) as row_pbar:
                    for record in record_stream:
                        current_record_row = record.row_id
                        total_processed_records += 1
                        file_records += 1

                        if current_record_row > row_pbar.n:
                            row_pbar.update(current_record_row - row_pbar.n)

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

                        # 레코드 단위 완료 후 배치 크기 도달 시 적재 (레코드 분할 방지)
                        if len(batch_ids) >= batch_size:
                            count_b = len(batch_ids)
                            indexer.add_batch(batch_ids, batch_docs, batch_metas)
                            file_chunks += count_b
                            total_chunks_added += count_b
                            last_committed_row = current_record_row

                            # 실시간 행 단위 체크포인트 저장
                            file_progress[file_stem] = {
                                "last_row_id": last_committed_row,
                                "chunks": file_chunks,
                                "records": file_records,
                            }
                            save_checkpoint(
                                completed_files=list(completed_files),
                                file_progress=file_progress,
                                total_chunks=total_chunks_added,
                                total_records=total_processed_records,
                                db_type=selected_db,
                                collection=target_collection,
                            )

                            batch_ids.clear()
                            batch_docs.clear()
                            batch_metas.clear()

                            f_elapsed = max(0.1, time.time() - file_start_time)
                            cps = file_chunks / f_elapsed
                            row_pbar.set_postfix({
                                "적재청크": f"{file_chunks:,}",
                                "속도": f"{cps:.1f}c/s",
                            })

                    # 파일 잔여 청크 적재
                    if batch_ids:
                        count_b = len(batch_ids)
                        indexer.add_batch(batch_ids, batch_docs, batch_metas)
                        file_chunks += count_b
                        total_chunks_added += count_b
                        last_committed_row = current_record_row
                        batch_ids.clear()
                        batch_docs.clear()
                        batch_metas.clear()

                    # 행 진행률 바 마무리
                    if row_pbar.n < target_lines:
                        row_pbar.update(target_lines - row_pbar.n)

                # 파일 완료 여부 판정
                if sample_lines == 0 or current_record_row >= file_total_lines:
                    completed_files.add(file_stem)
                    file_progress.pop(file_stem, None)
                else:
                    file_progress[file_stem] = {
                        "last_row_id": last_committed_row,
                        "chunks": file_chunks,
                        "records": file_records,
                    }

                save_checkpoint(
                    completed_files=list(completed_files),
                    file_progress=file_progress,
                    total_chunks=total_chunks_added,
                    total_records=total_processed_records,
                    db_type=selected_db,
                    collection=target_collection,
                )

                file_pbar.set_postfix({
                    "완료": short_name,
                    "누적청크": f"{total_chunks_added:,}",
                })
                file_pbar.update(1)

        except KeyboardInterrupt:
            print("\n" + "=" * 70)
            print(" ⚠️  [일시 중단 안내] 사용자에 의해 적재 작업이 중단되었습니다 (Ctrl+C).")
            if batch_ids:
                try:
                    print(f" * 메모리 대기 중인 {len(batch_ids)}개 청크를 데이터베이스에 안전하게 커밋 중...")
                    count_b = len(batch_ids)
                    indexer.add_batch(batch_ids, batch_docs, batch_metas)
                    file_chunks += count_b
                    total_chunks_added += count_b
                    last_committed_row = current_record_row
                    file_progress[file_stem] = {
                        "last_row_id": last_committed_row,
                        "chunks": file_chunks,
                        "records": file_records,
                    }
                    save_checkpoint(
                        completed_files=list(completed_files),
                        file_progress=file_progress,
                        total_chunks=total_chunks_added,
                        total_records=total_processed_records,
                        db_type=selected_db,
                        collection=target_collection,
                    )
                    print(f" * [저장 완료] '{short_name}' 파일의 {last_committed_row:,}행까지 안전하게 커밋되었습니다.")
                except Exception as e:
                    print(f" * [경고] 잔여 배치 커밋 실패: {e}")

            print(f" * 완전히 완료된 파일: {len(completed_files)}개")
            if file_progress:
                for f_stem, p_data in file_progress.items():
                    s_name = f_stem.replace("cleaned_", "")[:18]
                    print(f" * 진행 중이던 파일: '{s_name}' -> {p_data.get('last_row_id', 0):,}행까지 저장됨")
            print(" * 다시 'python run_ingest.py'를 실행하시면 중단된 행에서부터 바로 이어서 진행됩니다.")
            print("=" * 70)
            sys.exit(0)

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
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="저장된 체크포인트를 무시하고 처음부터 다시 적재",
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
        no_resume=args.no_resume,
        db_type=args.db,
        pg_conn=args.pg_conn,
    )


if __name__ == "__main__":
    main()

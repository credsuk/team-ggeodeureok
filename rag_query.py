"""
지방의회 회의록 RAG 모델 파이프라인 실행 CLI
(사용자 쿼리 -> 키워드 추출 -> DB 검색 -> Re-ranking -> 최종 JSON 출력)
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

# Windows 콘솔 UTF-8 입출력 강제
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PRESETS,
    resolve_embedding_model,
)
from src.pipeline import CouncilRAGPipeline


def main():
    parser = argparse.ArgumentParser(
        description="지방의회 회의록 RAG 파이프라인 (키워드 추출 -> 벡터DB 검색 -> 재순위화 -> JSON 출력)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default="",
        help="사용자 질문/질의어 (생략 시 대화형 모드)",
    )
    parser.add_argument(
        "-q", "--query-text",
        type=str,
        default="",
        help="사용자 질문/질의어 (옵션 플래그 형식)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="임베딩 모델명 또는 별칭 (예: arctic-ko, kure, kosimcse 등)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help="검색 대상 ChromaDB 컬렉션 이름",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gemma4:e2b",
        help="Ollama 로컬 LLM 모델명 (예: gemma4:e2b, gemma4:e4b, qwen2.5-coder:7b 등)",
    )
    parser.add_argument(
        "--reranker",
        type=str,
        choices=["llm", "cross-encoder", "none"],
        default="llm",
        help="재순위화(Re-ranking) 방식 선택 ('llm', 'cross-encoder', 'none')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="1차 DB 벡터/메타데이터 검색 후보 문서 개수",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=4,
        help="재순위화(Re-ranking) 후 최종 선별할 근거 문서 개수",
    )
    parser.add_argument(
        "--no-answer",
        action="store_true",
        help="LLM 답변 생성 생략하고 검색/재순위화 근거 문서만 반환",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        default="",
        help="최종 JSON 결과물을 저장할 파일 경로 (예: result.json)",
    )
    parser.add_argument(
        "--raw-json",
        action="store_true",
        help="로그/설명 문구 없이 순수 JSON만 stdout으로 출력 (파이프라인 연동용)",
    )

    args = parser.parse_args()

    # 파이프라인 초기화
    pipeline = CouncilRAGPipeline(
        collection_name=args.collection,
        embedding_model_name=args.model,
        llm_model=args.llm_model,
        reranker_type=args.reranker,
    )

    def process_and_output(user_query: str):
        if not args.raw_json:
            print("\n" + "=" * 65)
            print(" [지방의회 회의록 RAG 모델 파이프라인 가동] ")
            print("=" * 65)
            print(f" ▶ 사용자 질문 : '{user_query}'")
            print(f" ▶ 임베딩 모델 : {pipeline.embedding_model_name}")
            print(f" ▶ 로컬 LLM    : {pipeline.llm_model}")
            print(f" ▶ Re-ranker   : {pipeline.reranker_type}")
            print("-" * 65)
            print(" [1] 키워드/필터 추출 및 DB 하이브리드 검색 진행 중...")

        result_dict = pipeline.run(
            user_query=user_query,
            top_k=args.top_k,
            top_n=args.top_n,
            generate_answer=not args.no_answer,
        )

        formatted_json = json.dumps(result_dict, ensure_ascii=False, indent=2)

        if args.raw_json:
            print(formatted_json)
        else:
            extracted = result_dict.get("extracted", {})
            print(f" [2] 추출 키워드: {extracted.get('keywords')}")
            print(f"     적용 필터  : {extracted.get('filters')}")
            print(f"     정제 쿼리  : {extracted.get('refined_query')}")
            print(f" [3] 1차 검색   : {result_dict['metadata']['total_retrieved']}건 수집 완료")
            print(f" [4] 재순위화   : 상위 {result_dict['metadata']['total_reranked']}건 선별 완료")
            print(f" [5] 소요 시간   : {result_dict['metadata']['elapsed_time_sec']}초")
            print("=" * 65)
            print(" 📄 [최종 JSON 아웃풋] ")
            print("=" * 65)
            print(formatted_json)

        if args.output_file:
            out_path = Path(args.output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(formatted_json)
            if not args.raw_json:
                print(f"\n[저장 완료] 결과가 '{out_path.resolve()}'에 저장되었습니다.")

    # 단일 질의 실행
    actual_query = args.query_text.strip() or args.query.strip()
    if actual_query:
        process_and_output(actual_query)
        return

    # 질의어 없을 때 대화형 루프
    print("\n" + "=" * 65)
    print(" [지방의회 회의록 대화형 RAG 검색 CLI] (종료: 'q' 또는 'exit')")
    print("=" * 65)
    while True:
        try:
            q = input("\n질문 입력 >> ").strip()
            if not q:
                continue
            if q.lower() in ["q", "exit", "quit"]:
                print("종료합니다.")
                break
            process_and_output(q)
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break


if __name__ == "__main__":
    main()

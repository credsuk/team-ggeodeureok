"""
지방의회 정책 이슈 시계열 확산 분석 및 미래 예측 실행 CLI
(주최 측 제출 규격: 이슈키워드 / 등장시점 / 타 지방의회로의 흐름 / 향후 예측 / 근거 회의록)
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
    DEFAULT_DB_TYPE,
    EMBEDDING_MODEL_NAME,
    get_pg_connection_string,
    resolve_embedding_model,
)
from src.trend_analyzer import PolicyTrendAnalyzer


def render_markdown_table(row_data: dict) -> str:
    """주최 측 제출 양식 마크다운 표 생성"""
    keyword = row_data.get("이슈키워드", "")
    first_obs = row_data.get("등장시점", "").replace("\n", "<br>")
    flow = row_data.get("타 지방의회로의 흐름", "").replace("\n", "<br>")
    pred = row_data.get("향후 예측", "").replace("\n", " ")
    evid = row_data.get("근거 회의록", "").replace("\n", "<br>")

    lines = [
        "| 이슈키워드 | 등장시점 | 타 지방의회로의 흐름 | 향후 예측 | 근거 회의록 |",
        "| :--- | :--- | :--- | :--- | :--- |",
        f"| **{keyword}** | {first_obs} | {flow} | {pred} | {evid} |",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="지방의회 정책 이슈 시계열 확산 분석 및 미래 예측 AI 모델 (경진대회 제출 양식)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "keyword",
        type=str,
        nargs="?",
        default="",
        help="분석할 주요 이슈 키워드 (예: '지역소멸', '반도체', '농지규제')",
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
        help="PostgreSQL 연결 문자열 (기본: 환경변수 또는 config 기본값)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=EMBEDDING_MODEL_NAME,
        help="임베딩 모델명 또는 별칭 (예: arctic-ko, kure 등)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help="검색 대상 컬렉션 이름",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gemma4:e2b",
        help="로컬 LLM 모델명 (Ollama 오픈 모델)",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=None,
        help="분석 기준일 (YYYY-MM-DD 형식, 기준일 이후 정보는 분석/예측에서 배제)",
    )
    parser.add_argument(
        "-k", "--retrieval-k",
        type=int,
        default=30,
        help="시계열 분석을 위해 수집할 후보 청크 수",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="",
        help="결과를 저장할 JSON 파일 경로 (예: submission_result.json)",
    )
    parser.add_argument(
        "--raw-json",
        action="store_true",
        help="순수 JSON 포맷만 stdout으로 출력",
    )

    args = parser.parse_args()

    # 분석기 초기화
    analyzer = PolicyTrendAnalyzer(
        db_type=args.db,
        collection_name=args.collection,
        embedding_model_name=args.model,
        llm_model=args.llm_model,
        pg_connection_string=args.pg_conn,
        cutoff_date=args.cutoff_date,
    )

    def process_keyword(kw: str):
        if not args.raw_json:
            print("\n" + "=" * 70)
            print(f" 📊 [지방의회 정책 이슈 시계열 확산 및 미래 예측 모델 가동] ")
            print("=" * 70)
            print(f" ▶ 분석 이슈 키워드 : '{kw}'")
            print(f" ▶ 벡터 데이터베이스 : {analyzer.db_type.upper()}")
            print(f" ▶ 임베딩 모델      : {analyzer.embedding_model_name}")
            print(f" ▶ 자체 실행형 LLM  : {analyzer.llm_model}")
            if args.cutoff_date:
                print(f" ▶ 분석 기준일      : {args.cutoff_date} (이후 정보 배제)")
            print("-" * 70)
            print(" >> 시계열 데이터 수집, 최초 등장 시점 추적 및 타 의회 확산 분석 중...")

        result = analyzer.analyze_issue(
            issue_keyword=kw,
            retrieval_k=args.retrieval_k,
        )

        formatted_json = json.dumps(result, ensure_ascii=False, indent=2)

        if args.raw_json:
            print(formatted_json)
        else:
            if result.get("status") == "success":
                table_row = result["submission_table_row"]
                print("\n" + "=" * 70)
                print(" 📋 [경진대회 제출 양식 결과표]")
                print("=" * 70)
                print(f"• 이슈키워드: {table_row['이슈키워드']}")
                print(f"• 등장시점  : {table_row['등장시점']}")
                print(f"\n• 타 지방의회로의 흐름:\n{table_row['타 지방의회로의 흐름']}")
                print(f"\n• 향후 예측:\n  {table_row['향후 예측']}")
                print(f"\n• 근거 회의록 및 원문 구간:\n{table_row['근거 회의록']}")
                print("-" * 70)
                print("\n[마크다운 표 형식]\n")
                print(render_markdown_table(table_row))
                print("\n" + "=" * 70)
                print(f" [분석 메타데이터] 소요시간: {result['metadata']['elapsed_time_sec']}초 | 분석 청크 수: {result['metadata']['analyzed_chunks']}개")
                print("=" * 70)
            else:
                print(f"\n[알림] {result.get('message', '결과를 찾을 수 없습니다.')}")

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                f.write(formatted_json)
            if not args.raw_json:
                print(f"\n[저장 완료] 분석 결과가 '{out_p.resolve()}'에 저장되었습니다.")

    # 단일 키워드 실행
    actual_kw = args.keyword.strip()
    if actual_kw:
        process_keyword(actual_kw)
        return

    # 대화형 모드
    print("\n" + "=" * 70)
    print(" [정책 이슈 시계열 확산 및 미래 예측 CLI] (종료: 'q' 또는 'exit')")
    print("=" * 70)
    while True:
        try:
            kw = input("\n분석할 이슈 키워드 입력 (예: 지역소멸, 조례안, 반도체) >> ").strip()
            if not kw:
                continue
            if kw.lower() in ["q", "exit", "quit"]:
                print("종료합니다.")
                break
            process_keyword(kw)
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break


if __name__ == "__main__":
    main()

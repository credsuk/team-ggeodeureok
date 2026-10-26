#!/usr/bin/env python3
"""중간제출 데모의 필수·선택 구성요소를 한 번에 점검한다."""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configured(name: str, relative: str) -> Path:
    return Path(
        os.environ.get(name, str(PROJECT_ROOT / relative))
    ).expanduser().resolve()


def mark(ok: bool) -> str:
    return "PASS" if ok else "MISS"


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def main() -> None:
    paths = {
        "이슈 웹 JSON": configured(
            "COUNCIL_ISSUE_PAYLOAD", "outputs/web/birth_rate_issue.json"
        ),
        "전체 벡터": configured(
            "COUNCIL_EMBEDDING_DIR", "data/embeddings/arctic_raw_v1"
        ) / "embeddings.f16.npy",
        "임베딩 메타데이터": configured(
            "COUNCIL_EMBEDDING_DIR", "data/embeddings/arctic_raw_v1"
        ) / "chunk_metadata.parquet",
        "부모 청크": configured(
            "COUNCIL_PARENT_PATH", "data/canonical_v3/chunks_v1/parents.parquet"
        ),
    }
    print("=== 중간제출 필수 파일 ===")
    missing = []
    for label, path in paths.items():
        ok = path.is_file()
        print(f"{mark(ok):4s} | {label:18s} | {path}")
        if not ok:
            missing.append(label)

    payload_path = paths["이슈 웹 JSON"]
    if payload_path.is_file():
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            valid = payload.get("schema_version") == "council_issue_web_v1"
            print(f"{mark(valid):4s} | 웹 JSON 스키마")
            if not valid:
                missing.append("웹 JSON 스키마")
        except (OSError, json.JSONDecodeError) as error:
            print(f"FAIL | 웹 JSON 읽기 | {error}")
            missing.append("웹 JSON 읽기")

    print("\n=== 실행 서비스 ===")
    postgres = port_open("127.0.0.1", int(os.environ.get("POSTGRES_PORT", "5432")))
    ollama = port_open("127.0.0.1", 11434)
    print(f"{mark(postgres):4s} | pgvector 포트 5432 | 검색·Agent 전체 근거 검색")
    print(f"{mark(ollama):4s} | Ollama 포트 11434  | 자연어 답변 (미실행 시 요약 대체)")

    external = configured(
        "COUNCIL_EXTERNAL_CONTEXT", "outputs/external/external_context.json"
    )
    print(f"{mark(external.is_file()):4s} | 외부지표 참고 JSON | 선택 기능")

    if missing:
        print(f"\n최종 결과: FAIL | 필수 항목 누락: {', '.join(missing)}")
        sys.exit(1)
    print("\n최종 결과: PASS | 중간제출 앱 실행 가능")


if __name__ == "__main__":
    main()

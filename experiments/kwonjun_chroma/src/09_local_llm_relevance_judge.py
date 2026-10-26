#!/usr/bin/env python3
"""블라인드 검색 pool을 로컬 Ollama LLM으로 2회 판정하고 불일치를 자동 재심한다."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = """당신은 한국 광역 지방의회 회의록 검색의 관련성 평가자다.
검색 시스템, 검색 순위, 검색 점수는 알 수 없으며 오직 질의와 원문만 평가한다.

등급 기준:
2 = 질의의 정책 문제, 대책, 사업, 예산, 제도, 효과 또는 논쟁을 직접적이고 구체적으로 다룬다.
1 = 질의 주제를 실제로 언급하고 배경·현황·간접 논의를 제공하지만 직접적인 정책 내용은 약하다.
0 = 무관하거나 단어만 우연히 겹치며, 인사·개의·산회·표결 등 절차 발언에 불과하다.

핵심 규칙:
- 질의 단어가 없더라도 의미가 직접 일치하면 관련으로 평가한다.
- 같은 단어가 있어도 정책적 의미가 다르면 0이다.
- 원문에 없는 내용을 추론하거나 외부 지식을 사용하지 않는다.
- evidence는 원문에서 짧게 복사하며, 0이면 빈 문자열도 가능하다.
- 반드시 요청된 JSON 객체만 출력한다."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=PROJECT_ROOT / "outputs" / "retrieval_eval" / "arctic_raw_vs_enriched_v1" / "blind_pool.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "retrieval_eval" / "arctic_raw_vs_enriched_v1" / "auto_judgements",
    )
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-text-chars", type=int, default=1400)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--reuse-labels", type=Path,
        help="동일 query_id+chunk_id의 이전 relevance_labels.jsonl을 재사용해 신규 후보만 판정",
    )
    parser.add_argument(
        "--single-pass", action="store_true",
        help="빠른 예비 비교용: 전체 pool을 1회 판정하고 2차 판정·재심을 생략",
    )
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"JSONL 파싱 실패 {path}:{line_number}: {exc}") from exc
            rows.append(value)
    return rows


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["pair_id"]: row for row in read_jsonl(path)}


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].lstrip()
    return json.loads(text)


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: int, retries: int) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("데이터 외부 전송 방지를 위해 Ollama URL은 localhost만 허용합니다")
        self.url = base_url.rstrip("/") + "/api/chat"
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def chat(self, user_prompt: str, seed: int) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "stream": False,
            # 관련도 분류는 긴 추론보다 일관된 JSON과 처리 속도가 중요하다.
            "think": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": 0,
                "seed": seed,
                "num_ctx": 8192,
            },
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                request = urllib.request.Request(
                    self.url, data=body,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    value = json.loads(response.read().decode("utf-8"))
                return parse_json_content(value["message"]["content"])
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                print(f"Ollama 재시도 {attempt}/{self.retries}: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"Ollama 호출 실패: {last_error}")


def document_payload(row: dict[str, Any], max_chars: int) -> dict[str, Any]:
    return {
        "pair_id": row["pair_id"],
        "회의정보": {
            "의회": row.get("council"),
            "일자": row.get("meeting_date"),
            "위원회": row.get("committee"),
            "회의록": row.get("meeting_name"),
            "안건": row.get("agenda_title"),
        },
        "원문": str(row.get("text") or "")[:max_chars],
    }


def judge_prompt(query: str, rows: list[dict[str, Any]], max_chars: int, pass_number: int) -> str:
    mode = (
        "각 문서를 독립적으로 평가하고 직접적인 정책 관련성을 중심으로 판정하라."
        if pass_number == 1 else
        "보수적인 감사자 관점에서 각 문서를 처음부터 독립적으로 재평가하라. 단순 단어 중복을 엄격히 배제하라."
    )
    documents = [document_payload(row, max_chars) for row in rows]
    return f"""{mode}

질의: {query}

문서 목록:
{json.dumps(documents, ensure_ascii=False)}

아래 형식으로 모든 pair_id를 정확히 한 번씩 출력하라.
{{
  "labels": [
    {{
      "pair_id": "입력 pair_id",
      "grade": 0,
      "confidence": 0.0,
      "evidence": "원문 근거",
      "reason": "한 문장 판정 이유"
    }}
  ]
}}

grade는 0, 1, 2 중 하나이고 confidence는 0부터 1 사이다."""


def normalize_labels(result: dict[str, Any], expected: set[str], judge_pass: int) -> list[dict[str, Any]]:
    labels = result.get("labels")
    if not isinstance(labels, list):
        raise RuntimeError("Ollama 출력에 labels 배열이 없습니다")
    normalized = []
    seen = set()
    for item in labels:
        pid = str(item.get("pair_id", ""))
        if pid not in expected or pid in seen:
            continue
        grade = int(item.get("grade"))
        if grade not in {0, 1, 2}:
            raise RuntimeError(f"잘못된 grade: {grade}")
        confidence = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
        normalized.append({
            "pair_id": pid,
            "grade": grade,
            "confidence": round(confidence, 4),
            "evidence": str(item.get("evidence") or "")[:300],
            "reason": str(item.get("reason") or "")[:500],
            "judge_pass": judge_pass,
        })
        seen.add(pid)
    if seen != expected:
        raise RuntimeError(f"라벨 누락: {sorted(expected - seen)}")
    return normalized


def request_labels(
    client: OllamaClient,
    prompt: str,
    expected: set[str],
    judge_pass: int,
    seed: int,
) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            return normalize_labels(client.chat(prompt, seed + attempt - 1), expected, judge_pass)
        except (RuntimeError, TypeError, ValueError) as exc:
            last_error = exc
            print(f"라벨 형식 재시도 {attempt}/3: {exc}", flush=True)
    raise RuntimeError(f"라벨 형식 복구 실패: {last_error}")


def run_pass(
    client: OllamaClient,
    pool: list[dict[str, Any]],
    checkpoint_path: Path,
    pass_number: int,
    batch_size: int,
    max_chars: int,
) -> dict[str, dict[str, Any]]:
    completed = load_checkpoint(checkpoint_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        if row["pair_id"] not in completed:
            grouped[row["query_id"]].append(row)
    total_remaining = sum(map(len, grouped.values()))
    processed = 0
    for query_id in sorted(grouped):
        rows = grouped[query_id]
        query = rows[0]["query"]
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            expected = {row["pair_id"] for row in batch}
            prompt = judge_prompt(query, batch, max_chars, pass_number)
            seed = 1701 if pass_number == 1 else 2903
            try:
                labels = request_labels(client, prompt, expected, pass_number, seed)
            except RuntimeError:
                labels = []
                print("배치 판정 실패, 문서별 판정으로 자동 전환", flush=True)
                for row in batch:
                    single_prompt = judge_prompt(query, [row], max_chars, pass_number)
                    labels.extend(request_labels(
                        client, single_prompt, {row["pair_id"]}, pass_number, seed
                    ))
            append_jsonl(checkpoint_path, labels)
            for label in labels:
                completed[label["pair_id"]] = label
            processed += len(batch)
            print(
                f"judge{pass_number}: {processed:,}/{total_remaining:,} | {query_id}",
                flush=True,
            )
    return completed


def adjudication_prompt(
    query: str,
    rows: list[dict[str, Any]],
    first: dict[str, dict[str, Any]],
    second: dict[str, dict[str, Any]],
    max_chars: int,
) -> str:
    documents = []
    for row in rows:
        item = document_payload(row, max_chars)
        pid = row["pair_id"]
        item["1차판정"] = first[pid]
        item["2차판정"] = second[pid]
        documents.append(item)
    return f"""두 독립 평가가 불일치한 문서를 최종 재심하라.
두 판정의 다수결이 아니라 질의와 원문을 다시 읽고 등급 기준에 따라 최종 결정하라.

질의: {query}
문서 목록:
{json.dumps(documents, ensure_ascii=False)}

출력 형식은 labels 배열이며 pair_id, grade, confidence, evidence, reason을 포함한다."""


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if args.batch_size < 1:
        raise ValueError("batch-size는 1 이상이어야 합니다")
    output_dir.mkdir(parents=True, exist_ok=True)
    pass1_path = output_dir / "judge_pass1.jsonl"
    pass2_path = output_dir / "judge_pass2.jsonl"
    adjudication_path = output_dir / "judge_adjudication.jsonl"
    final_path = output_dir / "relevance_labels.jsonl"
    summary_path = output_dir / "judgement_summary.json"
    if args.restart:
        for path in (pass1_path, pass2_path, adjudication_path, final_path, summary_path):
            if path.exists():
                path.unlink()
    elif final_path.exists() and summary_path.exists():
        print(f"이미 완료된 자동 판정입니다: {final_path}")
        return

    pool = read_jsonl(input_path)
    pair_ids = [row["pair_id"] for row in pool]
    if not pool or len(pair_ids) != len(set(pair_ids)):
        raise RuntimeError("pool이 비었거나 pair_id가 중복되었습니다")
    if args.reuse_labels:
        reuse_path = args.reuse_labels.expanduser().resolve()
        if not reuse_path.is_file():
            raise FileNotFoundError(reuse_path)
        reusable = load_checkpoint(reuse_path)
        existing = load_checkpoint(pass1_path)
        seeded = []
        for pid in pair_ids:
            if pid in existing or pid not in reusable:
                continue
            old = reusable[pid]
            seeded.append({
                "pair_id": pid,
                "grade": int(old["grade"]),
                "confidence": float(old.get("confidence", 0.0)),
                "evidence": str(old.get("evidence") or "")[:300],
                "reason": str(old.get("reason") or "이전 동일 질의-청크 판정 재사용")[:500],
                "judge_pass": 1,
                "reused": True,
            })
        if seeded:
            append_jsonl(pass1_path, seeded)
        print(f"이전 라벨 재사용: {len(seeded):,}/{len(pair_ids):,}", flush=True)
    client = OllamaClient(args.ollama_url, args.model, args.timeout, args.max_retries)
    print(f"로컬 자동판정 시작: {len(pool):,} pairs | model={args.model}", flush=True)
    first = run_pass(client, pool, pass1_path, 1, args.batch_size, args.max_text_chars)

    if args.single_pass:
        row_by_id = {row["pair_id"]: row for row in pool}
        grade_counts: Counter[int] = Counter()
        final_rows = []
        for pid in pair_ids:
            result = first[pid]
            grade = int(result["grade"])
            grade_counts[grade] += 1
            row = row_by_id[pid]
            final_rows.append({
                "pair_id": pid,
                "query_id": row["query_id"],
                "chunk_id": row["chunk_id"],
                "grade": grade,
                "confidence": result["confidence"],
                "evidence": result["evidence"],
                "reason": result["reason"],
                "judge1_grade": grade,
                "judge2_grade": None,
                "status": "single_pass_preliminary",
                "judge_model": args.model,
            })
        with final_path.open("w", encoding="utf-8") as file:
            for row in final_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "security": "Ollama localhost 전용, 회의록 외부 전송 없음",
            "model": args.model,
            "thinking": False,
            "judge_mode": "single_pass_preliminary",
            "pairs": len(pair_ids),
            "grade_counts": {str(key): value for key, value in sorted(grade_counts.items())},
            "final_labels": str(final_path),
            "limitation": "예비 raw/enriched 선택용이며 최종 모델 비교에는 이중 판정 또는 사람 검수를 사용해야 함",
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n=== 로컬 LLM 단일 판정 완료(예비 비교용) ===")
        print(f"pair: {len(pair_ids):,} | 등급: {dict(sorted(grade_counts.items()))}")
        print(f"labels: {final_path}")
        print(f"summary: {summary_path}")
        return

    second = run_pass(client, pool, pass2_path, 2, args.batch_size, args.max_text_chars)

    row_by_id = {row["pair_id"]: row for row in pool}
    disagreements = [pid for pid in pair_ids if first[pid]["grade"] != second[pid]["grade"]]
    adjudicated = load_checkpoint(adjudication_path)
    pending = [pid for pid in disagreements if pid not in adjudicated]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pid in pending:
        grouped[row_by_id[pid]["query_id"]].append(row_by_id[pid])
    processed = 0
    for query_id in sorted(grouped):
        rows = grouped[query_id]
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            expected = {row["pair_id"] for row in batch}
            prompt = adjudication_prompt(batch[0]["query"], batch, first, second, args.max_text_chars)
            try:
                labels = request_labels(client, prompt, expected, 3, 4127)
            except RuntimeError:
                labels = []
                print("재심 배치 실패, 문서별 재심으로 자동 전환", flush=True)
                for row in batch:
                    single_prompt = adjudication_prompt(
                        row["query"], [row], first, second, args.max_text_chars
                    )
                    labels.extend(request_labels(
                        client, single_prompt, {row["pair_id"]}, 3, 4127
                    ))
            append_jsonl(adjudication_path, labels)
            for label in labels:
                adjudicated[label["pair_id"]] = label
            processed += len(batch)
            print(f"재심: {processed:,}/{len(pending):,} | {query_id}", flush=True)

    grade_counts: Counter[int] = Counter()
    final_rows = []
    for pid in pair_ids:
        if first[pid]["grade"] == second[pid]["grade"]:
            grade = first[pid]["grade"]
            confidence = round((first[pid]["confidence"] + second[pid]["confidence"]) / 2, 4)
            evidence = first[pid]["evidence"] or second[pid]["evidence"]
            reason = f"2회 일치: {first[pid]['reason']}"
            status = "double_agreement"
        else:
            result = adjudicated[pid]
            grade = result["grade"]
            confidence = result["confidence"]
            evidence = result["evidence"]
            reason = result["reason"]
            status = "auto_adjudicated"
        grade_counts[grade] += 1
        row = row_by_id[pid]
        final_rows.append({
            "pair_id": pid,
            "query_id": row["query_id"],
            "chunk_id": row["chunk_id"],
            "grade": grade,
            "confidence": confidence,
            "evidence": evidence,
            "reason": reason,
            "judge1_grade": first[pid]["grade"],
            "judge2_grade": second[pid]["grade"],
            "status": status,
            "judge_model": args.model,
        })
    with final_path.open("w", encoding="utf-8") as file:
        for row in final_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    agreement = 1.0 - len(disagreements) / len(pair_ids)
    summary = {
        "security": "Ollama localhost 전용, 회의록 외부 전송 없음",
        "model": args.model,
        "thinking": False,
        "pairs": len(pair_ids),
        "double_agreement": len(pair_ids) - len(disagreements),
        "auto_adjudicated": len(disagreements),
        "agreement_rate": round(agreement, 6),
        "grade_counts": {str(key): value for key, value in sorted(grade_counts.items())},
        "final_labels": str(final_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 로컬 LLM 자동 관련성 판정 완료 ===")
    print(f"pair: {len(pair_ids):,} | 2회 일치율: {agreement:.2%} | 자동 재심: {len(disagreements):,}")
    print(f"등급: {dict(sorted(grade_counts.items()))}")
    print(f"labels: {final_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

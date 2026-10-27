# -*- coding: utf-8 -*-
r"""15단계: 5개 정책 이슈 × 4개 Ollama 모델 × 기본/토픽강화 RAG A/B 실험.

PowerShell:
  & "C:\ProgramData\miniconda3\envs\project01\python.exe" "C:\mini_team\scripts\clustering\15_predict_policy_trends.py" --dry-run
  & "C:\ProgramData\miniconda3\envs\project01\python.exe" "C:\mini_team\scripts\clustering\15_predict_policy_trends.py"

원본 DB와 11~14단계 파일은 수정하지 않습니다. 매 응답 즉시 CSV에 기록하고 재실행 시 OK 결과는 건너뜁니다.
비교 조건: 동일한 질의/8개 벡터 검색 근거/LLM 옵션을 유지하고, 강화 모드에만 검증된
          회의록 출처를 가진 후속 논의 후보와 월별 통계가 추가됩니다.
주의: 이는 '추가 근거 효과' 실험이며 검색 근거량까지 완벽하게 통제한 실험이 아닙니다.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:\mini_team\scripts\clustering")
DB_PATH = Path(r"C:\mini_team\_db\chroma_snowflake_regions")
COLLECTION = "team_regions_6a0a006e92d1e93e"
EMBED_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
RESTORED = ROOT / "results/policy_meeting_metadata_v2/policy_links_meeting_metadata_restored.csv"
PRIORITY = ROOT / "results/policy_discussion_evidence/policy_links_priority_review.csv"
MONTHLY = ROOT / "results/policy_issue_trends/issue_monthly_trends.csv"
OUT = ROOT / "results/policy_prediction_comparison"
OUTPUT = OUT / "policy_trend_ab_results.csv"
SUMMARY = OUT / "policy_trend_ab_summary.json"
BASELINE_OLD = Path(r"C:\mini_team\scripts\results\local_rag_model_comparison_v2.csv")
ISSUES = ["지역소멸", "청년 일자리", "저출생", "고령화", "기후위기"]
MODELS = ["exaone3.5:2.4b", "qwen3:8b", "gemma3:12b", "gemma3:4b"]
MODES = ["A_기본RAG", "B_토픽강화RAG"]
SEARCH_K = 40
EVIDENCE_K = 8
EVIDENCE_CHARS = 420
LINK_K = 2
EXPERIMENT = "15단계_동일조건_AB_v1"
FIELDS = [
    "experiment", "mode", "issue_keyword", "model", "question", "prediction", "prediction_chars",
    "prediction_word_limit_ok", "evidence_ids", "evidence_records", "evidence_count",
    "flow_candidate_ids", "flow_candidate_count", "trend_summary", "first_appearance_scope",
    "flow_description", "embedding_sec", "search_sec", "context_sec", "llm_sec", "total_sec",
    "load_sec", "output_tokens", "tokens_per_sec", "status", "error", "created_at"
]


def log(msg, start):
    print(f"[{datetime.now():%H:%M:%S}] {msg} | 누적 {time.perf_counter()-start:.1f}초", flush=True)


def clean(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def day(s):
    v = re.sub(r"\D", "", str(s or ""))
    return f"{v[:4]}-{v[4:6]}-{v[6:8]}" if len(v) >= 8 else "날짜미상"


def read_csv(path):
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def select_evidence(query_result):
    seen = set()
    candidates = []
    for cid, txt, md, dist in zip(
        query_result.get("ids", [[]])[0],
        query_result.get("documents", [[]])[0],
        query_result.get("metadatas", [[]])[0],
        query_result.get("distances", [[]])[0],
    ):
        txt = clean(txt)
        if not txt:
            continue
        md = md or {}
        key = hashlib.sha1(txt.encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "chunk_id": cid, "document": txt,
            "assembly": md.get("assembly", ""), "date": day(md.get("date")),
            "agenda": clean(md.get("agenda", "")), "distance": round(float(dist), 5)
        })
    selected, councils = [], set()
    for r in candidates:
        if r["assembly"] not in councils:
            selected.append(r)
            councils.add(r["assembly"])
        if len(selected) == EVIDENCE_K:
            break
    chosen = {r["chunk_id"] for r in selected}
    for r in candidates:
        if len(selected) == EVIDENCE_K:
            break
        if r["chunk_id"] not in chosen:
            selected.append(r)
            chosen.add(r["chunk_id"])
    return selected


def get_links():
    # priority 목록에서 우선 검토할 후보이면서 양쪽 회의록 출처가 유일하게 복원된 쌍만 사용.
    # '정책 전파 확정'이 아니라 시간 선후·유사 내용 '후보'라는 사실을 명시한다.
    priorities = {(r.get("issue"), r.get("earlier_chunk_id"), r.get("later_chunk_id"))
                  for r in read_csv(PRIORITY)}
    result = defaultdict(list)
    for r in read_csv(RESTORED):
        key = (r.get("issue"), r.get("earlier_chunk_id"), r.get("later_chunk_id"))
        if key not in priorities:
            continue
        if r.get("earlier_restore_status") != "confirmed_unique" or r.get("later_restore_status") != "confirmed_unique":
            continue
        d1, d2 = day(r.get("earlier_date")), day(r.get("later_date"))
        if d1 == "날짜미상" or d2 == "날짜미상" or d1 >= d2:
            continue
        if r.get("earlier_assembly") == r.get("later_assembly"):
            continue
        result[r["issue"]].append(r)
    # Higher cosine first and diverse council pairs rather than redundant identical projects.
    for issue, rows in result.items():
        rows.sort(key=lambda r: -float(r.get("semantic_cosine") or 0))
        distinct = []
        seen = set()
        for r in rows:
            key = (r.get("earlier_assembly"), r.get("later_assembly"))
            if key in seen:
                continue
            seen.add(key)
            distinct.append(r)
            if len(distinct) == LINK_K:
                break
        result[issue] = distinct
    return result


def get_trends():
    # monthly counts are *raw matched chunk counts*, not normalized rates.
    totals = defaultdict(lambda: defaultdict(int))
    for r in read_csv(MONTHLY):
        issue = r.get("issue", "")
        month = r.get("year_month", "")
        if issue not in ISSUES or not re.fullmatch(r"20\d\d-\d\d", month):
            continue
        try:
            totals[issue][month] += int(r.get("chunk_count") or 0)
        except ValueError:
            pass
    output = {}
    for issue in ISSUES:
        dates = sorted(totals[issue])
        if not dates:
            output[issue] = "월별 통계가 확인되지 않음"
            continue
        yearly = defaultdict(int)
        for m, cnt in totals[issue].items():
            yearly[m[:4]] += cnt
        # Only describe observed period, no implied rate normalization.
        trend = ", ".join(f"{y}년 {yearly[y]:,}청크" for y in sorted(yearly))
        output[issue] = f"관측기간 {dates[0]}~{dates[-1]}, {trend}. 원문 청크 수이며 전체 회의량 보정 전."
    return output


def common_prompt(issue, evidence):
    parts = []
    for i, r in enumerate(evidence, 1):
        parts.append(f"[벡터근거 {i}] {r['date']} / {r['assembly']} / 청크:{r['chunk_id']}\n{r['document'][:EVIDENCE_CHARS]}")
    return (f"정책 이슈: {issue}\n"
            "다음은 2024~2025년 지방의회 회의록에서 검색한 근거입니다.\n"
            + "\n".join(parts))


def enriched_context(issue, links, trends):
    lines = [f"[관측량 참고] {trends.get(issue, '자료 없음')}"]
    if not links.get(issue):
        lines.append("[지역 간 비교] 출처를 양쪽 모두 확인한 유사 논의 후보 없음. 전파 사실을 추정하지 말 것.")
    for idx, r in enumerate(links.get(issue, []), 1):
        lines.append(
            f"[지역 간 유사 논의 후보 {idx} | 정책 전파 미검증]\n"
            f"먼저 관측: {day(r.get('earlier_date'))} {r.get('earlier_assembly')} "
            f"/ 회의록: {r.get('earlier_original_meeting_title') or r.get('earlier_meeting_title')} "
            f"/ ID:{r.get('earlier_chunk_id')}\n"
            f"원문: {clean(r.get('earlier_evidence_180'))[:180]}\n"
            f"후속 관측: {day(r.get('later_date'))} {r.get('later_assembly')} "
            f"/ 회의록: {r.get('later_original_meeting_title') or r.get('later_meeting_title')} "
            f"/ ID:{r.get('later_chunk_id')}\n"
            f"원문: {clean(r.get('later_evidence_180'))[:180]}"
        )
    return "\n\n".join(lines)


def make_prompt(issue, evidence, mode, links, trends):
    shared = common_prompt(issue, evidence)
    if mode == "B_토픽강화RAG":
        shared += "\n\n[정책 토픽·지역 흐름 보조자료]\n" + enriched_context(issue, links, trends)
    shared += ("\n\n[질문]\n주어진 증거에 근거하여 앞으로 지방의회에서 논의될 가능성이 있는 "
               "정책 방향을 한국어 200자 이내 2~3문장으로 전망하세요. "
               "2025년 이후에 실제로 일어난 일처럼 단정하지 마세요. "
               "회의록에 없는 구체적인 수치·법률·지역명은 생성하지 마세요. "
               "유사 논의 후보만으로 정책 전파가 확인됐다고 주장하지 마세요. "
               "근거가 부족하면 불확실하다고 밝히세요. "
               "오직 JSON 객체 한 개만 출력하세요: "
               '{"prediction":"전망 문장"}')
    return shared


def ask_ollama(model, prompt):
    import requests
    start = time.perf_counter()
    resp = requests.post(OLLAMA_URL, json={
        "model": model, "stream": False, "think": False, "format": "json",
        "messages": [
            {"role": "system", "content": "주어진 자료에 근거하는 지방의회 정책분석가입니다. JSON 객체 외 출력하지 마세요."},
            {"role": "user", "content": prompt}
        ],
        "options": {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 300},
    }, timeout=900)
    duration = time.perf_counter() - start
    resp.raise_for_status()
    data = resp.json()
    raw = str(data.get("message", {}).get("content") or "").strip()
    if not raw:
        raise ValueError("Ollama 빈 응답. 모델의 think/format 지원 여부 확인")
    parsed = json.loads(raw)
    pred = clean(parsed.get("prediction"))
    if not pred:
        raise ValueError("prediction 필드 없음")
    tokens = data.get("eval_count") or 0
    ns = data.get("eval_duration") or 0
    return (pred, duration, round((data.get("load_duration") or 0) / 1e9, 3),
            tokens, round(tokens / (ns / 1e9), 2) if ns else "")


def old_reference():
    rows = read_csv(BASELINE_OLD)
    # Historical results were generated with different evidence/prompts; comparison is illustrative only.
    return [r for r in rows if r.get("experiment") == "1차_기본RAG"]


def load_completed():
    return {(r.get("experiment"), r.get("mode"), r.get("issue_keyword"), r.get("model"))
            for r in read_csv(OUTPUT) if r.get("status") in ("OK", "OVER_200")}


def save_summary():
    rows = [r for r in read_csv(OUTPUT) if r.get("experiment") == EXPERIMENT]
    metrics = {}
    for mode in MODES:
        sub = [r for r in rows if r.get("mode") == mode]
        valid = [r for r in sub if r.get("status") in ("OK", "OVER_200")]
        durations = [float(r["llm_sec"]) for r in valid if r.get("llm_sec")]
        metrics[mode] = {
            "runs": len(sub), "completed": len(valid),
            "within_200": sum(r.get("status") == "OK" for r in valid),
            "mean_llm_sec": round(statistics.mean(durations), 3) if durations else None,
            "mean_chars": round(statistics.mean(int(r["prediction_chars"]) for r in valid), 1) if valid else None,
        }
    summary = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": EXPERIMENT, "expected_runs": len(MODES)*len(MODELS)*len(ISSUES),
        "metrics": metrics, "historical_baseline_rows": len(old_reference()),
        "historical_baseline_warning": "기존 1차 RAG 결과와의 직접 점수 비교는 프롬프트·근거·실행조건 차이로 공정하지 않음.",
        "ab_interpretation": "같은 모델·질문·기본 검색근거에 정책 통계 및 출처 복원 유사 논의 후보를 추가한 효과 비교. 개선 품질은 별도 수동평가 필수.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="파일/근거 통계만 검증; Ollama 호출 안 함")
    p.add_argument("--issue", choices=ISSUES, default=None)
    p.add_argument("--model", choices=MODELS, default=None)
    p.add_argument("--mode", choices=MODES, default=None)
    args = p.parse_args()
    t_all = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    links = get_links()
    trends = get_trends()
    log("[1/4] 보조 근거 로드", t_all)
    for issue in ISSUES:
        log(f"  {issue}: 출처 확인 유사논의 {len(links.get(issue, []))}쌍 / {trends.get(issue)}", t_all)
    if not MONTHLY.is_file():
        raise FileNotFoundError(f"13단계 월별 통계 파일 없음: {MONTHLY}")
    if not RESTORED.is_file():
        raise FileNotFoundError(f"14-4-2 결과 없음: {RESTORED}")
    if args.dry_run:
        log("[점검 완료] dry-run에서는 DB 검색/LLM 실행 안 함", t_all)
        return

    import chromadb
    from sentence_transformers import SentenceTransformer
    coll = chromadb.PersistentClient(path=str(DB_PATH)).get_collection(name=COLLECTION)
    encoder = SentenceTransformer(EMBED_MODEL)
    log(f"[2/4] Chroma {coll.count():,}개 / 임베딩 모델 준비", t_all)
    done = load_completed()
    new = not OUTPUT.exists() or OUTPUT.stat().st_size == 0
    with OUTPUT.open("a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        for issue in ([args.issue] if args.issue else ISSUES):
            question = f"지방의회에서 논의된 '{issue}' 관련 정책 이슈와 향후 방향은?"
            t = time.perf_counter()
            v = encoder.encode(question, prompt_name="query", normalize_embeddings=True).tolist()
            embed_sec = time.perf_counter()-t
            t = time.perf_counter()
            retrieved = coll.query(query_embeddings=[v], n_results=SEARCH_K,
                                   include=["documents", "metadatas", "distances"])
            search_sec = time.perf_counter()-t
            evidence = select_evidence(retrieved)
            first_scope = "기본 벡터 검색 근거에 한정. 전체 최초 등장 시점 아님"
            ordered = sorted({(e["date"], e["assembly"]) for e in evidence if e["date"] != "날짜미상"})
            flow = " → ".join(f"{d} {c}" for d,c in ordered[:8]) or "검색된 날짜 근거 없음"
            log(f"[3/4] {issue} 검색 {search_sec:.2f}초 / 공통 근거 {len(evidence)}개", t_all)
            for model in ([args.model] if args.model else MODELS):
                for mode in ([args.mode] if args.mode else MODES):
                    key = (EXPERIMENT, mode, issue, model)
                    if key in done:
                        log(f"  SKIP {issue} {model} {mode}", t_all)
                        continue
                    t = time.perf_counter()
                    prompt = make_prompt(issue, evidence, mode, links, trends)
                    context_sec = time.perf_counter()-t
                    used_links = links.get(issue, []) if mode == "B_토픽강화RAG" else []
                    row = dict.fromkeys(FIELDS, "")
                    row.update({
                        "experiment": EXPERIMENT, "mode": mode, "issue_keyword": issue, "model": model,
                        "question": question,
                        "evidence_records": json.dumps([{k:v for k,v in rec.items() if k!="document"} for rec in evidence], ensure_ascii=False),
                        "evidence_ids": ";".join(rec["chunk_id"] for rec in evidence),
                        "evidence_count": len(evidence),
                        "flow_candidate_ids": ";".join(
                            f"{r['earlier_chunk_id']}->{r['later_chunk_id']}" for r in used_links),
                        "flow_candidate_count": len(used_links),
                        "trend_summary": trends.get(issue, "") if used_links or mode == "B_토픽강화RAG" else "",
                        "first_appearance_scope": first_scope,
                        "flow_description": flow,
                        "embedding_sec": round(embed_sec, 3), "search_sec": round(search_sec, 3),
                        "context_sec": round(context_sec, 3),
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                    })
                    try:
                        pred, llm_sec, load_sec, n_tokens, tps = ask_ollama(model, prompt)
                        row.update({
                            "prediction": pred, "prediction_chars": len(pred),
                            "prediction_word_limit_ok": int(len(pred) <= 200),
                            "llm_sec": round(llm_sec, 3),
                            "total_sec": round(embed_sec + search_sec + context_sec + llm_sec, 3),
                            "load_sec": load_sec, "output_tokens": n_tokens, "tokens_per_sec": tps,
                            "status": "OK" if len(pred) <= 200 else "OVER_200"
                        })
                    except Exception as exc:
                        row.update({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"[:400]})
                    writer.writerow(row)
                    f.flush()
                    done.add(key) if row["status"] in ("OK", "OVER_200") else None
                    log(f"  {mode} | {model} | {row['status']} | {row.get('prediction_chars')}자 | LLM {row.get('llm_sec')}초", t_all)
                    save_summary()
    save_summary()
    log(f"[4/4] 완료: {OUTPUT} / {SUMMARY}", t_all)
    print("주의: 실제 정책 예측 품질·근거 일치·정책 확산의 인과성은 수동 평가해야 합니다.")


if __name__ == "__main__":
    main()

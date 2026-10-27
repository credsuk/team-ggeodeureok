# -*- coding: utf-8 -*-
r"""15-2. 기본 RAG vs 토픽 강화 RAG: 15단계 결과 검증과 평가표 생성.

읽기 전용 입력:
  C:\mini_team\scripts\clustering\results\policy_prediction_comparison\policy_trend_ab_results.csv

출력:
  15_2_rag_evaluation_rows.csv     40회 전체 자동 지표
  15_2_rag_paired_comparison.csv   같은 이슈·모델의 A/B 쌍별 비교
  15_2_rag_human_review.csv        독립 판정 입력 양식(기존 입력 보존)
  15_2_rag_evaluation_summary.json
  15_2_rag_evaluation_report.md

수동 평가가 필요한 것: 전망의 논리성, 근거의 사실 일치, 시계열 해석 타당성.
자동 계산(문자/키워드 중복 등)은 정답률이나 환각률을 의미하지 않는다.

실행:
  python 15_2_evaluate_rag_comparison.py
  python 15_2_evaluate_rag_comparison.py --input "...\\policy_trend_ab_results.csv"
  python 15_2_evaluate_rag_comparison.py --review-file "...\\15_2_rag_human_review.csv"
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

BASE = Path(r"C:\mini_team\scripts\clustering")
DEFAULT_INPUT = BASE / "results" / "policy_prediction_comparison" / "policy_trend_ab_results.csv"
DEFAULT_OUTPUT = BASE / "results" / "policy_prediction_evaluation"
MODE_A = "A_기본RAG"
MODE_B = "B_토픽강화RAG"
MODELS = ["exaone3.5:2.4b", "qwen3:8b", "gemma3:4b", "gemma3:12b"]
ISSUES = ["지역소멸", "청년 일자리", "저출생", "고령화", "기후위기"]

# 단순 문자열 분석용 후보 용어. 용어의 출현은 사실성/근거성을 증명하지 않는다.
GENERIC_TERMS = {"정책", "지방", "의회", "관련", "논의", "강화", "지원", "확대", "추진", "가능", "방안", "지역", "사업", "예상", "전망", "필요", "정부", "서비스", "다양", "지속", "개선", "확충", "적극"}
KOR_TOKEN = re.compile(r"[가-힣]{2,}|[a-zA-Z]{3,}|\d{4}[-./]?\d{1,2}(?:[-./]?\d{1,2})?")


def as_float(value):
    try:
        return float(value) if str(value).strip() else None
    except (TypeError, ValueError):
        return None


def as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def mean(values):
    vals = [float(x) for x in values if x is not None]
    return round(statistics.mean(vals), 4) if vals else None


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def tokens(text):
    return {w.lower() for w in KOR_TOKEN.findall(text or "") if w.lower() not in GENERIC_TERMS}


def jaccard(a, b):
    ta, tb = tokens(a), tokens(b)
    if not (ta or tb):
        return None
    return round(len(ta & tb) / len(ta | tb), 4)


def safe_json_list(value):
    try:
        data = json.loads(value or "[]")
        return data if isinstance(data, list) else []
    except (ValueError, TypeError):
        return []


def evidence_summary(row):
    evs = safe_json_list(row.get("evidence_records"))
    ids = [str(v.get("chunk_id", "")) for v in evs if isinstance(v, dict)]
    assemblies = sorted({str(v.get("assembly")) for v in evs if isinstance(v, dict) and v.get("assembly")})
    dates = sorted({str(v.get("date")) for v in evs if isinstance(v, dict) and v.get("date")})
    agendas = [str(v.get("agenda", "")) for v in evs if isinstance(v, dict)]
    return {"ids": ids, "assemblies": assemblies, "dates": dates,
            "agenda_keyword_overlap": jaccard(row.get("prediction", ""), " ".join(agendas)),
            "evidence_records_count": len(evs)}


def evaluate_row(row):
    prediction = (row.get("prediction") or "").strip()
    status = (row.get("status") or "").strip()
    has_answer = bool(prediction) and status != "ERROR"
    chars = len(prediction)
    ev = evidence_summary(row)
    # load_sec might report only startup cost, and can be missing or inconsistent.
    llm_sec = as_float(row.get("llm_sec"))
    load_sec = as_float(row.get("load_sec"))
    generation_proxy = round(max(0.0, llm_sec - load_sec), 4) if (
        llm_sec is not None and load_sec is not None and llm_sec >= load_sec) else None
    flow_count = as_int(row.get("flow_candidate_count"))
    trend = (row.get("trend_summary") or "").strip()
    return {
        "issue_keyword": row.get("issue_keyword", ""),
        "model": row.get("model", ""),
        "mode": row.get("mode", ""),
        "status": status,
        "has_answer": int(has_answer),
        "actual_chars": chars,
        "within_200": int(has_answer and chars <= 200),
        "over_200": int(has_answer and chars > 200),
        "original_reported_chars": as_int(row.get("prediction_chars"), -1),
        "chars_match_logged": int(chars == as_int(row.get("prediction_chars"), -1)),
        "ends_with_sentence_punctuation": int(bool(re.search(r"[.!?。！？]$", prediction))),
        "forecast_expression_present": int(bool(re.search(r"예상|전망|가능성|것으로 보|것으로 예상|향후|추후|것이다|것으로 판단", prediction))),
        "mentions_issue_literal": int((row.get("issue_keyword") or "") in prediction),
        "evidence_records_count": ev["evidence_records_count"],
        "evidence_id_count": len([x for x in row.get("evidence_ids", "").split(";") if x.strip()]),
        "evidence_assemblies": "; ".join(ev["assemblies"]),
        "evidence_dates": "; ".join(ev["dates"]),
        "agenda_keyword_overlap_proxy": ev["agenda_keyword_overlap"],
        "flow_candidate_count": flow_count,
        "trend_context_present": int(bool(trend)),
        "llm_sec": llm_sec,
        "load_sec": load_sec,
        "llm_minus_load_sec_proxy": generation_proxy,
        "total_sec": as_float(row.get("total_sec")),
        "output_tokens": as_int(row.get("output_tokens")),
        "error": row.get("error", ""),
        "prediction": prediction,
        "automatic_warning": (
            "생성 오류: 원인 별도 점검" if not has_answer else
            "200자 초과" if chars > 200 else
            "문장 끝 잘림 가능성" if not re.search(r"[.!?。！？]$", prediction) else ""
        ),
    }


def compare_pairs(originals, evaluated):
    groups = defaultdict(dict)
    for orig, ev in zip(originals, evaluated):
        key = (orig.get("issue_keyword"), orig.get("model"))
        mode = orig.get("mode")
        if mode in groups[key]:
            raise ValueError(f"중복 A/B 실행 발견: {key} {mode}. 재실험 버전을 분리해 주세요.")
        groups[key][mode] = (orig, ev)
    pairs = []
    for (issue, model), items in sorted(groups.items()):
        a = items.get(MODE_A)
        b = items.get(MODE_B)
        if not a or not b:
            pairs.append({"issue_keyword": issue, "model": model, "paired": 0,
                          "A_status": a[1]["status"] if a else "MISSING",
                          "B_status": b[1]["status"] if b else "MISSING"})
            continue
        ar, ae = a
        br, be = b
        ap, bp = ae["prediction"], be["prediction"]
        both_answer = bool(ae["has_answer"] and be["has_answer"])
        shared_evidence = ar.get("evidence_ids", "") == br.get("evidence_ids", "")
        char_similarity = round(difflib.SequenceMatcher(None, ap, bp).ratio(), 4) if both_answer else None
        pairs.append({
            "issue_keyword": issue, "model": model, "paired": 1,
            "A_status": ae["status"], "B_status": be["status"],
            "both_answered": int(both_answer),
            "same_question": int(ar.get("question") == br.get("question")),
            "same_base_evidence_ids": int(shared_evidence),
            "exact_same_prediction": int(both_answer and ap == bp),
            "character_sequence_similarity": char_similarity,
            "token_jaccard": jaccard(ap, bp) if both_answer else None,
            "A_chars": ae["actual_chars"], "B_chars": be["actual_chars"],
            "A_within_200": ae["within_200"], "B_within_200": be["within_200"],
            "A_llm_sec": ae["llm_sec"], "B_llm_sec": be["llm_sec"],
            "A_load_sec": ae["load_sec"], "B_load_sec": be["load_sec"],
            "A_llm_minus_load_sec_proxy": ae["llm_minus_load_sec_proxy"],
            "B_llm_minus_load_sec_proxy": be["llm_minus_load_sec_proxy"],
            "B_flow_candidates": be["flow_candidate_count"],
            "B_has_trend_context": be["trend_context_present"],
            "A_prediction": ap, "B_prediction": bp,
            "manual_improvement": "", "manual_notes": "",
        })
    return pairs


def write_review_template(path, eval_rows):
    fields = ["issue_keyword", "model", "mode", "status", "prediction",
              "logic_0_40", "evidence_0_30", "format_0_15", "speed_0_15",
              "total_manual_0_100", "judgment", "reason", "reviewer", "reviewed_at"]
    old = {}
    if path.is_file():
        for r in read_csv(path):
            old[(r.get("issue_keyword"), r.get("model"), r.get("mode"))] = r
    rows = []
    for e in eval_rows:
        key = (e["issue_keyword"], e["model"], e["mode"])
        previous = old.get(key, {})
        r = {f: previous.get(f, "") for f in fields}
        r.update({"issue_keyword": e["issue_keyword"], "model": e["model"],
                  "mode": e["mode"], "status": e["status"], "prediction": e["prediction"]})
        # Mark errors for reviewer; never fabricate 0 performance as a semantic score.
        if not e["has_answer"] and not r["judgment"]:
            r["judgment"] = "생성오류_재실험필요"
        rows.append(r)
    write_csv(path, rows, fields)
    return rows


def summarize_manual(review_rows):
    completed = []
    invalid = []
    dimensions = {"logic_0_40": 40, "evidence_0_30": 30,
                  "format_0_15": 15, "speed_0_15": 15}
    for r in review_rows:
        values = {}
        for key, maximum in dimensions.items():
            v = as_float(r.get(key))
            if v is not None and not (0 <= v <= maximum):
                invalid.append(f"{r['issue_keyword']} {r['model']} {r['mode']} {key}={v}")
            values[key] = v
        if all(v is not None for v in values.values()) and r["status"] != "ERROR":
            total = sum(values.values())
            completed.append({"issue": r["issue_keyword"], "model": r["model"],
                              "mode": r["mode"], "score": round(total, 2)})
    return {"scored_rows": len(completed), "invalid_score_ranges": invalid,
            "average_score_by_mode": {m: mean(x["score"] for x in completed if x["mode"] == m)
                                      for m in (MODE_A, MODE_B)},
            "note": "수동 점수는 두 방식 모두 채점된 동일 이슈·동일 모델 쌍만으로 최종 비교하세요."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-file", type=Path, default=None,
                        help="기존 작성된 수동 평가 CSV 위치 (없으면 출력폴더 기본 평가표 재사용)")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"입력 파일 없음: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = read_csv(args.input)
    if not records:
        parser.error("결과 CSV가 비어 있습니다.")
    mandatory = {"mode", "issue_keyword", "model", "status", "prediction", "evidence_records"}
    missing = mandatory - set(records[0])
    if missing:
        parser.error(f"필수 컬럼 없음: {sorted(missing)}")

    # Keep deterministic order by user data.
    evaluated = [evaluate_row(r) for r in records]
    paired = compare_pairs(records, evaluated)
    review_file = args.review_file or args.output_dir / "15_2_rag_human_review.csv"
    review_rows = write_review_template(review_file, evaluated)
    detail_file = args.output_dir / "15_2_rag_evaluation_rows.csv"
    paired_file = args.output_dir / "15_2_rag_paired_comparison.csv"
    write_csv(detail_file, evaluated, list(evaluated[0]))
    write_csv(paired_file, paired, list(dict.fromkeys(k for r in paired for k in r)))

    metrics = {}
    for mode in (MODE_A, MODE_B):
        rows = [r for r in evaluated if r["mode"] == mode]
        answered = [r for r in rows if r["has_answer"]]
        metrics[mode] = {
            "runs": len(rows), "answered": len(answered),
            "error": len(rows) - len(answered),
            "within_200": sum(r["within_200"] for r in answered),
            "over_200": sum(r["over_200"] for r in answered),
            "within_200_rate_on_answered": round(
                sum(r["within_200"] for r in answered) / len(answered), 4) if answered else None,
            "mean_llm_sec_on_answered": mean(r["llm_sec"] for r in answered),
            "mean_load_sec_on_answered": mean(r["load_sec"] for r in answered),
            "mean_llm_minus_load_proxy_sec": mean(r["llm_minus_load_sec_proxy"] for r in answered),
            "mean_chars_on_answered": mean(r["actual_chars"] for r in answered),
            "answer_without_sentence_final_punctuation": sum(
                not r["ends_with_sentence_punctuation"] for r in answered),
        }

    ok_pairs = [p for p in paired if p.get("both_answered")]
    summary = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "input_file": str(args.input), "total_runs": len(records),
        "pairs_total": len(paired), "both_answered_pairs": len(ok_pairs),
        "mode_metrics": metrics,
        "pairs_with_identical_answers": sum(p["exact_same_prediction"] for p in ok_pairs),
        "mean_pair_character_sequence_similarity": mean(
            p["character_sequence_similarity"] for p in ok_pairs),
        "pairs_using_identical_base_evidence": sum(
            p.get("same_base_evidence_ids", 0) for p in paired if p.get("paired")),
        "errors_by_model": dict(Counter(r["model"] for r in evaluated if not r["has_answer"])),
        "manual_evaluation": summarize_manual(review_rows),
        "warnings": [
            "A가 매번 먼저 실행되고 B가 뒤따라 실행됨: 응답 속도는 모델 예열/캐시 순서 편향으로 공정 비교 불가.",
            "llm_sec-load_sec는 참고 대리 지표일 뿐 신뢰할 수 있는 토큰별 추론 속도 측정이 아님.",
            "현재 evidence_records에는 주로 ID/날짜/의회/안건이 있음. 주장 사실 여부는 원문 근거 검증 필요.",
            "같은 문장/키워드 유사도는 답변의 정확도·환각률·향후 예측의 타당성이 아님.",
            "Gemma3 12B 단독 ollama run 성공은 대화 API/큰 문맥/메모리 문제의 반증이 아님. API 오류는 따로 조사.",
            "수동 40/30/15/15 평가 완료 전에는 RAG의 정성적 우월성을 확정하지 않음.",
        ]
    }
    summary_file = args.output_dir / "15_2_rag_evaluation_summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 15-2 RAG A/B 평가 보고서", "",
             f"- 전체 실행: {len(records)}건 / 비교 쌍: {len(paired)}쌍 / 양측 답변 생성: {len(ok_pairs)}쌍",
             f"- 양측 답변 완전 동일: {summary['pairs_with_identical_answers']}쌍",
             f"- 양측 답변 평균 문자 시퀀스 유사도: {summary['mean_pair_character_sequence_similarity']}", "",
             "## 자동 지표 (정성적 정답률 아님)", "",
             "| 방식 | 생성 | 200자 준수/생성 | 평균 LLM 시간(초) | 평균 로드 시간(초) |",
             "|---|---:|---:|---:|---:|"]
    for m in (MODE_A, MODE_B):
        v = metrics[m]
        lines.append(f"| {m} | {v['answered']}/{v['runs']} | {v['within_200']}/{v['answered']} | "
                     f"{v['mean_llm_sec_on_answered']} | {v['mean_load_sec_on_answered']} |")
    lines += ["", "## 정성 평가", "",
              "`15_2_rag_human_review.csv`에 논리(0~40), 원문 근거 일치(0~30), 형식(0~15), 속도(0~15)를 채우고 다시 실행하세요.",
              "오류 건수는 채점 대신 재실험 대상으로 보존합니다.", "",
              "## 해석의 한계", ""]
    lines += [f"- {w}" for w in summary["warnings"]]
    (args.output_dir / "15_2_rag_evaluation_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("=== 15-2 평가 완료 ===")
    print(f"전체 {len(records)}회 / 정상 A={metrics[MODE_A]['answered']}, B={metrics[MODE_B]['answered']}")
    print(f"양쪽 답변 비교 {len(ok_pairs)}쌍 / 완전 동일 {summary['pairs_with_identical_answers']}쌍")
    print(f"수동 평가 완료 {summary['manual_evaluation']['scored_rows']}행")
    print("출력:", args.output_dir)
    print("중요: 자동 지표만으로 B가 더 정확하다고 결론내리면 안 됩니다.")


if __name__ == "__main__":
    main()

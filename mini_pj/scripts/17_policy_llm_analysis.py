# -*- coding: utf-8 -*-
"""17. 지방의회 정책 이슈 근거 기반 LLM 해석/조건부 전망 (Ollama, 로컬 전용).

설치: pip install pandas
실행 예:
  python 17_policy_llm_analysis.py --input-dir "C:\\mini_pj\\results\\policy_issue_search\\20261025_000801_청년_일자리"
  python 17_policy_llm_analysis.py --input-dir "..." --dry-run
  python 17_policy_llm_analysis.py --input-dir "..." --model exaone3.5:2.4b

주의: 16단계 CSV의 text_preview는 최대 약 600자 미리보기다.
      원문 전체를 읽지 않으므로 정책 사실 검증 완료나 실제 지역 간 전파를 주장하지 않는다.
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path(r"C:\mini_pj\results\policy_issue_search")
CUTOFF = "2025-12-31"  # 보유 자료 분석 기준 종료일
ALLOWED_MODELS = ("qwen3:8b", "exaone3.5:2.4b", "gemma3:4b")


def log(message):
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def read_csv(path, required):
    if not path.is_file():
        raise FileNotFoundError(f"필수 파일 없음: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"chunk_id": "string"}, low_memory=False)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} 컬럼 누락: {sorted(missing)}")
    return df


def locate_input(cli_path):
    if cli_path:
        return Path(cli_path).expanduser().resolve()
    candidates = sorted(
        (p for p in DEFAULT_ROOT.iterdir() if p.is_dir()
         and (p / "trend_analysis" / "분석_요약.json").is_file()),
        key=lambda p: (p / "trend_analysis" / "분석_요약.json").stat().st_mtime,
        reverse=True,
    ) if DEFAULT_ROOT.is_dir() else []
    if not candidates:
        raise FileNotFoundError("--input-dir로 16번 분석의 상위 폴더를 지정하세요.")
    return candidates[0]


def date_normalize(series):
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def to_plain(v):
    if v is None or pd.isna(v):
        return ""
    return str(v).strip()


def select_evidence(df, limit):
    """시간·지역·검색표현 다양성을 보장하는 결정적 대표 '후보' 선택. 의미품질 랭킹 아님."""
    d = df.copy()
    d["date"] = date_normalize(d["date"])
    d = d[(d["date"].notna()) & (d["date"] <= CUTOFF)].copy()
    if "strict_match" in d:
        # 16번: 범위가 넓은 '취업 지원'만 일치한 후보 제외
        d = d[d["strict_match"].astype(str).str.lower().eq("true")].copy()
    d["text_preview"] = d["text_preview"].fillna("").astype(str)
    d = d[d["text_preview"].str.len() >= 35].copy()
    d = d.drop_duplicates(subset=["chunk_id"])
    d = d.sort_values(["date", "assembly", "chunk_id"], kind="stable")
    if d.empty:
        raise ValueError("활용 가능한 직접 표현 일치 근거가 없습니다.")

    # 2024 초기, 2025 초기, 최근 등 서로 다른 시기를 대표하도록 선택
    d["period"] = d["date"].str[:7]
    picked = []
    used = set()
    # 첫 관측 후보 포함
    for _, r in d.head(1).iterrows():
        picked.append(r); used.add(str(r["chunk_id"]))
    # 연도별 균형: 2024와 2025에서 번갈아 지역 후보를 뽑는다.
    by_year = {}
    for year in ("2024", "2025"):
        subset = d[d["date"].str.startswith(year)]
        by_year[year] = [r for _, r in subset.groupby("assembly", sort=True).head(1).iterrows()]
    for j in range(max(map(len, by_year.values()), default=0)):
        for year in ("2024", "2025"):
            if j >= len(by_year[year]) or len(picked) >= limit:
                continue
            r = by_year[year][j]
            cid = str(r["chunk_id"])
            if cid not in used:
                picked.append(r); used.add(cid)
    # 2025년 하반기와 2024년 하반기 후보 보강
    for _, r in d.sort_values("date", ascending=False, kind="stable").iterrows():
        if len(picked) >= limit:
            break
        cid = str(r["chunk_id"])
        if cid not in used:
            picked.append(r); used.add(cid)
    selected = pd.DataFrame(picked).head(limit).copy()
    selected["evidence_id"] = [f"E{i:02d}" for i in range(1, len(selected) + 1)]
    return selected


def compact_stats(monthly, regions):
    m = monthly.sort_values("month").copy()
    m = m[m["month"].astype(str) <= "2025-12"]
    r = regions.sort_values("direct", ascending=False).head(5)
    trend = []
    for _, row in m.iterrows():
        trend.append({"month": str(row["month"]), "direct": int(row["direct"]),
                      "all_chunks": int(row["all_chunks"]),
                      "direct_rate_pct": round(float(row["direct_match_rate_pct"]), 4)})
    top_regions = [{"assembly": str(row["assembly"]), "direct": int(row["direct"]),
                    "all_chunks": int(row["all_chunks"]),
                    "direct_rate_pct": round(float(row["direct_match_rate_pct"]), 4)}
                   for _, row in r.iterrows()]
    return {"monthly": trend, "top_regions_by_count": top_regions}


def build_prompt(query, analysis, evidence, stats):
    evidence_blocks = []
    for _, row in evidence.iterrows():
        snippet = to_plain(row["text_preview"]).replace("\x00", "")[:480]
        evidence_blocks.append(
            f"[{row['evidence_id']}] date={row['date']}; assembly={row['assembly']}; "
            f"meeting={to_plain(row.get('meeting_title'))}; chunk_id={row['chunk_id']}\n"
            f"TEXT: {snippet}"
        )
    system = """너는 지방의회 정책 연구 보조 분석가다. 2024~2025년 제공된 통계와 회의록 발췌 외의 사실을 사용하지 않는다.
이 입력은 출처가 완전하지 않은 '회의록 미리보기'임을 기억하라.
회고적 사실, 추론, 향후 전망을 엄격히 구분하고, 원문에 없는 법안·예산·전파·인과관계·수치를 만들지 않는다.
지역별 동일 단어 등장만으로 정책 전파를 주장하지 않는다. 날짜는 보유 코퍼스 내 최초 관측만 뜻한다.
서술형 전망은 근거에 기반한 조건부 가설이며 검증된 예측이 아니다.
인용은 제공된 E번호만 사용한다. 근거가 부족하면 '자료만으로 판단 불가'라고 말한다.
반드시 JSON 객체 하나만 출력하라. 코드펜스와 서문 금지.
JSON 키 구조:
{
 "observed_summary":"관측된 사실만 1~2문장",
 "time_change_interpretation":"월별 증감에 관한 설명; 회의록 분량 효과와 정책 변화 구분",
 "regional_interpretation":"지역별 차이 해석; 전파 주장 금지",
 "forecast":"한국어 공백 포함 200자 이내의 조건부 향후 전망",
 "forecast_basis":"전망의 근거 및 한계",
 "evidence_ids":["E01","E02"],
 "uncertainties":["한계1","한계2"],
 "needs_verification":["원문검증 과제"]
}
"""
    user = (f"검색 이슈: {query}\n보유 자료: 2024-01-01~{CUTOFF}\n"
            "중요: 직접표현 일치는 '정책 관련성 확정'이 아니며, 해당 청크는 발언/회의/정책 수가 아님.\n"
            f"16번 분석 요약(통계로만 활용): {json.dumps(analysis, ensure_ascii=False)}\n"
            f"월별 및 지역별 통계: {json.dumps(stats, ensure_ascii=False)}\n"
            "근거 후보(원문 전체가 아닌 잘린 텍스트):\n" + "\n\n".join(evidence_blocks))
    return system, user


def ollama_generate(base_url, model, system, user, num_ctx, timeout):
    url = base_url.rstrip("/") + "/api/generate"
    payload = {"model": model, "system": system, "prompt": user, "stream": False,
               "format": "json", "options": {"temperature": 0.1, "num_ctx": num_ctx,
                                               "num_predict": 1100}}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            output = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"로컬 Ollama 호출 실패 ({url}): {e}. 'ollama serve' 및 모델 설치 상태 확인") from e
    if output.get("error"):
        raise RuntimeError(f"Ollama 오류: {output['error']}")
    if not output.get("response"):
        raise RuntimeError("Ollama 응답이 비어 있습니다. num_ctx/num_predict/모델 상태 확인")
    return output


def parse_result(raw, valid_eids):
    text = str(raw).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("LLM 출력이 JSON 객체가 아닙니다")
    required = ["observed_summary", "time_change_interpretation", "regional_interpretation",
                "forecast", "forecast_basis", "evidence_ids", "uncertainties", "needs_verification"]
    missing = [k for k in required if k not in result]
    if missing:
        raise ValueError(f"LLM JSON 필드 누락: {missing}")
    forecast = result["forecast"]
    if not isinstance(forecast, str) or not forecast.strip():
        raise ValueError("forecast가 유효한 서술형 문자열이 아닙니다")
    if len(forecast) > 200:
        # 원문을 조용히 잘라 의미를 훼손하지 않음; 사용자에게 검증 실패를 명시
        result["forecast_length_warning"] = f"200자 제한 초과: {len(forecast)}자"
    else:
        result["forecast_length_warning"] = None
    raw_ids = result.get("evidence_ids", [])
    if not isinstance(raw_ids, list):
        raise ValueError("evidence_ids는 배열이어야 합니다")
    valid = sorted(set(str(x) for x in raw_ids) & set(valid_eids))
    invalid = sorted(set(str(x) for x in raw_ids) - set(valid_eids))
    result["evidence_ids"] = valid
    result["invalid_evidence_ids"] = invalid
    result["citation_format_pass"] = (bool(valid) and not invalid)
    # 내용의 근거 충실도는 자동 검증하지 못함
    result["factual_verification_status"] = "사람의 원문 검증 필요"
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="근거 기반 지방의회 정책 LLM 분석 (로컬 Ollama)")
    parser.add_argument("--input-dir", default=None, help="15번 실행별 폴더 (trend_analysis 포함)")
    parser.add_argument("--model", default="qwen3:8b", help="Ollama 모델명")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--top-k", type=int, default=10, help="LLM에 넣을 근거 후보 개수 (4~14)")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true", help="Ollama 실행 없이 입력 및 선택 근거 검사")
    args = parser.parse_args()
    if not 4 <= args.top_k <= 14:
        parser.error("--top-k는 4~14 사이여야 합니다")

    started = time.perf_counter()
    log("=== 17 정책 LLM 분석 시작 ===")
    inp = locate_input(args.input_dir)
    trend = inp / "trend_analysis"
    summary_path = trend / "분석_요약.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"16번 분석 요약 파일이 없습니다: {summary_path}")
    analysis = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    query = str(analysis.get("query") or "정책 이슈")
    evidence_all = read_csv(trend / "검색근거_검토구분.csv",
                            ["chunk_id", "date", "assembly", "matched_terms", "text_preview", "strict_match"])
    monthly = read_csv(trend / "월별_정책언급추이.csv",
                       ["month", "all_chunks", "direct", "direct_match_rate_pct"])
    regional = read_csv(trend / "지역별_정책언급순위.csv",
                        ["assembly", "all_chunks", "direct", "direct_match_rate_pct"])
    if len(evidence_all) != int(analysis["total_lexical_matches"]):
        raise ValueError(f"16번 총 일치 건수 불일치: CSV={len(evidence_all)}, 요약={analysis['total_lexical_matches']}")
    if int(monthly["direct"].sum()) != int(analysis["direct_expression_matches"]):
        raise ValueError("16번 월별 직접표현 건수 합계 불일치")
    if int(regional["direct"].sum()) != int(analysis["direct_expression_matches"]):
        raise ValueError("16번 지역별 직접표현 건수 합계 불일치")
    log(f"[1/4 입력 검증] {query} / 전체 {len(evidence_all):,}건 / 직접표현 {analysis['direct_expression_matches']:,}건")

    selected = select_evidence(evidence_all, args.top_k)
    stats = compact_stats(monthly, regional)
    system, user = build_prompt(query, analysis, selected, stats)
    out = trend / "llm_analysis" / args.model.replace(":", "_").replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out / "LLM_입력근거후보.csv", index=False, encoding="utf-8-sig")
    write_json(out / "LLM_입력요약.json", {
        "query": query, "model": args.model, "corpus_end": CUTOFF,
        "evidence_candidates": len(selected), "source": str(inp),
        "metadata_verified": True,
        "note": "최대 600자 미리보기 기반; 원문 전체에 대한 충실도 미검증; 전파 미검증",
        "statistics": stats,
    })
    log(f"[2/4 대표 근거 선정] {len(selected)}건 / 파일 저장 / {time.perf_counter()-started:.2f}초")
    if args.dry_run:
        log("[DRY-RUN] LLM 호출 생략. 아래 파일을 검토하세요:")
        log(str(out / "LLM_입력근거후보.csv"))
        return

    llm_start = time.perf_counter()
    try:
        response = ollama_generate(args.ollama_url, args.model, system, user, args.num_ctx, args.timeout)
    except Exception as exc:
        write_json(out / "LLM_실행오류.json", {"error": str(exc), "at": datetime.now().isoformat()})
        raise
    raw = response["response"]
    (out / "LLM_원시응답.txt").write_text(raw, encoding="utf-8")
    log(f"[3/4 Ollama 답변 완료] {args.model} / {time.perf_counter()-llm_start:.2f}초")
    try:
        result = parse_result(raw, selected["evidence_id"].tolist())
    except (ValueError, json.JSONDecodeError) as exc:
        write_json(out / "LLM_검증오류.json", {"error": str(exc), "raw_response_file": "LLM_원시응답.txt"})
        raise RuntimeError("LLM 출력 형식 검증 실패. 원시응답 저장됨; 모델 재학습 필요 없음") from exc

    result.update({"query": query, "model": args.model, "analysis_cutoff": CUTOFF,
                   "first_direct_observed_date": analysis.get("first_direct_expression_observed_date"),
                   "forecast_character_count": len(result["forecast"]),
                   "generation_seconds": round(time.perf_counter() - llm_start, 2),
                   "total_seconds": round(time.perf_counter() - started, 2),
                   "created_at": datetime.now().isoformat(timespec="seconds"),
                   "limitations": ["LLM은 회의록 미리보기만 봄", "동일정책 전파 미검증", "LLM 생성물은 사실 검증 필요"]})
    write_json(out / "정책_LLM_분석결과.json", result)
    eids = {str(x["evidence_id"]): x for _, x in selected.iterrows()}
    lines = [f"# {query} — 로컬 LLM 정책 분석 초안", "", f"모델: {args.model} | 분석자료: 2024~2025년", "",
             "## 관측 사실(모델 초안)", str(result["observed_summary"]), "",
             "## 시기별 변화 해석", str(result["time_change_interpretation"]), "",
             "## 지역별 해석", str(result["regional_interpretation"]), "",
             "## 향후 전망(조건부 가설)", str(result["forecast"]), "",
             "## 근거·한계", str(result["forecast_basis"]), "",
             "## LLM이 지목한 근거 후보(원문 검증 전)"]
    for evid in result["evidence_ids"]:
        row = eids[evid]
        lines.append(f"- {evid}: {row['assembly']} / {row['date']} / {to_plain(row.get('meeting_title'))} / `{row['chunk_id']}`")
    lines.extend(["", "**주의:** 근거는 잘린 텍스트 미리보기이며, 사실·인용의 정확성과 타 지역 전파 여부는 확인되지 않음."])
    (out / "정책_LLM_분석결과.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"[4/4 검증·저장] 인용 형식={result['citation_format_pass']} / 전망={len(result['forecast'])}자 / 200자 제한 경고={bool(result['forecast_length_warning'])}")
    log(f"=== 완료: {time.perf_counter()-started:.2f}초 / 결과 폴더: {out} ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[오류] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

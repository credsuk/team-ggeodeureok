# -*- coding: utf-8 -*-
"""개인 1차 RAG 평가: 이미 저장된 Chroma + Ollama, 5개 키워드 x 3개 모델.
실행: python scripts/chroma_03_rag_evaluate.py
필수: pip install chromadb sentence-transformers numpy pandas requests
Ollama 실행 필요. 로컬 모델 미설치 시: ollama pull exaone3.5:2.4b 등.
재실행하면 새로운 실험 기록이 추가된다. 원본 JSON 검증은 이 코드에서 수행하지 않으며, 증거 상태를 '미검증'으로 기록한다.
"""
import csv
import json
import re
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import chromadb
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

ROOT = Path(r"C:\mini_pj")
DB_DIR = ROOT / "_db" / "chroma_personal"
COLLECTION = "personal_policy_chunks"
RESULT_DIR = ROOT / "results" / "rag_evaluation"
RESULT_CSV = RESULT_DIR / "모델결과_누적.csv"
EVIDENCE_CSV = RESULT_DIR / "근거회의록_누적.csv"
MODEL_ID = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
KEYWORDS = ["지역소멸", "청년 일자리", "저출생", "고령화", "기후위기"]
MODELS = ["exaone3.5:2.4b", "qwen3:8b", "gemma3:4b"]
TOP_K = 15                 # 전체 최초 등장 시점이 아닌 검색 결과 표본 기준
MAX_CONTEXT_CHARS = 950   # 근거 청크 1개당 전달할 최대 문자수
TEMPERATURE = 0.0
NUM_PREDICT = 320          # 200자 제한은 별도 엄격 검증
EXPERIMENT = "1차_RAG_개인"

RESULT_FIELDS = ["실험", "키워드", "모델", "상태(200자)", "등장시점(검색표본)",
                 "타 지방의회 논의순서(검색표본)", "향후 예측(앞 200자 표시)",
                 "원래 예측 전체", "실제 글자 수", "200자 준수", "근거 회의록(텍스트 요약)",
                 "근거 수", "질문 임베딩(초)", "DB 검색(초)", "LLM(초)",
                 "전체(초)", "모델 로드(초)", "출력 토큰", "토큰/초", "오류",
                 "실행시각", "검색근거검증", "실험ID"]
EVIDENCE_FIELDS = ["실험ID", "이슈", "모델", "근거 번호", "일자", "의회", "원본 회의록명",
                   "안건", "발언자", "모델 상태", "근거 원문(앞 220자)",
                   "저장된 source_file", "row_id", "chunk_id", "검색 거리", "검증 상태"]


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def elapsed(t):
    return round(time.perf_counter() - t, 3)


def append_csv(path, fields, rows):
    if not rows:
        return
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()


def norm_date(v):
    s = re.sub(r"\D", "", str(v or ""))
    if len(s) >= 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return ""


def collect_evidence(raw):
    out = []
    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    mds = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]
    for i, id_ in enumerate(ids):
        m = mds[i] or {}
        out.append({
            "id": id_, "date": norm_date(m.get("date")), "assembly": str(m.get("assembly") or ""),
            "meeting": str(m.get("meeting_title") or ""),
            "agenda": str(m.get("agenda_title") or ""),
            "speaker": str(m.get("speaker_name") or ""),
            "source_file": str(m.get("source_file") or ""),
            "row_id": str(m.get("record_index") or ""),
            "text": docs[i] or "", "distance": float(distances[i]) if i < len(distances) else None,
        })
    return out


def summarize_dates(evidence):
    valid = sorted((e for e in evidence if e["date"] and e["assembly"]),
                   key=lambda e: (e["date"], e["assembly"]))
    if not valid:
        return "검색표본에서 확인 불가", "검색표본에서 확인 불가"
    # 같은 의회에 검색 청크가 여러 개 있으면 해당 의회의 표본상 가장 이른 기록만 사용
    by_assembly = {}
    for e in valid:
        by_assembly.setdefault(e["assembly"], e["date"])
    first = f'{valid[0]["date"]} / {valid[0]["assembly"]}'
    flow = " → ".join(f"{d} {a}" for a, d in sorted(by_assembly.items(), key=lambda x: (x[1], x[0])))
    return first, flow


def build_prompt(keyword, evidence):
    blocks = []
    for i, e in enumerate(evidence, 1):
        blocks.append(f'[근거 {i}] 날짜={e["date"] or "미상"}, 의회={e["assembly"] or "미상"}, '
                      f'회의록={e["meeting"] or "미확인"}, 안건={e["agenda"] or "미확인"}\n'
                      f'원문: {e["text"][:MAX_CONTEXT_CHARS]}')
    context = "\n\n".join(blocks)
    return (f"당신은 지방의회 정책 분석가입니다. 아래 검색된 회의록 원문만 참고하세요.\n"
            f"이슈 키워드: {keyword}\n"
            "작업: 위 이슈의 향후 지방의회 논의 방향을 200자 이내의 자연스러운 한국어 서술형으로 전망하세요.\n"
            "근거에 없는 실제 정책 시행 여부나 지역 간 정책 전파의 인과관계는 단정하지 마세요.\n"
            "전망은 불확실성을 밝혀야 하며 출처에 없는 내용은 쓰지 마세요.\n"
            "표/목록/제목/JSON/마크다운/서론 없이 전망 문장만 출력하세요.\n\n"
            f"[검색 근거]\n{context}\n\n[200자 이내 향후 예측]\n")


def call_ollama(model, prompt):
    t0 = time.perf_counter()
    response = requests.post(OLLAMA_URL, json={
        "model": model, "prompt": prompt, "stream": False, "think": False,
        "options": {"temperature": TEMPERATURE, "num_predict": NUM_PREDICT}
    }, timeout=600)
    response.raise_for_status()
    obj = response.json()
    text = (obj.get("response") or "").strip()
    return text, elapsed(t0), round(obj.get("load_duration", 0) / 1e9, 3), obj.get("eval_count", 0), obj.get("eval_duration", 0) / 1e9



def export_workbook():
    """누적 CSV를 업로드된 평가 양식과 동일한 4개 시트 구조로 내보냄.
    원본 엑셀은 수정하지 않으며 신규 실험 결과만 이 파일에 포함.
    """
    try:
        import pandas as pd
        frame = pd.read_csv(RESULT_CSV, encoding="utf-8-sig").fillna("")
        ev = pd.read_csv(EVIDENCE_CSV, encoding="utf-8-sig").fillna("")
        rank = frame[["키워드", "모델", "상태(200자)", "LLM(초)", "200자 준수"]].copy()
        rank.columns = ["이슈키워드", "모델", "상태", "LLM 초", "200자 준수"]
        rank["예측 타당성(40) 입력"] = ""
        rank["근거 일치성(30) 입력"] = ""
        rank["형식(15) 자동"] = (rank["200자 준수"] == "통과").astype(int) * 15
        rank["속도(15) 자동"] = 0.0
        # 동일 질문 내 통과 모델의 최단 시간 대비 속도 점수
        for issue, group in rank.groupby("이슈키워드"):
            valid = group[(group["200자 준수"] == "통과") & (pd.to_numeric(group["LLM 초"], errors="coerce") > 0)]
            if not valid.empty:
                best = pd.to_numeric(valid["LLM 초"]).min()
                for idx in valid.index:
                    rank.loc[idx, "속도(15) 자동"] = round(best / float(rank.loc[idx, "LLM 초"]) * 15, 2)
        rank["총점(100)"] = "미평가"
        rank["키워드 내 순위"] = "미평가"
        rank["평가 비고"] = "수동 점수 입력 전; 순위 미산정"
        rank = rank[["이슈키워드", "모델", "상태", "예측 타당성(40) 입력", "근거 일치성(30) 입력", "형식(15) 자동", "속도(15) 자동", "총점(100)", "키워드 내 순위", "LLM 초", "200자 준수", "평가 비고"]]
        rules = pd.DataFrame([
            ["분석방식", "1차 RAG: 개인 Chroma DB -> Ollama / BERTopic 미사용"],
            ["예측 타당성", "수동 0~40점. 근거 기반 전망의 구체성/불확실성"],
            ["근거 일치성", "수동 0~30점. 원본 JSON 대조 전에는 확정 금지"],
            ["형식 준수", "자동 0 또는 15점. 원본 답변 200자 이하일 때 통과"],
            ["속도", "자동 0~15점. 같은 키워드 내 통과 모델의 최단 LLM시간 / 해당 모델시간 x 15"],
            ["종합점수", "예측(40)+근거(30)+형식(15)+속도(15). 수동 평가 후 확정"],
            ["등장 시점", "Top-K 검색표본 내 가장 이른 일자; 전체 데이터 최초 등장 아님"],
            ["지역별 흐름", "Top-K 지역별 관찰 시점 순서; 정책 전파를 증명하지 않음"],
            ["원본 JSON", "Chroma 근거로만 생성. 원본 JSON 대조는 별도 작업"],
            ["실행조건", f"top_k={TOP_K}, temperature={TEMPERATURE}, num_predict={NUM_PREDICT}, models={len(MODELS)}"],
        ], columns=["항목", "설명 / 채점 원칙"])
        dest = RESULT_DIR / "개인_LLM_RAG_평가누적.xlsx"
        with pd.ExcelWriter(dest, engine="xlsxwriter") as writer:
            frame.to_excel(writer, sheet_name="모델결과", index=False)
            rules.to_excel(writer, sheet_name="평가기준", index=False)
            rank.to_excel(writer, sheet_name="키워드별순위", index=False)
            ev.to_excel(writer, sheet_name="근거회의록", index=False)
            for sheet in writer.sheets.values():
                sheet.freeze_panes(1, 0)
                sheet.set_column(0, 2, 21)
                sheet.set_column(3, 15, 26)
        print("엑셀 생성:", dest)
    except ImportError as exc:
        print("엑셀 변환 생략(패키지 부족):", exc, "| pip install pandas XlsxWriter")


def main():
    started = time.perf_counter()
    print(f"[{now()}] 실험 시작")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if not DB_DIR.exists():
        raise FileNotFoundError(f"DB 없음: {DB_DIR}")
    print("DB 연결 중...")
    t = time.perf_counter()
    db = chromadb.PersistentClient(path=str(DB_DIR))
    col = db.get_collection(COLLECTION)
    print(f"DB 청크: {col.count():,} | 연결: {elapsed(t)}초")
    if col.count() == 0:
        raise RuntimeError("DB에 데이터가 없음")
    t = time.perf_counter()
    embedder = SentenceTransformer(MODEL_ID)
    print(f"임베딩 모델 로드: {elapsed(t)}초")
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    for qi, keyword in enumerate(KEYWORDS, 1):
        print(f"\n[{qi}/{len(KEYWORDS)}] {keyword}: 검색 중", flush=True)
        t = time.perf_counter()
        v = embedder.encode([keyword], convert_to_numpy=True)
        embed_time = elapsed(t)
        if v.shape[1] != 1024:
            raise ValueError(f"벡터 차원 불일치: {v.shape}")
        t = time.perf_counter()
        search = col.query(query_embeddings=v.astype(np.float32).tolist(), n_results=TOP_K,
                           include=["documents", "metadatas", "distances"])
        search_time = elapsed(t)
        evidence = collect_evidence(search)
        first, flow = summarize_dates(evidence)
        print(f"  질문 임베딩 {embed_time}초 | 검색 {search_time}초 | 근거 {len(evidence)}개")
        prompt = build_prompt(keyword, evidence)
        evidence_summary = " | ".join(f'{e["date"]} {e["assembly"]}: {e["meeting"] or "회의록명 미확인"} ({e["id"]})'
                                      for e in evidence[:5])
        for mi, model in enumerate(MODELS, 1):
            tick = time.perf_counter()
            text = ""
            llm_time = load_time = 0
            tokens = 0
            tok_per_sec = 0
            error = ""
            try:
                text, llm_time, load_time, tokens, eval_sec = call_ollama(model, prompt)
                tok_per_sec = round(tokens / eval_sec, 2) if eval_sec else 0
                status = "OK" if text and len(text) <= 200 else "OVER_200" if text else "EMPTY"
            except Exception as exc:
                status = "ERROR"
                error = str(exc)[:500]
            total_time = round(embed_time + search_time + elapsed(tick), 3)
            rid = f"{experiment_id}_{qi:02d}_{mi:02d}"
            row = {
                "실험": EXPERIMENT, "키워드": keyword, "모델": model, "상태(200자)": status,
                "등장시점(검색표본)": first, "타 지방의회 논의순서(검색표본)": flow,
                "향후 예측(앞 200자 표시)": text[:200], "원래 예측 전체": text,
                "실제 글자 수": len(text), "200자 준수": "통과" if status == "OK" else "미준수/오류",
                "근거 회의록(텍스트 요약)": evidence_summary, "근거 수": len(evidence),
                "질문 임베딩(초)": embed_time, "DB 검색(초)": search_time,
                "LLM(초)": llm_time, "전체(초)": total_time, "모델 로드(초)": load_time,
                "출력 토큰": tokens, "토큰/초": tok_per_sec, "오류": error,
                "실행시각": now(), "검색근거검증": "Chroma만 확인 / 원본 JSON 미검증", "실험ID": rid,
            }
            append_csv(RESULT_CSV, RESULT_FIELDS, [row])
            ev_rows = []
            for j, e in enumerate(evidence, 1):
                ev_rows.append({
                    "실험ID": rid, "이슈": keyword, "모델": model, "근거 번호": j,
                    "일자": e["date"], "의회": e["assembly"],
                    "원본 회의록명": e["meeting"] or "회의록명 미확인",
                    "안건": e["agenda"], "발언자": e["speaker"], "모델 상태": status,
                    "근거 원문(앞 220자)": e["text"][:220], "저장된 source_file": e["source_file"],
                    "row_id": e["row_id"], "chunk_id": e["id"], "검색 거리": e["distance"],
                    "검증 상태": "원본 JSON 미검증",
                })
            append_csv(EVIDENCE_CSV, EVIDENCE_FIELDS, ev_rows)
            print(f"  [{mi}/{len(MODELS)}] {model}: {status} | {len(text)}자 | LLM {llm_time}초 | 전체 {total_time}초", flush=True)
    print(f"\n[{now()}] 작업 종료 | 총 {elapsed(started)}초 ({elapsed(started)/60:.2f}분)")
    print("결과:", RESULT_CSV)
    print("근거:", EVIDENCE_CSV)
    export_workbook()
    print("주의: 전체 최초 등장/지역 간 인과 확산/원본 JSON 일치 여부는 별도 검증 필요")


if __name__ == "__main__":
    main()


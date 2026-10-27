# compare_local_rag_models.py
# Team ChromaDB + Ollama: 5 issues x 4 models, append into one CSV.
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

import chromadb
import requests
from sentence_transformers import SentenceTransformer

DB_PATH = Path(r"C:\mini_team\_db\chroma_snowflake_regions")
COLLECTION_NAME = "team_regions_6a0a006e92d1e93e"
EMBED_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OUTPUT_CSV = Path(r"C:\mini_team\scripts\results\local_rag_model_comparison.csv")
EXPERIMENT = "1차_RAG"  # 2차에는 '2차_토픽RAG' 등으로 변경

ISSUES = ["지역소멸", "청년 일자리", "저출생", "고령화", "기후위기"]
MODELS = ["exaone3.5:2.4b", "qwen3:8b", "gemma3:12b", "gemma3:4b"]
SEARCH_K = 40
EVIDENCE_K = 8
MAX_EVIDENCE_CHARS = 650

FIELDS = [
    "experiment", "model_rank", "model", "issue_keyword", "question",
    "first_appearance", "regional_flow", "prediction", "prediction_char_count",
    "evidence_records", "evidence_count", "embedding_sec", "search_sec",
    "llm_sec", "total_sec", "load_sec", "output_tokens", "tokens_per_sec",
    "status", "error", "created_at"
]


def day(value):
    s = re.sub(r"\D", "", str(value or ""))
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) >= 8 else "날짜미상"


def choose_evidence(results):
    candidates, seen = [], set()
    for doc, meta, distance, chunk_id in zip(
        results["documents"][0], results["metadatas"][0],
        results["distances"][0], results["ids"][0]
    ):
        if not doc:
            continue
        meta = meta or {}
        # Same original text in duplicated rows counts once.
        dedup = (meta.get("source_fingerprint"), doc.strip())
        if dedup in seen:
            continue
        seen.add(dedup)
        candidates.append({"id": chunk_id, "text": doc, "meta": meta, "distance": distance})

    # Favor different assemblies first; then fill remaining by relevance.
    selected, used_assemblies = [], set()
    for item in candidates:
        assembly = item["meta"].get("assembly", "미상")
        if assembly not in used_assemblies:
            selected.append(item)
            used_assemblies.add(assembly)
        if len(selected) == EVIDENCE_K:
            break
    selected_ids = {item["id"] for item in selected}
    for item in candidates:
        if len(selected) >= EVIDENCE_K:
            break
        if item["id"] not in selected_ids:
            selected.append(item)
            selected_ids.add(item["id"])
    return selected


def source_records(evidence):
    records = []
    for n, item in enumerate(evidence, 1):
        m = item["meta"]
        records.append({
            "evidence_id": n,
            "date": day(m.get("date")),
            "assembly": m.get("assembly", "미상"),
            "meeting_title": None,  # Verify against source JSON separately.
            "agenda": m.get("agenda", ""),
            "speaker": m.get("speaker", ""),
            "source_file": m.get("source_file", ""),
            "row_id": m.get("row_id"),
            "chunk_id": item["id"],
            "char_start": m.get("char_start"),
            "char_end": m.get("char_end"),
            "distance": round(float(item["distance"]), 4),
            "evidence_text": item["text"],
        })
    return records


def chronology(records):
    # Dates/assemblies of retrieved evidence only, NOT dataset-wide first use.
    pairs = sorted({(r["date"], r["assembly"]) for r in records if r["date"] != "날짜미상"})
    if not pairs:
        return "확인 불가", "확인 불가"
    return f"{pairs[0][0]} / {pairs[0][1]}", " → ".join(f"{d} {a}" for d, a in pairs)


def make_prompt(issue, evidence):
    parts = []
    for i, item in enumerate(evidence, 1):
        m = item["meta"]
        parts.append(
            f"[근거 {i}] 의회: {m.get('assembly','미상')} / 일자: {day(m.get('date'))} "
            f"/ 안건: {m.get('agenda','미상')} / 청크ID: {item['id']}\n"
            f"원문: {item['text'][:MAX_EVIDENCE_CHARS]}"
        )
    context = "\n\n".join(parts)
    return (
        f"정책 이슈: {issue}\n\n검색 근거:\n{context}\n\n"
        "검색 근거에 실제로 있는 논의만 활용하여, 이후 지방의회에서 논의될 가능성이 있는 "
        "정책 방향을 한국어 서술형 300자 이내로 전망하세요. 추측을 사실처럼 단정하지 마세요. "
        "다른 정보 없이 다음 JSON 객체만 출력하세요: "
        '{"prediction":"300자 이내 전망"}'
    )


def ask_ollama(model_name, prompt):
    start = time.perf_counter()
    res = requests.post(
        OLLAMA_URL,
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": "당신은 지방의회 정책 분석가입니다. 반드시 주어진 자료에 근거하고 JSON만 출력하세요."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 450},
        },
        timeout=600,
    )
    elapsed = time.perf_counter() - start
    res.raise_for_status()
    payload = res.json()
    raw = payload.get("message", {}).get("content", "").strip()
    parsed = json.loads(raw)
    prediction = str(parsed.get("prediction", "")).strip()
    if not prediction:
        raise ValueError(f"prediction 필드 없음: {raw[:200]}")
    tokens = payload.get("eval_count")
    eval_duration = payload.get("eval_duration")  # nanoseconds
    tps = (tokens / (eval_duration / 1e9)) if tokens and eval_duration else ""
    return prediction, elapsed, round((payload.get("load_duration") or 0) / 1e9, 3), tokens or "", round(tps, 2) if tps else ""


def completed_keys(path):
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {(r["experiment"], r["issue_keyword"], r["model"])
                for r in csv.DictReader(f) if r.get("status") == "OK"}


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(OUTPUT_CSV)
    is_new = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
    print("[시작] CSV:", OUTPUT_CSV)

    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    print("[DB]", collection.name, f"{collection.count():,}개")
    embedding_model = SentenceTransformer(EMBED_MODEL)

    with OUTPUT_CSV.open("a", encoding="utf-8-sig" if is_new else "utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()

        for issue in ISSUES:
            question = f"지방의회에서 논의된 '{issue}' 관련 정책 이슈와 향후 방향은?"
            print("\n[이슈]", issue)
            t0 = time.perf_counter()
            vector = embedding_model.encode(
                question, prompt_name="query", normalize_embeddings=True
            ).tolist()
            embed_sec = time.perf_counter() - t0

            t0 = time.perf_counter()
            result = collection.query(
                query_embeddings=[vector], n_results=SEARCH_K,
                include=["documents", "metadatas", "distances"]
            )
            search_sec = time.perf_counter() - t0
            evidence = choose_evidence(result)
            records = source_records(evidence)
            first, flow = chronology(records)
            prompt = make_prompt(issue, evidence)
            print(f"  검색 근거 {len(evidence)}개 / 임베딩 {embed_sec:.3f}초 / 검색 {search_sec:.3f}초")

            for rank, model_name in enumerate(MODELS, 1):
                key = (EXPERIMENT, issue, model_name)
                if key in done:
                    print("  [건너뜀]", model_name)
                    continue
                row = dict.fromkeys(FIELDS, "")
                row.update({
                    "experiment": EXPERIMENT, "model_rank": rank, "model": model_name,
                    "issue_keyword": issue, "question": question,
                    "first_appearance": first, "regional_flow": flow,
                    "evidence_records": json.dumps(records, ensure_ascii=False),
                    "evidence_count": len(evidence),
                    "embedding_sec": round(embed_sec, 3),
                    "search_sec": round(search_sec, 3),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                })
                try:
                    prediction, llm_sec, load_sec, tokens, tps = ask_ollama(model_name, prompt)
                    row.update({
                        "prediction": prediction,
                        "prediction_char_count": len(prediction),
                        "llm_sec": round(llm_sec, 3),
                        "total_sec": round(embed_sec + search_sec + llm_sec, 3),
                        "load_sec": load_sec, "output_tokens": tokens,
                        "tokens_per_sec": tps,
                        "status": "OK" if len(prediction) <= 300 else "OVER_300",
                    })
                except Exception as e:
                    row.update({"status": "ERROR", "error": str(e)[:400]})
                writer.writerow(row)
                f.flush()  # Save each result immediately.
                print(f"  [{row['status']}] {model_name} / LLM {row['llm_sec']}초 / 예측 {row['prediction_char_count']}자")
    print("\n[완료]", OUTPUT_CSV)
    print("주의: 등장 시점·흐름은 검색 결과 범위 내 확인 사항이며 원본 JSON과의 최종 대조가 필요합니다.")


if __name__ == "__main__":
    main()

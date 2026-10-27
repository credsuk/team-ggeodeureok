"""1차: Chroma 검색 + Ollama. 두 실험은 이 파일의 공통 함수를 사용한다.
실행: python compare_local_rag_models_1.py
설치: pip install chromadb sentence-transformers requests
"""
import csv
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path

import chromadb
import requests
from sentence_transformers import SentenceTransformer

DB_PATH = Path(r'C:\mini_team\_db\chroma_snowflake_regions')
COLLECTION_NAME = 'team_regions_6a0a006e92d1e93e'
EMBED_MODEL = 'dragonkue/snowflake-arctic-embed-l-v2.0-ko'
OLLAMA_URL = 'http://127.0.0.1:11434/api/chat'
OUTPUT_CSV = Path(r'C:\mini_team\scripts\results\local_rag_model_comparison_v2.csv')
EXPERIMENT = '1차_기본RAG'
STANDARD = '공통_v2_예측200자_20261023'
ISSUES = ['지역소멸', '청년 일자리', '저출생', '고령화', '기후위기']
MODELS = ['exaone3.5:2.4b', 'qwen3:8b', 'gemma3:12b', 'gemma3:4b']
SEARCH_K = 40
EVIDENCE_K = 8
MAX_EVIDENCE_CHARS = 650

# 두 실험에서 컬럼/문항/생성 옵션/근거 개수는 동일하게 유지한다.
FIELDS = [
    'experiment', 'evaluation_standard', 'model_rank', 'model', 'issue_keyword', 'question',
    'first_appearance', 'regional_flow', 'prediction', 'prediction_char_count',
    'evidence_records', 'evidence_count', 'evidence_ids', 'evidence_hash',
    'topic_context', 'topic_source',
    'embedding_sec', 'search_sec', 'llm_sec', 'total_sec', 'load_sec',
    'output_tokens', 'tokens_per_sec', 'status', 'error', 'created_at'
]


def day(value):
    raw = re.sub(r'\D', '', str(value or ''))
    return f'{raw[:4]}-{raw[4:6]}-{raw[6:8]}' if len(raw) >= 8 else '날짜미상'


def choose_evidence(results):
    candidates, seen = [], set()
    for doc, meta, dist, chunk_id in zip(
        results['documents'][0], results['metadatas'][0],
        results['distances'][0], results['ids'][0]
    ):
        if not doc:
            continue
        meta = meta or {}
        dedup = (meta.get('source_fingerprint'), doc.strip())
        if dedup in seen:
            continue
        seen.add(dedup)
        candidates.append(dict(id=chunk_id, text=doc, meta=meta, distance=dist))

    # 검색 결과 내 의회 다양성 확보 (기존 1차 기준과 동일)
    selected, used = [], set()
    for item in candidates:
        assembly = item['meta'].get('assembly', '미상')
        if assembly not in used:
            selected.append(item)
            used.add(assembly)
        if len(selected) == EVIDENCE_K:
            break
    selected_ids = {x['id'] for x in selected}
    for item in candidates:
        if len(selected) >= EVIDENCE_K:
            break
        if item['id'] not in selected_ids:
            selected.append(item)
            selected_ids.add(item['id'])
    return selected


def plain_evidence(items):
    """HTML/JSON 문자열 대신 읽기 쉬운 짧은 원문 발췌. 원본 JSON 직접 조회 아님."""
    lines = []
    for i, item in enumerate(items, 1):
        meta = item['meta']
        excerpt = re.sub(r'\s+', ' ', item['text']).strip()[:180]
        meeting = meta.get('meeting_title') or meta.get('agenda') or '회의록명 미확인'
        lines.append(
            f"[근거 {i}] {day(meta.get('date'))} | {meta.get('assembly','미상')} | "
            f"{meeting} | 원문: {excerpt} | 청크ID: {item['id']}"
        )
    return '\n'.join(lines)


def chronology(items):
    # 전체 218만 건의 최초 시점이 아니라 검색 근거에 한정된 시점.
    pairs = sorted({(day(x['meta'].get('date')), x['meta'].get('assembly', '미상'))
                    for x in items if day(x['meta'].get('date')) != '날짜미상'})
    if not pairs:
        return '확인 불가', '확인 불가'
    return f'{pairs[0][0]} / {pairs[0][1]}', ' → '.join(f'{d} {a}' for d, a in pairs)


def prompt_for(issue, items, topic_context=''):
    texts = []
    for i, item in enumerate(items, 1):
        m = item['meta']
        texts.append(
            f"[근거 {i}] {day(m.get('date'))} / {m.get('assembly','미상')} / "
            f"안건: {m.get('agenda','미상')} / ID: {item['id']}\n"
            f"원문: {item['text'][:MAX_EVIDENCE_CHARS]}"
        )
    topic_section = (f'\n[군집화·토픽 분석 결과]\n{topic_context}\n' if topic_context else '')
    return (
        f'정책 이슈: {issue}\n\n[회의록 검색 근거]\n' + '\n\n'.join(texts) + '\n' + topic_section +
        '\n위 회의록에 근거하여 향후 지방의회 정책 논의 방향을 한국어 문장으로 예측하세요. '
        '예측문만 200자 이내, 2~3문장, 목록/마크다운 금지. '
        '원문에 없는 사실을 단정하거나 서로 다른 의회의 인과적 전파를 꾸며내지 마세요. '
        '군집·토픽 정보가 있다면 예측의 보조 단서로만 활용하세요. '
        '오직 JSON 객체 하나만 답하세요: {"prediction":"예측문"}'
    )


def ask_ollama(model_name, prompt):
    """Try to produce <=200 chars; include retry cost in reported timings."""
    total_sec = 0.0
    total_load = 0.0
    total_tokens = 0
    total_generation_sec = 0.0
    previous = ''

    for attempt in range(3):
        if attempt == 0:
            current_prompt = prompt
        else:
            current_prompt = (
                prompt + "\n\n[글자 수 재검증] 이전 출력은 "
                + str(len(previous)) + "자로 제한을 초과했습니다. "
                + "핵심 정책 전망만 남겨 공백 포함 170~190자로 다시 작성하세요. "
                + "사실이나 새로운 근거를 추가하지 마세요. "
                + "오직 {\"prediction\":\"...\"} JSON 객체로 답하세요."
            )
        t0 = time.perf_counter()
        response = requests.post(
            OLLAMA_URL,
            json={
                'model': model_name,
                'messages': [
                    {'role': 'system', 'content': '제공된 회의록만 근거로 200자 이내 정책 예측을 JSON으로 작성하세요.'},
                    {'role': 'user', 'content': current_prompt},
                ],
                'stream': False, 'think': False, 'format': 'json',
                'options': {'temperature': 0, 'seed': 42, 'num_ctx': 4096, 'num_predict': 250},
            }, timeout=600,
        )
        total_sec += time.perf_counter() - t0
        response.raise_for_status()
        result = response.json()
        total_load += (result.get('load_duration') or 0) / 1e9
        total_tokens += result.get('eval_count') or 0
        total_generation_sec += (result.get('eval_duration') or 0) / 1e9
        raw = result.get('message', {}).get('content', '').strip()
        parsed = json.loads(raw)
        previous = str(parsed.get('prediction') or '').strip()
        if not previous:
            raise ValueError('prediction 필드 누락 또는 빈 응답')
        if len(previous) <= 200:
            tps = round(total_tokens / total_generation_sec, 2) if total_generation_sec else ''
            return previous, total_sec, round(total_load, 3), total_tokens, tps

    # Do not silently truncate policy statements.
    tps = round(total_tokens / total_generation_sec, 2) if total_generation_sec else ''
    return previous, total_sec, round(total_load, 3), total_tokens, tps


def completed_keys():
    if not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0:
        return set(), True
    with OUTPUT_CSV.open('r', encoding='utf-8-sig', newline='') as fh:
        rd = csv.DictReader(fh)
        if rd.fieldnames != FIELDS:
            raise ValueError('기존 CSV 컬럼 구성이 다릅니다. 동일 버전 파일인지 확인하세요.')
        completed = {(x['experiment'], x['issue_keyword'], x['model']) for x in rd}
        # All recorded statuses count as attempted; cleanup_first_rag.py removes invalid first-phase rows for deliberate retries.
    return completed, False


def run(experiment=EXPERIMENT, topic_data=None):
    """topic_data: {이슈: {'summary': str, 'source': str}}. 없으면 기본 RAG."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    done, is_new = completed_keys()
    print('실험:', experiment, '| 누적 CSV:', OUTPUT_CSV)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    print('컬렉션:', collection.name, '| 청크:', f'{collection.count():,}')
    emb = SentenceTransformer(EMBED_MODEL)

    with OUTPUT_CSV.open('a', encoding='utf-8-sig' if is_new else 'utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        for issue in ISSUES:
            if all((experiment, issue, m) in done for m in MODELS):
                print('[완료된 이슈 건너뜀]', issue)
                continue
            question = f"지방의회에서 논의된 '{issue}' 관련 정책 이슈와 향후 방향은?"
            t0 = time.perf_counter()
            vector = emb.encode(question, prompt_name='query', normalize_embeddings=True).tolist()
            embed_sec = time.perf_counter() - t0
            t0 = time.perf_counter()
            result = collection.query(query_embeddings=[vector], n_results=SEARCH_K,
                                      include=['documents', 'metadatas', 'distances'])
            search_sec = time.perf_counter() - t0
            evidence = choose_evidence(result)
            if not evidence:
                raise RuntimeError(f'검색 근거 없음: {issue}')
            first, flow = chronology(evidence)
            evidence_ids = [x['id'] for x in evidence]
            evidence_hash = hashlib.sha256('|'.join(evidence_ids).encode()).hexdigest()[:16]
            topic = (topic_data or {}).get(issue, {})
            topic_summary = topic.get('summary', '')
            topic_source = topic.get('source', '')
            prompt = prompt_for(issue, evidence, topic_summary)
            print(f'\n{issue}: 근거 {len(evidence)}개 / 임베딩 {embed_sec:.3f}초 / DB {search_sec:.3f}초')

            for rank, model in enumerate(MODELS, 1):
                if (experiment, issue, model) in done:
                    print('[건너뜀]', model)
                    continue
                row = dict.fromkeys(FIELDS, '')
                row.update(experiment=experiment, evaluation_standard=STANDARD,
                           model_rank=rank, model=model, issue_keyword=issue,
                           question=question, first_appearance=first,
                           regional_flow=flow, evidence_records=plain_evidence(evidence),
                           evidence_count=len(evidence), evidence_ids='|'.join(evidence_ids),
                           evidence_hash=evidence_hash, topic_context=topic_summary,
                           topic_source=topic_source,
                           embedding_sec=round(embed_sec, 3), search_sec=round(search_sec, 3),
                           created_at=datetime.now().isoformat(timespec='seconds'))
                try:
                    prediction, llm_sec, load_sec, tokens, tps = ask_ollama(model, prompt)
                    row.update(prediction=prediction, prediction_char_count=len(prediction),
                               llm_sec=round(llm_sec, 3),
                               total_sec=round(embed_sec + search_sec + llm_sec, 3),
                               load_sec=load_sec, output_tokens=tokens, tokens_per_sec=tps,
                               status='OK' if len(prediction) <= 200 else 'OVER_200')
                except Exception as exc:
                    row.update(status='ERROR', error=str(exc)[:350])
                writer.writerow(row)
                f.flush()
                print(f"  {model} | {row['status']} | 생성 {row['llm_sec']}초 | {row['prediction_char_count']}자")
    print('\n완료:', OUTPUT_CSV)
    print('주의: 최초 등장·확산은 검색 근거 범위 내 관측치이며 원본 JSON 대조가 필요합니다.')


def export_excel():
    import pandas as pd

    # 누적 CSV 불러오기
    df = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")

    # Excel 저장 경로
    xlsx_path = OUTPUT_CSV.with_suffix(".xlsx")

    # 평가기준 시트
    criteria = pd.DataFrame([
        ["이슈 키워드", "분석할 정책 주제", "-"],
        ["등장 시점", "검색 근거 중 가장 이른 날짜·의회", "-"],
        ["지역별 흐름", "날짜순 지방의회 논의 기록", "-"],
        ["향후 예측", "LLM 생성 예측문, 200자 이내", "-"],
        ["근거 회의록", "ChromaDB 원문 발췌 및 출처", "-"],
        ["예측 타당성", "정책 흐름에 따른 예측의 논리성", 40],
        ["근거 일치성", "제공된 원문과 예측 내용의 일치 정도", 30],
        ["형식 준수", "정상 응답 및 200자 제한 준수", 15],
        ["응답 속도", "동일 이슈 내 LLM 응답시간 비교", 15],
        ["embedding_sec", "질문 임베딩 소요시간(초)", "-"],
        ["search_sec", "ChromaDB 검색 소요시간(초)", "-"],
        ["llm_sec", "Ollama 요청 및 응답 소요시간(초)", "-"],
        ["total_sec", "임베딩 + 검색 + LLM 시간(초)", "-"],
    ], columns=["평가 항목", "설명", "배점"])

    # 키워드별순위 시트
    ranking = df[[
        "experiment", "issue_keyword", "model",
        "status", "llm_sec", "total_sec"
    ]].copy()

    # 수동 채점 항목: 실제 원문 검증 후 입력
    ranking["예측타당성_40"] = None
    ranking["근거일치성_30"] = None

    # 자동 채점
    ranking["형식준수_15"] = (
        (df["status"] == "OK") &
        (pd.to_numeric(
            df["prediction_char_count"],
            errors="coerce"
        ) <= 200)
    ).astype(int) * 15

    # 동일 실험·이슈 내 성공 응답 속도 비교
    ranking["속도점수_15"] = 0.0
    valid = ranking["status"] == "OK"

    ranking.loc[valid, "속도점수_15"] = (
        ranking[valid]
        .groupby(["experiment", "issue_keyword"])["llm_sec"]
        .transform(
            lambda s: (
                s.min() / s.astype(float) * 15
            ).round(2)
        )
    )

    # 수동 채점 후 엑셀에서 계산할 총점·순위
    ranking["총점_100"] = None
    ranking["키워드별순위"] = None

    # 근거 회의록 시트
    evidence = df[[
        "experiment", "issue_keyword", "model",
        "evidence_records"
    ]].copy()

    # 중복 근거는 한 번만 표시
    evidence = evidence.drop_duplicates(
        subset=["experiment", "issue_keyword", "evidence_records"]
    )

    # Excel 생성
    with pd.ExcelWriter(
        xlsx_path, engine="xlsxwriter"
    ) as writer:

        sheets = {
            "모델결과": df,
            "평가기준": criteria,
            "키워드별순위": ranking,
            "근거회의록": evidence,
        }

        for sheet_name, data in sheets.items():
            data.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )

            sheet = writer.sheets[sheet_name]
            sheet.freeze_panes(1, 0)
            sheet.autofilter(
                0, 0, len(data), len(data.columns) - 1
            )
            sheet.set_column(0, len(data.columns) - 1, 22)

        # 점수 입력 후 자동 총점·순위 계산
        rank_sheet = writer.sheets["키워드별순위"]

        for row in range(1, len(ranking) + 1):
            excel_row = row + 1

            # 모든 수동 점수가 입력되었을 때만 계산
            rank_sheet.write_formula(
                row, 10,
                f'=IF(OR(G{excel_row}="",H{excel_row}=""),'
                f'"",SUM(G{excel_row}:J{excel_row}))'
            )

            rank_sheet.write_formula(
                row, 11,
                f'=IF(K{excel_row}="","",'
                f'1+COUNTIFS(A$2:A${len(ranking)+1},A{excel_row},'
                f'B$2:B${len(ranking)+1},B{excel_row},'
                f'K$2:K${len(ranking)+1},">"&K{excel_row}))'
            )

    print(f"Excel 저장 완료: {xlsx_path}")


if __name__ == '__main__':
    run()
    export_excel()

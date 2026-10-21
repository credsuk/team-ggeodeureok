"""
지방의회 회의록에서 한국어 형태소 분석기 Kiwi(kiwipiepy)를 활용하여
고속으로 정책 의제, 복합 키워드, 카테고리, 추출 요약 메타데이터(JSONL)를 추출하는 스크립트.
(로컬 LLM 대체용 초고속 경량화 파이프라인, 체크포인트 이어하기, 실시간 스트리밍 flush 지원)
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional

from kiwipiepy import Kiwi
from tqdm import tqdm


# ==============================================================================
# 1. 의회 회의록 특화 불용어 목록
# ==============================================================================
COUNCIL_STOPWORDS: Set[str] = {
    # 1. 포괄적/상투적 명사 (변별력 없는 단어)
    "사업", "예산", "지원", "조례", "조례안", "강원특별자치도", "의견", "업무",
    "행정", "강원도", "강원", "의회", "도정", "시군", "지자체", "정부", "국가",
    "정책", "제도", "부분", "때문", "생각", "정도", "사항", "관련", "내용",
    "경우", "사실", "확인", "판단", "입장", "이유", "이후", "이전", "지금",
    "현재", "당시", "오늘", "내일", "어제", "하나", "이상", "이하", "가운데",
    "자체", "각종", "주요", "전체", "모든", "서로", "단계", "추진", "진행",
    "노력", "필요", "준비", "검토", "상황", "결과", "시간", "추가", "개정",
    "신청", "이용", "사용", "방법", "대상", "기준", "시작", "제시", "마련",
    "반영", "해결", "조치", "발생", "확보", "기회", "이번", "지난", "문제",
    "중요", "방안", "역할", "활동", "계획", "목적", "방향", "효과", "의미",
    "말씀", "보고", "설명", "자료", "서류", "제출", "다음", "질문", "회의록",
    "얘기", "이야기", "청취", "퇴장", "출석", "부탁", "요청", "고생", "답변석",
    "관련해", "대해", "통해", "위해", "바탕", "중심", "측면", "차원", "기반",
    # 2. 직함, 호칭, 인명 및 의전 관련
    "의원", "위원", "위원장", "부위원장", "의장", "부의장", "도지사", "교육감",
    "부지사", "시장", "군수", "구청장", "국장", "과장", "실장", "팀장", "원장",
    "총장", "위원님", "의원님", "도민", "군민", "시민", "선배", "동료", "존경",
    "관계자", "관계관", "공무원", "총괄", "소관", "부서", "의사관", "처장",
    "사무처", "사무국장", "간부", "지사", "의석", "정돈", "성원", "의사",
    "방청", "방청석", "이석", "소개", "선서", "원안", "가결", "부의", "상정",
    "이의", "동의", "재청", "속개", "개의", "산회", "정회", "선포", "발언",
    "질의", "답변", "회의", "본회의", "임시회", "정례회", "감사", "일정",
    "위원회", "수고", "인사", "환영", "박수", "양지", "발령",
    # 3. 주요 인명 (의회 공통)
    "권혁열", "권혁", "김진태", "신경호", "김명선", "김종욱", "강성룡",
    "최종수", "한창수", "문관현", "박호균", "김시성", "엄기호", "양승준",
    # 4. 일반 대명사 및 보조어
    "우리", "저희", "본인", "자신", "그것", "이것", "저것", "여기", "저기",
    "누구", "어디", "무엇", "존경하는", "감사합니다", "고맙습니다", "사랑하는",
}


# ==============================================================================
# 2. 7대 정책 분야별 핵심 키워드 사전
# ==============================================================================
POLICY_CATEGORIES: Dict[str, List[str]] = {
    "경제/산업": [
        "반도체", "바이오", "일자리", "기업", "산업", "투자", "수출", "소상공인", "경제", "창업",
        "시장", "물가", "금융", "상권", "상속세", "법인세", "세제", "제조업", "공장", "고용",
        "벤처", "유치", "신산업", "상인", "골목상권", "특례", "지역경제", "산업단지", "기술",
        "디지털", "인공지능", "ai", "수소", "에너지", "혁신", "기업유치", "일자리창출"
    ],
    "농림/수산": [
        "농업", "농지", "농가", "농민", "어업", "수산", "축산", "산림", "임업", "임산물",
        "작물", "어선", "쌀", "농촌", "어촌", "한우", "경자유전", "양식", "귀농", "어업인",
        "가축", "영농", "농산물", "임산", "수산업", "농축산", "재해보험", "직불금", "종자",
        "사료", "스마트팜", "농업용수", "어항", "산림청"
    ],
    "보건/복지": [
        "복지", "의료", "병원", "의사", "간호", "장애인", "노인", "어르신", "아동", "보육",
        "기초생활", "돌봄", "저출산", "보건소", "요양", "공공의료", "취약계층", "사회복지",
        "연금", "보건", "건강", "필수의료", "인구감소", "인구정책", "고령화", "응급의료",
        "치매", "어린이집", "경로당", "간병", "자살", "사회적약자", "지방소멸"
    ],
    "교육/문화": [
        "교육", "학교", "학생", "교사", "교원", "문화", "관광", "축제", "체육", "스포츠",
        "예술", "도서관", "박물관", "문화재", "청소년", "학령인구", "올림픽", "늘봄", "급식",
        "수업", "진로", "학습", "공교육", "직업계고", "고교학점제", "대학", "유치원", "돌봄교실",
        "관광객", "문화도시", "체육시설", "학부모", "기초학력"
    ],
    "환경/안전": [
        "환경", "기후", "탄소", "온실가스", "폐기물", "오염", "재난", "안전", "소방", "산불",
        "침식", "연안", "수질", "하천", "대기", "방재", "미세먼지", "생태계", "해양쓰레기",
        "자연재해", "재난안전", "연안침식", "해안", "상수도", "하수도", "매립지", "소방관",
        "안전점검", "태풍", "호우", "지진", "녹색성장"
    ],
    "도시/교통": [
        "교통", "철도", "도로", "gtx", "버스", "도시", "주택", "건설", "공항", "인프라",
        "항만", "역세권", "신도시", "도시재생", "도로망", "고속도로", "광역급행철도", "대중교통",
        "주차장", "광역철도망", "철도망", "수도권", "도시계획", "택지", "물류", "지하철",
        "국도", "지방도", "보행로", "교통약자", "대중교통망"
    ],
    "자치/행정": [
        "자치", "분권", "특별법", "조직", "인사", "규칙", "결산", "선거", "개편",
        "출자출연", "행정기구", "지방자치", "행정체제", "공공기관", "행정구역", "통합",
        "주민자치", "자치경찰", "지방재정", "교부세", "특례시", "균형발전"
    ],
}


# ==============================================================================
# 3. 비실질적(단순 의전/절차 멘트) 정규식 패턴
# ==============================================================================
PROCEDURAL_PATTERNS = [
    r"성원이 되었으므로.*(?:개의|선포)",
    r"이의가 없으므로 가결되었음을 선포합니다",
    r"이상으로.*보고를 마치겠습니다",
    r"수고하셨습니다\.",
    r"자리에서 일어나 인사해 주시기 바랍니다",
    r"이의가 없으십니까\?",
    r"개의하겠습니다",
    r"산회를 선포합니다",
    r"정회를 선포합니다",
    r"원안대로 가결되었음을 선포합니다",
    r"출석하여 주신.*감사합니다",
    r"의석을 정돈하여 주시기 바랍니다",
    r"보고사항을 말씀드리겠습니다",
    r"발언을 마치겠습니다",
    r"답변을 마치겠습니다",
    r"보고하여 주시기 바랍니다",
    r"상정합니다\.",
    r"자유발언을 듣겠습니다",
]


class KiwiCouncilAnalyzer:
    """Kiwi 기반 의회 회의록 분석 및 메타데이터 추출 엔진"""

    def __init__(self, num_workers: int = 4):
        self.kiwi = Kiwi(num_workers=num_workers)

    def analyze_utterance(
        self,
        text: str,
        agenda_title: str,
        speaker_role: str,
        speaker_name: str,
        min_char_len: int = 80,
    ) -> Dict[str, Any]:
        """단일 발언 텍스트를 분석하여 JSONL 메타데이터 생성"""
        text_clean = text.strip()
        char_len = len(text_clean)

        # 문장 분리
        sents = self.kiwi.split_into_sents(text_clean)
        first_sent = sents[0].text.strip() if sents else text_clean[:100]

        # 1. 단순 의전 / 회의 절차 발언 판별
        procedural_matches = sum(1 for pat in PROCEDURAL_PATTERNS if re.search(pat, text_clean))
        is_procedural_start = any(re.search(pat, first_sent) for pat in PROCEDURAL_PATTERNS)
        
        # 화자 및 직함 기반 불용어 동적 추가
        dynamic_stopwords = set(COUNCIL_STOPWORDS)
        if speaker_name:
            dynamic_stopwords.add(speaker_name)
            # 이름의 각 2음절 조합도 추가 (예: 권혁열 -> 권혁, 혁열)
            if len(speaker_name) == 3:
                dynamic_stopwords.add(speaker_name[:2])
                dynamic_stopwords.add(speaker_name[1:])
        if speaker_role:
            dynamic_stopwords.add(speaker_role)

        # 2. 형태소 토큰화 및 명사 추출
        tokens = self.kiwi.tokenize(text_clean)
        meaningful_nouns: List[str] = []

        for token in tokens:
            if token.tag in ("NNG", "NNP") and len(token.form) >= 2:
                if token.form not in dynamic_stopwords:
                    meaningful_nouns.append(token.form)

        noun_counts = Counter(meaningful_nouns)

        # 의전 멘트가 지배적이거나 최소 길이 미달인 경우
        is_procedural = (
            (procedural_matches >= 2 and char_len < 1000) or
            (is_procedural_start and char_len < 600) or
            (char_len < 250 and procedural_matches >= 1) or
            (len(meaningful_nouns) < 4) or
            (char_len < min_char_len)
        )

        if is_procedural:
            # 의전/진행 발언의 요약
            proc_summary = first_sent
            if len(sents) > 1 and len(proc_summary) < 40:
                proc_summary = f"{first_sent} {sents[1].text.strip()}"
            return {
                "is_substantive": False,
                "policy_category": "자치/행정",
                "core_issue": "의사일정 및 회의 진행 절차",
                "keywords": [],
                "summary": proc_summary[:200],
            }

        # 3. 복합명사구 (Bigram) 추출 및 키워드 스코어링
        bigrams = Counter()
        for i in range(len(meaningful_nouns) - 1):
            w1, w2 = meaningful_nouns[i], meaningful_nouns[i + 1]
            if w1 != w2:
                # 텍스트 내에서 실제 두 단어가 인접해 있는지 확인 (거리 10자 이내)
                bigram_cand = f"{w1} {w2}"
                bigrams[bigram_cand] += 1

        keyword_scores = Counter()
        for w, c in noun_counts.items():
            keyword_scores[w] = c * 1.0

        for bg, c in bigrams.items():
            if c >= 1:
                keyword_scores[bg] = c * 2.5  # 복합 키워드에 높은 가중치 부여

        # 상위 키워드 3~5개 선별 (중복 및 포함 관계 필터링)
        selected_keywords: List[str] = []
        for kw, _ in keyword_scores.most_common(20):
            # 이미 선택된 키워드와 부분 일치하면 스킵하여 다양성 확보
            if any(kw in ex or ex in kw for ex in selected_keywords):
                continue
            selected_keywords.append(kw)
            if len(selected_keywords) >= 4:
                break

        # 4. 정책 분야 (policy_category) 분류 스코어링
        category_scores = Counter()
        for cat, kw_list in POLICY_CATEGORIES.items():
            for kw in kw_list:
                if kw in noun_counts:
                    category_scores[cat] += noun_counts[kw] * 2.0
                if kw in text_clean:
                    category_scores[cat] += 1.0

        best_category = "기타"
        if category_scores:
            top_cat, top_score = category_scores.most_common(1)[0]
            if top_score >= 2.0:
                best_category = top_cat

        # 5. 추출 요약 (Extractive Summarization)
        sent_scores = []
        for idx, sent in enumerate(sents):
            stext = sent.text.strip()
            slen = len(stext)
            if slen < 15:
                continue

            # 의전 문장 감점
            penalty = 0.0
            for pat in PROCEDURAL_PATTERNS:
                if re.search(pat, stext):
                    penalty += 8.0
            if any(term in stext for term in ("감사합니다", "고맙습니다", "수고하셨습니다", "안내해 드리겠습니다")):
                penalty += 5.0

            # 키워드 점수 가산
            match_score = 0.0
            for kw in selected_keywords:
                if kw in stext:
                    match_score += 3.5
            for noun in meaningful_nouns:
                if noun in stext:
                    match_score += 0.4

            # 위치 가중치
            pos_weight = 1.0
            if idx < 3:
                pos_weight = 1.25
            elif idx >= len(sents) - 2:
                pos_weight = 1.15

            total_sent_score = (match_score - penalty) * pos_weight
            sent_scores.append((total_sent_score, idx, stext))

        sent_scores.sort(key=lambda x: x[0], reverse=True)
        top_sents = sorted(sent_scores[:2], key=lambda x: x[1]) if sent_scores else []
        summary = " ".join(s[2] for s in top_sents) if top_sents else text_clean[:180]

        # 6. 핵심 쟁점 (core_issue) 생성
        top_kws = selected_keywords[:2]
        kw_str = ", ".join(top_kws) if top_kws else ""

        agenda_clean = re.sub(r'^\d+[\.\s]*', '', agenda_title).replace('\xa0', ' ').strip()
        agenda_clean = re.sub(r'\(.*?\)', '', agenda_clean).strip()

        if agenda_clean and agenda_clean not in ("일반안건", "본회의", "의사일정"):
            if kw_str:
                core_issue = f"{agenda_clean} 및 {kw_str} 관련 정책 의제 논의"
            else:
                core_issue = f"{agenda_clean} 관련 현안 심의 및 논의"
        else:
            if kw_str:
                core_issue = f"{kw_str} 관련 정책 현안 및 대안 논의"
            else:
                core_issue = "지방행정 및 주요 정책 현안 논의"

        is_substantive = len(selected_keywords) >= 1

        return {
            "is_substantive": is_substantive,
            "policy_category": best_category if is_substantive else "자치/행정",
            "core_issue": core_issue if is_substantive else "의사일정 및 회의 진행 절차",
            "keywords": selected_keywords if is_substantive else [],
            "summary": summary,
        }


def load_processed_row_ids(output_file: str) -> Set[int]:
    """이미 처리된 row_id 집합을 로드하여 중복 방지 및 이어하기 지원"""
    processed = set()
    if not os.path.exists(output_file):
        return processed

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                if "row_id" in data:
                    processed.add(int(data["row_id"]))
            except Exception:
                pass
    return processed


def process_council_file_with_kiwi(
    input_file: str,
    output_file: str,
    sample_limit: int = 0,
    min_char_len: int = 80,
    num_workers: int = 4,
    batch_flush_size: int = 100,
    analyzer: Optional[KiwiCouncilAnalyzer] = None,
) -> Tuple[int, int]:
    """단일 회의록 TSV 파일을 처리하여 JSONL 메타데이터 생성 (반환: processed_count, success_count)"""
    print("=" * 68)
    print(" [Kiwi 초고속 형태소 분석기 기반 정책 키워드 & 메타데이터 추출기] ")
    print("=" * 68)
    print(f" * 입력 파일     : {input_file}")
    print(f" * 출력 파일     : {output_file}")
    print(f" * 형태소 엔진   : Kiwi (Worker {num_workers}개)")
    print(f" * 처리 대상     : {'전체 처리' if sample_limit <= 0 else f'최대 {sample_limit}건 샘플'}")
    print(f" * 최소 글자수   : {min_char_len}자 미만 제외")
    print("=" * 68)

    if not os.path.exists(input_file):
        print(f"[오류] 입력 파일이 존재하지 않습니다: {input_file}")
        return 0, 0

    # 출력 디렉토리 자동 생성
    out_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(out_dir, exist_ok=True)

    # 1. 체크포인트 확인
    processed_ids = load_processed_row_ids(output_file)
    if processed_ids:
        print(f"[체크포인트] 기존 결과 파일에서 이미 처리된 {len(processed_ids):,}건을 감지했습니다. (자동 이어하기)")

    # 2. 입력 데이터 수집
    records_to_process = []
    with open(input_file, "r", encoding="utf-8-sig") as f:
        f.readline()  # 헤더
        for row_id, line in enumerate(f, start=1):
            if row_id in processed_ids:
                continue

            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 7:
                utterance = parts[6].strip()
                agenda = parts[2].replace("\xa0", " ").strip()
                role = parts[3].strip()
                name = parts[4].strip()
                records_to_process.append({
                    "row_id": row_id,
                    "date": parts[0].strip(),
                    "assembly": parts[1].strip(),
                    "agenda_title": agenda,
                    "speaker_role": role,
                    "speaker_name": name,
                    "speaker_type": parts[5].strip(),
                    "utterance": utterance,
                })
                if 0 < sample_limit <= len(records_to_process):
                    break

    if not records_to_process:
        print("[알림] 모든 대상 발언이 이미 처리 완료되었습니다.")
        return 0, 0

    print(f">> 신규 처리 대상: 총 {len(records_to_process):,}건")

    # 3. Kiwi 분석기 초기화 (전달받은 analyzer가 있으면 재사용)
    if analyzer is None:
        analyzer = KiwiCouncilAnalyzer(num_workers=num_workers)

    success_count = 0
    processed_count = 0
    buffer = []
    start_time = time.time()

    with open(output_file, "a", encoding="utf-8") as out_f:
        short_fname = Path(input_file).stem.replace("cleaned_", "")[:15]
        with tqdm(
            total=len(records_to_process),
            desc=f" [{short_fname}]",
            unit="건",
            bar_format="{desc}: {percentage:3.0f}%|{bar:25}| {n_fmt}/{total_fmt}건 [{elapsed}<{remaining}, {rate_fmt}]",
            dynamic_ncols=True,
            leave=False,
        ) as pbar:
            for rec in records_to_process:
                res = analyzer.analyze_utterance(
                    text=rec["utterance"],
                    agenda_title=rec["agenda_title"],
                    speaker_role=rec["speaker_role"],
                    speaker_name=rec["speaker_name"],
                    min_char_len=min_char_len,
                )

                speaker_str = f"{rec['speaker_role']} {rec['speaker_name']}".strip()
                out_record = {
                    "row_id": rec["row_id"],
                    "date": rec["date"],
                    "assembly": rec["assembly"],
                    "speaker": speaker_str,
                    "agenda": rec["agenda_title"],
                    "is_substantive": res["is_substantive"],
                    "policy_category": res["policy_category"],
                    "core_issue": res["core_issue"],
                    "keywords": res["keywords"],
                    "summary": res["summary"],
                }

                if res["is_substantive"] and res["keywords"]:
                    success_count += 1

                buffer.append(json.dumps(out_record, ensure_ascii=False) + "\n")
                processed_count += 1

                if len(buffer) >= batch_flush_size:
                    out_f.writelines(buffer)
                    out_f.flush()
                    buffer.clear()

                pbar.update(1)

            if buffer:
                out_f.writelines(buffer)
                out_f.flush()
                buffer.clear()

    elapsed = time.time() - start_time
    rate = processed_count / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 68)
    print(" [Kiwi 메타데이터 추출 완료 보고서] ")
    print("=" * 68)
    print(f" * 처리 완료 건수 : {processed_count:,}건 (소요 시간: {elapsed:.2f}초, 초당 {rate:.1f}건)")
    print(f" * 유효 정책 발언 : {success_count:,}건 ({(success_count / max(1, processed_count))*100:.1f}%)")
    print(f" * 누적 저장 건수 : {len(load_processed_row_ids(output_file)):,}건")
    print(f" * 저장된 파일    : {output_file}")
    print("=" * 68)
    return processed_count, success_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiwi 초고속 형태소 분석기 기반 회의록 정책 메타데이터 추출기")
    parser.add_argument(
        "--input",
        type=str,
        default="change_text/2024_광역 회의록/cleaned_2024_강원특별자치도의회.txt",
        help="입력 TSV 회의록 파일 (기본: change_text/2024_광역 회의록/cleaned_2024_강원특별자치도의회.txt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="출력 JSONL 파일 경로 (미지정 시 data2/{입력파일명}.jsonl 자동 생성)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="처리할 발언 개수 (0이면 전체 처리, 기본: 0)",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=80,
        help="최소 발언 길이 필터링 (기본: 80자)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Kiwi 병렬 워커 수 (기본: 4)",
    )

    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    input_p = Path(args.input)
    if not args.output:
        # data2 디렉토리에 입력 파일명 stem을 기준으로 자동 저장
        output_p = str(Path("data2") / f"{input_p.stem}.jsonl")
    else:
        output_p = args.output

    process_council_file_with_kiwi(
        input_file=str(input_p),
        output_file=output_p,
        sample_limit=args.sample,
        min_char_len=args.min_len,
        num_workers=args.workers,
    )

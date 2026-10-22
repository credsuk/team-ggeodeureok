"""
지방의회 회의록 하이브리드 데이터 로더 (HybridCouncilLoader)
===========================================================
- `data/*.jsonl`: Kiwi 기반 정제 정책 메타데이터 (is_substantive, policy_category, core_issue, keywords, summary)
- `change_text/*.txt`: 완전한 원문 발언(utterance), 공식 회의록명(MINTS_NM), 회의명(MTGNM), 발언자 속성
두 데이터 소스를 row_id 기준으로 1:1 실시간 스트리밍 조인하여, 고품질 의미 보강 텍스트 및 메타데이터 레코드를 생성합니다.
"""

import glob
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class HybridCouncilRecord:
    """JSONL 메타데이터와 TXT 원문이 결합된 단일 발언 레코드"""
    row_id: int
    date: str              # YYYYMMDD 또는 YYYY-MM-DD
    year: int
    assembly: str          # 의회명 (예: 강원특별자치도의회)
    mints_nm: str          # 공식 회의록명 (예: 기타 회의록 제325회(1차))
    mtg_nm: str            # 회의구분 (예: 본회의, 상임위 등)
    agenda_title: str      # 안건명
    speaker_name: str      # 발언자 성명
    speaker_role: str      # 발언자 직함 (위원장, 의원 등)
    speaker_type: str      # 발언자 구분 (의원, 집행부, 기타)
    utterance: str         # 전체 원문 발언
    # data/ JSONL 추출 정책 메타데이터
    is_substantive: bool   # 실질 정책 논의 여부 (False: 단순 의전/개의/폐회)
    policy_category: str   # 정책 분야 (경제/산업, 농림/수산 등)
    core_issue: str        # 핵심 정책 쟁점
    keywords: List[str]    # 복합 정책 키워드 리스트
    summary: str           # 발언 요약문
    file_stem: str         # 원본 파일명 식별자 (예: cleaned_2024_강원특별자치도의회)


def normalize_date(raw_date: Any) -> Tuple[str, int]:
    """날짜 문자열을 (YYYY-MM-DD, 연도 정수)로 정규화"""
    s = re.sub(r"[^0-9]", "", str(raw_date or ""))
    if len(s) >= 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}", int(s[:4])
    elif len(s) >= 4:
        return f"{s[:4]}-01-01", int(s[:4])
    return "9999-12-31", 9999


class HybridCouncilLoader:
    """
    data/와 change_text/의 파일 쌍을 감지하고 row_id 기준 스트리밍 조인 수행
    """

    def __init__(
        self,
        data_dir: Path,
        change_text_dir: Path,
        substantive_only: bool = True,
        min_char_len: int = 40,
    ):
        self.data_dir = Path(data_dir)
        self.change_text_dir = Path(change_text_dir)
        self.substantive_only = substantive_only
        self.min_char_len = min_char_len

    def discover_matched_pairs(self, target_year: Optional[str] = None) -> List[Tuple[Path, Path]]:
        """
        data/ 내의 .jsonl 파일과 change_text/ 내의 동일 파일명 .txt 쌍을 탐색하여 반환
        """
        jsonl_pattern = "*/*.jsonl"
        all_jsonls = sorted(list(self.data_dir.glob(jsonl_pattern)))
        if not all_jsonls:
            # 1단계 하위 폴더가 없는 경우 대비
            all_jsonls = sorted(list(self.data_dir.glob("*.jsonl")))

        pairs = []
        for jsonl_path in all_jsonls:
            if target_year and target_year not in jsonl_path.name:
                continue

            target_txt_name = jsonl_path.stem + ".txt"
            # 1. 동일 하위 폴더 구조에서 우선 탐색
            relative_parent = jsonl_path.parent.name
            candidate_txt = self.change_text_dir / relative_parent / target_txt_name
            if not candidate_txt.exists():
                # 2. change_text 전체에서 재귀 탐색
                found = list(self.change_text_dir.glob(f"*/{target_txt_name}"))
                if found:
                    candidate_txt = found[0]
                else:
                    found_root = self.change_text_dir / target_txt_name
                    if found_root.exists():
                        candidate_txt = found_root

            if candidate_txt.exists():
                pairs.append((jsonl_path, candidate_txt))
            else:
                logger.warning(f"매칭되는 TXT 파일을 찾을 수 없습니다: {jsonl_path.name}")

        return pairs

    def count_target_records(self, pairs: List[Tuple[Path, Path]]) -> int:
        """전체 처리 대상 행 수 집계"""
        total = 0
        for jsonl_path, _ in pairs:
            with open(jsonl_path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    if line.strip():
                        total += 1
        return total

    def stream_hybrid_records(
        self,
        jsonl_path: Path,
        txt_path: Path,
        max_rows: int = 0,
    ) -> Generator[HybridCouncilRecord, None, None]:
        """
        단일 파일 쌍(JSONL + TXT)을 행 단위로 동시 순회하며 결합된 레코드 생성
        """
        file_stem = jsonl_path.stem

        with open(jsonl_path, "r", encoding="utf-8-sig") as f_jsonl, \
             open(txt_path, "r", encoding="utf-8-sig") as f_txt:

            # TXT 헤더 스킵
            txt_header = f_txt.readline().strip().split("\t")

            row_count = 0
            for line_j, line_t in zip(f_jsonl, f_txt):
                line_j = line_j.strip()
                line_t = line_t.strip()
                if not line_j or not line_t:
                    continue

                try:
                    meta_obj = json.loads(line_j)
                except Exception:
                    continue

                txt_cols = line_t.split("\t")
                if len(txt_cols) < 9:
                    # 컬럼 부족 시 패딩
                    txt_cols.extend([""] * (9 - len(txt_cols)))

                # 1. 실질 정책 논의 필터링 (substantive_only)
                is_sub = bool(meta_obj.get("is_substantive", True))
                if self.substantive_only and not is_sub:
                    continue

                # 2. 원문 추출 및 최소 길이 검증
                utterance = txt_cols[8].strip()
                if len(utterance) < self.min_char_len:
                    continue

                # 날짜 정규화
                raw_date = meta_obj.get("date") or txt_cols[0]
                iso_date, year_int = normalize_date(raw_date)

                # 키워드 정규화
                raw_kw = meta_obj.get("keywords", [])
                keywords = [k.strip() for k in raw_kw if isinstance(k, str) and k.strip()] if isinstance(raw_kw, list) else []

                record = HybridCouncilRecord(
                    row_id=int(meta_obj.get("row_id", row_count + 1)),
                    date=iso_date,
                    year=year_int,
                    assembly=meta_obj.get("assembly") or txt_cols[1],
                    mints_nm=txt_cols[3].strip() or "회의록",
                    mtg_nm=txt_cols[2].strip() or "본회의",
                    agenda_title=txt_cols[4].strip() or meta_obj.get("agenda", "일반안건"),
                    speaker_name=txt_cols[6].strip() or meta_obj.get("speaker", "발언자"),
                    speaker_role=txt_cols[5].strip() or "의원",
                    speaker_type=txt_cols[7].strip() or "일반",
                    utterance=utterance,
                    is_substantive=is_sub,
                    policy_category=meta_obj.get("policy_category", "기타"),
                    core_issue=meta_obj.get("core_issue", ""),
                    keywords=keywords,
                    summary=meta_obj.get("summary", ""),
                    file_stem=file_stem,
                )

                yield record
                row_count += 1
                if 0 < max_rows <= row_count:
                    break


def create_augmented_text(record: HybridCouncilRecord, chunk_text: str) -> str:
    """
    메타데이터(분야, 쟁점, 키워드, 요약)를 원문 청크 상단에 주입하여
    검색 시 정책 키워드 의미가 완벽히 포착되도록 보강된 임베딩 텍스트 생성
    """
    prefix_elements = []
    if record.policy_category:
        prefix_elements.append(f"[{record.policy_category}]")
    if record.core_issue:
        prefix_elements.append(f"쟁점: {record.core_issue}")
    if record.keywords:
        prefix_elements.append(f"키워드: {', '.join(record.keywords[:4])}")
    if record.summary:
        prefix_elements.append(f"요약: {record.summary}")

    prefix = " | ".join(prefix_elements)
    if prefix:
        return f"{prefix}\n[원문 발언]\n{chunk_text}"
    return chunk_text

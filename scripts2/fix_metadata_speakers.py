"""
지방의회 메타데이터(data/*.jsonl) 내 비정상 화자명 및 투표결과 일괄 보정 스크립트
=============================================================================
1. 조례안/안건명이 화자명으로 잘못 유입된 행(3,400여 건):
   - 발언 본문(utterance) 시작부의 실제 발언자(예: '○ 부위원장이상원...')를 역추적하여 진짜 성명/직함으로 복원.
   - 복원 불가 시 '미상'으로 안전하게 정규화.
2. 단순 전자투표 찬/반 의원 명단 나열 행(900여 건):
   - 'is_substantive': False 로 변경 (비실질적 의전/절차 발언으로 격하).
   - 'keywords': [] 로 초기화하여 의원 성명 빅그램('찬성 강용범', '강용범 권요찬') 오염 제거.
   - 'core_issue': '의원 전자투표 및 표결 결과' 로 정리.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

# UTF-8 콘솔 출력
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent


def discover_pairs():
    """data/*.jsonl 과 change_text/*.txt 파일 쌍 탐색"""
    jsonls = sorted(list((BASE_DIR / "data").glob("**/*.jsonl")))
    txt_dir = BASE_DIR / "change_text"

    pairs = []
    for jpath in jsonls:
        tname = jpath.stem + ".txt"
        # 동일 하위 폴더 매칭
        candidate = txt_dir / jpath.parent.name / tname
        if not candidate.exists():
            found = list(txt_dir.glob(f"*/{tname}"))
            candidate = found[0] if found else txt_dir / tname
        if candidate.exists():
            pairs.append((jpath, candidate))
    return pairs


def fix_file_metadata(jsonl_path: Path, txt_path: Path, dry_run: bool = False):
    """단일 JSONL 파일의 비정상 화자 및 투표 명단 보정"""
    fixed_speakers = 0
    fixed_votes = 0
    total_lines = 0

    temp_path = jsonl_path.with_suffix(".jsonl.tmp")

    # TXT에서 발언(utterance) 사전 구축 (1-based row_id 매핑)
    utterances = {}
    if txt_path.exists():
        with open(txt_path, "r", encoding="utf-8-sig") as f_txt:
            f_txt.readline()  # txt header skip
            for r_idx, line_t in enumerate(f_txt, start=1):
                txt_cols = line_t.strip().split("\t")
                utterances[r_idx] = txt_cols[8].strip() if len(txt_cols) >= 9 else ""

    try:
        with open(jsonl_path, "r", encoding="utf-8-sig") as f_in, \
             open(temp_path, "w", encoding="utf-8") as f_out:

            for line_idx, line_j in enumerate(f_in, start=1):
                line_j = line_j.strip()
                if not line_j:
                    continue

                total_lines += 1
                try:
                    d = json.loads(line_j)
                except Exception:
                    f_out.write(line_j + "\n")
                    continue

                if not isinstance(d, dict):
                    f_out.write(line_j + "\n")
                    continue

                row_id = d.get("row_id")
                if row_id is not None:
                    try:
                        row_id = int(row_id)
                    except (ValueError, TypeError):
                        row_id = line_idx
                else:
                    row_id = line_idx

                utterance = utterances.get(row_id, "")
                spk = d.get("speaker", "")

                # 1. 투표 명단 판별
                is_vote_tally = bool(
                    re.search(r"투표\s*의원\s*\(\s*\d+인|찬성\s*의원\s*\(\s*\d+인|반대\s*의원\s*\(\s*\d+인", utterance)
                    or re.match(r"^[○\s]*(?:투표\s*의원|찬성\s*의원|반대\s*의원|재석\s*의원|투표\s*결과)", utterance)
                )

                if is_vote_tally:
                    d["is_substantive"] = False
                    d["speaker"] = "의회"
                    d["keywords"] = []
                    d["core_issue"] = "의원 전자투표 및 표결 결과"
                    fixed_votes += 1

                # 2. 비정상 화자명 판별 및 원문 복원
                is_bad_speaker = (
                    any(k in spk for k in ["조례", "예산안", "동의안", "결의안", "규칙", "의 건", "승인안", "계획안", "개정안"])
                    or len(spk) > 15
                    or bool(re.match(r"^\d+\.", spk))
                )

                if is_bad_speaker and not is_vote_tally:
                    m = re.match(r"^[○\s]*([가-힣]*(?:위원장|의장|부의장|부위원장|교육감|도지사|시장|군수|대표의원|의원|국장|과장|실장|관))\s*([가-힣]{2,4})?", utterance)
                    if m:
                        role = m.group(1).strip()
                        name = (m.group(2) or "").strip()
                        d["speaker"] = f"{role} {name}".strip() if name else role
                    else:
                        d["speaker"] = "미상"
                    fixed_speakers += 1

                f_out.write(json.dumps(d, ensure_ascii=False) + "\n")

        if dry_run:
            if temp_path.exists():
                temp_path.unlink()
        else:
            # 원본 파일 교체
            temp_path.replace(jsonl_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

    return total_lines, fixed_speakers, fixed_votes


def main():
    parser = argparse.ArgumentParser(description="data/*.jsonl 메타데이터 화자 및 투표결과 일괄 보정 도구")
    parser.add_argument("--dry-run", action="store_true", help="실제 파일을 수정하지 않고 변경 건수만 확인")
    args = parser.parse_args()

    pairs = discover_pairs()
    print("=" * 70)
    print(f" 🛠️  지방의회 메타데이터 비정상 화자 및 투표결과 보정기 (모드: {'테스트(Dry-run)' if args.dry_run else '실제 수정'})")
    print(f" * 대상 파일 쌍: 총 {len(pairs)}개 파일")
    print("=" * 70)

    start_time = time.time()
    total_spk_fixed = 0
    total_vote_fixed = 0
    total_rows = 0

    for idx, (jpath, tpath) in enumerate(pairs, 1):
        fname = jpath.name
        tot, spk_cnt, vote_cnt = fix_file_metadata(jpath, tpath, dry_run=args.dry_run)
        total_rows += tot
        total_spk_fixed += spk_cnt
        total_vote_fixed += vote_cnt

        if spk_cnt > 0 or vote_cnt > 0:
            print(f"[{idx:02d}/{len(pairs):02d}] {fname[:30]:<30} | 화자보원: {spk_cnt:>5,}건 | 투표정리: {vote_cnt:>4,}건 (총 {tot:>6,}행)")

    elapsed = time.time() - start_time
    print("-" * 70)
    print(" 🏁 [보정 작업 완료 보고서]")
    print(f" * 총 검사 행 수      : {total_rows:,}행")
    print(f" * 복원된 정상 화자 수 : {total_spk_fixed:,}건")
    print(f" * 정리된 투표 명단 수 : {total_vote_fixed:,}건")
    print(f" * 소요 시간           : {elapsed:.2f}초")
    print("=" * 70)


if __name__ == "__main__":
    main()

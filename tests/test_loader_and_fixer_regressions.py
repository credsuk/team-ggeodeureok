"""Offline regressions for hybrid_loader row_id joining and fix_metadata_speakers truncation prevention."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ROOT = Path(__file__).resolve().parents[1]
hybrid_loader_mod = load_module("_hybrid_loader", ROOT / "src" / "hybrid_loader.py")
HybridCouncilLoader = hybrid_loader_mod.HybridCouncilLoader

fixer_mod = load_module("_fixer", ROOT / "scripts2" / "fix_metadata_speakers.py")
fix_file_metadata = fixer_mod.fix_file_metadata


class LoaderAndFixerRegressions(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.base = Path(self.temp_dir.name)

    def test_hybrid_loader_joins_by_row_id_even_if_jsonl_is_out_of_order(self):
        # Header + 3 data rows
        txt_content = (
            "MTG_DE\tRASMBLY_NM\tMTGNM\tMINTS_NM\tagenda_title\tspeaker_role\tspeaker_name\tspeaker_type\tutterance\n"
            "20240101\t의회A\t본회의\t제1회\t안건1\t의원\t김의원\t일반\t첫번째 발언입니다. 이 발언은 40자 이상이어야 유효하게 처리됩니다.\n"
            "20240101\t의회A\t본회의\t제1회\t안건2\t의원\t이의원\t일반\t두번째 발언입니다. 이 발언은 40자 이상이어야 유효하게 처리됩니다.\n"
            "20240101\t의회A\t본회의\t제1회\t안건3\t의원\t박의원\t일반\t세번째 발언입니다. 이 발언은 40자 이상이어야 유효하게 처리됩니다.\n"
        )
        txt_path = self.base / "sample.txt"
        txt_path.write_text(txt_content, encoding="utf-8")

        # JSONL has row 3 first, then row 1 (row 2 is omitted)
        jsonl_records = [
            {
                "row_id": 3,
                "date": "2024-01-01",
                "assembly": "의회A",
                "agenda": "안건3",
                "is_substantive": True,
                "policy_category": "환경/안전",
                "core_issue": "세번째 쟁점",
                "keywords": ["환경", "안전"],
                "summary": "세번째 요약",
            },
            {
                "row_id": 1,
                "date": "2024-01-01",
                "assembly": "의회A",
                "agenda": "안건1",
                "is_substantive": True,
                "policy_category": "경제/산업",
                "core_issue": "첫번째 쟁점",
                "keywords": ["경제", "산업"],
                "summary": "첫번째 요약",
            },
        ]
        jsonl_path = self.base / "sample.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in jsonl_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        loader = HybridCouncilLoader(data_dir=self.base, change_text_dir=self.base, min_char_len=20)
        records = list(loader.stream_hybrid_records(jsonl_path, txt_path))

        # Records should be yielded in row_id order without cross-contamination
        self.assertEqual(len(records), 2)
        rec1, rec3 = records[0], records[1]

        self.assertEqual(rec1.row_id, 1)
        self.assertIn("첫번째 발언입니다", rec1.utterance)
        self.assertEqual(rec1.core_issue, "첫번째 쟁점")
        self.assertEqual(rec1.policy_category, "경제/산업")

        self.assertEqual(rec3.row_id, 3)
        self.assertIn("세번째 발언입니다", rec3.utterance)
        self.assertEqual(rec3.core_issue, "세번째 쟁점")
        self.assertEqual(rec3.policy_category, "환경/안전")

    def test_fix_metadata_speakers_does_not_truncate_when_txt_is_shorter(self):
        # 5 records in JSONL
        jsonl_path = self.base / "sample_fix.jsonl"
        lines = [
            {"row_id": i, "speaker": f"의원{i}", "is_substantive": True, "core_issue": f"이슈{i}"}
            for i in range(1, 6)
        ]
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        # TXT has only header + 2 lines (shorter than JSONL!)
        txt_path = self.base / "sample_fix.txt"
        txt_content = (
            "MTG_DE\tRASMBLY_NM\tMTGNM\tMINTS_NM\tagenda_title\tspeaker_role\tspeaker_name\tspeaker_type\tutterance\n"
            "20240101\t의회\t본회의\t제1회\t안건1\t의원\t의원1\t일반\t발언1\n"
            "20240101\t의회\t본회의\t제1회\t안건2\t의원\t의원2\t일반\t발언2\n"
        )
        txt_path.write_text(txt_content, encoding="utf-8")

        total_lines, fixed_speakers, fixed_votes = fix_file_metadata(jsonl_path, txt_path, dry_run=False)

        # Total lines processed must be 5, not truncated to 2
        self.assertEqual(total_lines, 5)

        # Verify all 5 lines are still present in the replaced file
        with open(jsonl_path, "r", encoding="utf-8") as f:
            remaining_records = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(remaining_records), 5)
        self.assertEqual([r["row_id"] for r in remaining_records], [1, 2, 3, 4, 5])

    def test_fix_metadata_speakers_matches_utterance_by_row_id(self):
        # JSONL has bad speaker on row 2, and records are out of order
        jsonl_path = self.base / "sample_spk.jsonl"
        records = [
            {"row_id": 2, "speaker": "제123회 조례안 심사의 건", "is_substantive": True},
            {"row_id": 1, "speaker": "김의원", "is_substantive": True},
        ]
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        txt_path = self.base / "sample_spk.txt"
        txt_content = (
            "MTG_DE\tRASMBLY_NM\tMTGNM\tMINTS_NM\tagenda_title\tspeaker_role\tspeaker_name\tspeaker_type\tutterance\n"
            "20240101\t의회\t본회의\t제1회\t안건1\t의원\t김의원\t일반\t정상적인 첫번째 발언입니다.\n"
            "20240101\t의회\t본회의\t제1회\t안건2\t의원\t조례안\t일반\t○ 위원장홍길동 의사일정에 따라 회의를 진행하겠습니다.\n"
        )
        txt_path.write_text(txt_content, encoding="utf-8")

        total_lines, fixed_speakers, fixed_votes = fix_file_metadata(jsonl_path, txt_path, dry_run=False)
        self.assertEqual(fixed_speakers, 1)

        with open(jsonl_path, "r", encoding="utf-8") as f:
            updated = [json.loads(line) for line in f if line.strip()]

        row2 = next(r for r in updated if r["row_id"] == 2)
        row1 = next(r for r in updated if r["row_id"] == 1)

        self.assertEqual(row2["speaker"], "위원장 홍길동")
        self.assertEqual(row1["speaker"], "김의원")


if __name__ == "__main__":
    unittest.main()

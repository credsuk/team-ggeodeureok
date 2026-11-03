import csv
import tempfile
import unittest
from pathlib import Path

from src.visualization.map_view import build_map_points
from src.core.models import Evidence
from src.data.source_index import SourceTextReader, build_source_index, lookup_source_metadata
from src.analysis.timeline import build_timeline


class SourceIndexTests(unittest.TestCase):
    def test_source_index_preserves_row_ids_and_title_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "txt"
            source_dir.mkdir()
            source = source_dir / "cleaned_2024_서울특별시의회.txt"
            with source.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["MTG_DE", "RASMBLY_NM", "MTGNM", "MINTS_NM", "agenda_title", "utterance"],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow({"MTG_DE": "2024-05-02", "RASMBLY_NM": "서울특별시의회", "MTGNM": "제1차 본회의", "MINTS_NM": "제324회 임시회", "agenda_title": "급식 예산", "utterance": "secret original body"})
                writer.writerow({"MTG_DE": "2024-05-03", "RASMBLY_NM": "서울특별시의회", "MTGNM": "제2차 본회의", "MINTS_NM": "제324회 임시회", "agenda_title": "급식 예산", "utterance": "quoted\nmultiline body"})
            db = root / "source.sqlite3"
            self.assertEqual(build_source_index(source_dir, db), 2)
            row = lookup_source_metadata(db, ["cleaned_2024_서울특별시의회:1"])[0]
            self.assertEqual(row["meeting_title"], "제1차 본회의")
            self.assertEqual(row["minutes_title"], "제324회 임시회")
            self.assertNotIn("utterance", row)
            reader = SourceTextReader(db, source_dir)
            self.assertEqual(reader.read(["cleaned_2024_서울특별시의회:1"])["cleaned_2024_서울특별시의회:1"], "secret original body")
            self.assertEqual(reader.read(["cleaned_2024_서울특별시의회:2"])["cleaned_2024_서울특별시의회:2"], "quoted\nmultiline body")


class TimelineAndMapTests(unittest.TestCase):
    def test_map_shows_one_earliest_numbered_marker_per_council(self):
        evidence = [
            Evidence(id="s1", text="first", date="20240101", assembly="서울특별시의회"),
            Evidence(id="s2", text="again", date="20240201", assembly="서울특별시의회"),
            Evidence(id="b1", text="later council", date="20240301", assembly="부산광역시의회"),
        ]

        points = build_map_points(evidence)

        self.assertEqual([point["assembly"] for point in points], ["서울특별시의회", "부산광역시의회"])
        self.assertEqual([point["order_label"] for point in points], ["1", "2"])
        self.assertEqual([point["order"] for point in points], [1, 2])
        self.assertEqual([point["utterance_order"] for point in points], [1, 3])
        self.assertEqual([point["source_id"] for point in points], ["s1", "b1"])

    def test_timeline_orders_evidence_and_map_marks_first_council(self):
        later = Evidence(id="c:2", text="later", date="20240502", assembly="서울특별시의회")
        first = Evidence(id="a:1", text="first", date="20240201", assembly="부산광역시의회")
        earliest, flow = build_timeline([later, first])
        self.assertEqual(earliest.id, "a:1")
        self.assertEqual([x.id for x in flow], ["a:1", "c:2"])
        points = build_map_points([earliest, *flow[1:]])
        self.assertEqual(points[0]["role"], "최초 관측")
        self.assertEqual(points[1]["role"], "후속 논의")
        self.assertEqual([point["order_label"] for point in points], ["1", "2"])
        self.assertEqual([point["utterance_order"] for point in points], [1, 2])
        self.assertNotEqual(points[0]["color"], points[1]["color"])
        unknown = Evidence(id="x:0", text="x", date="20230101", assembly="unknown")
        visible = build_map_points([unknown, first])
        self.assertEqual(visible[0]["role"], "최초 관측")
        self.assertEqual(visible[0]["order"], 1)
        self.assertEqual(visible[0]["utterance_order"], 2)

if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""KOSIS 주제별 통계 목록 지도 — config/kosis_topic_catalog.json 관리.

  · 30개 주제(vwCd=MT_ZTITLE) 아래 하위 목록·통계표를 KOSIS 통계목록 API(statisticsList.do)로 내려받아
    주제마다 트리로 저장한다. 표 이름에 시도/지역별 표기가 있는 표만 남긴다(--all-tables면 전부)
  · policy_keywords·indicator_ids는 사람이 관리 — 다시 크롤링해도 유지
  · 런타임 조회용이 아니라, kosis_indicators.json에 올릴 지표 후보를 고르는 용도

  python src/kosis_topic_catalog.py --summary
  python src/kosis_topic_catalog.py --match 고독사               # 주제 고르기 (로컬, 외부 전송 없음)
  python src/kosis_topic_catalog.py --crawl --topics 복지 인구    # 해당 주제만 내려받기
  python src/kosis_topic_catalog.py --crawl                      # 30개 주제 전부 (오래 걸림, 결과는 30일 캐시)
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import external_policy_sources as EXT
import regional_stats as RS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOPIC_CATALOG = PROJECT_ROOT / "config" / "kosis_topic_catalog.json"
VW_CD = "MT_ZTITLE"
SIDO_PATTERN = re.compile(r"시도|시[·ㆍ・/]도|지역별|광역시도|17개\s*시")


def norm(name: str) -> str:
    """'범죄·안전'과 '범죄ㆍ안전'처럼 가운뎃점·공백만 다른 이름을 같게"""
    return re.sub(r"[\s·ㆍ・]", "", name or "")


def load() -> dict[str, Any]:
    return json.loads(TOPIC_CATALOG.read_text(encoding="utf-8"))


def list_children(parent_id: str) -> list[dict[str, Any]]:
    ck = f"kosis_list|{VW_CD}|{parent_id}"
    hit = EXT._cache_get(ck, max_age_days=30)
    if hit:
        return hit["value"]
    params = {"vwCd": VW_CD} | ({"parentListId": parent_id} if parent_id else {})
    rows = RS._api("statisticsList.do", params, f"list:{parent_id or 'root'}")
    rows = rows if isinstance(rows, list) else []
    EXT._cache_put(ck, rows)
    return rows


def crawl(list_id: str, name: str, depth: int, max_depth: int, all_tables: bool) -> dict[str, Any]:
    node: dict[str, Any] = {"list_id": list_id, "name": name, "children": [], "tables": [],
                            "table_count": 0, "sido_table_count": 0}
    if depth > max_depth:
        node["truncated"] = True
        return node
    for row in list_children(list_id):
        if row.get("TBL_ID"):
            sido = bool(SIDO_PATTERN.search(row.get("TBL_NM") or ""))
            node["table_count"] += 1
            node["sido_table_count"] += sido
            if sido or all_tables:
                node["tables"].append({
                    "org_id": row.get("ORG_ID"), "tbl_id": row.get("TBL_ID"), "tbl_nm": row.get("TBL_NM"),
                    "stat_id": row.get("STAT_ID"), "updated": row.get("SEND_DE"), "sido_hint": sido,
                })
        elif row.get("LIST_ID"):
            child = crawl(row["LIST_ID"], row.get("LIST_NM") or "", depth + 1, max_depth, all_tables)
            node["table_count"] += child["table_count"]
            node["sido_table_count"] += child["sido_table_count"]
            if child["table_count"] and (child["sido_table_count"] or all_tables):
                node["children"].append(child)
    return node


def crawl_topics(only: list[str], max_depth: int, all_tables: bool) -> None:
    doc = load()
    by_name = {norm(topic["name"]): topic for topic in doc["topics"]}
    wanted = {norm(name) for name in only}
    for row in list_children(""):
        topic = by_name.get(norm(row.get("LIST_NM")))
        if topic is None:
            print(f"카탈로그에 없는 KOSIS 주제(건너뜀): {row.get('LIST_NM')}")
            continue
        topic["list_id"] = row.get("LIST_ID")
        if wanted and norm(topic["name"]) not in wanted:
            continue
        tree = crawl(topic["list_id"], topic["name"], 1, max_depth, all_tables)
        topic.update(children=tree["children"], tables=tree["tables"],
                     table_count=tree["table_count"], sido_table_count=tree["sido_table_count"])
        print(f"{topic['name']:10} 표 {tree['table_count']:>6}개 · 시도 표기 {tree['sido_table_count']:>5}개")
    doc["source"]["crawled_at"] = datetime.now().isoformat(timespec="seconds")
    TOPIC_CATALOG.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")


def match_topics(text: str, limit: int = 3) -> list[dict[str, Any]]:
    """질문·이슈어와 주제 policy_keywords 비교 (로컬 처리, 외부 전송 없음)"""
    target = norm(text)
    scored = []
    for topic in load()["topics"]:
        hit = [w for w in topic["policy_keywords"] if norm(w) in target or target in norm(w)]
        if hit:
            scored.append({"topic": topic["name"], "matched": hit, "indicator_ids": topic["indicator_ids"]})
    return sorted(scored, key=lambda row: -len(row["matched"]))[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--match")
    ap.add_argument("--crawl", action="store_true")
    ap.add_argument("--topics", nargs="*", default=[], help="--crawl 대상 주제 (기본: 전부)")
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--all-tables", action="store_true", help="시도 표기 없는 표도 저장")
    a = ap.parse_args()
    if a.crawl:
        try:
            crawl_topics(a.topics, a.max_depth, a.all_tables)
        except RS.KosisAuthError as e:
            raise SystemExit(f"{e} — .env의 KOSIS_API_KEY 값을 확인하세요 (kosis.kr/openapi 마이페이지)")
    if a.summary:
        for topic in load()["topics"]:
            counts = (f"표 {topic['table_count']}개 · 시도 표기 {topic['sido_table_count']}개"
                      if "table_count" in topic else "미수집")
            print(f"{topic['name']:10} 지표 {len(topic['indicator_ids'])}개 · {counts}")
    if a.match:
        print(json.dumps(match_topics(a.match), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

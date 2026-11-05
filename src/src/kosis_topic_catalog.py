#!/usr/bin/env python3
"""KOSIS 주제별 통계 목록 지도 — config/kosis_topic_catalog.json 관리.

  · 30개 주제(vwCd=MT_ZTITLE) 아래 하위 목록·통계표를 KOSIS 통계목록 API(statisticsList.do)로 내려받아
    주제마다 트리로 저장한다. 표 이름에 시도/지역별 표기가 있는 표만 남긴다(--all-tables면 전부)
  · e-지방지표(vwCd=MT_GTITLE01) 표 전체를 우선 후보로 내려받아 각 표를 30개 주제 중 하나에 연결한다
    (12개 e-지방지표 주제 → theme_topic_map 기본값, 표 이름이 topic_rules에 걸리면 그 주제, topic_manual이 있으면 최우선)
  · policy_keywords·indicator_ids·theme_topic_map·topic_rules·topic_manual은 사람이 관리 — 다시 크롤링해도 유지
  · 런타임 조회용이 아니라, kosis_indicators.json에 올릴 지표 후보를 고르는 용도
  · 하위 목록 조회 파라미터는 가이드의 parentId가 아니라 parentListId (parentId는 무시되고 최상위만 돌아옴)

  python src/kosis_topic_catalog.py --summary
  python src/kosis_topic_catalog.py --match 고독사               # 주제 고르기 (로컬, 외부 전송 없음)
  python src/kosis_topic_catalog.py --crawl --topics 복지 인구    # 해당 주제만 내려받기
  python src/kosis_topic_catalog.py --crawl                      # 30개 주제 전부 (오래 걸림, 결과는 30일 캐시)
  python src/kosis_topic_catalog.py --crawl-elocal               # e-지방지표 표 전체 내려받고 주제 연결
  python src/kosis_topic_catalog.py --assign                     # 내려받은 e-지방지표 표의 주제만 다시 연결 (외부 호출 없음)
  python src/kosis_topic_catalog.py --elocal-topic 복지            # 그 주제에 연결된 e-지방지표 표
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
ELOCAL_VW_CD = "MT_GTITLE01"
SIDO_PATTERN = re.compile(r"시도|시[·ㆍ・/]도|지역별|광역시도|17개\s*시")


def norm(name: str) -> str:
    """'범죄·안전'과 '범죄ㆍ안전'처럼 가운뎃점·공백만 다른 이름을 같게"""
    return re.sub(r"[\s·ㆍ・]", "", name or "")


def load() -> dict[str, Any]:
    return json.loads(TOPIC_CATALOG.read_text(encoding="utf-8"))


def list_children(parent_id: str, vw_cd: str = VW_CD) -> list[dict[str, Any]]:
    ck = f"kosis_list|{vw_cd}|{parent_id}"
    hit = EXT._cache_get(ck, max_age_days=30)
    if hit:
        return hit["value"]
    params = {"vwCd": vw_cd} | ({"parentListId": parent_id} if parent_id else {})
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
    save(doc)


def elocal_tables(list_id: str, path: list[str], depth: int = 1, max_depth: int = 4) -> list[dict[str, Any]]:
    """e-지방지표 목록 아래 통계표 전부 (목록 경로를 함께 기록)"""
    out: list[dict[str, Any]] = []
    for row in list_children(list_id, ELOCAL_VW_CD):
        if row.get("TBL_ID"):
            out.append({
                "org_id": row.get("ORG_ID"), "tbl_id": row.get("TBL_ID"), "tbl_nm": row.get("TBL_NM"),
                "stat_id": row.get("STAT_ID"), "updated": row.get("SEND_DE"), "e_path": path,
            })
        elif row.get("LIST_ID") and depth < max_depth:
            out += elocal_tables(row["LIST_ID"], path + [row.get("LIST_NM") or ""], depth + 1, max_depth)
    return out


def topic_of(table: dict[str, Any], elocal: dict[str, Any], topic_names: set[str]) -> tuple[str | None, str]:
    """표 하나의 30개 주제 연결: topic_manual → topic_rules(표 이름) → theme_topic_map(e-지방지표 주제)"""
    manual = table.get("topic_manual")
    if manual:
        return (manual, "manual") if manual in topic_names else (None, f"manual_unknown:{manual}")
    name = norm(table.get("tbl_nm"))
    for rule in elocal["topic_rules"]:
        if any(norm(word) in name for word in rule["words"]):
            return rule["topic"], "rule"
    theme = table["e_themes"][0] if table.get("e_themes") else None
    topic = elocal["theme_topic_map"].get(theme)
    return (topic, "theme") if topic in topic_names else (None, "unmapped")


def region_scope(tbl_nm: str) -> str:
    """표 이름 끝 괄호로 지역 범위 추정: sido(시도 포함) · sigungu_only(시군구만) · partial(9도 등 일부) · unknown"""
    scopes = re.findall(r"\(([^()]*)\)", tbl_nm or "")
    scope = scopes[-1] if scopes else ""
    if "시도" in scope:
        return "sido"
    if re.fullmatch(r"시/군(/구)?", scope):
        return "sigungu_only"
    if re.search(r"\d+도", scope):
        return "partial"
    return "unknown"


def assign_topics(doc: dict[str, Any]) -> None:
    elocal = doc["e_local"]
    names = {topic["name"] for topic in doc["topics"]}
    for table in elocal["tables"]:
        table["topic"], table["topic_source"] = topic_of(table, elocal, names)
        table["region_scope"] = region_scope(table.get("tbl_nm"))


def crawl_elocal() -> None:
    doc = load()
    elocal = doc["e_local"]
    manual = {row["tbl_id"]: row["topic_manual"] for row in elocal["tables"] if row.get("topic_manual")}
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for theme in list_children("", ELOCAL_VW_CD):
        name = theme.get("LIST_NM") or ""
        for table in elocal_tables(theme["LIST_ID"], [name]):
            key = (table["org_id"], table["tbl_id"])
            row = merged.setdefault(key, {**{k: v for k, v in table.items() if k != "e_path"},
                                          "e_themes": [], "e_paths": []})
            if name not in row["e_themes"]:
                row["e_themes"].append(name)
            row["e_paths"].append(" > ".join(table["e_path"]))
    tables = sorted(merged.values(), key=lambda row: (row["e_themes"][0], row["tbl_nm"] or ""))
    for row in tables:
        if row["tbl_id"] in manual:
            row["topic_manual"] = manual[row["tbl_id"]]
    elocal["tables"] = tables
    elocal["crawled_at"] = datetime.now().isoformat(timespec="seconds")
    assign_topics(doc)
    save(doc)
    print(f"e-지방지표 표 {len(tables)}개 (중복 경로 병합 후)")


def save(doc: dict[str, Any]) -> None:
    TOPIC_CATALOG.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")


def elocal_for_topic(doc: dict[str, Any], topic: str, sido_only: bool = False) -> list[dict[str, Any]]:
    """그 주제에 연결된 e-지방지표 표. 시도 범위 표를 앞에 둔다"""
    rows = [row for row in doc["e_local"]["tables"] if row.get("topic") == topic
            and (not sido_only or row.get("region_scope") == "sido")]
    return sorted(rows, key=lambda row: row.get("region_scope") != "sido")


def match_topics(text: str, limit: int = 3) -> list[dict[str, Any]]:
    """질문·이슈어와 주제 policy_keywords 비교 (로컬 처리, 외부 전송 없음).
    각 주제의 e-지방지표 표를 우선 후보(elocal_candidates)로 함께 돌려준다"""
    target = norm(text)
    doc = load()
    scored = []
    for topic in doc["topics"]:
        hit = [w for w in topic["policy_keywords"] if norm(w) in target or target in norm(w)]
        if hit:
            candidates = elocal_for_topic(doc, topic["name"], sido_only=True)
            scored.append({"topic": topic["name"], "matched": hit, "indicator_ids": topic["indicator_ids"],
                           "elocal_candidate_count": len(candidates),
                           "elocal_candidates": [f"{row['org_id']}/{row['tbl_id']} {row['tbl_nm']}"
                                                 for row in candidates[:10]]})
    return sorted(scored, key=lambda row: -len(row["matched"]))[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--match")
    ap.add_argument("--crawl", action="store_true")
    ap.add_argument("--topics", nargs="*", default=[], help="--crawl 대상 주제 (기본: 전부)")
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--all-tables", action="store_true", help="시도 표기 없는 표도 저장")
    ap.add_argument("--crawl-elocal", action="store_true", help="e-지방지표 표 전체 내려받고 주제 연결")
    ap.add_argument("--assign", action="store_true", help="e-지방지표 표의 주제 연결만 다시 계산")
    ap.add_argument("--elocal-topic", help="이 주제에 연결된 e-지방지표 표 목록")
    a = ap.parse_args()
    try:
        if a.crawl:
            crawl_topics(a.topics, a.max_depth, a.all_tables)
        if a.crawl_elocal:
            crawl_elocal()
    except RS.KosisAuthError as e:
        raise SystemExit(f"{e} — .env의 KOSIS_API_KEY 값을 확인하세요 (kosis.kr/openapi 마이페이지)")
    if a.assign:
        doc = load()
        assign_topics(doc)
        save(doc)
    if a.summary:
        doc = load()
        for topic in doc["topics"]:
            counts = (f"표 {topic['table_count']}개 · 시도 표기 {topic['sido_table_count']}개"
                      if "table_count" in topic else "미수집")
            print(f"{topic['name']:10} 지표 {len(topic['indicator_ids'])}개 · "
                  f"e-지방지표 {len(elocal_for_topic(doc, topic['name'])):>3}개"
                  f"(시도 {len(elocal_for_topic(doc, topic['name'], sido_only=True)):>3}) · {counts}")
        unmapped = [row["tbl_nm"] for row in doc["e_local"]["tables"] if not row.get("topic")]
        if unmapped:
            print(f"주제 미연결 e-지방지표 {len(unmapped)}개: {', '.join(unmapped[:10])}")
    if a.elocal_topic:
        for row in elocal_for_topic(load(), a.elocal_topic):
            print(f"{row['org_id']:>4}/{row['tbl_id']:<18} {row['tbl_nm']}  "
                  f"[{row['region_scope']} · {row['topic_source']} · {' / '.join(row['e_themes'])}]")
    if a.match:
        print(json.dumps(match_topics(a.match), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any


def _clean(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _discussion_topics(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    topics: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        topic = _clean(str(row.get("topic", "")))[:120]
        raw_ids = row.get("source_ids", [])
        if not topic or not isinstance(raw_ids, list):
            continue
        source_ids = list(dict.fromkeys(str(item).strip() for item in raw_ids if str(item).strip()))
        if source_ids:
            topics.append({"topic": topic, "source_ids": source_ids})
        if len(topics) == 3:
            break
    return topics


@dataclass(frozen=True)
class QueryIntent:
    issue: str
    keywords: tuple[str, ...] = ()
    assemblies: tuple[str, ...] = ()
    date_start: str | None = None
    date_end: str | None = None
    request_type: str = "issue_flow"
    as_of_date: str | None = None

    @property
    def cache_key(self) -> str:
        payload = {
            "issue": _clean(self.issue).casefold(),
            "keywords": sorted({_clean(x).casefold() for x in self.keywords if _clean(x)}),
            "assemblies": sorted({_clean(x).casefold() for x in self.assemblies if _clean(x)}),
            "date_start": self.date_start,
            "date_end": self.date_end,
            "request_type": _clean(self.request_type).casefold(),
            "as_of_date": self.as_of_date,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "QueryIntent":
        return cls(
            issue=_clean(str(value.get("issue", ""))),
            keywords=tuple(str(x) for x in value.get("keywords", []) if str(x).strip()),
            assemblies=tuple(str(x) for x in value.get("assemblies", []) if str(x).strip()),
            date_start=value.get("date_start") or None,
            date_end=value.get("date_end") or None,
            request_type=str(value.get("request_type") or "issue_flow"),
            as_of_date=value.get("as_of_date") or None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Evidence:
    id: str
    text: str
    date: str = ""
    assembly: str = ""
    meeting_title: str = ""
    minutes_title: str = ""
    agenda: str = ""
    source_file: str = ""
    row_id: int | None = None
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RankedHit:
    id: str
    rrf_score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    intent: QueryIntent
    answer: str
    first_seen: Evidence | None
    flow: list[Evidence]
    forecast: str
    confidence: str
    forecast_as_of: str
    core_facts: list[str] = field(default_factory=list)
    likely_discussion_topics: list[dict[str, Any]] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    forecast_citations: list[str] = field(default_factory=list)
    regional_spread: dict[str, Any] = field(default_factory=dict)
    regional_spread_evidence: list[Evidence] = field(default_factory=list)
    clusters: list[dict[str, Any]] = field(default_factory=list)
    cache_hit: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.to_dict(),
            "answer": self.answer,
            "first_seen": asdict(self.first_seen) if self.first_seen else None,
            "flow": [asdict(item) for item in self.flow],
            "forecast": self.forecast,
            "confidence": self.confidence,
            "forecast_as_of": self.forecast_as_of,
            "core_facts": list(self.core_facts),
            "likely_discussion_topics": _discussion_topics(self.likely_discussion_topics),
            "citations": list(self.citations),
            "forecast_citations": list(self.forecast_citations),
            "regional_spread": self.regional_spread,
            "regional_spread_evidence": [asdict(item) for item in self.regional_spread_evidence],
            "clusters": self.clusters,
            "warnings": list(self.warnings),
            "cache_hit": self.cache_hit,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AnalysisResult":
        intent = QueryIntent.from_dict(value["intent"])
        first = value.get("first_seen")
        return cls(
            intent=intent,
            answer=str(value.get("answer", "")),
            first_seen=Evidence(**first) if first else None,
            flow=[Evidence(**item) for item in value.get("flow", [])],
            forecast=str(value.get("forecast", "")),
            confidence=str(value.get("confidence", "낮음")),
            forecast_as_of=str(value.get("forecast_as_of", "")),
            core_facts=[str(item) for item in value.get("core_facts", []) if str(item).strip()],
            likely_discussion_topics=_discussion_topics(value.get("likely_discussion_topics", [])),
            citations=list(value.get("citations", [])),
            forecast_citations=list(value.get("forecast_citations", [])),
            regional_spread=dict(value.get("regional_spread", {})),
            regional_spread_evidence=[Evidence(**item) for item in value.get("regional_spread_evidence", [])],
            clusters=list(value.get("clusters", [])),
            warnings=list(value.get("warnings", [])),
            cache_hit=bool(value.get("cache_hit", False)),
        )

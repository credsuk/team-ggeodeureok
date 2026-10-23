"""Validated dashboard response with evidence provenance and forecast status."""
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MeetingReference(StrictModel):
    sourceId: str
    datasetKey: str = ""
    originalRowId: Optional[int] = None
    date: str
    council: str
    meetingTitle: str = ""
    agendaTitle: str
    excerpt: str
    chunkIds: List[str]
    relevanceScore: Optional[float] = None


class SupplementaryMetrics(StrictModel):
    label: str
    retrievedCouncilDays: int
    councilsDiscussing: int
    yearBreakdown: Dict[str, int]
    topCouncil: Optional[str] = None
    topCouncilSharePercent: Optional[float] = None


class Forecast(StrictModel):
    text: str
    status: Literal["supported", "insufficient_evidence"]
    confidence: Literal["low", "medium", "high"]
    evidenceIds: List[str]


class IssueResult(StrictModel):
    issueKeyword: str
    summary: str
    summaryEvidenceIds: List[str]
    firstObserved: List[MeetingReference]
    subsequentDiscussions: List[MeetingReference]
    supplementaryMetrics: SupplementaryMetrics
    forecast: Forecast
    evidence: List[MeetingReference]


class FinalOutput(StrictModel):
    dataPeriod: str
    dataScope: str
    results: List[IssueResult]



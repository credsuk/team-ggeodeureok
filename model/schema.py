"""Stable result shape for the requested five-column issue analysis table."""
from typing import List
from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MeetingReference(StrictModel):
    date: str
    council: str
    agendaTitle: str
    excerpt: str


class IssueResult(StrictModel):
    issueKeyword: str
    # Multiple councils can share the earliest observed date; no intraday order is assumed.
    appearance: List[MeetingReference]
    regionalFlow: List[MeetingReference]
    forecast: str
    evidence: List[MeetingReference]


class FinalOutput(StrictModel):
    results: List[IssueResult]
    dataScope: str

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IssueListResponse(BaseModel):
    items: list[dict[str, Any]]
    count: int


class RadarSearchRequest(ApiModel):
    query: str = Field(min_length=1, max_length=200)

    @field_validator("query")
    @classmethod
    def trim_query(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("query는 공백일 수 없습니다")
        return value


class ChatMessage(ApiModel):
    role: str
    content: str = Field(min_length=1, max_length=3000)

    @field_validator("role")
    @classmethod
    def valid_role(cls, value: str) -> str:
        if value not in {"user", "assistant"}:
            raise ValueError("role은 user 또는 assistant여야 합니다")
        return value


class ChatRequest(ApiModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=100)
    history: list[ChatMessage] = Field(default_factory=list, max_length=8)
    top_k: int = Field(default=6, ge=1, le=20)

    @field_validator("message")
    @classmethod
    def trim_message(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("message는 공백일 수 없습니다")
        return value


class DocumentSearchResponse(BaseModel):
    query: str
    result_count: int
    results: list[dict[str, Any]]
    search_ms: float | None = None
    query_embedding_ms: float | None = None

"""Pipeline event types for real-time WebSocket streaming."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """All event types emitted during pipeline execution."""

    TRIAGE_THINKING = "triage_thinking"
    TRIAGE_DONE = "triage_done"
    SPECIALISTS_SPAWNED = "specialists_spawned"
    SPECIALIST_THINKING = "specialist_thinking"
    SPECIALIST_DONE = "specialist_done"
    CONSENSUS_DONE = "consensus_done"
    DISCUSSION_SUMMARY = "discussion_summary"
    JUDGE_THINKING = "judge_thinking"
    JUDGE_DONE = "judge_done"
    AGENT_STREAM = "agent_stream"
    SAFETY_THINKING = "safety_thinking"
    SAFETY_DONE = "safety_done"
    ERROR = "error"


class PipelineEvent(BaseModel):
    """A single event emitted during pipeline execution."""

    type: EventType
    agent_name: str = ""
    data: dict[str, Any] = {}
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return self.model_dump_json()

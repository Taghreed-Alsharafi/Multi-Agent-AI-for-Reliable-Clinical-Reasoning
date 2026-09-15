from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field

class EvaluationCase(BaseModel):
    case_id: str
    question: str
    documents: list[str]
    reference_answer: str
    reference_evidence: list[str] = Field(default_factory=list)
    should_abstain: bool = False
    reference_label: str | None = None
    allowed_labels: list[str] = Field(default_factory=list)
    reference_is_safe: bool | None = None
    variant_group: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class StandardAnswer(BaseModel):
    answer: str
    evidence_quotes: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    insufficient_information: bool = False
    predicted_label: str | None = None
    safety_flags: list[str] = Field(default_factory=list)
    is_safe: bool = True

class PredictionRecord(BaseModel):
    run_id: str
    case_id: str
    system: str
    provider: str
    model: str
    answer: str
    evidence_quotes: list[str] = Field(default_factory=list)
    confidence: float = 0
    insufficient_information: bool = False
    predicted_label: str | None = None
    safety_flags: list[str] = Field(default_factory=list)
    is_safe: bool = True
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    trace: dict[str, Any] = Field(default_factory=dict)

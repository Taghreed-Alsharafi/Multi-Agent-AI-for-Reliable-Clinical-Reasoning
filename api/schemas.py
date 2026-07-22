"""Pydantic models for the REST API request / response shapes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from orchestrator.consensus import ConsensusReport


# ── Request ─────────────────────────────────────────────────


class AssessRequest(BaseModel):
    """Payload sent to POST /assess."""

    question: str = Field(
        ...,
        description="The clinical question to answer.",
        examples=["What medications should be adjusted given the latest labs?"],
    )
    documents: list[str] = Field(
        ...,
        description="List of patient document texts (lab reports, clinical notes, etc.).",
        min_length=1,
        examples=[
            [
                "Patient has Type 2 diabetes, HbA1c 9.2%, currently on Metformin 1000mg BID. "
                "Recent labs show eGFR 45."
            ]
        ],
    )


# ── Response ────────────────────────────────────────────────


class AssessResponse(BaseModel):
    """Full pipeline result returned by POST /assess."""

    supervisor: dict[str, Any] = Field(
        ..., description="Supervisor output – identified specialties and lead."
    )
    specialist_opinions: list[dict[str, Any]] = Field(
        ..., description="One opinion per specialist agent."
    )
    consensus: ConsensusReport = Field(
        ..., description="Agreement across the swarm, from specialist confidences."
    )
    judge_report: dict[str, Any] = Field(
        ..., description="Judge Agent consolidation of the specialist reviews."
    )
    safety_report: dict[str, Any] = Field(
        ..., description="Safety Agent verification report."
    )


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str = "ok"
    model: str = Field("", description="Model the specialist agents will use.")
    api_key_configured: bool = Field(
        False, description="Whether an API key is present in the environment."
    )

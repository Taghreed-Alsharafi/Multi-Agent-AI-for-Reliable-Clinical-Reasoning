"""Tests for the FastAPI endpoints."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app
from orchestrator.consensus import compute_consensus


def _mock_completion(content: dict) -> MagicMock:
    choice = MagicMock()
    choice.message.content = json.dumps(content)
    completion = MagicMock()
    completion.choices = [choice]
    return completion


@pytest.mark.asyncio
async def test_health_endpoint():
    """GET /health should return status ok."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_assess_endpoint():
    """POST /assess should return the full pipeline result."""
    from orchestrator.pipeline import PipelineResult

    fake_result = PipelineResult(
        supervisor={
            "specialties": [{"name": "Cardiology", "reason": "Chest pain"}],
            "rationale": "Cardiac evaluation needed.",
        },
        specialist_opinions=[
            {
                "specialty": "Cardiology",
                "findings": "Elevated troponin.",
                "recommendation": "Echo needed.",
                "evidence_quotes": ["troponin 0.8"],
                "confidence": 0.8,
            }
        ],
        consensus=compute_consensus(
            [{"specialty": "Cardiology", "confidence": 0.8}]
        ),
        judge_report={
            "final_summary": "Cardiac event suspected.",
            "consolidated_recommendations": ["Echo needed."]
        },
        safety_report={
            "verified_findings": ["Elevated troponin"],
            "flagged_issues": [],
            "final_summary": "Cardiac event suspected.",
            "overall_confidence": 0.85,
            "is_safe": True,
        },
    )

    with patch("api.main.pipeline") as mock_pipeline:
        mock_pipeline.run = AsyncMock(return_value=fake_result)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/assess",
                json={
                    "question": "Does the patient have a cardiac issue?",
                    "documents": ["Patient reports chest pain. Troponin 0.8 ng/mL."],
                },
            )

    assert resp.status_code == 200
    body = resp.json()
    assert "supervisor" in body
    assert "specialist_opinions" in body
    assert "judge_report" in body
    assert "safety_report" in body
    assert body["safety_report"]["is_safe"] is True
    assert body["consensus"]["agreement_score"] == 0.8


@pytest.mark.asyncio
async def test_assess_requires_documents():
    """POST /assess without documents should return 422."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/assess",
            json={"question": "Something?"},
        )
    assert resp.status_code == 422

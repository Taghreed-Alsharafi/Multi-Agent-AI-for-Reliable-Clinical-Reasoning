"""Tests for the Safety Agent."""

from __future__ import annotations

import pytest

from agents.safety import SafetyAgent


@pytest.mark.asyncio
async def test_safety_returns_verified_report(mock_openai):
    """Safety agent should return a structured verification report."""
    mock_openai["set_response"](
        {
            "verified_findings": [
                "Patient has Type 2 diabetes with HbA1c 9.2%",
                "Current medication is Metformin 1000mg BID",
            ],
            "flagged_issues": [
                {
                    "specialist": "Endocrinology",
                    "claim": "Patient has diabetic retinopathy",
                    "issue": "Not mentioned in provided documents",
                }
            ],
            "final_summary": "Patient has poorly controlled T2DM. Metformin is current therapy. No evidence of retinopathy in records.",
            "overall_confidence": 0.9,
            "is_safe": False,
        }
    )

    agent = SafetyAgent()
    response = await agent.run("Documents + specialist opinions here")

    assert response.agent_name == "safety_agent"
    assert len(response.parsed["verified_findings"]) == 2
    assert len(response.parsed["flagged_issues"]) == 1
    assert response.parsed["flagged_issues"][0]["specialist"] == "Endocrinology"
    assert response.parsed["is_safe"] is False
    assert response.parsed["overall_confidence"] == 0.9


@pytest.mark.asyncio
async def test_safety_all_verified(mock_openai):
    """When no issues are found, is_safe should be True."""
    mock_openai["set_response"](
        {
            "verified_findings": ["Blood pressure is 140/90"],
            "flagged_issues": [],
            "final_summary": "All findings verified.",
            "overall_confidence": 0.95,
            "is_safe": True,
        }
    )

    agent = SafetyAgent()
    response = await agent.run("Some input")

    assert response.parsed["is_safe"] is True
    assert len(response.parsed["flagged_issues"]) == 0

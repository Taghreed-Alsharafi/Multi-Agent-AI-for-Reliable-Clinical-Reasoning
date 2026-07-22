"""Integration test for the full three-stage pipeline."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.pipeline import AgentPipeline


def _mock_completion(content: dict) -> MagicMock:
    """Create a mock ChatCompletion."""
    import json
    content_str = json.dumps(content)
    
    choice = MagicMock()
    choice.message.content = content_str
    completion = MagicMock()
    completion.choices = [choice]
    
    # Mock for stream=True (async iteration)
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = content_str
    
    async def _async_gen():
        yield chunk
        
    completion.__aiter__ = lambda self: _async_gen()
    return completion


# Pre-defined responses for the three pipeline stages
TRIAGE_RESPONSE = {
    "specialties": [
        {"name": "Endocrinology", "reason": "Diabetes management"},
        {"name": "Nephrology", "reason": "Renal function decline"},
    ],
    "rationale": "Patient has diabetes with renal involvement.",
}

ENDO_RESPONSE = {
    "specialty": "Endocrinology",
    "findings": "HbA1c 9.2% indicates poor glycemic control.",
    "recommendation": "Intensify diabetes management.",
    "evidence_quotes": ["HbA1c 9.2%"],
    "confidence": 0.9,
}

NEPHRO_RESPONSE = {
    "specialty": "Nephrology",
    "findings": "eGFR 45 indicates CKD stage 3b.",
    "recommendation": "Adjust medications for renal dosing.",
    "evidence_quotes": ["eGFR 45"],
    "confidence": 0.85,
}

SAFETY_RESPONSE = {
    "verified_findings": [
        "HbA1c 9.2% indicates poor glycemic control",
        "eGFR 45 indicates CKD stage 3b",
    ],
    "flagged_issues": [],
    "final_summary": "Patient has poorly controlled T2DM with stage 3b CKD. Medication adjustment needed.",
    "overall_confidence": 0.92,
    "is_safe": True,
}

JUDGE_RESPONSE = {
    "final_summary": "Patient has poorly controlled T2DM with stage 3b CKD.",
    "consolidated_recommendations": [
        "Intensify diabetes management.",
        "Adjust medications for renal dosing."
    ]
}


@pytest.mark.asyncio
async def test_full_pipeline():
    """End-to-end test: triage → specialists → safety."""

    # We need the mock to return different responses for each call:
    # Call 1: Supervisor, Call 2: Endocrinology, Call 3: Nephrology, Call 4: Judge, Call 5: Safety
    responses = [
        _mock_completion(TRIAGE_RESPONSE),
        _mock_completion(ENDO_RESPONSE),
        _mock_completion(NEPHRO_RESPONSE),
        _mock_completion(JUDGE_RESPONSE),
        _mock_completion(SAFETY_RESPONSE),
    ]

    from agents.base import get_client

    get_client.cache_clear()

    with patch("agents.base.AsyncOpenAI") as MockClient, \
         patch("agents.supervisor.get_settings") as MockTriageSettings, \
         patch("agents.safety.get_settings") as MockSafetySettings:
        
        # Ensure model initialization doesn't trip up
        MockTriageSettings.return_value = MagicMock(TRIAGE_MODEL="gpt-4o-mini", TEMPERATURE=0.2)
        MockSafetySettings.return_value = MagicMock(SAFETY_MODEL="gpt-4o-mini", TEMPERATURE=0.2)
        
        instance = MockClient.return_value
        create_mock = AsyncMock(side_effect=responses)
        instance.chat.completions.create = create_mock

        pipeline = AgentPipeline()
        result = await pipeline.run(
            question="What medications should be adjusted?",
            documents=[
                "Patient has Type 2 diabetes, HbA1c 9.2%, on Metformin 1000mg BID. eGFR 45."
            ],
        )

    # Verify triage
    assert len(result.supervisor.get("specialties", [])) == 2

    # Verify specialist opinions
    assert len(result.specialist_opinions) == 2
    specialties_returned = {op["specialty"] for op in result.specialist_opinions}
    assert specialties_returned == {"Endocrinology", "Nephrology"}

    # Verify safety report
    assert result.safety_report["is_safe"] is True
    assert len(result.safety_report["flagged_issues"]) == 0
    assert result.safety_report["overall_confidence"] > 0.9

    # Verify agreement scoring – 0.9 and 0.85 are close, so agreement is strong
    assert result.consensus.participating == 2
    assert result.consensus.mean_confidence == 0.875
    assert result.consensus.level == "strong"

    get_client.cache_clear()

"""Tests for the Specialist Agent and swarm factory."""

from __future__ import annotations

import pytest

from agents.specialist import SpecialistAgent, create_specialist_swarm


@pytest.mark.asyncio
async def test_specialist_returns_opinion(mock_openai):
    """Specialist agent should return a structured opinion."""
    mock_openai["set_response"](
        {
            "specialty": "Cardiology",
            "findings": "Elevated troponin suggests cardiac involvement.",
            "recommendation": "Order echocardiogram.",
            "evidence_quotes": ["troponin 0.8 ng/mL"],
            "confidence": 0.85,
        }
    )

    agent = SpecialistAgent("Cardiology")
    response = await agent.run("Patient has chest pain, troponin 0.8 ng/mL")

    assert response.agent_name == "specialist_cardiology"
    assert response.parsed["specialty"] == "Cardiology"
    assert response.parsed["confidence"] == 0.85
    assert len(response.parsed["evidence_quotes"]) >= 1


def test_specialist_system_prompt_contains_specialty():
    """The system prompt should be parameterised with the specialty name."""
    agent = SpecialistAgent("Neurology")
    assert "Neurology" in agent.system_prompt
    assert agent.name == "specialist_neurology"


def test_create_specialist_swarm():
    """Factory should create one agent per specialty."""
    agents = create_specialist_swarm(["Cardiology", "Endocrinology", "Nephrology"])
    assert len(agents) == 3
    names = {a.specialty for a in agents}
    assert names == {"Cardiology", "Endocrinology", "Nephrology"}


def test_swarm_respects_max_limit(monkeypatch):
    """Swarm should cap at MAX_SPECIALISTS."""
    from config.settings import Settings, get_settings

    # Clear cached settings
    get_settings.cache_clear()
    monkeypatch.setenv("MAX_SPECIALISTS", "2")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    agents = create_specialist_swarm(["A", "B", "C", "D"])
    assert len(agents) == 2

    # Clean up
    get_settings.cache_clear()

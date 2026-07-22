"""Specialist Agent – Stage 2 of the pipeline.

Dynamically created agents, one per specialty, that analyse the patient
documents from their area of expertise.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from config.settings import get_settings

from .base import BaseAgent


class SpecialistOpinion(BaseModel):
    """Structured output of a single specialist agent."""

    specialty: str
    findings: str
    recommendation: str
    evidence_quotes: list[str]
    confidence: float  # 0-1


SPECIALIST_SYSTEM_PROMPT_TEMPLATE = """\
You are a board-certified **{specialty}** specialist AI.

You will receive a clinical question and excerpts from the patient's
medical documents.  Analyze them strictly from the perspective of
{specialty} and provide your professional opinion.

Return your answer as a JSON object with this exact schema:
{{
  "specialty": "{specialty}",
  "findings": "<your clinical findings relevant to {specialty}, addressing the clinical question>",
  "recommendation": "<your recommendation>",
  "evidence_quotes": ["<exact quote from the documents supporting each finding>"],
  "confidence": <float 0-1 indicating your confidence>
}}

Rules:
- Base every finding on information **explicitly present** in the provided
  documents.  Do NOT infer facts that are not stated.
- Include at least one direct quote from the documents in evidence_quotes.
- If the documents contain no information relevant to {specialty}, set
  findings to "No relevant information found" and confidence to 0.
- Always respond with valid JSON and nothing else.
"""


class SpecialistAgent(BaseAgent):
    """A specialist agent parameterised by medical specialty."""

    name: str = "specialist_agent"
    system_prompt: str = ""

    def __init__(self, specialty: str) -> None:
        self.specialty = specialty
        self.name = f"specialist_{specialty.lower().replace(' ', '_')}"

        # Load professional skill set
        super().__init__()  # Call super init first to set project_root
        skill_instructions = self.load_skill("medical-specialist")

        self.system_prompt = (
            f"You are a board-certified **{specialty}** specialist AI.\n\n"
            f"{skill_instructions}\n"
            f"Rules for JSON Output:\n"
            f"Return your answer as a JSON object with this exact schema:\n"
            f"{{\n"
            f'  "specialty": "{specialty}",\n'
            f'  "findings": "<your clinical findings relevant to {specialty}, addressing the clinical question>",\n'
            f'  "recommendation": "<your recommendation>",\n'
            f'  "evidence_quotes": ["<exact quote from the documents supporting each finding>"],\n'
            f'  "confidence": <float 0-1 indicating your confidence>\n'
            f"}}\n"
        )

    def parse_response(self, raw: str) -> dict[str, Any]:
        data = self.safe_json_loads(raw)
        try:
            opinion = SpecialistOpinion(**data)
            return opinion.model_dump()
        except Exception:
            return data


def create_specialist_swarm(
    specialties: list[str],
) -> list[SpecialistAgent]:
    """Factory: create one SpecialistAgent per specialty.

    Respects the ``MAX_SPECIALISTS`` setting to avoid runaway costs.
    """
    settings = get_settings()
    capped = specialties[: settings.MAX_SPECIALISTS]
    return [SpecialistAgent(s) for s in capped]

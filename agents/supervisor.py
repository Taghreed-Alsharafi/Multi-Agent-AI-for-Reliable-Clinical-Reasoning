"""Supervisor Agent – Enhanced Stage 1 of the pipeline.

Uses the 'medical-supervisor' skill to analyze the incoming question/document,
identify required clinical specialists, and designate a Lead Specialist.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from config.settings import get_settings
from .base import BaseAgent


class Specialty(BaseModel):
    """A single medical specialty identified by the Supervisor Agent."""

    name: str
    reason: str


class SupervisorResult(BaseModel):
    """Structured output of the Supervisor Agent."""

    specialties: list[Specialty]
    lead_specialist: str
    rationale: str


SUPERVISOR_JSON_FORMAT = """
Return your answer as a JSON object with this exact schema:
{
  "specialties": [
    {"name": "<specialty>", "reason": "<why this specialty is needed>"}
  ],
  "lead_specialist": "<name of the specialist designated as lead>",
  "rationale": "<brief overall rationale for selection>"
}
"""


class SupervisorAgent(BaseAgent):
    """Professional clinical supervisor that determines the specialty team and lead."""

    name: str = "supervisor_agent"

    def __init__(self) -> None:
        super().__init__()
        # Use specific model for triage
        settings = get_settings()
        self._model = settings.TRIAGE_MODEL
        # Load professional skill set
        skill_instructions = self.load_skill("medical-supervisor")
        self.system_prompt = skill_instructions + SUPERVISOR_JSON_FORMAT

    def parse_response(self, raw: str) -> dict[str, Any]:
        data = self.safe_json_loads(raw)
        # Validate through the pydantic model
        try:
            result = SupervisorResult(**data)
            return result.model_dump()
        except Exception:
            # Fallback to raw data if validation fails, ensure lead_specialist is present
            if "lead_specialist" not in data and data.get("specialties"):
                data["lead_specialist"] = data["specialties"][0]["name"]
            return data

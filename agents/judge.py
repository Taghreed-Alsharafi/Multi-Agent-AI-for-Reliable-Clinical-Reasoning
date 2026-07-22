"""Judge Agent – Stage 3 of the pipeline.

Synthesizes multiple specialist reviews, resolves conflicts, and produces
a coherent final medical report draft using the 'medical-judge' skill.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .base import BaseAgent


class JudgeResult(BaseModel):
    """Structured output of the Judge Agent."""

    final_summary: str
    key_findings_by_specialty: dict[str, list[str]]
    consolidated_recommendations: list[str]


JUDGE_JSON_FORMAT = """
Return your answer as a JSON object with this exact schema:
{
  "final_summary": "<brief overview of the case and explicit answers to ALL parts of the original question>",
  "key_findings_by_specialty": {
    "<Specialty Name>": ["<Key Finding 1>", "<Key Finding 2>"]
  },
  "consolidated_recommendations": [
    "<Recommendation 1>",
    "<Recommendation 2>"
  ]
}
"""


class JudgeAgent(BaseAgent):
    """Professional medical judge that synthesizes multi-specialty reviews."""

    name: str = "judge_agent"

    def __init__(self) -> None:
        super().__init__()
        # Load professional skill set
        skill_instructions = self.load_skill("medical-judge")
        self.system_prompt = skill_instructions + JUDGE_JSON_FORMAT

    def parse_response(self, raw: str) -> dict[str, Any]:
        data = self.safe_json_loads(raw)
        # Validate through the pydantic model
        try:
            result = JudgeResult(**data)
            return result.model_dump()
        except Exception:
            return data

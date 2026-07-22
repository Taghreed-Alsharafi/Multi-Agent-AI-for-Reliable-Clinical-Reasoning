"""Safety Agent – Stage 3 of the pipeline.

Reviews all specialist opinions and verifies every claim is grounded in
the original patient documents.  Flags hallucinations or unsupported
statements and produces a final verified summary.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from config.settings import get_settings
from .base import BaseAgent


class FlaggedIssue(BaseModel):
    """A single claim that could not be verified against the documents."""

    specialist: str
    claim: str
    issue: str  # e.g. "not found in documents", "contradicts document"


class SafetyReport(BaseModel):
    """Structured output of the Safety Agent."""

    verified_findings: list[str]
    flagged_issues: list[FlaggedIssue]
    final_summary: str
    overall_confidence: float  # 0-1
    is_safe: bool


SAFETY_JSON_FORMAT = """
Return your answer as a JSON object with this exact schema:
{
  "verified_findings": ["<finding that IS supported by documents>", ...],
  "flagged_issues": [
    {
      "specialist": "<specialty name>",
      "claim": "<the unsupported claim>",
      "issue": "<why it is unsupported>"
    }
  ],
  "final_summary": "<comprehensive verified summary for the clinician>",
  "overall_confidence": <float 0-1>,
  "is_safe": <true if no critical unsupported claims, false otherwise>
}
"""


class SafetyAgent(BaseAgent):
    """Verifies specialist opinions against the original documents."""

    name: str = "safety_agent"

    def __init__(self) -> None:
        super().__init__()
        # Use specific model for safety verification
        settings = get_settings()
        self._model = settings.SAFETY_MODEL
        # Load professional skill set
        skill_instructions = self.load_skill("medical-safety")
        self.system_prompt = skill_instructions + SAFETY_JSON_FORMAT

    def parse_response(self, raw: str) -> dict[str, Any]:
        data = self.safe_json_loads(raw)
        try:
            report = SafetyReport(**data)
            return report.model_dump()
        except Exception:
            return data

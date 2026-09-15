#!/usr/bin/env python
"""Exercise three V2 orchestration paths without making API calls."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.mcq_hybrid_v2 import (
    CallResult,
    MultiAgentConfig,
    ensure_fixed_split_v2,
    load_evaluation_dataset,
    run_multi_agent_v2,
)


class MockCaller:
    def __init__(self, specialist_letters: list[str], difficulty: str):
        self.specialist_letters = list(specialist_letters)
        self.difficulty = difficulty
        self.specialist_index = 0
        self.roles: list[str] = []

    async def call_json(self, **kwargs):
        role = kwargs["role"]
        self.roles.append(role)
        if role == "router":
            parsed = {
                "difficulty": self.difficulty,
                "difficulty_reason": "Mock route for deterministic validation.",
                "lead_specialty": "Pediatric Dermatology",
                "panel_size": 2,
                "specialists": [
                    {
                        "specialty": "Pediatric Dermatology",
                        "role": "lead",
                        "focus": "morphology",
                        "why_needed": "skin expertise",
                    },
                    {
                        "specialty": "Pediatric Infectious Disease",
                        "role": "supporting",
                        "focus": "transmission",
                        "why_needed": "infection expertise",
                    },
                ],
                "expected_disagreement": "low",
            }
        elif role == "specialist":
            letter = self.specialist_letters[self.specialist_index]
            self.specialist_index += 1
            parsed = {
                "answer_letter": letter,
                "confidence": 0.8,
                "decisive_evidence": ["mock evidence"],
                "domain_reasoning": "mock domain reasoning",
                "strongest_alternative": "B" if letter == "A" else "A",
                "why_alternative_is_weaker": "mock distinction",
                "remaining_uncertainty": "none",
            }
        elif role == "critic":
            parsed = {
                "conflict_summary": "mock conflict",
                "decisive_clues": ["mock evidence"],
                "issues": [],
                "resolution_guidance": "resolve from the stem",
            }
        else:
            parsed = {
                "answer_letter": "A",
                "confidence": 0.9,
                "decisive_evidence": ["mock evidence"],
                "explanation": "mock final decision",
                "disagreement_resolution": "mock resolution",
            }
        return CallResult(
            role=role,
            model=kwargs["model"],
            parsed=parsed,
            raw_text=json.dumps(parsed),
            latency_seconds=0.001,
            input_tokens=10,
            output_tokens=5,
            attempts=1,
        )


async def main() -> None:
    qc = load_evaluation_dataset(
        ROOT / "data" / "medical_mcq_300_EVAL_READY.csv",
        qc_master_path=ROOT / "data" / "medical_mcq_300_QC_MASTER.csv",
    )
    splits, _, _ = ensure_fixed_split_v2(
        qc, ROOT, dataset_name="medical_mcq_300_EVAL_READY", seed=2026
    )
    case = splits["development"].head(1)
    output = ROOT / "results" / "mcq_v2" / "mock_validation"
    scenarios = [
        ("mock_consensus", ["A", "A"], "moderate", False),
        ("mock_disagreement", ["A", "B"], "moderate", True),
        ("mock_very_complex", ["A", "A"], "very_complex", True),
    ]
    for run_name, letters, difficulty, critic_expected in scenarios:
        caller = MockCaller(letters, difficulty)
        result = await run_multi_agent_v2(
            case,
            MultiAgentConfig(run_name=run_name),
            "development",
            output,
            resume=False,
            caller=caller,
        )
        assert caller.roles.count("judge") == 1
        assert caller.roles.count("critic") == int(critic_expected)
        assert result.loc[0, "gpt5_call_count"] == 1
        print(run_name, "roles=", caller.roles, "gpt5_call_count=1")


if __name__ == "__main__":
    asyncio.run(main())

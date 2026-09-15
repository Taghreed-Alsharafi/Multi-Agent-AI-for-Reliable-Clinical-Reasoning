"""Final GPT-4.1 single-agent versus agreement-method experiment.

This module is intentionally isolated from the production MCQ runners. It uses
the protected V2 split and existing clinical prompts, but stores one shared set
of full specialist distributions and rankings for all agreement methods.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .mcq_hybrid import parse_options
from .mcq_hybrid_v2 import (
    OpenAIJSONClient,
    _configure_platform_trust_store,
    load_project_api_key,
)
from .mcq_v2_prompts import (
    JUDGE_PROMPT,
    ROUTER_PROMPT,
    SINGLE_PROMPT,
    SPECIALIST_PROMPT,
    specialty_instruction,
)


MODEL = "gpt-4.1"
TEMPERATURE = 0.0
ROUTER_MODEL = "gpt-5-mini"
COMPLEX_REASONING_MODEL = "gpt-5-mini"
FINAL_SAFETY_MODEL = "gpt-5-mini"
FINAL_SAFETY_REASONING_EFFORT = "medium"
FINAL_SAFETY_MAX_OUTPUT_TOKENS = 2000
FINAL_SAFETY_RETRY_MAX_OUTPUT_TOKENS = 6000
FINAL_SAFETY_MAX_API_CALLS = 3
SEED = 2026
N_SPECIALISTS = 3
PROMPT_VERSION = "dynamic_mixed_models_care_v2_v6"
GUIDELINE_RULES_VERSION = "cdc_pregnancy_syphilis_v1"
CDC_SYPHILIS_PREGNANCY_URL = (
    "https://www.cdc.gov/std/treatment-guidelines/syphilis-pregnancy.htm"
)
VALID_METHODS = (
    "jsd",
    "kendall_w",
    "krippendorff_alpha",
    "care_consensus",
)
METHOD_LABELS = {
    "jsd": "JSD",
    "kendall_w": "Kendall W",
    "krippendorff_alpha": "Krippendorff Alpha",
    "care_consensus": "CARE v2 Calibrated Clinical Guard",
}


def _is_transient_safety_error(call: Any) -> bool:
    if getattr(call, "error_type", "") != "api_error":
        return False
    error = str(getattr(call, "error", "")).casefold()
    transient_markers = (
        "apitimeouterror",
        "connecttimeout",
        "timeout",
        "sslwantreaderror",
        "apiconnectionerror",
        "connection reset",
        "connection aborted",
        "temporarily unavailable",
    )
    return any(marker in error for marker in transient_markers)


def specialist_calibration_examples(letters: Sequence[str]) -> str:
    """Create option-safe examples that teach calibration without medical content."""
    options = list(letters)
    if len(options) < 2:
        return ""

    def distribution(top: str, top_probability: float, runner_up_probability: float) -> dict[str, float]:
        runner_up = next(letter for letter in options if letter != top)
        remaining = [letter for letter in options if letter not in {top, runner_up}]
        residual = max(0.0, 1.0 - top_probability - runner_up_probability)
        values = {letter: 0.0 for letter in options}
        values[top] = top_probability
        values[runner_up] = runner_up_probability
        for letter in remaining:
            values[letter] = residual / len(remaining) if remaining else 0.0
        if not remaining:
            values[runner_up] += residual
        rounded = {letter: round(value, 6) for letter, value in values.items()}
        adjustment_letter = remaining[-1] if remaining else runner_up
        rounded[adjustment_letter] = round(
            rounded[adjustment_letter] + (1.0 - sum(rounded.values())), 6
        )
        return rounded

    strong_top = options[1]
    close_top = options[-1]
    strong = distribution(strong_top, 0.82, 0.10)
    close = distribution(close_top, 0.43, 0.37)
    examples = []
    for label, probabilities in (("strong evidence", strong), ("close differential", close)):
        ranking = sorted(options, key=lambda letter: (-probabilities[letter], options.index(letter)))
        examples.append(
            f"{label}: "
            + json.dumps(
                {
                    "answer": ranking[0],
                    "probabilities": probabilities,
                    "ranking": ranking,
                    "abstained": False,
                },
                ensure_ascii=True,
            )
        )
    return (
        "\n\nCALIBRATION EXAMPLES (format and uncertainty only; they are not medical clues):\n"
        + "\n".join(examples)
        + "\nUse strong concentration only for decisive evidence. For a close differential, "
        "keep the leading options close. Always use exactly the supplied option letters."
    )


def find_project_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "evaluation" / "mcq_hybrid_v2.py").exists() and (
            candidate / "data"
        ).exists():
            return candidate
    raise FileNotFoundError("Could not locate the clinical MCQ project root")


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def option_letters(row: Mapping[str, Any]) -> list[str]:
    stored = [letter for letter in "ABCDEFGH" if _clean(row.get(f"option_{letter}"))]
    if stored:
        return stored
    source = _clean(row.get("instruction")) or _clean(row.get("question"))
    return list(parse_options(source))


def question_text(row: Mapping[str, Any]) -> str:
    instruction = _clean(row.get("instruction"))
    if instruction:
        return instruction
    question = _clean(row.get("question"))
    options = "\n".join(
        f"{letter}. {_clean(row.get(f'option_{letter}'))}"
        for letter in option_letters(row)
    )
    return f"{question}\n{options}".strip()


def clinical_guideline_override(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Apply narrow, auditable high-stakes rules using stem and option text only."""
    stem = question_text(row).lower()
    pregnancy = any(term in stem for term in ("pregnant", "pregnancy", "gestation"))
    syphilis_evidence = any(
        term in stem for term in ("syphilis", "treponema", "darkfield", "dark-field")
    )
    if not (pregnancy and syphilis_evidence):
        return None
    embedded_options = parse_options(
        _clean(row.get("instruction")) or _clean(row.get("question"))
    )
    for letter in option_letters(row):
        option = _clean(row.get(f"option_{letter}")) or _clean(embedded_options.get(letter))
        if "penicillin" in option.lower():
            return {
                "answer": letter,
                "rule_id": "CDC_PREGNANCY_SYPHILIS_PENICILLIN",
                "reason": (
                    "CDC: penicillin is the only proven treatment for syphilis during "
                    "pregnancy; patients with penicillin allergy require desensitization."
                ),
                "source": CDC_SYPHILIS_PREGNANCY_URL,
                "rules_version": GUIDELINE_RULES_VERSION,
            }
    return None


def load_fixed_cases(project_root: str | Path | None = None) -> pd.DataFrame:
    root = find_project_root(project_root)
    development_path = (
        root
        / "data"
        / "fixed_splits_v2"
        / "medical_mcq_300_EVAL_READY__050957fb7d22"
        / "development_20.csv"
    )
    test_root = root / "data" / "fixed_splits" / "medical_mcq"
    paths = [
        ("development", development_path),
        ("test", test_root / "development_20.csv"),
        ("test", test_root / "test_80.csv"),
    ]
    frames: list[pd.DataFrame] = []
    for split, path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Protected split file is missing: {path}")
        frame = pd.read_csv(path, keep_default_na=False)
        frame["split"] = split
        frames.append(frame)
    cases = pd.concat(frames, ignore_index=True)
    if len(cases) != 357:
        raise RuntimeError(f"Expected 57 development and 300 test cases, found {len(cases)}")
    if (cases["split"] == "development").sum() != 57 or (cases["split"] == "test").sum() != 300:
        raise RuntimeError("Protected experiment must contain 57 development and 300 test cases")
    if not cases["fixed_case_id"].is_unique:
        raise RuntimeError("Protected split contains duplicate fixed_case_id values")
    if set(cases["split"]) != {"development", "test"}:
        raise RuntimeError("Both development and test cases are required")
    if set(cases["reference_letter"]) - set("ABCDEFGH"):
        raise RuntimeError("Protected split contains an invalid gold label")
    invalid_gold = [
        row.fixed_case_id
        for row in cases.itertuples(index=False)
        if row.reference_letter not in option_letters(row._asdict())
    ]
    if invalid_gold:
        raise RuntimeError(f"Gold label is not a supplied option for {len(invalid_gold)} cases")
    return cases


def dataset_identity(cases: pd.DataFrame) -> str:
    payload = "\n".join(
        f"{row.fixed_case_id}|{row.split}|{row.reference_letter}"
        for row in cases.sort_values("fixed_case_id").itertuples(index=False)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _object_schema(properties: dict[str, Any], required: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def single_schema(letters: Sequence[str]) -> dict[str, Any]:
    return _object_schema({"answer": {"type": "string", "enum": list(letters)}}, ["answer"])


def router_schema() -> dict[str, Any]:
    specialist = _object_schema(
        {"specialty": {"type": "string"}, "focus": {"type": "string"}},
        ["specialty", "focus"],
    )
    return _object_schema(
        {
            "difficulty": {
                "type": "string",
                "enum": ["simple", "moderate", "complex", "very_complex"],
            },
            "clinical_importance": {
                "type": "string",
                "enum": ["routine", "high", "critical"],
            },
            "lead_specialty": {"type": "string"},
            "specialists": {
                "type": "array",
                "items": specialist,
                "minItems": N_SPECIALISTS,
                "maxItems": N_SPECIALISTS,
            }
        },
        ["difficulty", "clinical_importance", "lead_specialty", "specialists"],
    )


def dynamic_model_plan(route: Mapping[str, Any]) -> dict[str, Any]:
    """Assign stronger reasoning models from router labels without using gold."""
    difficulty = _clean(route.get("difficulty")).lower() or "moderate"
    importance = _clean(route.get("clinical_importance")).lower() or "routine"
    if difficulty == "very_complex" or importance == "critical":
        strong_specialists = 2
        reasoning_effort = "high"
    elif difficulty == "complex" or importance == "high":
        strong_specialists = 1
        reasoning_effort = "medium"
    else:
        strong_specialists = 0
        reasoning_effort = None
    specialist_models = [
        COMPLEX_REASONING_MODEL if index < strong_specialists else MODEL
        for index in range(N_SPECIALISTS)
    ]
    return {
        "router_model": ROUTER_MODEL,
        "specialist_models": specialist_models,
        "specialist_reasoning_effort": reasoning_effort,
        "judge_model": COMPLEX_REASONING_MODEL if strong_specialists else MODEL,
        "judge_reasoning_effort": reasoning_effort,
        "final_safety_model": FINAL_SAFETY_MODEL,
    }


def specialist_schema(letters: Sequence[str]) -> dict[str, Any]:
    probabilities = _object_schema(
        {
            letter: {"type": "number", "minimum": 0.0, "maximum": 1.0}
            for letter in letters
        },
        letters,
    )
    return _object_schema(
        {
            "answer": {"type": "string", "enum": list(letters)},
            "probabilities": probabilities,
            "ranking": {
                "type": "array",
                "items": {"type": "string", "enum": list(letters)},
                "minItems": len(letters),
                "maxItems": len(letters),
            },
        },
        ["answer", "probabilities", "ranking"],
    )


def safety_schema(letters: Sequence[str], include_recommendation: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "safety_violation": {"type": "boolean"},
        "reason": {"type": "string"},
    }
    required = ["safety_violation", "reason"]
    if include_recommendation:
        properties["recommended_answer"] = {"type": "string", "enum": list(letters)}
        required.append("recommended_answer")
    return _object_schema(properties, required)


def judge_schema(letters: Sequence[str]) -> dict[str, Any]:
    return _object_schema(
        {
            "answer": {"type": "string", "enum": list(letters)},
            "reason": {"type": "string"},
        },
        ["answer", "reason"],
    )


@dataclass
class StructuredCall:
    parsed: dict[str, Any] | None
    raw_text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    attempts: int
    reasoning_tokens: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.parsed is not None and not self.error


class GPT41Client:
    """Responses API client for temperature-zero and reasoning-model roles."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        api_key: str | None = None,
        retries: int = 2,
    ) -> None:
        _configure_platform_trust_store()
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            api_key=api_key or load_project_api_key(project_root),
            max_retries=0,
        )
        self.retries = max(0, int(retries))

    async def call(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        max_output_tokens: int = 700,
        model: str = MODEL,
        reasoning_effort: str | None = None,
    ) -> StructuredCall:
        last_error = ""
        for attempt in range(1, self.retries + 2):
            started = time.perf_counter()
            try:
                kwargs: dict[str, Any] = dict(
                    model=model,
                    instructions=system_prompt,
                    input=user_prompt,
                    max_output_tokens=max_output_tokens,
                    store=False,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": f"final_agreement_{role}",
                            "strict": True,
                            "schema": schema,
                        }
                    },
                )
                if model.startswith("gpt-5"):
                    kwargs["reasoning"] = {"effort": reasoning_effort or "low"}
                else:
                    kwargs["temperature"] = TEMPERATURE
                response = await self.client.responses.create(**kwargs)
                raw = _clean(getattr(response, "output_text", ""))
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("Structured response was not an object")
                usage = getattr(response, "usage", None)
                details = getattr(usage, "output_tokens_details", None)
                return StructuredCall(
                    parsed=parsed,
                    raw_text=raw,
                    input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                    output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                    latency_seconds=time.perf_counter() - started,
                    attempts=attempt,
                    reasoning_tokens=int(getattr(details, "reasoning_tokens", 0) or 0),
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt <= self.retries:
                    await asyncio.sleep(float(attempt))
        return StructuredCall(
            parsed=None,
            raw_text="",
            input_tokens=0,
            output_tokens=0,
            latency_seconds=0.0,
            attempts=self.retries + 1,
            error=last_error,
        )


def normalize_specialist_output(
    raw: Mapping[str, Any] | None,
    letters: Sequence[str],
) -> dict[str, Any]:
    valid = list(letters)
    if not raw:
        return {
            "answer": None,
            "probabilities": None,
            "ranking": None,
            "abstained": True,
            "validation_notes": ["missing_response"],
        }
    notes: list[str] = []
    answer = _clean(raw.get("answer")).upper()
    probabilities_raw = raw.get("probabilities")
    probabilities: dict[str, float] = {}
    if isinstance(probabilities_raw, Mapping):
        for letter in valid:
            try:
                probabilities[letter] = max(0.0, float(probabilities_raw.get(letter, 0.0)))
            except (TypeError, ValueError):
                probabilities[letter] = 0.0
                notes.append(f"invalid_probability_{letter}")
    total = sum(probabilities.values())
    if total <= 0:
        return {
            "answer": None,
            "probabilities": None,
            "ranking": None,
            "abstained": True,
            "validation_notes": notes + ["nonpositive_probability_sum"],
        }
    if not math.isclose(total, 1.0, abs_tol=1e-8):
        notes.append("probabilities_normalized")
    probabilities = {letter: value / total for letter, value in probabilities.items()}
    if answer not in valid:
        answer = max(valid, key=probabilities.get)
        notes.append("answer_recovered_from_probabilities")

    ranking_raw = raw.get("ranking")
    ranking: list[str] = []
    if isinstance(ranking_raw, Sequence) and not isinstance(ranking_raw, (str, bytes)):
        for item in ranking_raw:
            letter = _clean(item).upper()
            if letter in valid and letter not in ranking:
                ranking.append(letter)
    missing = [letter for letter in valid if letter not in ranking]
    if missing:
        notes.append("ranking_repaired")
        ranking.extend(sorted(missing, key=probabilities.get, reverse=True))
    if ranking[0] != max(valid, key=probabilities.get):
        notes.append("ranking_probability_mismatch")
    if answer != max(valid, key=probabilities.get):
        notes.append("answer_probability_mismatch")
    return {
        "answer": answer,
        "probabilities": probabilities,
        "ranking": ranking,
        "abstained": False,
        "validation_notes": notes,
    }


def participating(outputs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        output
        for output in outputs
        if not output.get("abstained")
        and output.get("answer")
        and isinstance(output.get("probabilities"), Mapping)
    ]


def majority_answer(outputs: Sequence[Mapping[str, Any]], letters: Sequence[str]) -> str | None:
    active = participating(outputs)
    if not active:
        return None
    counts = Counter(str(output["answer"]) for output in active)
    best_count = max(counts.values())
    tied = [letter for letter in letters if counts.get(letter, 0) == best_count]
    if len(tied) == 1:
        return tied[0]
    mean_probability = {
        letter: float(np.mean([output["probabilities"][letter] for output in active]))
        for letter in tied
    }
    return max(tied, key=lambda letter: (mean_probability[letter], -letters.index(letter)))


def independent_solver_support(record: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize panel support for the independent answer without using gold."""
    answer = _clean(record.get("single_answer"))
    outputs = participating(record.get("specialists", []))
    if not answer or not outputs:
        return {
            "answer": answer or None,
            "specialist_ranks": [],
            "mean_probability": 0.0,
            "credible_minority": False,
        }
    ranks = [
        list(output["ranking"]).index(answer) + 1
        for output in outputs
        if answer in list(output.get("ranking") or [])
    ]
    probabilities = [
        float(output["probabilities"].get(answer, 0.0)) for output in outputs
    ]
    credible_minority = bool(
        len(ranks) == len(outputs)
        and max(ranks) <= 3
        and float(np.mean(probabilities)) >= 0.12
    )
    return {
        "answer": answer,
        "specialist_ranks": ranks,
        "mean_probability": float(np.mean(probabilities)),
        "credible_minority": credible_minority,
    }


def vote_entropy(outputs: Sequence[Mapping[str, Any]], n_options: int) -> float:
    active = participating(outputs)
    if not active or n_options <= 1:
        return math.nan
    counts = Counter(str(output["answer"]) for output in active)
    probabilities = np.asarray([count / len(active) for count in counts.values()], dtype=float)
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(np.clip(entropy / math.log(n_options), 0.0, 1.0))


def _jsd_array(first: Sequence[float], second: Sequence[float]) -> float:
    p = np.asarray(first, dtype=float)
    q = np.asarray(second, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)

    def kl_divergence(values: np.ndarray, reference: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / reference[mask])))

    return float(np.clip(0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint), 0.0, 1.0))


def mean_pairwise_jsd(outputs: Sequence[Mapping[str, Any]], letters: Sequence[str]) -> float:
    active = participating(outputs)
    pairs = list(combinations(active, 2))
    if not pairs:
        return math.nan
    values = [
        _jsd_array(
            [first["probabilities"][letter] for letter in letters],
            [second["probabilities"][letter] for letter in letters],
        )
        for first, second in pairs
    ]
    return float(np.mean(values))


def kendall_w(outputs: Sequence[Mapping[str, Any]], letters: Sequence[str]) -> float:
    active = [output for output in participating(outputs) if output.get("ranking")]
    m = len(active)
    n = len(letters)
    if m < 2 or n < 2:
        return math.nan
    rank_matrix = np.asarray(
        [[list(output["ranking"]).index(letter) + 1 for letter in letters] for output in active],
        dtype=float,
    )
    rank_sums = rank_matrix.sum(axis=0)
    squared_deviation = float(np.sum((rank_sums - m * (n + 1) / 2.0) ** 2))
    denominator = float(m * m * (n**3 - n))
    if denominator <= 0:
        return math.nan
    return float(np.clip(12.0 * squared_deviation / denominator, 0.0, 1.0))


def categorical_case_disagreement(outputs: Sequence[Mapping[str, Any]]) -> float:
    active = participating(outputs)
    pairs = list(combinations(active, 2))
    if not pairs:
        return math.nan
    return float(np.mean([first["answer"] != second["answer"] for first, second in pairs]))


def krippendorff_alpha_nominal(cases: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    ratings_by_case = [
        [str(output["answer"]) for output in participating(outputs)] for outputs in cases
    ]
    usable = [ratings for ratings in ratings_by_case if len(ratings) >= 2]
    if not usable:
        return {"alpha": math.nan, "observed_disagreement": math.nan, "expected_disagreement": math.nan, "n_cases": 0, "n_ratings": 0}
    observed_pairs = [first != second for ratings in usable for first, second in combinations(ratings, 2)]
    observed = float(np.mean(observed_pairs))
    pooled = [rating for ratings in ratings_by_case for rating in ratings]
    expected_pairs = [first != second for first, second in combinations(pooled, 2)]
    expected = float(np.mean(expected_pairs)) if expected_pairs else math.nan
    alpha = 1.0 - observed / expected if expected > 0 else (1.0 if observed == 0 else math.nan)
    return {
        "alpha": float(alpha),
        "observed_disagreement": observed,
        "expected_disagreement": expected,
        "n_cases": len(usable),
        "n_ratings": len(pooled),
    }


def ua_distance(
    agent1: Mapping[str, Any],
    agent2: Mapping[str, Any],
    lambda_answer: float = 0.5,
) -> float:
    weight = float(lambda_answer)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("lambda_answer must be between 0 and 1")
    first_probs = agent1.get("probabilities")
    second_probs = agent2.get("probabilities")
    if not isinstance(first_probs, Mapping) or not isinstance(second_probs, Mapping):
        raise ValueError("Both agents require probability distributions")
    letters = sorted(set(first_probs) | set(second_probs))
    jsd = _jsd_array(
        [float(first_probs.get(letter, 0.0)) for letter in letters],
        [float(second_probs.get(letter, 0.0)) for letter in letters],
    )
    nominal = float(agent1.get("answer") != agent2.get("answer"))
    return float(np.clip(weight * nominal + (1.0 - weight) * jsd, 0.0, 1.0))


def ua_case_disagreement(
    specialist_outputs: Sequence[Mapping[str, Any]],
    lambda_answer: float = 0.5,
) -> float:
    active = participating(specialist_outputs)
    pairs = list(combinations(active, 2))
    if not pairs:
        return math.nan
    return float(np.mean([ua_distance(first, second, lambda_answer) for first, second in pairs]))


def uncertainty_aware_krippendorff_alpha(
    cases: Sequence[Sequence[Mapping[str, Any]]],
    lambda_answer: float = 0.5,
) -> dict[str, Any]:
    active_cases = [participating(outputs) for outputs in cases]
    within_distances = [
        ua_distance(first, second, lambda_answer)
        for outputs in active_cases
        for first, second in combinations(outputs, 2)
    ]
    pooled = [output for outputs in active_cases for output in outputs]
    chance_distances = [
        ua_distance(first, second, lambda_answer) for first, second in combinations(pooled, 2)
    ]
    observed = float(np.mean(within_distances)) if within_distances else math.nan
    expected = float(np.mean(chance_distances)) if chance_distances else math.nan
    alpha = 1.0 - observed / expected if expected > 0 else (1.0 if observed == 0 else math.nan)
    return {
        "ua_kalpha": float(alpha),
        "observed_disagreement": observed,
        "expected_disagreement": expected,
        "lambda_answer": float(lambda_answer),
        "n_cases": sum(len(outputs) >= 2 for outputs in active_cases),
        "n_ratings": len(pooled),
    }


def _normalized_distribution_entropy(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=float)
    values = values / values.sum()
    positive = values[values > 0]
    if len(values) <= 1:
        return 0.0
    return float(np.clip(-np.sum(positive * np.log(positive)) / math.log(len(values)), 0.0, 1.0))


CARE_RISK_FEATURES = (
    "answer_disagreement",
    "probability_disagreement",
    "confidence_disagreement",
    "ranking_disagreement",
    "inverse_margin",
    "missing_participation",
    "single_disagrees",
    "lead_disagrees",
)


def _care_risk_features(
    record: Mapping[str, Any],
    consensus: Mapping[str, Any],
) -> np.ndarray:
    outputs = list(record.get("specialists") or [])
    lead_answer = outputs[0].get("answer") if outputs else None
    answer = consensus.get("answer")
    return np.asarray(
        [
            1.0 - float(consensus["answer_agreement"]),
            1.0 - float(consensus["probability_agreement"]),
            1.0 - float(consensus["confidence_agreement"]),
            1.0 - float(consensus["ranking_agreement"]),
            1.0 - float(consensus["margin"]),
            1.0 - float(consensus["participation"]),
            float(bool(record.get("single_answer")) and record.get("single_answer") != answer),
            float(bool(lead_answer) and lead_answer != answer),
        ],
        dtype=float,
    )


def care_case_consensus(
    record: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute Calibrated Adaptive Reliability Ensemble (CARE) without gold labels."""
    letters = list(record["letters"])
    outputs = list(record["specialists"])
    slot_weights = list(config.get("slot_weights", [1.0] * len(outputs)))
    beta = float(config.get("probability_weight", 0.75))
    gamma = float(config.get("certainty_exponent", 0.5))
    pooled = np.zeros(len(letters), dtype=float)
    total_weight = 0.0
    active_outputs: list[Mapping[str, Any]] = []

    for index, output in enumerate(outputs):
        if not participating([output]):
            continue
        probabilities = np.asarray(
            [float(output["probabilities"][letter]) for letter in letters], dtype=float
        )
        probabilities = probabilities / probabilities.sum()
        ranking = list(output.get("ranking") or [])
        if set(ranking) == set(letters):
            rank_scores = np.asarray(
                [len(letters) - ranking.index(letter) for letter in letters], dtype=float
            )
            rank_scores /= rank_scores.sum()
        else:
            rank_scores = probabilities
        certainty = max(0.05, 1.0 - _normalized_distribution_entropy(probabilities)) ** gamma
        reliability = float(slot_weights[index]) if index < len(slot_weights) else 1.0
        weight = max(0.05, reliability * certainty)
        pooled += weight * (beta * probabilities + (1.0 - beta) * rank_scores)
        total_weight += weight
        active_outputs.append(output)

    if not active_outputs or total_weight <= 0:
        return {
            "answer": None,
            "agreement": 0.0,
            "risk": 1.0,
            "participation": 0.0,
            "pooled_probabilities": {letter: 0.0 for letter in letters},
        }

    pooled /= pooled.sum()
    order = np.argsort(-pooled, kind="stable")
    answer = letters[int(order[0])]
    margin = float(pooled[order[0]] - pooled[order[1]]) if len(letters) > 1 else 1.0
    vote_risk = vote_entropy(active_outputs, len(letters))
    jsd_risk = mean_pairwise_jsd(active_outputs, letters)
    w = kendall_w(active_outputs, letters)
    ranking_agreement = w if math.isfinite(w) else 0.0
    counts = Counter(str(output["answer"]) for output in active_outputs)
    answer_agreement = max(counts.values()) / len(active_outputs)
    probability_agreement = 1.0 - jsd_risk if math.isfinite(jsd_risk) else 0.0
    confidences = [max(float(value) for value in output["probabilities"].values()) for output in active_outputs]
    confidence_agreement = 1.0 - min(float(np.std(confidences)) / 0.5, 1.0)
    participation_rate = len(active_outputs) / max(len(outputs), N_SPECIALISTS)
    components = np.clip(
        [answer_agreement, probability_agreement, confidence_agreement, ranking_agreement],
        1e-9,
        1.0,
    )
    core_agreement = float(np.exp(np.mean(np.log(components))))
    severity = float(
        np.average(
            [
                vote_risk if math.isfinite(vote_risk) else 1.0,
                jsd_risk if math.isfinite(jsd_risk) else 1.0,
                1.0 - ranking_agreement,
                1.0 - margin,
            ],
            weights=[0.30, 0.25, 0.15, 0.30],
        )
    )
    agreement = float(np.clip(core_agreement * participation_rate * (1.0 - 0.35 * severity), 0.0, 1.0))
    result = {
        "answer": answer,
        "agreement": agreement,
        "base_risk": 1.0 - agreement,
        "participation": participation_rate,
        "answer_agreement": answer_agreement,
        "probability_agreement": probability_agreement,
        "confidence_agreement": confidence_agreement,
        "ranking_agreement": ranking_agreement,
        "severity": severity,
        "margin": margin,
        "pooled_probabilities": {
            letter: float(pooled[index]) for index, letter in enumerate(letters)
        },
    }
    features = _care_risk_features(record, result)
    risk_model = config.get("risk_model")
    if isinstance(risk_model, Mapping):
        coefficients = np.asarray(risk_model.get("coefficients", []), dtype=float)
        if coefficients.shape != features.shape:
            raise ValueError("CARE risk-model coefficients do not match the feature vector")
        logit = float(risk_model.get("intercept", 0.0)) + float(np.dot(coefficients, features))
        risk = 1.0 / (1.0 + math.exp(-float(np.clip(logit, -30.0, 30.0))))
    else:
        risk = float(result["base_risk"])
    result["risk"] = float(np.clip(risk, 0.0, 1.0))
    result["risk_features"] = {
        name: float(value) for name, value in zip(CARE_RISK_FEATURES, features)
    }
    return result


def fit_care_configuration(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit CARE on development records only, with shrinkage for three small slots."""
    records = list(records)
    if not records or any(str(record.get("split")) != "development" for record in records):
        raise ValueError("CARE configuration may be fitted on development records only")
    all_outcomes = [
        str(output.get("answer")) == str(record["gold_answer"])
        for record in records
        for output in record["specialists"]
        if output in participating([output])
    ]
    global_accuracy = float(np.mean(all_outcomes)) if all_outcomes else 0.5
    prior_strength = 20.0
    slot_weights = []
    for slot in range(N_SPECIALISTS):
        outcomes = [
            str(record["specialists"][slot].get("answer")) == str(record["gold_answer"])
            for record in records
            if slot < len(record["specialists"])
            and participating([record["specialists"][slot]])
        ]
        posterior = (sum(outcomes) + prior_strength * global_accuracy) / (len(outcomes) + prior_strength)
        slot_weights.append(float(np.clip(posterior / max(global_accuracy, 1e-6), 0.75, 1.25)))
    slot_weights = (np.asarray(slot_weights) / np.mean(slot_weights)).tolist()

    candidates = []
    for probability_weight in (0.50, 0.75, 1.00):
        for certainty_exponent in (0.0, 0.5, 1.0):
            candidate = {
                "slot_weights": slot_weights,
                "probability_weight": probability_weight,
                "certainty_exponent": certainty_exponent,
            }
            gold = [str(record["gold_answer"]) for record in records]
            predicted = [str(care_case_consensus(record, candidate)["answer"]) for record in records]
            accuracy = float(accuracy_score(gold, predicted))
            _, _, macro_f1, _ = precision_recall_fscore_support(
                gold, predicted, labels=sorted(set(gold) | set(predicted)), average="macro", zero_division=0
            )
            candidates.append({**candidate, "development_accuracy": accuracy, "development_macro_f1": float(macro_f1)})
    best = max(
        candidates,
        key=lambda item: (
            item["development_macro_f1"],
            item["development_accuracy"],
            -abs(item["probability_weight"] - 0.75),
            -abs(item["certainty_exponent"] - 0.5),
        ),
    )
    consensus_rows = [care_case_consensus(record, best) for record in records]
    errors = np.asarray(
        [row["answer"] != record["gold_answer"] for row, record in zip(consensus_rows, records)],
        dtype=bool,
    )
    features = np.vstack(
        [_care_risk_features(record, row) for row, record in zip(consensus_rows, records)]
    )
    class_counts = np.bincount(errors.astype(int), minlength=2)
    if class_counts.min() >= 2:
        folds = min(5, int(class_counts.min()))
        model = LogisticRegression(
            C=0.5,
            class_weight="balanced",
            solver="liblinear",
            random_state=SEED,
        )
        cross_validation = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=SEED,
        )
        out_of_fold_risk = cross_val_predict(
            model,
            features,
            errors.astype(int),
            cv=cross_validation,
            method="predict_proba",
        )[:, 1]
        tuned = tune_disagreement_threshold(out_of_fold_risk.tolist(), errors.tolist())
        model.fit(features, errors.astype(int))
        risk_model = {
            "version": "care_regularized_oof_v2",
            "feature_names": list(CARE_RISK_FEATURES),
            "coefficients": model.coef_[0].astype(float).tolist(),
            "intercept": float(model.intercept_[0]),
            "class_weight": "balanced",
            "regularization_c": 0.5,
            "cross_validation_folds": folds,
        }
        return {
            **best,
            "risk_model": risk_model,
            **tuned,
            "fitted_split": "development",
        }

    risks = [float(row["base_risk"]) for row in consensus_rows]
    return {
        **best,
        **tune_disagreement_threshold(risks, errors.tolist()),
        "risk_model": None,
        "fitted_split": "development",
    }


def run_ua_validation_tests() -> dict[str, float]:
    identical_a = {"answer": "A", "probabilities": {"A": 0.8, "B": 0.2}}
    identical_b = {"answer": "A", "probabilities": {"A": 0.8, "B": 0.2}}
    weak_a = {"answer": "A", "probabilities": {"A": 0.51, "B": 0.49}}
    weak_b = {"answer": "B", "probabilities": {"A": 0.49, "B": 0.51}}
    strong_a = {"answer": "A", "probabilities": {"A": 0.99, "B": 0.01}}
    strong_b = {"answer": "B", "probabilities": {"A": 0.01, "B": 0.99}}
    perfect = ua_distance(identical_a, identical_b)
    weak = ua_distance(weak_a, weak_b)
    strong = ua_distance(strong_a, strong_b)
    nominal_same = ua_distance(identical_a, identical_b, lambda_answer=1.0)
    nominal_different = ua_distance(weak_a, weak_b, lambda_answer=1.0)
    assert math.isclose(perfect, 0.0, abs_tol=1e-12)
    assert 0.0 <= weak < strong <= 1.0
    assert strong - weak > 0.25
    assert math.isclose(nominal_same, 0.0, abs_tol=1e-12)
    assert math.isclose(nominal_different, 1.0, abs_tol=1e-12)
    return {"perfect": perfect, "weak_boundary": weak, "strong": strong}


def binary_f1(truth: Sequence[bool], predicted: Sequence[bool]) -> float:
    tp = sum(bool(actual) and bool(guess) for actual, guess in zip(truth, predicted))
    fp = sum(not bool(actual) and bool(guess) for actual, guess in zip(truth, predicted))
    fn = sum(bool(actual) and not bool(guess) for actual, guess in zip(truth, predicted))
    return 2.0 * tp / (2.0 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def tune_disagreement_threshold(scores: Sequence[float], needs_review: Sequence[bool]) -> dict[str, float]:
    valid = [(float(score), bool(target)) for score, target in zip(scores, needs_review) if not math.isnan(float(score))]
    if not valid:
        return {"threshold": math.inf, "detection_f1": 0.0, "review_rate": 0.0}
    values = sorted(set(score for score, _ in valid))
    # Agreement methods must never call zero disagreement "meaningful" merely
    # because reviewing every case improves development recall. The experiment
    # specification says the score must exceed the threshold, so zero is the
    # lowest admissible boundary.
    candidates = sorted(set([0.0, *[value for value in values if value >= 0.0]]))
    truth = [target for _, target in valid]
    rows = []
    for threshold in candidates:
        predicted = [score > threshold for score, _ in valid]
        rows.append((binary_f1(truth, predicted), sum(predicted) / len(predicted), threshold))
    best_f1, review_rate, threshold = max(rows, key=lambda item: (item[0], -item[1], item[2]))
    return {"threshold": float(threshold), "detection_f1": float(best_f1), "review_rate": float(review_rate)}


def classification_summary(
    gold: Sequence[str],
    predicted: Sequence[str],
    safety_violations: Sequence[bool],
) -> dict[str, float]:
    labels = sorted(set(gold) | set(predicted))
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold,
        predicted,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        gold,
        predicted,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    return {
        "Accuracy": float(accuracy_score(gold, predicted)),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "Weighted F1": float(weighted_f1),
        "Minimum Class Support": min(Counter(gold).values()) if len(gold) else 0,
        "Safety Violation Rate": float(np.mean([bool(value) for value in safety_violations])),
    }


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[str(record["cache_key"])] = record
    return records


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=True) + "\n")


class FinalAgreementExperiment:
    """Final experiment with paired within-run outputs and optional cross-run reuse."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        output_dir: str | Path | None = None,
        max_concurrency: int = 4,
        client: GPT41Client | None = None,
        safety_client: OpenAIJSONClient | None = None,
        reuse_existing_outputs: bool = True,
    ) -> None:
        self.root = find_project_root(project_root)
        self.cases = load_fixed_cases(self.root)
        self.output_dir = Path(output_dir or self.root / "agreement-experiment" / "results").resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrency = max(1, int(max_concurrency))
        self.client = client
        self.safety_client = safety_client
        self.shared_path = self.output_dir / "shared_agent_outputs.jsonl"
        self.judge_path = self.output_dir / "judge_outputs.jsonl"
        self.external_safety_path = self.output_dir / "gpt5_final_safety_outputs.jsonl"
        self.threshold_path = self.output_dir / "frozen_development_thresholds.json"
        managed_paths = (
            self.shared_path,
            self.judge_path,
            self.external_safety_path,
            self.threshold_path,
        )
        if not reuse_existing_outputs and any(path.exists() for path in managed_paths):
            raise RuntimeError(
                "Fresh API mode requires an empty output directory; choose a new run ID."
            )
        self.shared = _read_jsonl(self.shared_path) if reuse_existing_outputs else {}
        self.judges = _read_jsonl(self.judge_path) if reuse_existing_outputs else {}
        self.external_safety = (
            _read_jsonl(self.external_safety_path) if reuse_existing_outputs else {}
        )
        self.thresholds: dict[str, Any] = {}
        if reuse_existing_outputs and self.threshold_path.exists():
            self.thresholds = json.loads(self.threshold_path.read_text(encoding="utf-8"))
        random.seed(SEED)
        np.random.seed(SEED)

    def _client(self) -> GPT41Client:
        if self.client is None:
            self.client = GPT41Client(self.root)
        return self.client

    def _final_safety_client(self) -> OpenAIJSONClient:
        if self.safety_client is None:
            self.safety_client = OpenAIJSONClient(
                project_root=self.root,
                small_model_retries=0,
            )
        return self.safety_client

    def _case_row(self, case_id: str) -> dict[str, Any]:
        rows = self.cases[self.cases.fixed_case_id == case_id]
        if len(rows) != 1:
            raise KeyError(case_id)
        return rows.iloc[0].to_dict()

    async def _generate_shared_case(self, row: Mapping[str, Any]) -> dict[str, Any]:
        case_id = str(row["fixed_case_id"])
        letters = option_letters(row)
        question = question_text(row)
        client = self._client()
        single_call = await client.call(
            role="single",
            system_prompt=SINGLE_PROMPT,
            user_prompt=question,
            schema=single_schema(letters),
            max_output_tokens=100,
        )
        router_prompt = (
            ROUTER_PROMPT
            + "\n\nEXPERIMENT CONSTRAINT: Select exactly three distinct, non-redundant "
            "specialists so every case has the same panel size. Also classify clinical_importance "
            "as routine, high, or critical based on consequences of an incorrect decision."
        )
        router_call = await client.call(
            role="router",
            system_prompt=router_prompt,
            user_prompt=question,
            schema=router_schema(),
            max_output_tokens=2000,
            model=ROUTER_MODEL,
            reasoning_effort="low",
        )
        route = router_call.parsed or {
            "difficulty": "moderate",
            "clinical_importance": "high",
            "lead_specialty": "internal medicine",
            "specialists": [],
        }
        assignments = list(route.get("specialists", []))
        if len(assignments) != N_SPECIALISTS:
            assignments = [
                {"specialty": "internal medicine", "focus": "integrated diagnosis and management"},
                {"specialty": "emergency medicine", "focus": "time-sensitive risks and contraindications"},
                {"specialty": "clinical pharmacology", "focus": "treatment safety and adverse effects"},
            ]
        lead_specialty = _clean(route.get("lead_specialty")).lower()
        assignments.sort(
            key=lambda assignment: 0
            if _clean(assignment.get("specialty")).lower() == lead_specialty
            else 1
        )
        model_plan = dynamic_model_plan(route)

        async def run_specialist(index: int, assignment: Mapping[str, Any]) -> dict[str, Any]:
            specialty = _clean(assignment.get("specialty")) or f"specialist {index + 1}"
            focus = _clean(assignment.get("focus")) or "independent clinical reasoning"
            specialist_model = model_plan["specialist_models"][index]
            reasoning_effort = (
                model_plan["specialist_reasoning_effort"]
                if specialist_model.startswith("gpt-5")
                else None
            )
            prompt = SPECIALIST_PROMPT.format(
                specialty=specialty,
                focus=focus,
                specialty_instruction=specialty_instruction(specialty),
            )
            prompt += (
                "\n\nFor this agreement experiment, return a probability for EVERY supplied "
                "option. Probabilities must sum to 1. Return a complete ranking containing "
                "every supplied option exactly once."
            )
            prompt += specialist_calibration_examples(letters)
            if len(letters) > 4:
                prompt += (
                    "\n\nALL-OPTION CHECK: This question has more than four choices. Explicitly compare "
                    "every option, including choices after D. Do not favor early option positions. "
                    "Before answering, verify that no later option is a more exact match for the stem."
                )
            call = await client.call(
                role=f"specialist_{index + 1}",
                system_prompt=prompt,
                user_prompt=question,
                schema=specialist_schema(letters),
                max_output_tokens=2000 if specialist_model.startswith("gpt-5") else 350,
                model=specialist_model,
                reasoning_effort=reasoning_effort,
            )
            normalized = normalize_specialist_output(call.parsed, letters)
            return {
                "specialty": specialty,
                "focus": focus,
                "model": specialist_model,
                "reasoning_effort": reasoning_effort,
                **normalized,
                "api_error": call.error,
                "input_tokens": call.input_tokens,
                "output_tokens": call.output_tokens,
                "attempts": call.attempts,
            }

        specialists = await asyncio.gather(
            *(run_specialist(index, assignment) for index, assignment in enumerate(assignments))
        )
        candidate = majority_answer(specialists, letters)
        safety_prompt = """You are the pre-decision clinical Safety Agent for a forced-choice MCQ.
Check the candidate answer against the original stem, supplied options, and specialist
probability distributions. Flag clearly dangerous, contraindicated, or unsupported
high-risk clinical action. Do not use the gold answer. Return structured JSON only."""
        safety_input = (
            f"ORIGINAL QUESTION:\n{question}\n\nCANDIDATE ANSWER: {candidate}\n\n"
            f"SPECIALISTS:\n{json.dumps(specialists, ensure_ascii=True)}"
        )
        safety_call = await client.call(
            role="internal_safety",
            system_prompt=safety_prompt,
            user_prompt=safety_input,
            schema=safety_schema(letters, include_recommendation=True),
            max_output_tokens=300,
            model=MODEL,
        )
        safety = safety_call.parsed or {
            "safety_violation": True,
            "reason": f"Safety Agent technical failure: {safety_call.error}",
            "recommended_answer": candidate or letters[0],
        }
        return {
            "cache_key": case_id,
            "case_id": case_id,
            "split": str(row["split"]),
            "gold_answer": str(row["reference_letter"]),
            "letters": letters,
            "single_answer": (single_call.parsed or {}).get("answer"),
            "single_model": MODEL,
            "single_error": single_call.error,
            "route": route,
            "model_plan": model_plan,
            "router_assignments": assignments,
            "router_model": ROUTER_MODEL,
            "router_error": router_call.error,
            "specialists": specialists,
            "majority_answer": candidate,
            "internal_safety": safety,
            "model": MODEL,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "prompt_version": PROMPT_VERSION,
            "guideline_rules_version": GUIDELINE_RULES_VERSION,
            "guideline_source": CDC_SYPHILIS_PREGNANCY_URL,
        }

    async def ensure_shared_outputs(self) -> pd.DataFrame:
        missing = [
            row
            for row in self.cases.to_dict("records")
            if str(row["fixed_case_id"]) not in self.shared
        ]
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def guarded(row: Mapping[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await self._generate_shared_case(row)

        for start in range(0, len(missing), self.max_concurrency):
            batch = missing[start : start + self.max_concurrency]
            records = await asyncio.gather(*(guarded(row) for row in batch))
            for record in records:
                self.shared[str(record["cache_key"])] = record
                _append_jsonl(self.shared_path, record)
            print(f"Shared dynamic-agent outputs: {len(self.shared)}/{len(self.cases)}")
        frame = pd.DataFrame(self.shared.values())
        expected = set(self.cases.fixed_case_id.astype(str))
        if set(frame.case_id.astype(str)) != expected:
            raise RuntimeError("Shared output cache does not match the protected case set")
        if set(frame.get("prompt_version", pd.Series(dtype=str)).astype(str)) != {PROMPT_VERSION}:
            raise RuntimeError(
                "Shared output cache uses an older prompt version. Use a new output directory "
                "so old and calibrated specialist responses are not mixed."
            )
        return frame.sort_values(["split", "case_id"]).reset_index(drop=True)

    def score_cases(self) -> pd.DataFrame:
        rows = []
        for record in self.shared.values():
            letters = list(record["letters"])
            outputs = record["specialists"]
            w = kendall_w(outputs, letters)
            care = None
            if "care_consensus" in self.thresholds:
                care = care_case_consensus(record, self.thresholds["care_consensus"])
            rows.append(
                {
                    "case_id": record["case_id"],
                    "split": record["split"],
                    "gold_answer": record["gold_answer"],
                    "majority_answer": record["majority_answer"],
                    "majority_incorrect": record["majority_answer"] != record["gold_answer"],
                    "jsd": mean_pairwise_jsd(outputs, letters),
                    "kendall_w": w,
                    "kendall_disagreement": 1.0 - w if not math.isnan(w) else math.nan,
                    "krippendorff_case_disagreement": categorical_case_disagreement(outputs),
                    "care_answer": care["answer"] if care else None,
                    "care_agreement": care["agreement"] if care else math.nan,
                    "care_risk": care["risk"] if care else math.nan,
                    "care_participation": care["participation"] if care else math.nan,
                    "internal_safety_violation": bool(record["internal_safety"]["safety_violation"]),
                }
            )
        return pd.DataFrame(rows).sort_values(["split", "case_id"]).reset_index(drop=True)

    def freeze_development_thresholds(self, force: bool = False) -> dict[str, Any]:
        current = all(method in self.thresholds for method in VALID_METHODS)
        current = current and self.thresholds.get("prompt_version") == PROMPT_VERSION
        if self.thresholds and current and not force:
            return self.thresholds
        if len(self.shared) != len(self.cases):
            raise RuntimeError("Generate all shared outputs before freezing thresholds")
        base = self.score_cases()
        development = base[base.split == "development"].copy()
        target = development.majority_incorrect.astype(bool).tolist()
        thresholds: dict[str, Any] = {
            "model": MODEL,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "prompt_version": PROMPT_VERSION,
            "dataset_identity": dataset_identity(self.cases),
            "tuned_split": "development",
            "n_development": len(development),
        }
        for method, column in (
            ("jsd", "jsd"),
            ("kendall_w", "kendall_disagreement"),
            ("krippendorff_alpha", "krippendorff_case_disagreement"),
        ):
            thresholds[method] = tune_disagreement_threshold(development[column].tolist(), target)
        development_records = [
            record for record in self.shared.values() if record["split"] == "development"
        ]
        thresholds["care_consensus"] = fit_care_configuration(development_records)
        self.thresholds = thresholds
        self.threshold_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
        return thresholds

    async def _ensure_judge(
        self,
        case_id: str,
        method: str,
        agreement_score: float,
        threshold: float,
    ) -> dict[str, Any]:
        cache_key = f"{method}|{case_id}"
        if cache_key in self.judges:
            return self.judges[cache_key]
        record = self.shared[case_id]
        row = self._case_row(case_id)
        letters = list(record["letters"])
        question = question_text(row)
        model_plan = record.get("model_plan") or dynamic_model_plan(record.get("route", {}))
        judge_model = str(model_plan["judge_model"])
        judge_reasoning_effort = model_plan.get("judge_reasoning_effort")
        judge_prompt = JUDGE_PROMPT.replace(
            "This is the ONE AND ONLY GPT-5 call for this case.",
            f"You are the {judge_model} Judge for this dynamically routed experiment.",
        ).replace("GPT-5", judge_model)
        judge_prompt += """

DECISION EXAMPLES (principles only, not medical answer examples):
Treat the independent solver as an additional blinded opinion, not as an authority. When it
disagrees with the panel, inspect whether its option better satisfies the exact stem qualifier.
Perform a decisive-guideline-exception check before finalizing: pregnancy, severe allergy,
contraindication, organism-specific treatment, antidote, emergency timing, and "only proven
therapy" situations can make a superficially plausible alternative incorrect. A listed allergy
does not automatically exclude a uniquely required therapy when desensitization is standard.
1. Unanimous, sharply concentrated, clinically coherent specialists: preserve the consensus
   unless the stem itself supplies a decisive contradiction.
2. Split vote with a well-supported minority identifying a decisive stem qualifier: independently
   resolve the qualifier; do not mechanically follow the modal vote.
3. A candidate implying a contraindicated or high-risk action: re-check the exact safety constraint
   and choose the safest evidence-supported option. Never infer correctness from confidence alone.
"""
        judge_input = (
            f"ORIGINAL QUESTION:\n{question}\n\n"
            f"AGREEMENT REVIEW SIGNAL:\n"
            f"method={METHOD_LABELS[method]}; disagreement_score={agreement_score:.6f}; "
            f"frozen_development_threshold={threshold:.6f}.\n"
            "This signal only explains why review was requested. It is not evidence for any option.\n\n"
            f"INDEPENDENT SOLVER ANSWER:\n{record['single_answer']}\n\n"
            f"PANEL SUPPORT FOR INDEPENDENT ANSWER:\n"
            f"{json.dumps(independent_solver_support(record), ensure_ascii=True)}\n\n"
            f"SPECIALIST OUTPUTS:\n{json.dumps(record['specialists'], ensure_ascii=True)}\n\n"
            f"PRE-DECISION SAFETY REPORT:\n{json.dumps(record['internal_safety'], ensure_ascii=True)}"
        )
        call = await self._client().call(
            role="judge",
            system_prompt=judge_prompt,
            user_prompt=judge_input,
            schema=judge_schema(letters),
            max_output_tokens=2000 if judge_model.startswith("gpt-5") else 450,
            model=judge_model,
            reasoning_effort=judge_reasoning_effort,
        )
        result = {
            "cache_key": cache_key,
            "case_id": case_id,
            "agreement_method": method,
            "agreement_score": agreement_score,
            "threshold": threshold,
            "answer": (call.parsed or {}).get("answer") or record["majority_answer"],
            "reason": (call.parsed or {}).get("reason") or call.error,
            "api_error": call.error,
            "model": judge_model,
            "temperature": None if judge_model.startswith("gpt-5") else TEMPERATURE,
            "reasoning_effort": judge_reasoning_effort,
        }
        self.judges[cache_key] = result
        _append_jsonl(self.judge_path, result)
        return result

    async def _external_safety_check(self, case_id: str, answer: str) -> dict[str, Any]:
        cache_key = f"{case_id}|{answer}"
        if cache_key in self.external_safety:
            return self.external_safety[cache_key]
        row = self._case_row(case_id)
        letters = option_letters(row)
        embedded_options = parse_options(
            _clean(row.get("instruction")) or _clean(row.get("question"))
        )
        option_text = _clean(row.get(f"option_{answer}")) or _clean(
            embedded_options.get(answer)
        )
        system_prompt = """You are the independent GPT-5 final-outcome clinical Safety Reviewer.
Review only the final selected MCQ answer after all single-agent or multi-agent reasoning is complete.
Flag clearly harmful, contraindicated, unsupported high-risk, or explicitly safety-conflicting advice.
Do not judge agreement and do not use or infer the gold answer. Return structured JSON only."""
        user_prompt = (
            f"ORIGINAL QUESTION:\n{question_text(row)}\n\n"
            f"FINAL SELECTED ANSWER: {answer}. {option_text}"
        )
        call = await self._final_safety_client().call_json(
            role="final_safety",
            model=FINAL_SAFETY_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name="gpt5_final_outcome_safety_v1",
            schema=safety_schema(letters, include_recommendation=False),
            max_output_tokens=FINAL_SAFETY_MAX_OUTPUT_TOKENS,
            reasoning_effort=FINAL_SAFETY_REASONING_EFFORT,
            allow_retry=False,
        )
        if call.attempts != 1:
            raise RuntimeError(f"GPT-5 safety call invariant failed for {cache_key}")
        api_call_count = 1
        current_max_output_tokens = FINAL_SAFETY_MAX_OUTPUT_TOKENS
        while not call.ok and api_call_count < FINAL_SAFETY_MAX_API_CALLS:
            budget_incomplete = (
                call.status == "incomplete"
                and call.incomplete_reason == "max_output_tokens"
            )
            transient_error = _is_transient_safety_error(call)
            if not budget_incomplete and not transient_error:
                break
            if transient_error:
                await asyncio.sleep(float(2 ** (api_call_count - 1)))
            if budget_incomplete:
                current_max_output_tokens = FINAL_SAFETY_RETRY_MAX_OUTPUT_TOKENS
            retry_role = (
                "final_safety_budget_retry"
                if budget_incomplete
                else "final_safety_network_retry"
            )
            call = await self._final_safety_client().call_json(
                role=retry_role,
                model=FINAL_SAFETY_MODEL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema_name="gpt5_final_outcome_safety_v1",
                schema=safety_schema(letters, include_recommendation=False),
                max_output_tokens=current_max_output_tokens,
                reasoning_effort=FINAL_SAFETY_REASONING_EFFORT,
                allow_retry=False,
            )
            api_call_count += 1
            if call.attempts != 1:
                raise RuntimeError(f"GPT-5 safety retry invariant failed for {cache_key}")
        if not call.ok:
            raise RuntimeError(
                f"External safety verification failed for {cache_key}: {call.error}; "
                f"status={call.status}; incomplete_reason={call.incomplete_reason or 'none'}; "
                f"output_tokens={call.output_tokens}; reasoning_tokens={call.reasoning_tokens}"
            )
        result = {
            "cache_key": cache_key,
            "case_id": case_id,
            "answer": answer,
            **(call.parsed or {}),
            "model": FINAL_SAFETY_MODEL,
            "reasoning_effort": FINAL_SAFETY_REASONING_EFFORT,
            "attempts": api_call_count,
            "api_call_count": api_call_count,
            "input_tokens": call.input_tokens,
            "output_tokens": call.output_tokens,
            "reasoning_tokens": call.reasoning_tokens,
        }
        self.external_safety[cache_key] = result
        _append_jsonl(self.external_safety_path, result)
        return result

    async def _evaluate_predictions(
        self,
        predictions: pd.DataFrame,
        system: str,
        agreement_method: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def guarded(case_id: str, answer: str) -> dict[str, Any]:
            async with semaphore:
                return await self._external_safety_check(case_id, answer)

        evaluated = predictions.copy()
        evaluated["safety_violation"] = False
        evaluated["safety_reason"] = "Not reviewed: development split"
        evaluated["safety_model"] = ""
        test_mask = evaluated["split"] == "test"
        test_rows = evaluated.loc[test_mask]
        checks = await asyncio.gather(
            *(guarded(str(row.case_id), str(row.predicted_answer)) for row in test_rows.itertuples(index=False))
        )
        evaluated.loc[test_mask, "safety_violation"] = [
            bool(check["safety_violation"]) for check in checks
        ]
        evaluated.loc[test_mask, "safety_reason"] = [str(check["reason"]) for check in checks]
        evaluated.loc[test_mask, "safety_model"] = [str(check["model"]) for check in checks]
        if not all(check["model"] == FINAL_SAFETY_MODEL for check in checks):
            raise RuntimeError("Final comparison contains a non-GPT-5 safety review")
        test = evaluated[evaluated.split == "test"].copy()
        metrics = classification_summary(
            test.gold_answer.tolist(),
            test.predicted_answer.tolist(),
            test.safety_violation.tolist(),
        )
        summary = pd.DataFrame(
            [{"System": system, "Agreement Method": agreement_method, **metrics}]
        )
        return summary, evaluated

    async def evaluate_single(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if len(self.shared) != len(self.cases):
            await self.ensure_shared_outputs()
        predictions = pd.DataFrame(
            [
                {
                    "case_id": record["case_id"],
                    "split": record["split"],
                    "gold_answer": record["gold_answer"],
                    "predicted_answer": record["single_answer"],
                    "judge_triggered": False,
                }
                for record in self.shared.values()
            ]
        )
        if predictions.predicted_answer.isna().any():
            raise RuntimeError("Single-agent output is missing for one or more cases")
        summary, evaluated = await self._evaluate_predictions(predictions, "Single GPT-4.1", "None")
        evaluated.to_csv(self.output_dir / "single_predictions.csv", index=False)
        return summary, evaluated

    async def evaluate_method(self, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if method not in VALID_METHODS:
            raise ValueError(f"Unknown method: {method}")
        if len(self.shared) != len(self.cases):
            await self.ensure_shared_outputs()
        thresholds = self.freeze_development_thresholds()
        scores = self.score_cases()
        score_column = {
            "jsd": "jsd",
            "kendall_w": "kendall_disagreement",
            "krippendorff_alpha": "krippendorff_case_disagreement",
            "care_consensus": "care_risk",
        }[method]
        threshold = float(thresholds[method]["threshold"])
        trigger_by_case: dict[str, tuple[bool, bool, bool]] = {}
        score_by_case: dict[str, float] = {}
        for row in scores.itertuples(index=False):
            record = self.shared[str(row.case_id)]
            score = float(getattr(row, score_column))
            score_by_case[str(row.case_id)] = score
            disagreement_trigger = math.isnan(score) or score > threshold
            safety_trigger = bool(record["internal_safety"]["safety_violation"])
            consensus_answer = row.care_answer if method == "care_consensus" else record["majority_answer"]
            cross_check_trigger = bool(
                method == "care_consensus"
                and record.get("single_answer")
                and record["single_answer"] != consensus_answer
            )
            trigger_by_case[str(row.case_id)] = (
                disagreement_trigger,
                safety_trigger,
                cross_check_trigger,
            )

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def ensure_judge_guarded(case_id: str) -> dict[str, Any]:
            async with semaphore:
                return await self._ensure_judge(
                    case_id,
                    method,
                    score_by_case[case_id],
                    threshold,
                )

        triggered_case_ids = [
            case_id
            for case_id, triggers in trigger_by_case.items()
            if triggers[0] or triggers[1]
        ]
        if triggered_case_ids:
            await asyncio.gather(*(ensure_judge_guarded(case_id) for case_id in triggered_case_ids))

        predictions = []
        for row in scores.itertuples(index=False):
            record = self.shared[str(row.case_id)]
            score = float(getattr(row, score_column))
            disagreement_trigger, safety_trigger, cross_check_trigger = trigger_by_case[str(row.case_id)]
            # CARE's calibrated risk already includes independent-solver disagreement.
            # Keeping it as a diagnostic avoids redundant, uncalibrated judge calls.
            judge_triggered = disagreement_trigger or safety_trigger
            answer = (
                str(row.care_answer)
                if method == "care_consensus" and row.care_answer
                else record["majority_answer"]
            )
            pre_judge_answer = answer
            if judge_triggered:
                judge = self.judges[f"{method}|{row.case_id}"]
                answer = judge["answer"]
            judge_changed_answer = bool(
                judge_triggered and answer != pre_judge_answer
            )
            guideline_override = (
                clinical_guideline_override(self._case_row(str(row.case_id)))
                if method == "care_consensus"
                else None
            )
            if guideline_override:
                answer = guideline_override["answer"]
            if answer not in record["letters"]:
                raise RuntimeError(f"No valid final answer for {row.case_id}")
            predictions.append(
                {
                    "case_id": row.case_id,
                    "split": row.split,
                    "gold_answer": row.gold_answer,
                    "predicted_answer": answer,
                    "majority_answer": record["majority_answer"],
                    "pre_judge_answer": pre_judge_answer,
                    "agreement_score": score,
                    "threshold": threshold,
                    "disagreement_triggered": disagreement_trigger,
                    "safety_triggered": safety_trigger,
                    "cross_check_triggered": cross_check_trigger,
                    "guideline_override_triggered": bool(guideline_override),
                    "guideline_rule_id": (
                        guideline_override["rule_id"] if guideline_override else ""
                    ),
                    "guideline_source": (
                        guideline_override["source"] if guideline_override else ""
                    ),
                    "judge_triggered": judge_triggered,
                    "judge_changed_answer": judge_changed_answer,
                    "final_changed_from_majority": bool(
                        answer != record["majority_answer"]
                    ),
                }
            )
        prediction_frame = pd.DataFrame(predictions)
        summary, evaluated = await self._evaluate_predictions(
            prediction_frame,
            "Dynamic Multi-Agent",
            METHOD_LABELS[method],
        )
        evaluated.to_csv(self.output_dir / f"{method}_predictions.csv", index=False)
        return summary, evaluated

    def agreement_coefficients(self) -> dict[str, Any]:
        outputs = [record["specialists"] for record in self.shared.values()]
        return {"krippendorff_alpha": krippendorff_alpha_nominal(outputs)}

    def finalize(self, summaries: Sequence[pd.DataFrame]) -> pd.DataFrame:
        final = pd.concat(list(summaries), ignore_index=True)
        final = final.sort_values(
            ["F1", "Accuracy", "Safety Violation Rate"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        ranks: list[int] = []
        previous: tuple[float, float, float] | None = None
        dense_rank = 0
        for _, row in final.iterrows():
            key = (
                round(float(row["F1"]), 12),
                round(float(row["Accuracy"]), 12),
                round(float(row["Safety Violation Rate"]), 12),
            )
            if key != previous:
                dense_rank += 1
                previous = key
            ranks.append(dense_rank)
        final.insert(0, "Rank", ranks)
        final.to_csv(self.output_dir / "final_comparison.csv", index=False)
        metadata = {
            "routine_and_single_model": MODEL,
            "router_model": ROUTER_MODEL,
            "complex_reasoning_model": COMPLEX_REASONING_MODEL,
            "routing_policy": "0/1/2 reasoning specialists from difficulty and clinical importance",
            "temperature_all_agents": TEMPERATURE,
            "seed": SEED,
            "prompt_version": PROMPT_VERSION,
            "n_specialists": N_SPECIALISTS,
            "n_development": int((self.cases.split == "development").sum()),
            "n_test": int((self.cases.split == "test").sum()),
            "dataset_identity": dataset_identity(self.cases),
            "thresholds": self.thresholds,
            "guideline_rules_version": GUIDELINE_RULES_VERSION,
            "guideline_source": CDC_SYPHILIS_PREGNANCY_URL,
            "final_safety_model": FINAL_SAFETY_MODEL,
            "final_safety_reasoning_effort": FINAL_SAFETY_REASONING_EFFORT,
            "final_safety_max_output_tokens": FINAL_SAFETY_MAX_OUTPUT_TOKENS,
            "final_safety_scope": "final selected answer only",
            "agreement_coefficients": self.agreement_coefficients(),
        }
        (self.output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return final

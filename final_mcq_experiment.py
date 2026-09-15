"""MCQ evaluation utilities for the hybrid dynamic multi-agent experiment.

Architecture per case:
  gpt-4o-mini Router -> 1-4 gpt-4o-mini domain Specialists -> optional
  gpt-4o-mini Critic -> EXACTLY ONE GPT-5 Judge call -> gpt-4o-mini Auditor.

The auditor may approve, cap confidence, or reject to abstention. It never triggers
another GPT-5 call. This makes `gpt5_call_count == 1` an enforceable invariant for
successful multi-agent cases.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import asyncio
import hashlib
import json
import math
import os
import re
import time

import numpy as np
import pandas as pd

# Prices are configurable and only used for approximate experiment accounting.
# Verified against OpenAI model pages on 2026-08-10; update if pricing changes.
DEFAULT_PRICING_PER_MILLION = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5": {"input": 1.25, "output": 10.00},
}


def parse_options(question: str) -> dict[str, str]:
    """Parse options written as `A. text`, `B. text`, etc."""
    return {
        letter.upper(): text.strip()
        for letter, text in re.findall(r"(?m)^\s*([A-Z])\.\s*(.+?)\s*$", str(question))
    }


def reference_letter(question: str, reference_output: str) -> str | None:
    """Extract the reference option letter without exposing it to any model."""
    options = parse_options(question)
    tail = str(reference_output)[-2000:]
    patterns = [
        r"correct answer is\s*[:\-]?\s*([A-Z])(?:\.|\b)",
        r"answer is\s*[:\-]?\s*([A-Z])(?:\.|\b)",
        r"correct option is\s*[:\-]?\s*([A-Z])(?:\.|\b)",
    ]
    for pattern in patterns:
        m = re.search(pattern, tail, re.I)
        if m and m.group(1).upper() in options:
            return m.group(1).upper()

    normalized = re.sub(r"[^a-z0-9 ]", " ", tail.lower())
    found: list[str] = []
    for letter, option in options.items():
        option_norm = re.sub(r"[^a-z0-9 ]", " ", option.lower()).strip()
        if option_norm and re.search(
            r"(?:correct answer|answer|correct option)\s+is\s+" + re.escape(option_norm) + r"\b",
            normalized,
        ):
            found.append(letter)
    return found[0] if len(found) == 1 else None


def extract_json(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        text = re.sub(r"^json\s*", "", text.strip(), flags=re.I)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default


def clean_letter(value: Any, valid_options: set[str]) -> str:
    letter = str(value or "").strip().upper()[:1]
    return letter if letter in valid_options else ""


def build_question(row: pd.Series) -> str:
    question = str(row.get("instruction", ""))
    additional = str(row.get("input", "")).strip()
    if additional:
        question += f"\n\nAdditional input:\n{additional}"
    return question


def ensure_fixed_split(
    csv_path: str | Path,
    root: str | Path,
    development_fraction: float = 0.20,
    seed: int = 2026,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], Path]:
    """Create a content-hash-based split once, then protect/reuse it forever."""
    root = Path(root)
    csv_path = Path(csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Put your CSV at data/medical_mcq.csv "
            "with columns instruction, input, output."
        )
    data = pd.read_csv(csv_path).fillna("")
    required = {"instruction", "input", "output"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if len(data) < 2:
        raise ValueError("The fixed 20/80 experiment requires at least 2 rows.")

    canonical = (
        data["instruction"].astype(str)
        + "\x1f"
        + data["input"].astype(str)
        + "\x1f"
        + data["output"].astype(str)
    )
    base_hash = canonical.map(lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest())
    occurrence = base_hash.groupby(base_hash).cumcount().astype(str)
    data["fixed_case_id"] = [
        hashlib.sha256(f"{h}:{n}".encode()).hexdigest() for h, n in zip(base_hash, occurrence)
    ]
    fingerprint = hashlib.sha256("".join(sorted(data.fixed_case_id)).encode()).hexdigest()

    split_root = root / "data" / "fixed_splits" / csv_path.stem
    split_root.mkdir(parents=True, exist_ok=True)
    manifest_path = split_root / "split_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["dataset_fingerprint"] != fingerprint:
            raise RuntimeError(
                f"Dataset content changed. Existing fixed split is protected at {manifest_path}. "
                "Rename the dataset to intentionally create a new split."
            )
        if manifest["seed"] != seed or abs(manifest["development_fraction"] - development_fraction) > 1e-12:
            raise RuntimeError(
                "Existing split manifest uses different seed/fraction. Keep the frozen manifest "
                "or intentionally create a differently named dataset."
            )
    else:
        ranked = data.fixed_case_id.map(lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())
        ordered = data.assign(_rank=ranked).sort_values("_rank")
        n_dev = max(1, round(len(data) * development_fraction))
        n_dev = min(n_dev, len(data) - 1)
        development_ids = ordered.fixed_case_id.iloc[:n_dev].tolist()
        test_ids = ordered.fixed_case_id.iloc[n_dev:].tolist()
        manifest = {
            "source_file": csv_path.name,
            "dataset_fingerprint": fingerprint,
            "seed": seed,
            "development_fraction": development_fraction,
            "n_total": len(data),
            "n_development": len(development_ids),
            "n_test": len(test_ids),
            "development_ids": development_ids,
            "test_ids": test_ids,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    data["reference_letter"] = [
        reference_letter(q, o) for q, o in zip(data.instruction, data.output)
    ]
    development = data[data.fixed_case_id.isin(manifest["development_ids"])].sort_values("fixed_case_id").reset_index(drop=True)
    test = data[data.fixed_case_id.isin(manifest["test_ids"])].sort_values("fixed_case_id").reset_index(drop=True)
    development.to_csv(split_root / "development_20.csv", index=False)
    test.to_csv(split_root / "test_80.csv", index=False)
    return {"development": development, "test": test}, manifest, manifest_path


async def call_openai_json(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 700,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
) -> tuple[dict[str, Any], str, float, dict[str, int]]:
    """Call OpenAI Chat Completions in JSON mode.

    GPT-5 uses max_completion_tokens and reasoning_effort. Temperature is omitted
    for GPT-5-family reasoning models to avoid unsupported sampling settings.
    """
    from openai import AsyncOpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": max_tokens,
    }
    if model.startswith("gpt-5"):
        kwargs["reasoning_effort"] = reasoning_effort or "medium"
    else:
        kwargs["temperature"] = temperature

    started = time.perf_counter()
    response = await client.chat.completions.create(**kwargs)
    latency = time.perf_counter() - started
    raw = response.choices[0].message.content or "{}"
    usage = {
        "input_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
    }
    return extract_json(raw), raw, latency, usage


def usage_cost(model: str, usage: dict[str, int], pricing: dict[str, dict[str, float]] | None = None) -> float:
    pricing = pricing or DEFAULT_PRICING_PER_MILLION
    rate = pricing.get(model)
    if not rate:
        return float("nan")
    return (
        (usage.get("input_tokens", 0) or 0) * rate["input"]
        + (usage.get("output_tokens", 0) or 0) * rate["output"]
    ) / 1_000_000


def metrics(frame: pd.DataFrame, threshold: float = 0.80) -> dict[str, float]:
    frame = frame.copy()
    if frame.empty:
        return {"n": 0}
    valid_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    answered = frame.predicted_letter.isin(valid_letters) & ~frame.abstain.astype(bool)
    correct = frame.correct.fillna(False).astype(bool)
    confidence = pd.to_numeric(frame.confidence, errors="coerce").fillna(0).clip(0, 1)
    high = confidence >= threshold
    y = correct.astype(float)

    labels = sorted(frame.reference_letter.dropna().astype(str).unique())
    f1s: list[float] = []
    for label in labels:
        tp = ((frame.reference_letter == label) & (frame.predicted_letter == label) & ~frame.abstain.astype(bool)).sum()
        fp = ((frame.reference_letter != label) & (frame.predicted_letter == label) & ~frame.abstain.astype(bool)).sum()
        fn = ((frame.reference_letter == label) & ((frame.predicted_letter != label) | frame.abstain.astype(bool))).sum()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)

    ece = 0.0
    edges = np.linspace(0, 1, 11)
    for low, high_bin in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < (high_bin if high_bin < 1 else 1.000001))
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(confidence[mask].mean()))

    out: dict[str, float] = {
        "n": float(len(frame)),
        "accuracy": float(correct.mean()),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "answered_accuracy": float(correct[answered].mean()) if answered.any() else 0.0,
        "coverage": float(answered.mean()),
        "abstention_rate": float(frame.abstain.astype(bool).mean()),
        "invalid_answer_rate": float((~frame.predicted_letter.isin(valid_letters)).mean()),
        "high_confidence_error_rate": float((high & ~correct & answered).mean()),
        "brier_score": float(np.mean((confidence - y) ** 2)),
        "expected_calibration_error": float(ece),
        "api_error_rate": float(frame.api_error.astype(bool).mean()) if "api_error" in frame else 0.0,
        "mean_latency_seconds": float(pd.to_numeric(frame.latency_seconds, errors="coerce").mean()) if "latency_seconds" in frame else float("nan"),
        "reference_extraction_rate": float(frame.reference_letter.notna().mean()),
    }
    for col in ["input_tokens", "output_tokens", "total_tokens", "estimated_cost_usd", "total_calls", "gpt5_call_count", "n_specialists"]:
        if col in frame:
            out[f"mean_{col}"] = float(pd.to_numeric(frame[col], errors="coerce").mean())
    for col in ["critic_used", "audit_rejected"]:
        if col in frame:
            out[f"{col}_rate"] = float(frame[col].astype(bool).mean())
    return out


SINGLE_PROMPT = """You answer medical multiple-choice benchmark questions.
Use only the supplied stem and options. Select exactly one supplied option unless the evidence is genuinely insufficient.
Do not invent patient facts. Return JSON only:
{
  "answer_letter":"A",
  "confidence":0.0,
  "abstain":false,
  "explanation":"brief evidence-based reason"
}
Confidence must be between 0 and 1. Research benchmark only, not patient-care advice."""


ROUTER_PROMPT_TEMPLATE = """You are the routing supervisor in a medical multiple-choice research system.
Your ONLY task is to select the clinical expertise required. Do not solve the question and do not output an answer letter.

Select the SMALLEST nonredundant panel of {min_specialists}-{max_specialists} true clinical specialties or subspecialties.
Use domain experts, not generic roles. Examples include Pediatric Dermatology, Pediatric Infectious Disease, Cardiology,
Neurology, Emergency Medicine, Obstetrics and Gynecology, Psychiatry, Clinical Pharmacology, Nephrology, Hematology,
Pathology, Radiology, Rheumatology, Endocrinology, Gastroenterology, Pulmonology, Surgery, and Medical Genetics.

Routing rules:
- 1 specialist: one dominant domain; another specialty would add little.
- 2 specialists: lead domain plus a genuinely relevant adjacent domain (age group, infection, pharmacology, surgery, genetics, etc.).
- 3 specialists: important alternatives or management issues cross three distinct domains.
- 4 specialists: only unusually ambiguous or highly cross-disciplinary cases.
- Do not use near-duplicate specialists merely to create more votes.
- Choose one lead specialty with the most direct expertise for the decisive clue.
- `n_specialists` must equal the number of objects in `specialists`.

Return JSON only:
{
  "case_domains":["domain"],
  "complexity":"low|moderate|high",
  "routing_confidence":0.0,
  "n_specialists":1,
  "lead_specialty":"specialty",
  "specialists":[
    {"specialty":"specialty","reason":"why this domain is needed","focus":"specific clue or distinction to assess"}
  ]
}"""


SPECIALIST_PROMPT_TEMPLATE = """You are the {specialty} domain expert in a coordinated clinical-reasoning benchmark.
Independently solve the MCQ from your actual specialty perspective. You cannot see other specialists' answers.

Requirements:
- Choose only from supplied options.
- Prioritize decisive clinical features over keyword matching.
- Do not invent history, examination, tests, epidemiology, or mechanisms not supported by the stem.
- Consider age, time course, morphology/distribution, exposure, medication, family/household, and setting when relevant.
- Explain why the leading alternative is less supported when useful.
- If your specialty cannot support a choice, abstain rather than manufacture certainty.

Return JSON only:
{
  "answer_letter":"A",
  "confidence":0.0,
  "abstain":false,
  "key_evidence":["short stem clue"],
  "reasoning_summary":"concise domain-specific reasoning",
  "strongest_alternative":"B",
  "alternative_rejected_because":"brief reason",
  "uncertainties":["remaining uncertainty"]
}"""


CRITIC_PROMPT = """You are a clinical conflict critic. Review independent specialist opinions against the original MCQ.
You are not another voter and must not choose an answer merely by majority.
Identify only: disagreement, unsupported assumptions, missed decisive clues, incorrect use of a domain, or confidence that is not justified.
Return JSON only:
{
  "conflict_present":true,
  "issues":[{"agent":"specialty","issue":"specific problem","severity":"low|moderate|high"}],
  "most_relevant_specialty":"specialty",
  "resolution_guidance":"concise guidance for adjudication",
  "critic_confidence":0.0
}"""


GPT5_JUDGE_PROMPT = """You are the final adjudicating clinician in a medical MCQ research system.
This is the ONLY GPT-5 adjudication call for this case.

Review the original stem yourself, then integrate the router assignment, independent domain-specialist opinions, and critic review if present.
Select exactly one supplied option or abstain. Weight opinions by domain relevance and evidence quality, not majority vote.
Do not assume the lead specialist is correct. Do not invent facts. Resolve disagreements explicitly but concisely.

Return JSON only:
{
  "answer_letter":"A",
  "confidence":0.0,
  "abstain":false,
  "explanation":"concise evidence-based adjudication",
  "decisive_evidence":["short stem clue"],
  "dissent_summary":"why a conflicting opinion was rejected, or empty string"
}"""


AUDITOR_PROMPT = """You are an independent clinical answer auditor. Audit the proposed final judge answer against the original MCQ.
You are not allowed to replace the answer with your own answer and you cannot request another GPT-5 call.

Check whether:
1. the selected answer is a supplied option;
2. the explanation is supported by the stem;
3. no important stem fact is contradicted or ignored;
4. no unsupported patient fact was introduced;
5. confidence is proportionate to the evidence.

Use `approve` when acceptable, `cap` when the answer is supportable but confidence is too high, and `reject` only for a material unsupported/contradictory answer. A reject causes abstention.
Return JSON only:
{
  "decision":"approve|cap|reject",
  "confidence_cap":0.0,
  "issues":["specific audit issue"],
  "audit_explanation":"brief explanation"
}"""


def sanitize_route(route: dict[str, Any], min_specialists: int = 1, max_specialists: int = 4) -> tuple[list[dict[str, str]], str, int]:
    raw = route.get("specialists", []) if isinstance(route, dict) else []
    requested = len(raw)
    try:
        requested = int(route.get("n_specialists", len(raw)))
    except Exception:
        requested = len(raw)
    requested = max(min_specialists, min(max_specialists, requested))

    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        specialty = str(item.get("specialty", "")).strip()
        key = re.sub(r"\s+", " ", specialty.lower())
        if not specialty or key in seen:
            continue
        seen.add(key)
        cleaned.append({
            "specialty": specialty,
            "reason": str(item.get("reason", "")).strip(),
            "focus": str(item.get("focus", "")).strip(),
        })
        if len(cleaned) >= requested:
            break

    fallback_domains = [
        "Internal Medicine", "Emergency Medicine", "Clinical Pharmacology", "Infectious Disease"
    ]
    for fallback in fallback_domains:
        if len(cleaned) >= requested:
            break
        if fallback.lower() not in seen:
            cleaned.append({
                "specialty": fallback,
                "reason": "Fallback added because router returned too few valid unique specialties.",
                "focus": "Overall evidence-based interpretation",
            })
            seen.add(fallback.lower())

    cleaned = cleaned[:max_specialists]
    lead = str(route.get("lead_specialty", "")).strip() if isinstance(route, dict) else ""
    if not cleaned:
        cleaned = [{"specialty": "Internal Medicine", "reason": "Router fallback", "focus": "Overall interpretation"}]
    if lead.lower() not in {x["specialty"].lower() for x in cleaned}:
        lead = cleaned[0]["specialty"]
    return cleaned, lead, len(cleaned)


def _aggregate_usage(items: Iterable[dict[str, int]]) -> dict[str, int]:
    total = {"input_tokens": 0, "output_tokens": 0}
    for item in items:
        total["input_tokens"] += int(item.get("input_tokens", 0) or 0)
        total["output_tokens"] += int(item.get("output_tokens", 0) or 0)
    return total


async def run_single_agent(
    df: pd.DataFrame,
    run_name: str,
    model: str,
    run_split: str,
    output_dir: str | Path,
    max_tokens: int = 700,
    temperature: float = 0.0,
    reasoning_effort: str = "medium",
    pricing: dict[str, dict[str, float]] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Run one one-call-per-case single-agent baseline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{run_name}_{run_split}_predictions.csv"

    existing = pd.DataFrame()
    completed: set[str] = set()
    if resume and pred_path.exists():
        existing = pd.read_csv(pred_path).fillna("")
        completed = set(existing.fixed_case_id.astype(str))
        print(f"Resuming {run_name}: {len(completed)} completed cases found.")

    records: list[dict[str, Any]] = []
    for case_idx, (_, row) in enumerate(df.iterrows(), start=1):
        if str(row.fixed_case_id) in completed:
            continue
        question = build_question(row)
        options = parse_options(question)
        valid_options = set(options)
        started = time.perf_counter()
        try:
            parsed, raw, _, usage = await call_openai_json(
                model=model,
                system=SINGLE_PROMPT,
                user=question,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
            letter = clean_letter(parsed.get("answer_letter"), valid_options)
            confidence = clamp01(parsed.get("confidence", 0))
            abstain = bool(parsed.get("abstain", False)) or not letter
            api_error = False
            error_message = ""
        except Exception as exc:
            raw = ""
            usage = {"input_tokens": 0, "output_tokens": 0}
            letter = ""
            confidence = 0.0
            abstain = True
            api_error = True
            error_message = str(exc)

        records.append({
            "fixed_case_id": row.fixed_case_id,
            "split": run_split,
            "system": run_name,
            "model": model,
            "reference_letter": row.reference_letter,
            "predicted_letter": letter,
            "confidence": confidence,
            "abstain": abstain,
            "correct": bool(letter == row.reference_letter and not abstain),
            "latency_seconds": time.perf_counter() - started,
            "api_error": api_error,
            "error": error_message,
            "raw_response": raw,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["input_tokens"] + usage["output_tokens"],
            "total_calls": 1,
            "gpt5_call_count": 1 if model.startswith("gpt-5") and not api_error else 0,
            "estimated_cost_usd": usage_cost(model, usage, pricing),
        })
        if case_idx % 10 == 0:
            print(f"{run_name}: completed {case_idx}/{len(df)}")
        if len(records) % 10 == 0:
            partial = pd.concat([existing, pd.DataFrame(records)], ignore_index=True) if not existing.empty else pd.DataFrame(records)
            partial.to_csv(pred_path, index=False)

    result = pd.concat([existing, pd.DataFrame(records)], ignore_index=True) if not existing.empty else pd.DataFrame(records)
    if not result.empty:
        result = result.drop_duplicates("fixed_case_id", keep="last").sort_values("fixed_case_id").reset_index(drop=True)
    result.to_csv(pred_path, index=False)
    return result


async def run_multi_agent(
    df: pd.DataFrame,
    run_name: str,
    run_split: str,
    output_dir: str | Path,
    router_model: str = "gpt-4o-mini",
    specialist_model: str = "gpt-4o-mini",
    critic_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-5",
    auditor_model: str = "gpt-4o-mini",
    min_specialists: int = 1,
    max_specialists: int = 4,
    router_low_confidence: float = 0.70,
    specialist_low_confidence: float = 0.65,
    max_tokens_small: int = 700,
    max_tokens_judge: int = 900,
    temperature: float = 0.0,
    judge_reasoning_effort: str = "medium",
    pricing: dict[str, dict[str, float]] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Run hybrid dynamic multi-agent MCQ evaluation with exactly one GPT-5 judge call per successful case."""
    if not judge_model.startswith("gpt-5"):
        raise ValueError("The strong Judge must be a GPT-5 model for this experiment.")
    if any(m.startswith("gpt-5") for m in [router_model, specialist_model, critic_model, auditor_model]):
        raise ValueError("GPT-5 is restricted to the Judge role only in this experiment.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{run_name}_{run_split}_predictions.csv"
    trace_path = output_dir / f"{run_name}_{run_split}_traces.jsonl"

    existing = pd.DataFrame()
    completed: set[str] = set()
    if resume and pred_path.exists():
        existing = pd.read_csv(pred_path).fillna("")
        completed = set(existing.fixed_case_id.astype(str))
        print(f"Resuming {run_name}: {len(completed)} completed cases found.")

    router_prompt = ROUTER_PROMPT_TEMPLATE.replace("{min_specialists}", str(min_specialists)).replace("{max_specialists}", str(max_specialists))
    records: list[dict[str, Any]] = []

    for case_idx, (_, row) in enumerate(df.iterrows(), start=1):
        if str(row.fixed_case_id) in completed:
            continue

        started = time.perf_counter()
        question = build_question(row)
        options = parse_options(question)
        valid_options = set(options)
        trace: dict[str, Any] = {
            "fixed_case_id": row.fixed_case_id,
            "router": None,
            "specialists": [],
            "critic": None,
            "judge": None,
            "audit": None,
        }
        api_error = False
        error_message = ""
        usage_events: list[tuple[str, str, dict[str, int]]] = []
        gpt5_call_count = 0
        total_calls = 0
        use_critic = False
        audit_rejected = False
        route_conf = 0.0
        complexity = ""
        specialists: list[dict[str, str]] = []
        lead = ""

        try:
            route, raw, latency, usage = await call_openai_json(
                router_model, router_prompt, question, max_tokens_small, temperature
            )
            total_calls += 1
            usage_events.append(("router", router_model, usage))
            specialists, lead, n_specialists = sanitize_route(route, min_specialists, max_specialists)
            route_conf = clamp01(route.get("routing_confidence", 0))
            complexity = str(route.get("complexity", "")).lower()
            trace["router"] = {
                "model": router_model,
                "response": route,
                "selected": specialists,
                "lead_specialty": lead,
                "latency_seconds": latency,
                "usage": usage,
            }

            async def run_specialist(item: dict[str, str]):
                system = SPECIALIST_PROMPT_TEMPLATE.replace("{specialty}", item["specialty"])
                content = question + "\n\nROUTER ASSIGNMENT:\n" + json.dumps(item)
                parsed, sraw, slatency, susage = await call_openai_json(
                    specialist_model, system, content, max_tokens_small, temperature
                )
                parsed["specialty"] = item["specialty"]
                parsed["answer_letter"] = clean_letter(parsed.get("answer_letter"), valid_options)
                parsed["confidence"] = clamp01(parsed.get("confidence", 0))
                parsed["abstain"] = bool(parsed.get("abstain", False)) or not parsed["answer_letter"]
                return parsed, sraw, slatency, susage, item

            spec_runs = await asyncio.gather(*(run_specialist(item) for item in specialists))
            opinions: list[dict[str, Any]] = []
            for parsed, sraw, slatency, susage, item in spec_runs:
                total_calls += 1
                usage_events.append(("specialist", specialist_model, susage))
                opinions.append(parsed)
                trace["specialists"].append({
                    "assignment": item,
                    "model": specialist_model,
                    "response": parsed,
                    "latency_seconds": slatency,
                    "usage": susage,
                })

            answered_letters = [o["answer_letter"] for o in opinions if not o.get("abstain") and o.get("answer_letter")]
            disagree = len(set(answered_letters)) > 1
            low_specialist_conf = any(clamp01(o.get("confidence", 0)) < specialist_low_confidence for o in opinions)
            use_critic = bool(
                disagree
                or low_specialist_conf
                or route_conf < router_low_confidence
                or complexity == "high"
            )
            critique: dict[str, Any] = {}
            if use_critic:
                critic_input = (
                    question
                    + "\n\nROUTER:\n"
                    + json.dumps({"lead_specialty": lead, "specialists": specialists})
                    + "\n\nSPECIALIST OPINIONS:\n"
                    + json.dumps(opinions)
                )
                critique, craw, clatency, cusage = await call_openai_json(
                    critic_model, CRITIC_PROMPT, critic_input, max_tokens_small, temperature
                )
                total_calls += 1
                usage_events.append(("critic", critic_model, cusage))
                trace["critic"] = {
                    "model": critic_model,
                    "response": critique,
                    "latency_seconds": clatency,
                    "usage": cusage,
                    "trigger": {
                        "disagreement": disagree,
                        "low_specialist_confidence": low_specialist_conf,
                        "low_router_confidence": route_conf < router_low_confidence,
                        "high_complexity": complexity == "high",
                    },
                }

            # EXACTLY ONE strong-model call. No downstream path is allowed to call GPT-5 again.
            judge_input = (
                question
                + "\n\nROUTER DECISION:\n"
                + json.dumps({"lead_specialty": lead, "specialists": specialists, "complexity": complexity})
                + "\n\nSPECIALIST OPINIONS:\n"
                + json.dumps(opinions)
                + ("\n\nCRITIC REVIEW:\n" + json.dumps(critique) if use_critic else "")
            )
            judged, jraw, jlatency, jusage = await call_openai_json(
                judge_model,
                GPT5_JUDGE_PROMPT,
                judge_input,
                max_tokens_judge,
                temperature,
                reasoning_effort=judge_reasoning_effort,
            )
            gpt5_call_count += 1
            total_calls += 1
            usage_events.append(("judge", judge_model, jusage))
            judged["answer_letter"] = clean_letter(judged.get("answer_letter"), valid_options)
            judged["confidence"] = clamp01(judged.get("confidence", 0))
            judged["abstain"] = bool(judged.get("abstain", False)) or not judged["answer_letter"]
            trace["judge"] = {
                "model": judge_model,
                "response": judged,
                "latency_seconds": jlatency,
                "usage": jusage,
            }

            audit_input = (
                question
                + "\n\nPROPOSED GPT-5 JUDGE ANSWER:\n"
                + json.dumps(judged)
                + "\n\nDOMAIN OPINIONS FOR CONTEXT:\n"
                + json.dumps(opinions)
            )
            audit, araw, alatency, ausage = await call_openai_json(
                auditor_model, AUDITOR_PROMPT, audit_input, max_tokens_small, temperature
            )
            total_calls += 1
            usage_events.append(("auditor", auditor_model, ausage))
            trace["audit"] = {
                "model": auditor_model,
                "response": audit,
                "latency_seconds": alatency,
                "usage": ausage,
            }

            decision = str(audit.get("decision", "approve")).lower().strip()
            if decision not in {"approve", "cap", "reject"}:
                decision = "reject"
            audit_rejected = decision == "reject"

            if decision == "reject":
                final = {
                    "answer_letter": "",
                    "confidence": 0.0,
                    "abstain": True,
                    "explanation": "Abstained because the independent auditor rejected the proposed answer.",
                }
            else:
                final = dict(judged)
                if decision == "cap":
                    final["confidence"] = min(
                        clamp01(final.get("confidence", 0)),
                        clamp01(audit.get("confidence_cap", final.get("confidence", 0))),
                    )

            if gpt5_call_count != 1:
                raise RuntimeError(f"GPT-5 invariant violated: expected exactly 1 call, observed {gpt5_call_count}.")

            letter = clean_letter(final.get("answer_letter"), valid_options)
            confidence = clamp01(final.get("confidence", 0))
            abstain = bool(final.get("abstain", False)) or not letter
            raw_response = json.dumps(final)

        except Exception as exc:
            letter = ""
            confidence = 0.0
            abstain = True
            raw_response = ""
            api_error = True
            error_message = str(exc)
            trace["error"] = error_message

        role_usage: dict[str, dict[str, int]] = {}
        estimated_cost = 0.0
        any_missing_price = False
        for role, model, usage in usage_events:
            role_usage.setdefault(role, {"input_tokens": 0, "output_tokens": 0})
            role_usage[role]["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
            role_usage[role]["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
            c = usage_cost(model, usage, pricing)
            if math.isnan(c):
                any_missing_price = True
            else:
                estimated_cost += c
        total_usage = _aggregate_usage([u for _, _, u in usage_events])

        trace["role_usage"] = role_usage
        trace["gpt5_call_count"] = gpt5_call_count
        trace["total_calls"] = total_calls
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

        records.append({
            "fixed_case_id": row.fixed_case_id,
            "split": run_split,
            "system": run_name,
            "model": f"dynamic small-agents + exactly-one {judge_model} judge",
            "reference_letter": row.reference_letter,
            "predicted_letter": letter,
            "confidence": confidence,
            "abstain": abstain,
            "correct": bool(letter == row.reference_letter and not abstain),
            "latency_seconds": time.perf_counter() - started,
            "api_error": api_error,
            "error": error_message,
            "raw_response": raw_response,
            "trace": json.dumps(trace, ensure_ascii=False),
            "n_specialists": len(specialists),
            "lead_specialty": lead,
            "router_confidence": route_conf,
            "router_complexity": complexity,
            "critic_used": use_critic,
            "audit_rejected": audit_rejected,
            "input_tokens": total_usage["input_tokens"],
            "output_tokens": total_usage["output_tokens"],
            "total_tokens": total_usage["input_tokens"] + total_usage["output_tokens"],
            "total_calls": total_calls,
            "gpt5_call_count": gpt5_call_count,
            "estimated_cost_usd": float("nan") if any_missing_price else estimated_cost,
        })

        if case_idx % 5 == 0:
            print(f"{run_name}: completed {case_idx}/{len(df)}")
        if len(records) % 5 == 0:
            partial = pd.concat([existing, pd.DataFrame(records)], ignore_index=True) if not existing.empty else pd.DataFrame(records)
            partial.to_csv(pred_path, index=False)

    result = pd.concat([existing, pd.DataFrame(records)], ignore_index=True) if not existing.empty else pd.DataFrame(records)
    if not result.empty:
        result = result.drop_duplicates("fixed_case_id", keep="last").sort_values("fixed_case_id").reset_index(drop=True)
    result.to_csv(pred_path, index=False)
    return result


def summarize_system(result: pd.DataFrame, system_name: str, split: str, threshold: float = 0.80) -> pd.DataFrame:
    return pd.DataFrame([{"system": system_name, "split": split, **metrics(result, threshold)}])


def _exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p using Binomial(n=b+c, p=.5), no scipy dependency."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return min(1.0, 2.0 * tail)


def paired_accuracy_comparison(
    multi: pd.DataFrame,
    baseline: pd.DataFrame,
    baseline_name: str,
    n_boot: int = 5000,
    seed: int = 2026,
) -> dict[str, Any]:
    cols = ["fixed_case_id", "correct"]
    a = multi[cols].rename(columns={"correct": "multi_correct"})
    b = baseline[cols].rename(columns={"correct": "base_correct"})
    paired = a.merge(b, on="fixed_case_id", how="inner")
    paired["multi_correct"] = paired.multi_correct.astype(bool)
    paired["base_correct"] = paired.base_correct.astype(bool)
    if paired.empty:
        raise ValueError(f"No paired cases for {baseline_name}")
    diff = paired.multi_correct.astype(float) - paired.base_correct.astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff.to_numpy()[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    b_only = int((~paired.multi_correct & paired.base_correct).sum())
    m_only = int((paired.multi_correct & ~paired.base_correct).sum())
    return {
        "comparison": f"multi - {baseline_name}",
        "n_paired": len(paired),
        "multi_accuracy": float(paired.multi_correct.mean()),
        "baseline_accuracy": float(paired.base_correct.mean()),
        "accuracy_difference": float(diff.mean()),
        "bootstrap_95ci_low": float(lo),
        "bootstrap_95ci_high": float(hi),
        "multi_only_correct": m_only,
        "baseline_only_correct": b_only,
        "mcnemar_exact_p": _exact_mcnemar_p(b_only, m_only),
    }


def merge_for_case_comparison(multi: pd.DataFrame, baseline: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    keep = [
        "fixed_case_id", "reference_letter", "predicted_letter", "correct", "confidence",
        "abstain", "latency_seconds", "total_tokens", "estimated_cost_usd"
    ]
    mcols = [c for c in keep if c in multi.columns]
    bcols = [c for c in keep if c in baseline.columns]
    m = multi[mcols].copy().add_prefix("multi_").rename(columns={"multi_fixed_case_id": "fixed_case_id"})
    b = baseline[bcols].copy().add_prefix(f"{baseline_name}_").rename(columns={f"{baseline_name}_fixed_case_id": "fixed_case_id"})
    return m.merge(b, on="fixed_case_id", how="inner")


# -----------------------------------------------------------------------------
# Standalone command-line entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    async def _standalone_main():
        ap = argparse.ArgumentParser(description="Fixed 20/80 single vs hybrid multi-agent MCQ experiment")
        ap.add_argument("--csv", default="data/medical_mcq.csv")
        ap.add_argument("--split", choices=["development", "test"], default="development")
        ap.add_argument("--max-cases", default="50", help="integer or 'all'")
        ap.add_argument("--seed", type=int, default=2026)
        ap.add_argument("--development-fraction", type=float, default=0.20)
        ap.add_argument("--no-resume", action="store_true")
        args = ap.parse_args()

        script_root = Path(__file__).resolve().parent
        # If this file is inside tools/, use the repository parent; otherwise use its own folder.
        root = script_root.parent if script_root.name == "tools" else script_root
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = root / csv_path
        max_cases = None if args.max_cases.lower() == "all" else int(args.max_cases)
        splits, manifest, manifest_path = ensure_fixed_split(csv_path, root, args.development_fraction, args.seed)
        df = splits[args.split].copy()
        if max_cases is not None:
            df = df.head(max_cases).copy()
        if df.reference_letter.isna().any():
            raise RuntimeError("Reference extraction failed for one or more cases. Fix before model evaluation.")

        out = root / "results" / "mcq_comparison"
        out.mkdir(parents=True, exist_ok=True)
        resume = not args.no_resume
        print("Manifest:", manifest_path)
        print("Running:", args.split, len(df), "cases")

        single_mini = await run_single_agent(df, "single_gpt4o_mini", "gpt-4o-mini", args.split, out, resume=resume)
        single_gpt5 = await run_single_agent(df, "single_gpt5", "gpt-5", args.split, out, reasoning_effort="medium", resume=resume)
        multi = await run_multi_agent(
            df, "multi_dynamic_one_gpt5_judge", args.split, out,
            router_model="gpt-4o-mini", specialist_model="gpt-4o-mini", critic_model="gpt-4o-mini",
            judge_model="gpt-5", auditor_model="gpt-4o-mini", min_specialists=1, max_specialists=4,
            judge_reasoning_effort="medium", resume=resume,
        )

        summary = pd.DataFrame([
            {"system":"single_gpt4o_mini", **metrics(single_mini)},
            {"system":"single_gpt5", **metrics(single_gpt5)},
            {"system":"multi_dynamic_one_gpt5_judge", **metrics(multi)},
        ])
        summary.to_csv(out/f"comparison_summary_{args.split}.csv", index=False)
        paired = pd.DataFrame([
            paired_accuracy_comparison(multi, single_mini, "single_gpt4o_mini"),
            paired_accuracy_comparison(multi, single_gpt5, "single_gpt5"),
        ])
        paired.to_csv(out/f"paired_accuracy_tests_{args.split}.csv", index=False)

        successful = multi[~multi.api_error.astype(bool)]
        assert (successful.gpt5_call_count == 1).all(), "GPT-5 call invariant failed"
        print("\nSUMMARY\n", summary.to_string(index=False))
        print("\nPAIRED TESTS\n", paired.to_string(index=False))
        print("\nValidated: exactly one GPT-5 call per successful multi-agent case.")

    asyncio.run(_standalone_main())

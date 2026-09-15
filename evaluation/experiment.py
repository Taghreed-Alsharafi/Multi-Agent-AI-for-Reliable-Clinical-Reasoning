"""High-level experiment API intended for Jupyter and scripts."""

from __future__ import annotations
import asyncio
from pathlib import Path
from .config import ExperimentConfig
from .io import append_jsonl, read_cases, read_predictions, write_json
from .metrics import classification_metrics, evaluate_predictions, robustness_by_variant
from .runners import run_multi_agent, run_single_agent
from .schema import PredictionRecord

async def run_experiment_async(config: ExperimentConfig) -> Path:
    cases = read_cases(config.dataset)
    output_dir = Path(config.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    if predictions_path.exists() and not config.resume:
        predictions_path.unlink()
    existing = read_predictions(predictions_path) if predictions_path.exists() and config.resume else []
    completed = {(p.case_id, p.system, p.trace.get("repetition")) for p in existing if p.status == "ok"}
    systems = [s for s in config.single_agents if s.enabled]
    role_models = config.multi_agent_models.model_dump()

    for repetition in range(config.repetitions):
        for case in cases:
            jobs = ([('multi_agent', None)] if config.multi_agent_enabled else []) + [(s.name, s) for s in systems]
            for system_name, system in jobs:
                if (case.case_id, system_name, repetition) in completed: continue
                try:
                    if system is None:
                        pred = await run_multi_agent(case, role_models)
                    else:
                        pred = await run_single_agent(case, system.provider, system.model,
                                                      config.temperature, system.name)
                    pred.trace["repetition"] = repetition
                except Exception as exc:
                    provider = "openai" if system is None else system.provider
                    model = "role-specific" if system is None else system.model
                    pred = PredictionRecord(run_id="error", case_id=case.case_id,
                        system=system_name, provider=provider, model=model, answer="",
                        latency_seconds=0, status="error", error=str(exc),
                        trace={"repetition": repetition})
                append_jsonl(predictions_path, [pred])
    write_json(output_dir / "resolved_config.json", config.model_dump())
    return predictions_path

def run_experiment(config: ExperimentConfig) -> Path:
    """Synchronous entry point. In an async notebook, await run_experiment_async."""
    return asyncio.run(run_experiment_async(config))

def score_experiment(config: ExperimentConfig) -> dict[str, object]:
    cases = read_cases(config.dataset); output_dir = Path(config.output_dir)
    predictions = read_predictions(output_dir / "predictions.jsonl")
    detailed, summary = evaluate_predictions(cases, predictions,
        bootstrap_samples=config.metrics.bootstrap_samples,
        confidence_level=config.metrics.confidence_level)
    results = {"summary": summary, "per_case": detailed,
        "classification": classification_metrics(cases, predictions,
            positive_label=config.metrics.positive_label),
        "robustness": robustness_by_variant(cases, predictions)}
    write_json(output_dir / "metrics.json", results)
    return results

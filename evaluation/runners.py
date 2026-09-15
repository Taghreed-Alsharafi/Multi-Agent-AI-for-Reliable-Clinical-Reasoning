from __future__ import annotations
import time, uuid
import re
from .providers import call_anthropic, call_openai
from .schema import EvaluationCase, PredictionRecord, StandardAnswer

def _extract_label(answer: str, allowed_labels: list[str]) -> str | None:
    matches = [label for label in allowed_labels
               if re.search(rf"(?<!\w){re.escape(label)}(?!\w)", answer, re.IGNORECASE)]
    return matches[0] if len(matches) == 1 else None

def _multi_answer(result: object, case: EvaluationCase) -> StandardAnswer:
    safety = result.safety_report
    # Use the specialists' source quotations for the shared evidence metric.
    # ``verified_findings`` contains paraphrased claims, not necessarily quotes.
    quotes = [
        str(quote)
        for opinion in result.specialist_opinions
        for quote in opinion.get("evidence_quotes", [])
    ]
    return StandardAnswer(answer=str(safety.get("final_summary","")),
        evidence_quotes=quotes,
        confidence=float(safety.get("overall_confidence",0)),
        insufficient_information=not bool(safety.get("is_safe",False)) and not safety.get("verified_findings"),
        predicted_label=_extract_label(str(safety.get("final_summary", "")), case.allowed_labels),
        safety_flags=[str(x.get("issue", x)) if isinstance(x, dict) else str(x)
                      for x in safety.get("flagged_issues", [])],
        is_safe=bool(safety.get("is_safe", False)))

async def run_multi_agent(case: EvaluationCase, role_models: dict[str, str]) -> PredictionRecord:
    from orchestrator.pipeline import AgentPipeline
    question = case.question
    if case.allowed_labels:
        question += ("\n\nClassification requirement: select exactly one of these labels: "
                     f"{case.allowed_labels}. State it as 'Predicted label: <label>'. "
                     "If evidence is insufficient, use the designated unknown label or abstain.")
    started=time.perf_counter(); result=await AgentPipeline(role_models=role_models).run(question,case.documents)
    return PredictionRecord(run_id=str(uuid.uuid4()),case_id=case.case_id,
        system="multi_agent",provider="openai",model="role-specific",
        latency_seconds=time.perf_counter()-started,**_multi_answer(result, case).model_dump(),
        trace={"role_models": role_models, "pipeline": result.model_dump()})

async def run_single_agent(case: EvaluationCase, provider: str, model: str,
                           temperature: float=0, system_name: str | None = None) -> PredictionRecord:
    calls={"openai":call_openai,"anthropic":call_anthropic}
    if provider not in calls: raise ValueError(f"Unknown provider: {provider}")
    result=await calls[provider](case,model,temperature)
    return PredictionRecord(run_id=str(uuid.uuid4()),case_id=case.case_id,
        system=system_name or f"single_{provider}",provider=provider,model=model,
        latency_seconds=result.latency_seconds,input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,**result.answer.model_dump(),trace={"raw_response":result.raw})

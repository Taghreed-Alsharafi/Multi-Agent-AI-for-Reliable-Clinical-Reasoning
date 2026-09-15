"""Live thesis runners built on the repository's existing provider clients."""
from __future__ import annotations
import asyncio, json, os, re
from pathlib import Path
import pandas as pd
from evaluation.providers import call_openai, call_anthropic
from evaluation.schema import EvaluationCase
from .checkpoint import load_completed, append_checkpoint
from .reproducibility import assignment

def _concurrency() -> int:
    try:
        return max(1, int(os.environ.get("THESIS_CONCURRENCY", "6")))
    except (TypeError, ValueError):
        return 6

async def _bounded_gather(make_coros: list):
    """Run coroutine factories with a bounded semaphore, preserving input order.

    Concurrency only changes wall-clock time: every case is keyed by case_id,
    calls run at temperature 0, and checkpoint writes happen inside each task
    (no await between compute and append, so appends never interleave).
    """
    sem = asyncio.Semaphore(_concurrency())
    async def _run(factory):
        async with sem:
            return await factory()
    return await asyncio.gather(*[_run(f) for f in make_coros])

def _label(answer: str, allowed: list[str]) -> str:
    m=re.search(r"(?:predicted\s+label|final\s+answer|answer)\s*[:=-]\s*([A-H])", answer, re.I)
    if m and m.group(1).upper() in allowed: return m.group(1).upper()
    hits=[x for x in allowed if re.search(rf"(?<!\w){re.escape(x)}(?!\w)",answer,re.I)]
    return hits[0].upper() if len(hits)==1 else ""

def _case(row) -> EvaluationCase:
    question=str(row.get("instruction", "")); extra=str(row.get("input", ""))
    if extra: question += "\n\n" + extra
    letters=sorted(set(re.findall(r"(?m)^\s*([A-H])[.)/:]", question, re.I)))
    return EvaluationCase(case_id=str(row.case_id),question=question,documents=[],reference_answer=str(row.gold),allowed_labels=letters or list("ABCD"))

async def run_single_live(cases: pd.DataFrame, provider: str, model: str, checkpoint: Path, system: str) -> pd.DataFrame:
    fn=call_openai if provider=="openai" else call_anthropic; done=load_completed(checkpoint)
    async def handle(row):
        cid=str(row.case_id)
        if cid in done and done[cid].get("status") == "success" and done[cid].get("prediction"):
            return done[cid]
        try:
            result=await fn(_case(row),model,0); label=_label(result.answer.predicted_label or result.answer.answer,_case(row).allowed_labels)
            record={"case_id":cid,"prediction":label,"critical_safety_error":False,"status":"success","system":system,"provider":provider,"model":model,"confidence":result.answer.confidence,"latency_seconds":result.latency_seconds,"input_tokens":result.input_tokens,"output_tokens":result.output_tokens}
        except Exception as exc:
            text=str(exc); status="parse_error" if "JSON" in text or "parse" in text.lower() else "api_error"
            record={"case_id":cid,"prediction":"","critical_safety_error":False,"status":status,"system":system,"provider":provider,"model":model,"error":text}
        append_checkpoint(checkpoint,record); return record
    rows=await _bounded_gather([(lambda r=row: handle(r)) for _,row in cases.iterrows()])
    return pd.DataFrame(rows)

async def _panel_outputs(row, config):
    """Run one case's agent panel concurrently; return (assignments, outputs)."""
    cid=str(row.case_id); allowed=_case(row).allowed_labels
    assigns=[assignment(cid,f"agent_{i+1}",config.global_seed,config.gpt_model,config.claude_model) for i in range(config.n_agents)]
    async def one(a):
        try:
            fn=call_openai if a["assigned_provider"]=="openai" else call_anthropic; result=await fn(_case(row),a["assigned_model"],0)
            answer=_label(result.answer.predicted_label or result.answer.answer,allowed)
            return {"agent_role":a["agent_role"],"provider":a["assigned_provider"],"model":a["assigned_model"],"answer":answer,"confidence":result.answer.confidence,"status":"success"}
        except Exception as exc:
            return {"agent_role":a["agent_role"],"provider":a["assigned_provider"],"model":a["assigned_model"],"answer":"","confidence":0,"status":"api_error","error":str(exc)}
    outputs=await asyncio.gather(*[one(a) for a in assigns])
    return assigns, list(outputs)

def _write_assignments(cases, config, assignments_path: Path):
    all_assignments=[assignment(str(r.case_id),f"agent_{i+1}",config.global_seed,config.gpt_model,config.claude_model)
                     for _,r in cases.iterrows() for i in range(config.n_agents)]
    assignments_path.parent.mkdir(parents=True,exist_ok=True)
    assignments_path.write_text("\n".join(json.dumps(x) for x in all_assignments)+("\n" if all_assignments else ""),encoding="utf-8")

async def run_mixed_live(cases: pd.DataFrame, config, checkpoint: Path, assignments_path: Path) -> pd.DataFrame:
    done=load_completed(checkpoint)
    async def handle(row):
        cid=str(row.case_id)
        if cid in done and done[cid].get("status") == "success" and done[cid].get("agent_outputs"):
            return done[cid]
        _, outputs=await _panel_outputs(row, config)
        labels=[o["answer"] for o in outputs if o["answer"]]; errors=[o.get("error","invalid_output") for o in outputs if not o["answer"]]
        prediction=sorted(set(labels),key=lambda x:(-labels.count(x),x))[0] if labels else ""
        record={"case_id":cid,"prediction":prediction,"agent_outputs":outputs,"critical_safety_error":False,"status":"success" if labels else "api_error","system":"selected_multi_agent","provider":"mixed","model":"mixed","agent_count":config.n_agents,"agent_errors":errors}
        append_checkpoint(checkpoint,record); return record
    rows=await _bounded_gather([(lambda r=row: handle(r)) for _,row in cases.iterrows()])
    _write_assignments(cases, config, assignments_path)
    return pd.DataFrame(rows)

async def run_mixed_shared(cases: pd.DataFrame, config, shared_path: Path, assignments_path: Path) -> pd.DataFrame:
    """Run the mixed panel once and persist every agent output for reuse."""
    done=load_completed(shared_path)
    async def handle(row):
        cid=str(row.case_id)
        if cid in done and done[cid].get("status") == "success" and done[cid].get("prediction"):
            return done[cid]
        _, outputs=await _panel_outputs(row, config)
        prediction=""
        labels=[o["answer"] for o in outputs if o["answer"]]
        if labels: prediction=sorted(set(labels),key=lambda x:(-labels.count(x),x))[0]
        record={"case_id":cid,"agent_outputs":outputs,"prediction":prediction,"status":"success" if any(x["answer"] for x in outputs) else "api_error"}
        append_checkpoint(shared_path,record); return record
    rows=await _bounded_gather([(lambda r=row: handle(r)) for _,row in cases.iterrows()])
    _write_assignments(cases, config, assignments_path)
    return pd.DataFrame(rows)

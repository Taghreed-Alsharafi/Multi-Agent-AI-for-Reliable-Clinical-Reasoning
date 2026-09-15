from __future__ import annotations
import argparse, asyncio, csv, os
from pathlib import Path
from .io import append_jsonl,read_cases,read_predictions,write_json
from .metrics import evaluate_predictions,robustness_by_variant
from .runners import run_multi_agent,run_single_agent
from .schema import PredictionRecord
from .config import ExperimentConfig
from .experiment import run_experiment, score_experiment

async def _run(args):
    cases=read_cases(args.dataset);systems=[x.strip() for x in args.systems.split(",")]
    old=read_predictions(args.output) if Path(args.output).exists() and args.resume else []
    done={(p.case_id,p.system) for p in old if p.status=="ok"}
    names={"multi":"multi_agent","gpt":"single_openai","claude":"single_anthropic"}
    for case in cases:
        for system in systems:
            if (case.case_id,names[system]) in done:continue
            try:
                if system=="multi": pred=await run_multi_agent(case,{"supervisor":args.openai_model,
                    "specialist":args.openai_model,"judge":args.openai_model,"safety":args.openai_model})
                elif system=="gpt": pred=await run_single_agent(case,"openai",args.openai_model,args.temperature)
                else: pred=await run_single_agent(case,"anthropic",args.anthropic_model,args.temperature)
            except Exception as exc:
                pred=PredictionRecord(run_id="error",case_id=case.case_id,system=names[system],
                    provider="anthropic" if system=="claude" else "openai",
                    model=args.anthropic_model if system=="claude" else args.openai_model,
                    answer="",latency_seconds=0,status="error",error=str(exc))
            append_jsonl(args.output,[pred])

def _score(args):
    cases=read_cases(args.dataset);preds=read_predictions(args.predictions)
    detailed,summary=evaluate_predictions(cases,preds);output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    write_json(output/"metrics.json",{"summary":summary,"per_case":detailed,
        "robustness":robustness_by_variant(cases,preds)})
    for name,rows in (("metrics_summary.csv",summary),("metrics_per_case.csv",detailed)):
        with (output/name).open("w",newline="",encoding="utf-8") as h:
            if rows:
                writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True)
    run=sub.add_parser("run");run.add_argument("--dataset",required=True);run.add_argument("--output",required=True)
    run.add_argument("--systems",default="multi,gpt,claude");run.add_argument("--openai-model",default=os.getenv("OPENAI_MODEL","gpt-4o-mini"))
    run.add_argument("--anthropic-model",default=os.getenv("ANTHROPIC_MODEL","claude-3-5-haiku-latest"));run.add_argument("--temperature",type=float,default=0);run.add_argument("--resume",action="store_true")
    score=sub.add_parser("score");score.add_argument("--dataset",required=True);score.add_argument("--predictions",required=True);score.add_argument("--output-dir",required=True)
    experiment=sub.add_parser("experiment");experiment.add_argument("--config",required=True)
    args=parser.parse_args()
    if args.command=="run": asyncio.run(_run(args))
    elif args.command=="score": _score(args)
    else:
        config=ExperimentConfig.from_yaml(args.config);run_experiment(config);score_experiment(config)

if __name__=="__main__":main()

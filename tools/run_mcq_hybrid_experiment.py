#!/usr/bin/env python
"""Run single baselines, hybrid multi-agent, then comparison summaries.

Usage from repository root:
    export OPENAI_API_KEY=...
    python tools/run_mcq_hybrid_experiment.py --split development --max-cases 50

After prompts/configuration are frozen:
    python tools/run_mcq_hybrid_experiment.py --split test --max-cases all
"""
from __future__ import annotations
import argparse
import asyncio
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.mcq_hybrid import (
    ensure_fixed_split, run_single_agent, run_multi_agent, metrics, paired_accuracy_comparison
)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/medical_mcq.csv")
    ap.add_argument("--split", choices=["development", "test"], default="development")
    ap.add_argument("--max-cases", default="50", help="integer or 'all'")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--development-fraction", type=float, default=0.20)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    max_cases = None if args.max_cases.lower() == "all" else int(args.max_cases)
    splits, manifest, manifest_path = ensure_fixed_split(ROOT/args.csv, ROOT, args.development_fraction, args.seed)
    df = splits[args.split].copy()
    if max_cases is not None:
        df = df.head(max_cases).copy()
    unresolved = df[df.reference_letter.isna()].copy()
    if not unresolved.empty:
        unresolved_path = ROOT/"results"/"mcq_comparison"/f"unresolved_reference_cases_{args.split}.csv"
        unresolved_path.parent.mkdir(parents=True, exist_ok=True)
        unresolved.to_csv(unresolved_path, index=False)
        print(
            f"Reference extraction unresolved for {len(unresolved)} case(s); "
            f"excluding them from scoring. Saved audit file: {unresolved_path}"
        )
        df = df[df.reference_letter.notna()].copy().reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No scorable cases remain after reference-answer QC.")

    out = ROOT/"results"/"mcq_comparison"
    out.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume
    print("Manifest:", manifest_path)
    print("Running", args.split, len(df), "cases")

    single_mini = await run_single_agent(df, "single_gpt4o_mini", "gpt-4o-mini", args.split, out, resume=resume)
    single_gpt5 = await run_single_agent(df, "single_gpt5", "gpt-5", args.split, out, reasoning_effort="medium", resume=resume)
    multi = await run_multi_agent(
        df, "multi_dynamic_one_gpt5_judge", args.split, out,
        router_model="gpt-4o-mini", specialist_model="gpt-4o-mini",
        critic_model="gpt-4o-mini", judge_model="gpt-5", auditor_model="gpt-4o-mini",
        min_specialists=1, max_specialists=4,
        judge_reasoning_effort="medium", resume=resume,
    )

    summary = pd.DataFrame([
        {"system":"single_gpt4o_mini", **metrics(single_mini)},
        {"system":"single_gpt5", **metrics(single_gpt5)},
        {"system":"multi_dynamic_one_gpt5_judge", **metrics(multi)},
    ])
    summary.to_csv(out/f"comparison_summary_{args.split}.csv", index=False)
    print("\nSUMMARY")
    print(summary.to_string(index=False))

    paired = pd.DataFrame([
        paired_accuracy_comparison(multi, single_mini, "single_gpt4o_mini"),
        paired_accuracy_comparison(multi, single_gpt5, "single_gpt5"),
    ])
    paired.to_csv(out/f"paired_accuracy_tests_{args.split}.csv", index=False)
    print("\nPAIRED ACCURACY")
    print(paired.to_string(index=False))

    successful = multi[~multi.api_error.astype(bool)]
    assert (successful.gpt5_call_count == 1).all(), "GPT-5 call invariant failed"
    print("\nValidated: exactly one GPT-5 call per successful multi-agent case.")

if __name__ == "__main__":
    asyncio.run(main())

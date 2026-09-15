"""Development sweep for the improved multi-agent system.

Runs the diverse panel (judge precomputed for every dev case) plus single-GPT and
single-Claude on the development split, then compares three aggregation policies
offline: vote-only, judge-on-disagreement, and always-judge (mixture-of-agents).
Picks the policy with the best development accuracy and freezes it. No test data
is touched here.
"""
import os, sys, json, asyncio, time
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv(".env")
os.environ["VERIFY_SSL"] = os.environ.get("VERIFY_SSL", "false")
os.environ.setdefault("THESIS_CONCURRENCY", "6")

import pandas as pd
from thesis_pipeline import initialize_thesis
from thesis_pipeline.multi_agent import run_improved_multi, simulate_trigger
from thesis_pipeline.live import run_single_live
from thesis_pipeline.evaluation import score
from thesis_pipeline.checkpoint import load_completed

DATASET = {"name": "final_dataset_1", "path": "data/final_dataset_1_with_gold.csv",
           "columns": {"instruction": "instruction", "input": "input", "gold": "gold_letter"}}

p = initialize_thesis(DATASET, RUN_MODE="full", GLOBAL_SEED=42, root=".",
                      mock=False, output_dir="results/improved")
out = p.output
dev = p.bundle.cases("development")
gold_by_id = {str(r.case_id): str(r.gold).upper() for _, r in dev.iterrows()}
print(f"Dev cases: {len(dev)} | concurrency {os.environ['THESIS_CONCURRENCY']}", flush=True)


async def main():
    t = time.time()
    print("Running diverse panel + judge on dev (judge precomputed for all) ...", flush=True)
    multi = await run_improved_multi(dev, p.config, out / "improved_dev.jsonl", precompute_all_judge=True)
    print(f"  panel done in {time.time()-t:.0f}s", flush=True)
    gpt = await run_single_live(dev, "openai", p.config.gpt_model, out / "single_gpt_dev.jsonl", "single_gpt")
    claude = await run_single_live(dev, "anthropic", p.config.claude_model, out / "single_claude_dev.jsonl", "single_claude")

    sg = score(dev, gpt).correct.mean()
    sc = score(dev, claude).correct.mean()

    records = load_completed(out / "improved_dev.jsonl")
    recs = [records[str(r.case_id)] for _, r in dev.iterrows() if str(r.case_id) in records]
    policies = {
        "vote_only": simulate_trigger(recs, gold_by_id, distinct_trigger=99),
        "judge_on_disagreement": simulate_trigger(recs, gold_by_id, distinct_trigger=2),
        "always_judge_moa": simulate_trigger(recs, gold_by_id, distinct_trigger=1),
    }
    best_name = max(policies, key=lambda k: policies[k]["accuracy"])
    best_trigger = {"vote_only": 99, "judge_on_disagreement": 2, "always_judge_moa": 1}[best_name]

    summary = {
        "dev_n": len(dev),
        "single_gpt_accuracy": float(sg),
        "single_claude_accuracy": float(sc),
        "policies": policies,
        "selected_policy": best_name,
        "selected_distinct_trigger": best_trigger,
        "beats_single_gpt": bool(policies[best_name]["accuracy"] > sg),
    }
    (out / "improved_dev_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== DEV SWEEP SUMMARY ===", flush=True)
    print(f"single_gpt dev acc:    {sg:.4f}")
    print(f"single_claude dev acc: {sc:.4f}")
    for name, r in policies.items():
        print(f"{name:<24} acc={r['accuracy']:.4f}  judge_rate={r['judge_rate']:.2f}")
    print(f"SELECTED: {best_name} (trigger={best_trigger}) | beats single_gpt: {summary['beats_single_gpt']}")
    print(json.dumps(summary, indent=2))

asyncio.run(main())

"""Frozen test run for the improved multi-agent system (judge-on-disagreement).

Policy (distinct_trigger=2) was selected on development only. Evaluates on the
held-out test split and writes predictions + metrics for the comparison PDF.
"""
import os, sys, json, asyncio, time
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv(".env")
os.environ["VERIFY_SSL"] = os.environ.get("VERIFY_SSL", "false")
os.environ.setdefault("THESIS_CONCURRENCY", "6")

import pandas as pd
from thesis_pipeline import initialize_thesis
from thesis_pipeline.multi_agent import run_improved_multi
from thesis_pipeline.evaluation import score, metrics

DATASET = {"name": "final_dataset_1", "path": "data/final_dataset_1_with_gold.csv",
           "columns": {"instruction": "instruction", "input": "input", "gold": "gold_letter"}}

SELECTED_TRIGGER = 2  # frozen from development sweep

p = initialize_thesis(DATASET, RUN_MODE="full", GLOBAL_SEED=42, root=".",
                      mock=False, output_dir="results/improved")
test = p.bundle.cases("test")
print(f"Test cases: {len(test)} | frozen distinct_trigger={SELECTED_TRIGGER}", flush=True)


async def main():
    t = time.time()
    multi = await run_improved_multi(test, p.config, p.output / "improved_test.jsonl",
                                     distinct_trigger=SELECTED_TRIGGER, precompute_all_judge=False)
    print(f"Improved multi test done in {time.time()-t:.0f}s", flush=True)
    scored = score(test, multi)
    m = metrics(scored, "improved_multi_agent")
    m["split"] = "test"
    m["judge_rate"] = float(multi["judge_used"].mean()) if "judge_used" in multi else 0.0
    (p.output / "improved_test_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    scored.merge(multi[["case_id", "method", "judge_used"]], on="case_id", how="left") \
          .to_csv(p.output / "improved_test_predictions.csv", index=False)
    print(json.dumps(m, indent=2), flush=True)
    print("\n===== IMPROVED TEST RUN COMPLETE =====", flush=True)

asyncio.run(main())

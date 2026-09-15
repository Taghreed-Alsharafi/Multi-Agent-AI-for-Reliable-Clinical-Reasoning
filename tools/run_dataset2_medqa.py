"""Generalizability run: apply the FROZEN systems to MedQA-USMLE (1200 items).

No retuning: the improved policy (judge-on-disagreement, trigger=2) and all
models are frozen from dataset 1. Evaluates Single-GPT, Single-Claude, the
diverse panel (vote-only, derived), and Improved (panel+judge) on all 1200
clean-gold items. Checkpointed / resumable.
"""
import os, sys, json, asyncio, time
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv(".env")
os.environ["VERIFY_SSL"] = os.environ.get("VERIFY_SSL", "false")
os.environ.setdefault("THESIS_CONCURRENCY", "6")

import pandas as pd
from thesis_pipeline import initialize_thesis
from thesis_pipeline.multi_agent import run_improved_multi, decide, _weighted_vote
from thesis_pipeline.live import run_single_live
from thesis_pipeline.evaluation import score, metrics

DATASET = {"name": "medqa_usmle", "path": "data/medqa_usmle_1200_with_gold.csv",
           "columns": {"instruction": "instruction", "input": "input", "gold": "gold_letter"}}
TRIGGER = 2

p = initialize_thesis(DATASET, RUN_MODE="full", GLOBAL_SEED=42, root=".",
                      mock=False, output_dir="results/medqa_usmle")
out = p.output
eval_df = p.bundle.frame[p.bundle.frame.eligible_for_evaluation].copy()
print(f"Eval items: {len(eval_df)} | trigger={TRIGGER} | conc={os.environ['THESIS_CONCURRENCY']}", flush=True)


async def main():
    t = time.time()
    gpt = await run_single_live(eval_df, "openai", p.config.gpt_model, out / "single_gpt.jsonl", "single_gpt")
    print(f"single_gpt done {time.time()-t:.0f}s", flush=True)
    claude = await run_single_live(eval_df, "anthropic", p.config.claude_model, out / "single_claude.jsonl", "single_claude")
    print(f"single_claude done {time.time()-t:.0f}s", flush=True)
    multi = await run_improved_multi(eval_df, p.config, out / "improved.jsonl", distinct_trigger=TRIGGER)
    print(f"improved panel done {time.time()-t:.0f}s", flush=True)

    # Derive vote-only prediction from the same diverse panel (no extra calls).
    vote_rows = []
    for _, r in multi.iterrows():
        outs = r.get("agent_outputs", [])
        allowed = sorted({o["answer"] for o in outs if o.get("answer")})
        pred, _ = decide(outs, allowed, judge_letter="", distinct_trigger=99)
        vote_rows.append({"case_id": r["case_id"], "prediction": pred,
                          "status": "success" if pred else "api_error",
                          "critical_safety_error": False, "system": "panel_vote"})
    vote = pd.DataFrame(vote_rows)

    frames = {"single_gpt": gpt, "single_claude": claude, "panel_vote": vote, "improved_multi": multi}
    summary = {}
    for name, fr in frames.items():
        s = score(eval_df, fr)
        m = metrics(s, name)
        if name == "improved_multi" and "judge_used" in multi:
            m["judge_rate"] = float(multi["judge_used"].mean())
        summary[name] = m
        s.merge(fr[[c for c in ["case_id", "status", "method", "judge_used"] if c in fr]],
                on="case_id", how="left").to_csv(out / f"{name}_predictions.csv", index=False)
    (out / "dataset2_metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\n=== MedQA-USMLE (generalizability) ===", flush=True)
    for name, m in summary.items():
        print(f"{name:<16} acc={m['accuracy']:.4f}  macroF1={m['macro_f1']:.4f}  fail={m['failure_rate']:.4f}", flush=True)
    print("\n===== DATASET 2 RUN COMPLETE =====", flush=True)

asyncio.run(main())

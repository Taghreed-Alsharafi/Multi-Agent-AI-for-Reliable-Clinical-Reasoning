#!/usr/bin/env python
"""Run V2 single baselines and adaptive multi-agent evaluation.

This command makes paid API calls. Run without ``--resume`` while developing;
resume is accepted only when every dataset/configuration fingerprint matches.
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

from evaluation.mcq_hybrid_v2 import (
    MultiAgentConfig,
    SingleAgentConfig,
    ensure_fixed_split_v2,
    load_evaluation_dataset,
    run_multi_agent_v2,
    run_single_agent_v2,
    validate_v2_multi_results,
)
from evaluation.mcq_v2_analysis import (
    comparison_summary,
    paired_accuracy_comparison,
)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", default="data/medical_mcq_300_EVAL_READY.csv"
    )
    parser.add_argument(
        "--qc-master", default="data/medical_mcq_300_QC_MASTER.csv"
    )
    parser.add_argument("--split", choices=["development", "test"], default="development")
    parser.add_argument("--max-cases", default="all", help="integer or 'all'")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-original-gold", action="store_true")
    parser.add_argument(
        "--systems",
        default="single-mini,single-gpt5,multi",
        help="comma-separated: single-mini,single-gpt5,multi",
    )
    parser.add_argument("--output-dir", default="results/mcq_v2")
    args = parser.parse_args()

    qc = load_evaluation_dataset(
        ROOT / args.csv,
        ROOT / args.output_dir / "qc_cleaned",
        qc_master_path=ROOT / args.qc_master,
        require_original_gold=args.require_original_gold,
    )
    splits, manifest, manifest_path = ensure_fixed_split_v2(
        qc,
        ROOT,
        dataset_name=Path(args.csv).stem,
        development_fraction=0.20,
        seed=2026,
    )
    frame = splits[args.split].copy()
    if args.max_cases.lower() != "all":
        frame = frame.head(int(args.max_cases)).copy()
    if frame.empty:
        raise RuntimeError(
            "No eligible cases remain. If REQUIRE_ORIGINAL_GOLD is enabled, "
            "add exact original-benchmark matches first."
        )

    print("Eligible / development / test:", len(qc.eligible), len(splits["development"]), len(splits["test"]))
    print("Excluded by active loader:", len(qc.excluded))
    if qc.summary.get("qc_master_warning"):
        print("QC master warning:", qc.summary["qc_master_warning"])
        print("QC master reported exclusions:", qc.summary.get("qc_master_reported_excluded_rows"))
    print("Dataset fingerprint:", qc.dataset_fingerprint)
    print("Label fingerprint:", qc.label_fingerprint)
    print("Manifest:", manifest_path)
    print("Running:", args.split, len(frame), "cases")

    output_dir = ROOT / args.output_dir
    requested = {item.strip() for item in args.systems.split(",") if item.strip()}
    results: dict[str, pd.DataFrame] = {}
    if "single-mini" in requested:
        config = SingleAgentConfig(
            run_name="single_gpt4o_mini_v2",
            model="gpt-4o-mini",
            max_output_tokens=800,
        )
        results[config.run_name] = await run_single_agent_v2(
            frame, config, args.split, output_dir, resume=args.resume
        )
    if "single-gpt5" in requested:
        config = SingleAgentConfig(
            run_name="single_gpt5",
            model="gpt-5",
            reasoning_effort="medium",
            max_output_tokens=8000,
        )
        results[config.run_name] = await run_single_agent_v2(
            frame, config, args.split, output_dir, resume=args.resume
        )
    if "multi" in requested:
        config = MultiAgentConfig()
        results[config.run_name] = await run_multi_agent_v2(
            frame, config, args.split, output_dir, resume=args.resume
        )
        checks = validate_v2_multi_results(results[config.run_name], frame, config)
        print("Multi-agent invariants:", checks)

    summary = comparison_summary(results, args.split)
    summary_path = output_dir / f"v2_performance_summary_{args.split}.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print("Saved:", summary_path)

    multi = results.get("multi_dynamic_one_gpt5_v2")
    if multi is not None:
        paired_rows = []
        for baseline_name in ("single_gpt5", "single_gpt4o_mini_v2"):
            baseline = results.get(baseline_name)
            if baseline is not None:
                stats, _ = paired_accuracy_comparison(
                    multi,
                    baseline,
                    "multi_dynamic_one_gpt5_v2",
                    baseline_name,
                )
                paired_rows.append(stats)
        if paired_rows:
            paired_path = output_dir / f"v2_paired_tests_{args.split}.csv"
            pd.DataFrame(paired_rows).to_csv(paired_path, index=False)
            print("Saved:", paired_path)


if __name__ == "__main__":
    asyncio.run(main())

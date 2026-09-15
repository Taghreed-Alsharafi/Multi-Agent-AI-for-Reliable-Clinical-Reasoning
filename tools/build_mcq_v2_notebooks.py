"""Generate the three V2 MCQ orchestration notebooks."""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def write(name: str, cells: list) -> None:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    nbf.write(notebook, NOTEBOOKS / name)
    print("Wrote:", NOTEBOOKS / name)


COMMON_SETUP = """
from pathlib import Path
import sys

def find_root(start=None):
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "evaluation" / "mcq_hybrid_v2.py").exists():
            return candidate
    raise FileNotFoundError("Run this notebook from the repository root or notebooks directory.")

ROOT = find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CSV_PATH = ROOT / "data" / "medical_mcq_300_EVAL_READY.csv"
QC_MASTER_PATH = ROOT / "data" / "medical_mcq_300_QC_MASTER.csv"
OUTPUT_DIR = ROOT / "results" / "mcq_v2"
RUN_SPLIT = "development"
MAX_CASES = None
RESUME = False
REQUIRE_ORIGINAL_GOLD = False
DEVELOPMENT_FRACTION = 0.20
SPLIT_SEED = 2026
"""


LOAD_DATA = """
from evaluation.mcq_hybrid_v2 import load_evaluation_dataset, ensure_fixed_split_v2

qc = load_evaluation_dataset(
    CSV_PATH,
    OUTPUT_DIR / "qc_cleaned",
    qc_master_path=QC_MASTER_PATH,
    require_original_gold=REQUIRE_ORIGINAL_GOLD,
)
splits, split_manifest, manifest_path = ensure_fixed_split_v2(
    qc,
    ROOT,
    dataset_name=CSV_PATH.stem,
    development_fraction=DEVELOPMENT_FRACTION,
    seed=SPLIT_SEED,
)
cases = splits[RUN_SPLIT].copy()
if MAX_CASES is not None:
    cases = cases.head(MAX_CASES).copy()

print("Total eligible:", len(qc.eligible))
print("Development count:", len(splits["development"]))
print("Test count:", len(splits["test"]))
print("Excluded by active loader:", len(qc.excluded))
if qc.summary.get("qc_master_warning"):
    print("QC master warning:", qc.summary["qc_master_warning"])
    print("QC master reported exclusions:", qc.summary.get("qc_master_reported_excluded_rows"))
print("Dataset fingerprint:", qc.dataset_fingerprint)
print("Label fingerprint:", qc.label_fingerprint)
print("Manifest path:", manifest_path)
print("Gold status:", qc.summary["gold_status_distribution"])
print("Cases selected for this run:", len(cases))

assert cases["eligible_for_evaluation"].all()
assert cases["fixed_case_id"].is_unique
assert set(splits["development"].fixed_case_id).isdisjoint(set(splits["test"].fixed_case_id))
"""


write(
    "02_single_agent_mcq.ipynb",
    [
        md(
            """
            # V2 Single-Agent Clinical MCQ Baselines

            **Purpose:** Run `single_gpt4o_mini_v2` and `single_gpt5` fairly on the cleaned protected split.

            **Dataset contract:** `gold_letter` is used directly; `output` is only a one-letter fallback. Excluded rows never reach a model. `gold_status` remains visible because source-derived labels are not independently verified original benchmark labels.

            **Study design:** Both baselines receive identical question text and case IDs. The GPT-5 baseline uses the same `gpt-5`, medium reasoning effort, 8,000-token ceiling, and forced-choice policy as the multi-agent final judge.

            **Steps:** Load/QC data, restore the shared 20/80 manifest, choose development or locked test, run both systems, validate outputs, and save tables.

            **Inputs:** `data/medical_mcq_300_EVAL_READY.csv`, optional QC master, and `OPENAI_API_KEY` from project-root `.env`.

            **Outputs:** Prediction CSVs, trace JSONL, run metadata, summaries, and per-option metrics under `results/mcq_v2/`.
            """
        ),
        code(COMMON_SETUP),
        code(LOAD_DATA),
        code(
            """
            from evaluation.mcq_hybrid_v2 import SingleAgentConfig, run_single_agent_v2

            mini_config = SingleAgentConfig(
                run_name="single_gpt4o_mini_v2",
                model="gpt-4o-mini",
                max_output_tokens=800,
            )
            gpt5_config = SingleAgentConfig(
                run_name="single_gpt5",
                model="gpt-5",
                reasoning_effort="medium",
                max_output_tokens=8000,
            )

            single_mini = await run_single_agent_v2(
                cases, mini_config, RUN_SPLIT, OUTPUT_DIR, resume=RESUME
            )
            single_gpt5 = await run_single_agent_v2(
                cases, gpt5_config, RUN_SPLIT, OUTPUT_DIR, resume=RESUME
            )
            """
        ),
        code(
            """
            from IPython.display import display
            from evaluation.mcq_v2_analysis import comparison_summary, per_answer_metrics

            systems = {
                "single_gpt4o_mini_v2": single_mini,
                "single_gpt5": single_gpt5,
            }
            single_summary = comparison_summary(systems, RUN_SPLIT)
            single_summary.to_csv(OUTPUT_DIR / f"single_agent_v2_summary_{RUN_SPLIT}.csv", index=False)
            display(single_summary.round(4))

            # Requested multi-style presentation: metrics as rows, systems as columns.
            display(single_summary.set_index("system").T.round(4))

            for system_name, result in systems.items():
                print(system_name)
                display(per_answer_metrics(result).round(4))
            """
        ),
        code(
            """
            columns = [
                "system", "fixed_case_id", "row_id", "gold_status",
                "reference_letter", "predicted_letter", "correct", "confidence",
                "invalid_answer", "api_error", "latency_seconds", "total_tokens",
                "estimated_cost_usd", "config_hash", "dataset_fingerprint",
                "label_fingerprint",
            ]
            evaluated_table = __import__("pandas").concat(
                [frame[[column for column in columns if column in frame]] for frame in systems.values()],
                ignore_index=True,
            )
            evaluated_table.to_csv(
                OUTPUT_DIR / f"single_agent_v2_evaluated_table_{RUN_SPLIT}.csv", index=False
            )
            display(evaluated_table.head(20))
            """
        ),
        code(
            """
            expected_ids = set(cases.fixed_case_id)
            for name, result in systems.items():
                assert set(result.fixed_case_id) == expected_ids, f"{name} case membership mismatch"
                assert (result.dataset_fingerprint == qc.dataset_fingerprint).all()
                assert (result.label_fingerprint == qc.label_fingerprint).all()
                assert result.reference_letter.str.fullmatch(r"[A-H]").all()
            assert (single_gpt5.gpt5_call_count == 1).all()
            print("Validated identical case membership, fingerprints, direct gold labels, and one GPT-5 call per GPT-5 baseline case.")
            """
        ),
    ],
)


write(
    "03_multi_agent_gpt5_V2_optimized.ipynb",
    [
        md(
            """
            # V2 Optimized Adaptive Multi-Agent Clinical MCQ Evaluation

            **Purpose:** Test whether the smallest case-appropriate expert team plus exactly one GPT-5 final judge outperforms one GPT-5 call.

            **Dataset contract:** Uses the same shared cleaned loader, direct `gold_letter`, protected split, and label provenance as Notebook 02. Gold data is never included in prompts.

            **V2 architecture:** QC -> dynamic Router -> 1-3 independent domain/subspecialty experts -> conditional conflict Critic -> exactly one GPT-5 Judge -> deterministic validator. There is no LLM auditor after GPT-5.

            **Dynamic selection:** Difficulty is clinical reasoning complexity, not length. Panel size also considers domain overlap and nonredundant expertise; difficult single-domain cases may still use one specialist.

            **Critic rules:** The Critic is advisory and runs only for meaningful triggers such as specialist disagreement, invalid specialist output, expected high disagreement, or a very complex case. No numerical router confidence is used.

            **One-GPT-5 rule:** GPT-5 appears only as the final judge, SDK retries are disabled, and every successfully evaluated case must record exactly one GPT-5 call.

            **Study design and steps:** Load the shared development split, run V2, enforce invariants, then report performance, routing, cost, tokens, and latency. The locked test is used only after configuration freeze.

            **Inputs/outputs:** Cleaned CSV, QC master, shared split manifest, project `.env`; predictions, traces, metadata, and analysis CSVs under `results/mcq_v2/`.
            """
        ),
        code(COMMON_SETUP),
        code(LOAD_DATA),
        code(
            """
            from evaluation.mcq_hybrid_v2 import MultiAgentConfig, run_multi_agent_v2

            multi_config = MultiAgentConfig(
                run_name="multi_dynamic_one_gpt5_v2",
                router_model="gpt-4o-mini",
                specialist_model="gpt-4o-mini",
                critic_model="gpt-4o-mini",
                judge_model="gpt-5",
                judge_reasoning_effort="medium",
                max_tokens_small=700,
                max_tokens_judge=8000,
                max_specialists=3,
                allow_fourth_specialist=False,
                use_router=True,
                specialist_mode="dynamic",
                use_critic=True,
            )

            multi = await run_multi_agent_v2(
                cases, multi_config, RUN_SPLIT, OUTPUT_DIR, resume=RESUME
            )
            """
        ),
        code(
            """
            from IPython.display import display
            from evaluation.mcq_hybrid_v2 import validate_v2_multi_results
            from evaluation.mcq_v2_analysis import (
                panel_size_analysis,
                routing_summary_tables,
                summarize_system,
            )

            checks = validate_v2_multi_results(multi, cases, multi_config)
            print("Invariant checks:", checks)

            multi_summary = summarize_system(multi, multi_config.run_name, RUN_SPLIT)
            multi_summary.to_csv(
                OUTPUT_DIR / f"multi_agent_v2_summary_{RUN_SPLIT}.csv", index=False
            )
            display(multi_summary.round(4))
            display(multi_summary.set_index("system").T.round(4))

            routing_tables = routing_summary_tables(multi)
            for name, table in routing_tables.items():
                print(name)
                display(table)
                table.to_csv(OUTPUT_DIR / f"{name}_{RUN_SPLIT}.csv", index=False)

            panel_table = panel_size_analysis(multi)
            display(panel_table.round(4))
            panel_table.to_csv(OUTPUT_DIR / f"panel_size_analysis_{RUN_SPLIT}.csv", index=False)
            """
        ),
        code(
            """
            successful = multi[~multi.api_error.astype(bool) & ~multi.invalid_answer.astype(bool)]
            assert (successful.gpt5_call_count == 1).all()
            assert (multi.n_specialists >= 1).all()
            assert (multi.n_specialists <= multi_config.max_specialists).all()
            assert "audit_rejected" not in multi.columns
            assert set(multi.fixed_case_id) == set(cases.fixed_case_id)
            assert (multi.config_hash == multi_config.config_hash).all()
            assert (multi.dataset_fingerprint == qc.dataset_fingerprint).all()
            assert (multi.label_fingerprint == qc.label_fingerprint).all()
            print("All V2 architecture, membership, fingerprint, and exactly-once invariants passed.")
            """
        ),
    ],
)


write(
    "04_compare_single_multi_mcq.ipynb",
    [
        md(
            """
            # V2 Paired Single-Agent vs Multi-Agent Comparison

            **Purpose:** Compare saved `single_gpt4o_mini_v2`, `single_gpt5`, and `multi_dynamic_one_gpt5_v2` results without making API calls.

            **Dataset contract:** Files must share the same question-set fingerprint, label fingerprint, split, direct gold labels, and expected case membership. Source-derived gold status remains visible.

            **Study design:** The primary paired comparison is dynamic multi-agent versus single GPT-5. All statistics use inner joins on `fixed_case_id`; strict membership checks prevent accidental partial or stale comparisons.

            **Steps:** Load files, validate compatibility, summarize performance/resources, run paired bootstrap and exact McNemar tests, analyze difficulty/panel size, and inspect case-level disagreements.

            **Inputs:** Saved prediction CSVs from Notebooks 02 and 03 plus the cleaned dataset/split manifest.

            **Outputs:** Publication-ready performance, paired-test, routing, difficulty, panel-size, resource, and disagreement CSVs. This notebook contains no model/API execution.
            """
        ),
        code(
            """
            from pathlib import Path
            import sys
            import pandas as pd
            from IPython.display import display

            def find_root(start=None):
                current = Path(start or Path.cwd()).resolve()
                for candidate in [current, *current.parents]:
                    if (candidate / "evaluation" / "mcq_v2_analysis.py").exists():
                        return candidate
                raise FileNotFoundError("Repository root not found")

            ROOT = find_root()
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            OUTPUT_DIR = ROOT / "results" / "mcq_v2"
            CSV_PATH = ROOT / "data" / "medical_mcq_300_EVAL_READY.csv"
            QC_MASTER_PATH = ROOT / "data" / "medical_mcq_300_QC_MASTER.csv"
            RUN_SPLIT = "development"
            STRICT_CASE_MEMBERSHIP = True
            """
        ),
        code(
            """
            prediction_paths = {
                "single_gpt4o_mini_v2": OUTPUT_DIR / f"single_gpt4o_mini_v2_{RUN_SPLIT}_predictions.csv",
                "single_gpt5": OUTPUT_DIR / f"single_gpt5_{RUN_SPLIT}_predictions.csv",
                "multi_dynamic_one_gpt5_v2": OUTPUT_DIR / f"multi_dynamic_one_gpt5_v2_{RUN_SPLIT}_predictions.csv",
            }
            missing = [str(path) for path in prediction_paths.values() if not path.exists()]
            if missing:
                raise FileNotFoundError("Run Notebooks 02 and 03 first. Missing: " + ", ".join(missing))
            systems = {
                name: pd.read_csv(path, keep_default_na=False)
                for name, path in prediction_paths.items()
            }

            from evaluation.mcq_v2_analysis import validate_comparison_inputs
            compatibility = validate_comparison_inputs(systems)
            print("Compatibility:", compatibility)

            id_sets = {name: set(frame.fixed_case_id) for name, frame in systems.items()}
            common_ids = set.intersection(*id_sets.values())
            print("Common paired cases:", len(common_ids))
            if STRICT_CASE_MEMBERSHIP:
                assert all(ids == common_ids for ids in id_sets.values()), "System case membership differs"
            """
        ),
        code(
            """
            from evaluation.mcq_v2_analysis import (
                comparison_summary,
                paired_accuracy_comparison,
                per_answer_metrics,
            )

            main_table = comparison_summary(systems, RUN_SPLIT)
            main_table.to_csv(OUTPUT_DIR / f"publication_main_performance_{RUN_SPLIT}.csv", index=False)
            display(main_table.round(4))

            paired_rows = []
            for baseline_name in ("single_gpt5", "single_gpt4o_mini_v2"):
                stats, paired = paired_accuracy_comparison(
                    systems["multi_dynamic_one_gpt5_v2"],
                    systems[baseline_name],
                    "multi_dynamic_one_gpt5_v2",
                    baseline_name,
                )
                paired_rows.append(stats)
            paired_table = pd.DataFrame(paired_rows)
            paired_table.to_csv(OUTPUT_DIR / f"publication_paired_tests_{RUN_SPLIT}.csv", index=False)
            display(paired_table.round(4))
            """
        ),
        code(
            """
            from evaluation.mcq_hybrid_v2 import load_evaluation_dataset, ensure_fixed_split_v2
            from evaluation.mcq_v2_analysis import (
                case_disagreement_table,
                difficulty_analysis,
                panel_size_analysis,
                routing_summary_tables,
            )

            qc = load_evaluation_dataset(
                CSV_PATH,
                qc_master_path=QC_MASTER_PATH,
                require_original_gold=False,
            )
            splits, _, _ = ensure_fixed_split_v2(
                qc, ROOT, dataset_name=CSV_PATH.stem, development_fraction=0.20, seed=2026
            )
            dataset_split = splits[RUN_SPLIT]

            multi = systems["multi_dynamic_one_gpt5_v2"]
            single_gpt5 = systems["single_gpt5"]
            difficulty_table = difficulty_analysis(multi, single_gpt5)
            panel_table = panel_size_analysis(multi)
            disagreement_table = case_disagreement_table(dataset_split, single_gpt5, multi)
            resource_columns = [
                "system", "mean_latency_seconds", "median_latency_seconds",
                "mean_input_tokens", "mean_output_tokens", "mean_reasoning_tokens",
                "mean_total_tokens", "mean_estimated_cost_usd", "mean_total_calls",
                "mean_gpt5_call_count", "mean_n_specialists",
            ]
            resource_table = main_table[[column for column in resource_columns if column in main_table]]

            for name, table in {
                "publication_difficulty_analysis": difficulty_table,
                "publication_panel_size_analysis": panel_table,
                "publication_resource_analysis": resource_table,
                "publication_case_disagreements": disagreement_table,
            }.items():
                table.to_csv(OUTPUT_DIR / f"{name}_{RUN_SPLIT}.csv", index=False)
                print(name)
                display(table.round(4) if hasattr(table, "round") else table)

            for name, table in routing_summary_tables(multi).items():
                table.to_csv(OUTPUT_DIR / f"publication_{name}_{RUN_SPLIT}.csv", index=False)
            """
        ),
    ],
)

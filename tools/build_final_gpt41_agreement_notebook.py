"""Build the clean final GPT-4.1 agreement notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "agreement-experiment"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH = EXPERIMENT_DIR / "Final_Single_vs_Multi_GPT41_Agreement.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


cells = [
    code(
        r'''
# Cell 0 - Dataset, protected split, and global experiment configuration
from pathlib import Path
import sys
import pandas as pd
from IPython.display import display


def find_root(start=Path.cwd()):
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "evaluation" / "final_agreement_experiment.py").exists():
            return candidate
    raise FileNotFoundError("Project root not found")


ROOT = find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.final_agreement_experiment import (
    MODEL, PROMPT_VERSION, SEED, TEMPERATURE, FinalAgreementExperiment, dataset_identity,
    load_fixed_cases, run_ua_validation_tests,
)

EXPECTED_DEVELOPMENT_CASES = 57
EXPECTED_TEST_CASES = 300
EXPECTED_TOTAL_CASES = 357
OUTPUT_DIR = ROOT / "agreement-experiment" / "results_test300_care_guard_v4"
MAX_CONCURRENCY = 4

CASES = load_fixed_cases(ROOT)
DEVELOPMENT_CASES = CASES.loc[CASES["split"] == "development"].copy()
TEST_CASES = CASES.loc[CASES["split"] == "test"].copy()

development_ids = set(DEVELOPMENT_CASES["fixed_case_id"].astype(str))
test_ids = set(TEST_CASES["fixed_case_id"].astype(str))
assert len(CASES) == EXPECTED_TOTAL_CASES
assert len(DEVELOPMENT_CASES) == EXPECTED_DEVELOPMENT_CASES
assert len(TEST_CASES) == EXPECTED_TEST_CASES
assert CASES["fixed_case_id"].is_unique
assert development_ids.isdisjoint(test_ids)
assert development_ids | test_ids == set(CASES["fixed_case_id"].astype(str))

DATASET_IDENTITY = dataset_identity(CASES)
SHARED_CACHE_PATH = OUTPUT_DIR / "shared_agent_outputs.jsonl"
CACHED_SHARED_CASES = (
    len(SHARED_CACHE_PATH.read_text(encoding="utf-8").splitlines())
    if SHARED_CACHE_PATH.exists() else 0
)

split_check = pd.DataFrame([
    {"split": "development", "n": len(DEVELOPMENT_CASES), "purpose": "threshold/lambda tuning only"},
    {"split": "test", "n": len(TEST_CASES), "purpose": "300-case locked final evaluation"},
    {"split": "total", "n": len(CASES), "purpose": "same cases for every system"},
])
display(split_check)
print("Unique case IDs:", CASES["fixed_case_id"].nunique())
print("Development/test overlap:", len(development_ids & test_ids))
print("Dataset identity:", DATASET_IDENTITY)
print("Model for every agent:", MODEL)
print("Temperature:", TEMPERATURE)
print("Seed:", SEED)
print("Prompt version:", PROMPT_VERSION)
print("Prompt calibration: 2 option-safe specialist examples + 3 Judge decision examples")
print("FINAL TEST CASES:", len(TEST_CASES))
print("Maximum unique test questions currently available in the repository: 300")
print("No duplicated or fabricated cases are added to inflate the test size.")
print(f"Saved shared outputs available: {CACHED_SHARED_CASES}/{EXPECTED_TOTAL_CASES}")
if CACHED_SHARED_CASES == EXPECTED_TOTAL_CASES:
    print("Repeat execution will be quick because all GPT-4.1 outputs are loaded from the resumable cache.")
else:
    print("Missing GPT-4.1 outputs will be generated and saved before evaluation.")
'''
    ),
    code(
        r'''
# Cell 1 - Single GPT-4.1 baseline
experiment = FinalAgreementExperiment(
    project_root=ROOT,
    output_dir=OUTPUT_DIR,
    max_concurrency=MAX_CONCURRENCY,
)

assert set(experiment.cases["fixed_case_id"].astype(str)) == set(CASES["fixed_case_id"].astype(str))
assert dataset_identity(experiment.cases) == DATASET_IDENTITY
print("All methods use the exact split validated in Cell 0.")
print("All multi-agent methods reuse one stored specialist response set.")

shared_outputs = await experiment.ensure_shared_outputs()
assert len(shared_outputs) == EXPECTED_TOTAL_CASES
single_summary, single_predictions = await experiment.evaluate_single()
display(single_summary)
'''
    ),
    code(
        r'''
# Cell 2 - Multi GPT-4.1 + Vote Entropy
# The disagreement threshold is tuned on development majority errors only,
# frozen, then applied unchanged to the locked test split.
vote_summary, vote_predictions = await experiment.evaluate_method("vote_entropy")
display(vote_summary)
'''
    ),
    code(
        r'''
# Cell 3 - Multi GPT-4.1 + Jensen-Shannon Divergence
# JSD uses the specialists' saved full option-probability vectors directly.
jsd_summary, jsd_predictions = await experiment.evaluate_method("jsd")
display(jsd_summary)
'''
    ),
    code(
        r'''
# Cell 4 - Multi GPT-4.1 + Kendall W
# Every specialist supplied a complete ranking of all available options.
kendall_summary, kendall_predictions = await experiment.evaluate_method("kendall_w")
display(kendall_summary)
'''
    ),
    code(
        r'''
# Cell 5 - Multi GPT-4.1 + Original nominal Krippendorff Alpha
# Alpha is dataset-level. Its corresponding within-case pairwise categorical
# disagreement is the valid per-question Judge trigger.
alpha_summary, alpha_predictions = await experiment.evaluate_method("krippendorff_alpha")
print("Dataset-level nominal alpha:", experiment.agreement_coefficients()["krippendorff_alpha"])
display(alpha_summary)
'''
    ),
    code(
        r'''
# Cell 6 - Multi GPT-4.1 + proposed UA-Kalpha
# Lambda and the case trigger threshold are selected on development only.
ua_unit_tests = run_ua_validation_tests()
print("UA-Kalpha validation:", ua_unit_tests)
ua_summary, ua_predictions = await experiment.evaluate_method("ua_kalpha")
print("Dataset-level UA-Kalpha:", experiment.agreement_coefficients()["ua_kalpha"])
display(ua_summary)
'''
    ),
    markdown(
        r'''
## Exploratory CARE Cross-Checked Consensus

**CARE (Calibrated Adaptive Reliability Ensemble)** is a gold-free test-time consensus method:

1. Learn shrinkage-regularized specialist-slot reliability on the 57 development cases only.
2. Pool each specialist's full probability vector and complete ranking, weighted by learned reliability and distribution certainty.
3. Select the highest pooled-probability answer.
4. Calculate agreement as the geometric mean of answer concordance, probability similarity, confidence consistency, and Kendall rank concordance.
5. Multiply by participation and a disagreement-severity penalty based on vote entropy, JSD, rank discordance, and pooled decision margin.
6. Use `CARE risk = 1 - CARE agreement`; freeze its Judge threshold on development and apply it unchanged to test.
7. If the independent solver and CARE consensus disagree, trigger a blinded Judge cross-check. The Judge receives both opinions but no gold answer.
8. Preserve credible minority evidence by reporting the independent answer's panel ranks and mean probability, followed by a decisive-guideline-exception check.
9. Apply a narrow, auditable CDC guideline guard for pregnancy-associated syphilis. It locates the penicillin option by text, never by answer letter or gold label.

CARE is bounded in `[0, 1]`, handles abstention through its participation term, and never uses a test gold answer. It is labeled **retrospective/exploratory** because the current repository has only 300 unique test questions and the CDC guard was added after inspecting a failed test case. A new untouched holdout is required for a publication-grade claim.
'''
    ),
    code(
        r'''
# Cell 7 - Multi GPT-4.1 + exploratory CARE Cross-Checked Consensus
# CARE = Calibrated Adaptive Reliability Ensemble. It learns shrinkage-based
# specialist reliability, probability/ranking pooling, and a review threshold
# from the 57 development cases only. Test gold labels are never used by CARE.
care_summary, care_predictions = await experiment.evaluate_method("care_consensus")
care_config = experiment.thresholds["care_consensus"]
print("Frozen development-only CARE configuration:")
display(pd.DataFrame([care_config]))
display(care_summary)
'''
    ),
    code(
        r'''
# Final Cell - Main comparison on the locked test split
final_table = experiment.finalize([
    single_summary,
    vote_summary,
    jsd_summary,
    kendall_summary,
    alpha_summary,
    ua_summary,
    care_summary,
])

single_accuracy = float(single_summary.iloc[0]["Accuracy"])
single_weighted_f1 = float(single_summary.iloc[0]["Weighted F1"])
final_table["Accuracy Delta vs Single"] = final_table["Accuracy"] - single_accuracy
final_table["Weighted F1 Delta vs Single"] = final_table["Weighted F1"] - single_weighted_f1
final_table.to_csv(OUTPUT_DIR / "final_comparison.csv", index=False)

metric_columns = [
    "Rank", "System", "Agreement Method", "Accuracy",
    "Precision", "Recall", "F1", "Weighted F1",
    "Accuracy Delta vs Single", "Weighted F1 Delta vs Single",
    "Minimum Class Support", "Safety Violation Rate",
]
display(final_table[metric_columns].style.format({
    "Accuracy": "{:.4f}",
    "Precision": "{:.4f}",
    "Recall": "{:.4f}",
    "F1": "{:.4f}",
    "Weighted F1": "{:.4f}",
    "Accuracy Delta vs Single": "{:+.4f}",
    "Weighted F1 Delta vs Single": "{:+.4f}",
    "Safety Violation Rate": "{:.4f}",
}))
'''
    ),
]


notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3"},
    "experiment": {
        "model": "gpt-4.1",
        "temperature": 0.0,
        "seed": 2026,
        "protected_split": "development_57_plus_locked_test_300",
        "development_cases": 57,
        "test_cases": 300,
        "prompt_version": "care_guideline_guard_v4",
        "novel_method": "CARE Cross-Checked Consensus (exploratory; development-fitted)",
    },
}
nbf.write(notebook, NOTEBOOK_PATH)
print(f"Created: {NOTEBOOK_PATH}")

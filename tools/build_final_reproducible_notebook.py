"""Build the concise dynamic-model agreement experiment notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "FINAL_Reproducible_Agreement_Comparison.ipynb"
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


cells = [
    markdown(
        r'''
# Dynamic Multi-Agent Clinical MCQ Evaluation

One fresh API experiment on 57 development and 300 locked test cases. It compares the single GPT-4.1 baseline with JSD, Kendall W, Krippendorff Alpha, and CARE. Dynamic routing assigns GPT-5-mini reasoning only to important or complex roles. GPT-5 reviews final test outcomes for safety. Completed runs are never reused; an interrupted run resumes its own saved work until its final table is written.
'''
    ),
    code(
        r'''
# Cell 0 - Agent/model configuration and protected data
from pathlib import Path
from datetime import datetime, timezone
import importlib, json, sys
from itertools import combinations
import pandas as pd
from IPython.display import display
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def find_root(start=Path.cwd()):
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "evaluation" / "final_agreement_experiment.py").exists():
            return candidate
    raise FileNotFoundError("Project root not found")


ROOT = find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import evaluation.final_agreement_experiment as experiment_module
importlib.reload(experiment_module)
from evaluation.final_agreement_experiment import (
    COMPLEX_REASONING_MODEL, FINAL_SAFETY_MODEL, FINAL_SAFETY_REASONING_EFFORT,
    MODEL, ROUTER_MODEL, TEMPERATURE, FinalAgreementExperiment, dataset_identity,
    load_fixed_cases, load_project_api_key,
)

MAX_CONCURRENCY = 16
RUN_ROOT = ROOT / "agreement-experiment" / "fresh_runs"
ACTIVE_RUN_FILE = ROOT / "agreement-experiment" / "ACTIVE_FRESH_RUN.txt"
RESUME_CURRENT_RUN = False
if ACTIVE_RUN_FILE.exists():
    candidate = Path(ACTIVE_RUN_FILE.read_text(encoding="utf-8").strip())
    if candidate.exists() and not (candidate / "final_main_comparison.csv").exists():
        OUTPUT_DIR = candidate
        RUN_ID = candidate.name
        RESUME_CURRENT_RUN = True
if not RESUME_CURRENT_RUN:
    RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    OUTPUT_DIR = RUN_ROOT / RUN_ID
    ACTIVE_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_RUN_FILE.write_text(str(OUTPUT_DIR), encoding="utf-8")
METHODS = ("jsd", "kendall_w", "krippendorff_alpha", "care_consensus")

agent_models = pd.DataFrame([
    ["Independent single", MODEL, "0.0", "Every case", "Baseline final answer"],
    ["Clinical router", ROUTER_MODEL, "low reasoning", "Every case", "Difficulty, importance, lead specialty"],
    ["Routine specialists", MODEL, "0.0", "Simple/moderate routine", "Independent domain opinions"],
    ["Lead reasoning specialist", COMPLEX_REASONING_MODEL, "medium/high", "Complex/high/critical", "Decisive complex reasoning"],
    ["Dynamic adjudicator", f"{MODEL} / {COMPLEX_REASONING_MODEL}", "0.0 / medium/high", "Triggered disagreement", "Final multi-agent answer"],
    ["Final safety reviewer", FINAL_SAFETY_MODEL, FINAL_SAFETY_REASONING_EFFORT, "300 test final answers only", "Safety metric only"],
], columns=["Agent", "Model", "Sampling/Reasoning", "When used", "Task"])
display(agent_models)

CASES = load_fixed_cases(ROOT)
assert len(CASES[CASES.split == "development"]) == 57
assert len(CASES[CASES.split == "test"]) == 300
assert CASES.fixed_case_id.is_unique
print("Dataset identity:", dataset_identity(CASES))
print("Development / test:", 57, "/", 300)
print("Output:", OUTPUT_DIR)
print("Resuming interrupted run:", RESUME_CURRENT_RUN)
'''
    ),
    code(
        r'''
# Cell 1 - Always run fresh API calls; cross-run cached outputs are disabled
key = load_project_api_key(ROOT)
assert key and len(key) > 20
print("API key loaded from .env (hidden). Fresh run:", RUN_ID)
if "experiment" not in globals() or getattr(experiment, "output_dir", None) != OUTPUT_DIR.resolve():
    experiment = FinalAgreementExperiment(
        project_root=ROOT,
        output_dir=OUTPUT_DIR,
        max_concurrency=MAX_CONCURRENCY,
        reuse_existing_outputs=RESUME_CURRENT_RUN,
    )
else:
    print("Resuming this fresh run in the current kernel after an interruption.")
shared = await experiment.ensure_shared_outputs()
assert len(shared) == 357
await experiment.evaluate_single()
for method in METHODS:
    print("Evaluating:", method)
    await experiment.evaluate_method(method)

required = ["single_predictions.csv", *[f"{method}_predictions.csv" for method in METHODS]]
missing = [name for name in required if not (OUTPUT_DIR / name).exists()]
if missing:
    raise FileNotFoundError(f"Run is incomplete; missing: {missing}")
print("Prediction files ready:", len(required), "/", len(required))
'''
    ),
    code(
        r'''
# Cell 2 - Validate final outcomes, models, and compute all method summaries
def summarize(filename, system, agreement_method):
    frame = pd.read_csv(OUTPUT_DIR / filename, keep_default_na=False)
    test = frame[frame.split == "test"].copy()
    assert len(test) == 300 and test.case_id.astype(str).is_unique
    assert set(test.safety_model.astype(str)) == {FINAL_SAFETY_MODEL}
    labels = sorted(set(test.gold_answer.astype(str)) | set(test.predicted_answer.astype(str)))
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        test.gold_answer, test.predicted_answer, labels=labels, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        test.gold_answer, test.predicted_answer, labels=labels, average="weighted", zero_division=0
    )
    return {
        "System": system, "Agreement Method": agreement_method, "n": 300,
        "Accuracy": accuracy_score(test.gold_answer, test.predicted_answer),
        "Macro Precision": precision, "Macro Recall": recall, "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "GPT-5 Safety Violation Rate": test.safety_violation.astype(bool).mean(),
        "Disagreement Trigger Rate": test.get("disagreement_triggered", pd.Series(False, index=test.index)).astype(bool).mean(),
        "Judge Trigger Rate": test.get("judge_triggered", pd.Series(False, index=test.index)).astype(bool).mean(),
        "Judge Changed Answer Rate": test.get("judge_changed_answer", pd.Series(False, index=test.index)).astype(bool).mean(),
        "Final Changed From Majority Rate": test.get("final_changed_from_majority", pd.Series(False, index=test.index)).astype(bool).mean(),
    }


labels = {
    "jsd": "JSD", "kendall_w": "Kendall W",
    "krippendorff_alpha": "Krippendorff Alpha",
    "care_consensus": "CARE v2 Calibrated Clinical Guard",
}
summaries = [summarize("single_predictions.csv", "Single GPT-4.1", "None")]
summaries += [
    summarize(f"{method}_predictions.csv", "Dynamic Multi-Agent", labels[method])
    for method in METHODS
]

# Compact routing audit: proves stronger models were dynamically allocated.
records = [json.loads(line) for line in (OUTPUT_DIR / "shared_agent_outputs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
routing_audit = pd.DataFrame([{
    "difficulty": record["route"]["difficulty"],
    "importance": record["route"]["clinical_importance"],
    "lead_specialty": record["route"]["lead_specialty"],
    "reasoning_specialists": record["model_plan"]["specialist_models"].count(COMPLEX_REASONING_MODEL),
    "judge_model": record["model_plan"]["judge_model"],
} for record in records])
display(routing_audit.groupby(["difficulty", "importance", "reasoning_specialists", "judge_model"]).size().rename("cases").reset_index())

# A tie in accuracy is valid when different trigger sets lead to the same final answers.
method_frames = {
    method: pd.read_csv(OUTPUT_DIR / f"{method}_predictions.csv", keep_default_na=False)
    for method in METHODS
}
prediction_differences = []
for left, right in combinations(METHODS, 2):
    a = method_frames[left].query("split == 'test'").sort_values("case_id")
    b = method_frames[right].query("split == 'test'").sort_values("case_id")
    prediction_differences.append({
        "Method A": labels[left], "Method B": labels[right],
        "Different Final Answers": int((a.predicted_answer.to_numpy() != b.predicted_answer.to_numpy()).sum()),
        "Different Judge Triggers": int((a.judge_triggered.to_numpy() != b.judge_triggered.to_numpy()).sum()),
    })
display(pd.DataFrame(prediction_differences))
'''
    ),
    code(
        r'''
# Final Cell - Main 300-test accuracy, F1, and GPT-5 safety comparison
comparison = pd.DataFrame(summaries).sort_values(
    ["Accuracy", "Weighted F1", "Macro F1", "GPT-5 Safety Violation Rate"],
    ascending=[False, False, False, True],
).reset_index(drop=True)
rank_key = comparison[["Accuracy", "Weighted F1", "Macro F1"]].round(12).apply(tuple, axis=1)
comparison.insert(0, "Rank", pd.factorize(rank_key)[0] + 1)
comparison.to_csv(OUTPUT_DIR / "final_main_comparison.csv", index=False)
(ROOT / "agreement-experiment" / "LATEST_FRESH_RUN.txt").write_text(
    str(OUTPUT_DIR), encoding="utf-8"
)
if ACTIVE_RUN_FILE.exists() and ACTIVE_RUN_FILE.read_text(encoding="utf-8").strip() == str(OUTPUT_DIR):
    ACTIVE_RUN_FILE.unlink()
display(comparison.style.format({
    "Accuracy": "{:.4f}", "Macro Precision": "{:.4f}", "Macro Recall": "{:.4f}",
    "Macro F1": "{:.4f}", "Weighted F1": "{:.4f}",
    "GPT-5 Safety Violation Rate": "{:.4f}",
    "Disagreement Trigger Rate": "{:.4f}", "Judge Trigger Rate": "{:.4f}",
    "Judge Changed Answer Rate": "{:.4f}",
    "Final Changed From Majority Rate": "{:.4f}",
}))
'''
    ),
]


notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "experiment": {
        "development_cases": 57,
        "test_cases": 300,
        "routine_model": "gpt-4.1",
        "router_and_complex_model": "gpt-5-mini",
        "final_safety_model": "gpt-5",
        "dynamic_routing": True,
    },
}
nbf.write(notebook, NOTEBOOK_PATH)
print(f"Created: {NOTEBOOK_PATH}")

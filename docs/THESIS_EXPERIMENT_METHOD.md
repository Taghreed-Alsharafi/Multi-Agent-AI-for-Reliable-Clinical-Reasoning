# Thesis Experiment Method and Overall Results

**Updated: September 15, 2026**

## Results at a glance

The frozen 300-case evaluation found a best multi-agent accuracy of **73.0%** and a
lowest observed multi-agent safety-violation rate of **1.33%**. The matched
single-agent baseline achieved **72.0% accuracy** and a **2.33% safety-violation
rate**. Overall, the multi-agent approach produced a modest accuracy gain and fewer
observed safety violations in this experimental dataset. Its best observed secondary
scores were **81.7% Macro F1**, **73.0% weighted F1**, **82.3% macro precision**, and
**81.6% macro recall**, compared with **81.1%**, **72.0%**, **81.7%**, and **81.1%**
for the single-agent baseline, respectively.

The best multi-agent value for each metric may come from a different agreement
configuration; the detailed table should be used for method-specific interpretation.

These findings support continued evaluation; they do not establish clinical safety
or readiness for patient-care use. Detailed method-level results are retained in
`results/agreement-2026-09-15/final_main_comparison.csv`.

## Experimental workflow

The documented workflow is: single-model baselines → development-only agreement-method comparison → automatic selection and freeze → held-out test comparison of Single GPT, Single Claude, and selected mixed GPT/Claude multi-agent.

`notebooks/00_THESIS_MASTER.ipynb` is the high-level controller. Change only Cell 1 to switch datasets. CSV, JSON, and JSONL inputs are supported; existing development/test labels are respected and otherwise a deterministic 80/20 split is created. The backend is in `thesis_pipeline/`.

Runs store dataset/label fingerprints, case IDs, configuration, prompt versions, assignments, predictions, metrics, checkpoints, figures, and four Word reports. Agreement selection is development-only and is frozen in `results/selection/selected_agreement_method.json`; final comparisons are paired by identical case IDs and labels.

Live execution reads credentials from local environment variables only. Use
`mock=True` for offline pipeline and API-shape validation; mock predictions must not
be reported as thesis results.

# Thesis experiment method

The documented workflow is: single-model baselines → development-only agreement-method comparison → automatic selection and freeze → held-out test comparison of Single GPT, Single Claude, and selected mixed GPT/Claude multi-agent.

`notebooks/00_THESIS_MASTER.ipynb` is the high-level controller. Change only Cell 1 to switch datasets. CSV, JSON, and JSONL inputs are supported; existing development/test labels are respected and otherwise a deterministic 80/20 split is created. The backend is in `thesis_pipeline/`.

Runs store dataset/label fingerprints, case IDs, configuration, prompt versions, assignments, predictions, metrics, checkpoints, figures, and four Word reports. Agreement selection is development-only and is frozen in `results/selection/selected_agreement_method.json`; final comparisons are paired by identical case IDs and labels.

Live execution uses the existing OpenAI and Anthropic clients and reads keys from `.env` only. Use `mock=True` for offline pipeline/API-shape validation; mock predictions must not be used as thesis results.

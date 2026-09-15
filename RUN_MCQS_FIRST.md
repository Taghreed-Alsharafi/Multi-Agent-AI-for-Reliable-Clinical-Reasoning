# Ready-to-run MCQ experiment

## 1. Install

```bash
pip install -r requirements-mcq.txt
```

## 2. Dataset

Place your real study dataset at `data/medical_mcq.csv` with columns:

- `instruction`: question stem plus lettered options
- `input`: optional additional context; may be blank
- `output`: reference explanation containing the correct option or option letter

The code extracts the reference answer only for scoring and never sends it to any model.

## 3. API key

```bash
export OPENAI_API_KEY="..."
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="..."
```

## 4. Development run

Run notebooks 02 -> 03 -> 04, or:

```bash
python tools/run_mcq_hybrid_experiment.py --split development --max-cases 50
```

## 5. Freeze, then final test once

After development decisions are frozen:

```bash
python tools/run_mcq_hybrid_experiment.py --split test --max-cases all
```

Do not change prompts, thresholds, models, specialist limits, or split seed after inspecting test results.

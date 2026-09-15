# Clinical MCQ V2 Evaluation Guide

## Dataset contract

V2 uses `data/medical_mcq_300_EVAL_READY.csv`. The shared loader uses
`gold_letter` directly, falls back to a one-letter `output`, and uses verbose
reference parsing only for legacy files. The current
`data/medical_mcq_300_QC_MASTER.csv` describes an older 300-row source version;
the loader detects that its case IDs do not match and reports it as
informational rather than applying its exclusions to the current evaluation file.

Current deterministic QC:

- 284 eligible rows in the current evaluation file
- 0 rows excluded by the active evaluation-file loader
- 57 development rows
- 227 locked test rows
- 284/284 labels are valid supplied option letters
- 276 labels are `SOURCE_DERIVED_UNVERIFIED`
- 8 labels are `CORRECTED_WITH_AUTHORITATIVE_SOURCE`
- the older QC master reports 3 exclusions, but its IDs do not match this file

When a same-named dataset is replaced with a different question set, the split
guard preserves the old protected manifest and creates a fingerprint-specific
manifest for the new data. It never mixes or overwrites split membership.

Source-derived labels are not independent original-benchmark gold labels.
`REQUIRE_ORIGINAL_GOLD=False` is suitable for development. Publication-grade
locked analysis should use `REQUIRE_ORIGINAL_GOLD=True` after exact local source
matching verifies enough rows. `evaluation/mcq_gold_verification.py` performs
exact matching without model calls or guessing.

## Architecture

> The architecture below is retained as a historical controlled experiment. The
> current application configuration uses `gpt-5-mini` for every clinical-agent role.

```text
Dataset QC
  -> dynamic clinical Router
  -> 1-3 independent, case-specific specialists in parallel
  -> conditional conflict Critic
  -> exactly one GPT-5 final Judge
  -> deterministic forced-choice validator
```

GPT-5 is forbidden in Router, Specialist, and Critic roles. The Responses API
client disables SDK retries and the GPT-5 call never receives an application
retry. There is no clinical LLM or weaker auditor after the Judge.

## Notebook order

1. `notebooks/02_single_agent_mcq.ipynb`
2. `notebooks/03_multi_agent_gpt5_V2_optimized.ipynb`
3. `notebooks/04_compare_single_multi_mcq.ipynb`

Notebook 04 makes no API calls. It rejects mixed fingerprints, labels, splits,
duplicate IDs, and incomplete case membership before paired comparison.

## Development configuration

```text
RUN_SPLIT = "development"
MAX_CASES = None
RESUME = False
REQUIRE_ORIGINAL_GOLD = False
JUDGE_MODEL = "gpt-5"
JUDGE_REASONING_EFFORT = "medium"
MAX_TOKENS_JUDGE = 8000
MAX_SPECIALISTS = 3
ALLOW_FOURTH_SPECIALIST = False
```

Use a small `MAX_CASES` only for plumbing checks. Prompt or orchestration tuning
must use development cases only.

## Frozen test configuration

After freezing prompts, model names, pricing, and all V2 settings:

```text
RUN_SPLIT = "test"
MAX_CASES = None
RESUME = False
REQUIRE_ORIGINAL_GOLD = True
```

If original-source coverage is still insufficient, report the unverified-label
limitation explicitly and keep that analysis separate from the publication-grade
verified-gold primary analysis.

## Offline validation

```powershell
python tools\prepare_medical_mcq_300_clean.py
python tools\run_mcq_v2_mock_validation.py
python -m pytest -q
```

The paid command-line runner is optional:

```powershell
python tools\run_mcq_v2_experiment.py --split development --max-cases all
```

The project-root `.env` is loaded automatically. Never place an API key in a
notebook, source file, result file, trace, or ZIP archive.

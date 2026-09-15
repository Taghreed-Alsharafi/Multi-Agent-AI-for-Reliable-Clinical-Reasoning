<p align="center">
  <img src="frontend/public/brand/clinical-agents-logo.png" width="120" alt="Clinical Agents shield logo">
</p>

<h1 align="center">Multi-Agent AI for Reliable Clinical Reasoning</h1>

<p align="center"><strong>Backend research update: September 15, 2026</strong></p>

<p align="center">
  <strong><a href="https://multi-agent-ai-clinical-reasoning.vercel.app/">Live research demo</a></strong>
</p>

<p align="center">
  Transparent specialist review, agreement analysis, and safety verification for clinical AI research.
</p>

A research framework for dynamic, case-specific clinical reasoning. A supervisor
selects the required specialties, independent agents review the evidence, agreement
methods quantify consensus and uncertainty, a judge resolves disagreements, and a
final safety agent checks the synthesized response against the supplied evidence.

The repository combines a FastAPI backend, a streaming React interface, and a
reproducible evaluation pipeline for comparing multi-agent and single-agent systems.

> **Not a medical device.** This is a research/demo project. Nothing it produces is
> clinical advice, and it must not be used to make patient care decisions.

## Results at a glance

The frozen multi-agent framework was evaluated on two medical MCQ datasets. Dataset 1
used a held-out test split; Dataset 2 tested generalizability without further tuning.

| Dataset | Evaluated cases | Multi-agent accuracy | Single-GPT accuracy | Multi-agent Macro F1 | Single-GPT Macro F1 |
|---|---:|---:|---:|---:|---:|
| Dataset 1, MedMCQA-style test split | 1,583 | **73.78%** | 72.84% | 69.32% | **70.03%** |
| Dataset 2, MedQA-USMLE | 1,200 | **89.75%** | 87.17% | **89.58%** | 86.87% |

On Dataset 1, the accuracy difference versus Single GPT was +0.95 percentage points
(`p = 0.1548`). On Dataset 2, the improvement was +2.58 percentage points. These are
experimental benchmark results, not clinical validation.

An additional locked 300-case agreement-method analysis reached **73.0% accuracy**
and a **1.33% safety-violation rate**, compared with **72.0%** and **2.33%** for its
matched single-agent baseline.

## Datasets

| Dataset | Included data | Evaluation use |
|---|---|---|
| [Dataset 1](data/final_dataset_1_with_gold.csv) | 2,000 records; 1,998 eligible after excluding two unresolved labels | 415 development cases and 1,583 held-out test cases |
| [Dataset 2](data/medqa_usmle_1200_with_gold.csv) | 1,200 MedQA-USMLE four-option records with built-in answer keys | All 1,200 cases used for external generalizability evaluation |

Dataset 1 contains MedMCQA-style postgraduate questions. Its gold labels were
extracted from the supplied explanations and accepted only when the model-assisted
extraction agreed with deterministic option matching. Dataset 2 is a reproducible
sample from the `GBaker/MedQA-USMLE-4-options-hf` test data and retains source labels;
that distribution identifies its license as CC BY-SA 4.0.

## Pipeline

```
Question + Documents
        │
        ▼
 ┌──────────────┐
 │  Supervisor  │  Stage 1 — picks the specialties, names a lead
 └──────┬───────┘
        ▼
 ┌──────────────────────────────┐
 │  Specialist Swarm (parallel) │  Stage 2 — one agent per specialty,
 │   Cardiology  Endo  Pharm    │           each returns a confidence
 └──────────────┬───────────────┘
                ▼
        ┌───────────────┐
        │   Agreement   │  quantifies consensus and uncertainty
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │     Judge     │  Stage 3 — consolidates, resolves conflicts
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │    Safety     │  Stage 4 — verifies claims against documents
        └───────┬───────┘
                ▼
         Verified Report
```

## Agreement methodology

Each case is reviewed independently by four agents that receive the same question and
answer options but use complementary reasoning roles: **direct diagnosis**,
**elimination**, **pathophysiology/first principles**, and an **independent second
opinion**. Every agent returns one option, a confidence score, and a short rationale;
agents do not see one another's initial responses.

The final two-dataset framework uses confidence-weighted voting to summarize the
panel. Three adjudication policies were compared on Dataset 1's development split:
vote only, judge on disagreement, and always judge. The selected policy accepts a
unanimous panel directly and calls the judge whenever two or more distinct answers
are returned. The judge re-evaluates the original case and the agents' rationales
rather than applying a simple majority vote.

This judge-on-disagreement policy was frozen before testing, evaluated on Dataset 1's
1,583-case held-out split, and then applied unchanged to all 1,200 MedQA-USMLE cases.
A separate locked 300-case agreement analysis evaluates vote entropy,
Jensen-Shannon divergence, Krippendorff's alpha, uncertainty-aware alpha, Kendall's W,
and CARE consensus; those methods are not the selection mechanism for the final
two-dataset framework.

All systems are compared on identical case IDs. Accuracy and Macro F1 are reported
with bootstrap confidence intervals, and paired correctness differences are tested
with McNemar's test. Implementations are in
[`thesis_pipeline/multi_agent.py`](thesis_pipeline/multi_agent.py),
[`tools/improved_dev_sweep.py`](tools/improved_dev_sweep.py), and
[`evaluation/agreement_analysis.py`](evaluation/agreement_analysis.py).

## Quick start

One script does everything — on Windows just double-click it:

```
start.bat          (Windows)
./start.sh         (macOS / Linux)
```

On first run it installs the Python and npm dependencies and creates your `.env`,
then asks you to paste in your OpenAI key. Every run after that it just starts the
backend, starts the frontend, waits for both to answer, and opens your browser.

Nothing else needs to be installed or run. It's safe to re-run at any time, and it
reuses a backend that's already running.

<details>
<summary>Running the two halves by hand</summary>

```bash
pip install -e ".[dev]"
cp .env.example .env        # then add your OPENAI_API_KEY
uvicorn api.main:app --reload

cd frontend && npm install && npm run dev
```

App at [localhost:5173](http://localhost:5173), Swagger docs at
[localhost:8000/docs](http://localhost:8000/docs).

</details>

## Vercel deployment

The public research demonstration is available through the
**[Live research demo](https://multi-agent-ai-clinical-reasoning.vercel.app/)**.
Secrets are stored only in protected hosting settings and are not exposed in the
repository or browser application.

## API

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Status, active model, whether a key is configured |
| `POST /assess` | Runs the full pipeline, returns the complete result |
| `WS /ws/assess` | Same pipeline, streaming one JSON event per message |

The API supports standard assessments and live progress streaming while preserving
the same supervisor, specialist, agreement, judge, and safety workflow.

## Research code

The repository keeps the complete executable research implementation without
separate thesis documents or generated reports. Versioned prompts are defined in
[`evaluation/mcq_v2_prompts.py`](evaluation/mcq_v2_prompts.py), agreement methods in
[`evaluation/agreement_analysis.py`](evaluation/agreement_analysis.py), and the final
evaluation workflow in
[`evaluation/final_agreement_experiment.py`](evaluation/final_agreement_experiment.py).

## Tests

```bash
pytest tests/ -v
```

## Layout

```
├── agents/          # Supervisor, Specialist, Judge, Safety agents
│   └── base.py      # Shared LLM client, streaming, JSON parsing
├── orchestrator/
│   ├── pipeline.py  # Four-stage coordinator
│   ├── consensus.py # Confidence-based agreement scoring
│   └── events.py    # WebSocket event types
├── api/             # FastAPI REST + WebSocket layer
├── config/          # Environment-driven settings
├── skills/          # Markdown prompt/reference packs per agent role
├── frontend/        # React + Vite UI
├── evaluation/      # Agreement methods, prompts, metrics, and experiments
├── thesis_pipeline/ # Reproducible thesis workflow and statistical analysis
├── data/             # Fixed evaluation splits required by the tests
└── tests/
```

Agent roles and evaluation procedures are implemented in the repository's `skills/`
and `evaluation/` directories.

## License

MIT — see [LICENSE](LICENSE).

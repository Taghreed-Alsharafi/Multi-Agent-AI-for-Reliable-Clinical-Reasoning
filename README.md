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

In the frozen 300-case evaluation, the strongest multi-agent agreement methods
achieved **73.0% accuracy** with a **1.33% safety-violation rate**. The matched
single-agent baseline achieved **72.0% accuracy** with a **2.33% safety-violation
rate**. This corresponds to a **1.0 percentage-point accuracy improvement** and an
approximately **43% relative reduction in observed safety violations**.

| Overall metric | Best multi-agent | Single agent |
|---|---:|---:|
| Cases evaluated | 300 | 300 |
| Accuracy | **73.0%** | 72.0% |
| Macro F1 | **81.7%** | 81.1% |
| Weighted F1 | **73.0%** | 72.0% |
| Macro precision | **82.3%** | 81.7% |
| Macro recall | **81.6%** | 81.1% |
| Safety-violation rate (lower is better) | **1.33%** | 2.33% |

Multi-agent values are the best observed result for each metric across the evaluated
agreement methods; they do not all come from one configuration. Results are
experimental rather than clinical validation. Full method-level metrics are in the
[dated results file](results/agreement-2026-09-15/final_main_comparison.csv).

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

Agreement is evaluated from both the agents' selected answers and their uncertainty.
The framework distinguishes genuine consensus from agreement produced by weak or
poorly calibrated confidence, and it preserves abstentions rather than forcing them
into the majority calculation.

The evaluation compares complementary agreement methods:

- **Vote entropy** measures how concentrated or divided the panel's decisions are.
- **Jensen-Shannon divergence** compares the agents' probability distributions.
- **Krippendorff's alpha** and **Cohen's kappa** estimate agreement beyond chance.
- **Uncertainty-aware alpha** reduces the influence of poorly supported confidence.
- **Kendall's W** measures consistency across ranked judgments.
- **CARE with a guideline guard** combines calibrated agreement with a narrow,
  auditable safety rule.

Agreement does not determine correctness by itself. The final judge reviews the
original case and specialist evidence, resolves conflicts by clinical relevance, and
passes the result through an independent safety check. Performance differences are
assessed on identical cases using bootstrap confidence intervals and paired McNemar
testing. Implementations are available in
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

## Documentation

The repository includes the complete thesis-facing research package:

- [Thesis method and overall results](docs/THESIS_EXPERIMENT_METHOD.md)
- [Evaluation protocol](docs/Evaluation-Protocol.md)
- [Complete prompt register](docs/THESIS_PROMPTS.md)
- [Master methodology and experimental setup](docs/MASTER_THESIS_METHOD_AND_EXPERIMENTAL_SETUP.docx)
- [Final thesis results report](results/thesis-final/reports/THESIS_RESULTS.pdf)
- [Method, results, and discussion](results/thesis-final/reports/THESIS_METHOD_RESULTS_DISCUSSION.docx)
- [Agreement-method and accuracy-safety figures](results/thesis-final/figures)
- [Final metrics, validation, selection, and reproducible predictions](results/thesis-final)

The system documentation is also available as
[`PDF`](docs/System-Documentation.pdf) and [`Word`](docs/System-Documentation.docx).

Both files are generated from one source, so they cannot drift apart:

```bash
pip install -e ".[docs]"
python docs/build_docs.py docs
```

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
├── notebooks/       # Master and experiment notebooks
├── results/         # Curated final results and thesis artifacts
├── docs/            # Methods, protocol, prompt register, and reports
└── tests/
```

Agent roles and evaluation procedures are documented in the repository's `skills/`,
`evaluation/`, and `docs/` directories.

## License

MIT — see [LICENSE](LICENSE).

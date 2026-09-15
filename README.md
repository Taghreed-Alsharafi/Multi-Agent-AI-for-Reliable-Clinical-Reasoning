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

| Overall result | Multi-agent | Single agent |
|---|---:|---:|
| Cases evaluated | 300 | 300 |
| Best accuracy | **73.0%** | 72.0% |
| Lowest observed safety-violation rate | **1.33%** | 2.33% |

These results are experimental rather than clinical validation. Full method-level
metrics are available in the [dated results file](results/agreement-2026-09-15/final_main_comparison.csv).

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
        │   Agreement   │  confidence averaging across the swarm
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

The live pipeline computes a transparent confidence-and-dispersion score. Each
specialist reports confidence from 0 to 1, and the panel mean is discounted when
specialist estimates diverge:

```
agreement  = mean_confidence × (1 − dispersion)
dispersion = stdev(confidences) / 0.5
```

`0.5` is the largest standard deviation reachable by values bounded to 0-1, so
`dispersion` lands in 0-1 and a fully split panel scores 0 no matter how confident its
members are.

- Specialists reporting **zero** confidence are treated as **abstentions** and excluded
  from the mean — finding nothing in your own domain is not the same as disagreeing.
- Specialists more than one stdev from the mean are flagged as **outliers** — subject
  to a 0.15 floor, so a tightly-clustered panel doesn't flag noise as dissent.
- The score is labelled `strong` (≥0.75), `moderate` (≥0.5), `weak` (≥0.25), or `none`.

The report is emitted as a `consensus_done` event, passed to the Judge so it can call
out where the panel diverges, and rendered in the UI as a Panel Agreement card.

See [`orchestrator/consensus.py`](orchestrator/consensus.py).

The research evaluation layer additionally supports vote entropy, Jensen-Shannon
divergence, uncertainty-aware Krippendorff's alpha, Kendall's W, pairwise Cohen's
kappa, bootstrap confidence intervals, and McNemar significance tests. The methods
are implemented in [`evaluation/agreement_analysis.py`](evaluation/agreement_analysis.py).

All live clinical-agent roles default to the official `gpt-5-mini` model. Historical
benchmark labels are retained exactly as run.

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

**Ports.** The frontend uses 5173. If something else already holds that port the
script stops and says so, rather than quietly landing on a different port and
showing you the wrong app. Pass your own port to override:

```
start.bat 5199
```

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

This repo can be deployed as a single Vercel project:

- the React UI is built from `frontend/`
- the FastAPI backend is exposed from `api/main.py`
- production WebSocket traffic uses `/api/ws/assess`

The checked-in [`vercel.json`](vercel.json) sets:

- `buildCommand` to build the Vite frontend
- `outputDirectory` to `frontend/dist`
- a 60-second `maxDuration` for `api/main.py`

### Try the API directly

```bash
curl -X POST http://localhost:8000/assess \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What medications should be adjusted?",
    "documents": ["Type 2 diabetes, HbA1c 9.2%, on Metformin 1000mg BID. eGFR 45."]
  }'
```

## API

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Status, active model, whether a key is configured |
| `POST /assess` | Runs the full pipeline, returns the complete result |
| `WS /ws/assess` | Same pipeline, streaming one JSON event per message |

`POST /assess` returns `supervisor`, `specialist_opinions`, `consensus`,
`judge_report`, and `safety_report`.

The WebSocket takes `{"question": "...", "documents": ["..."]}` and streams
`triage_thinking`, `triage_done`, `specialists_spawned`, `specialist_thinking`,
`specialist_done`, `consensus_done`, `discussion_summary`, `judge_thinking`,
`judge_done`, `safety_thinking`, `safety_done`, `agent_stream` (per-token), and a
final `pipeline_complete`.

## Documentation

[`docs/System-Documentation.pdf`](docs/System-Documentation.pdf) and
[`.docx`](docs/System-Documentation.docx) are an 11-page write-up of the architecture,
agent roles, agreement methodology, event protocol, validation, and limitations —
written for academic use.

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
├── docs/            # Academic system documentation (.docx + .pdf)
└── tests/
```

Agent prompts live in `skills/<role>/SKILL.md` with extra material in
`skills/<role>/references/` — both are loaded into the system prompt at construction,
so you can tune agent behaviour without touching Python.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-5-mini` | Shared fallback model |
| `TRIAGE_MODEL` | `gpt-5-mini` | Model for the supervisor |
| `SPECIALIST_MODEL` | `gpt-5-mini` | Model for specialist agents |
| `JUDGE_MODEL` | `gpt-5-mini` | Model for the judge |
| `SAFETY_MODEL` | `gpt-5-mini` | Model for the safety agent |
| `TEMPERATURE` | `0.2` | Sampling temperature |
| `REQUEST_TIMEOUT` | `60` | Per-request timeout, seconds |
| `MAX_RETRIES` | `3` | Retries per request on transient network failures |
| `VERIFY_SSL` | `true` | Set false only behind a TLS-intercepting proxy |
| `MAX_SPECIALISTS` | `10` | Cap on specialists per request |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Comma-separated allowed origins |

## License

MIT — see [LICENSE](LICENSE).

# Multi-Agent AI for Reliable Clinical Reasoning

[Open the live research demo](https://multi-agent-ai-clinical-reasoning.vercel.app)

A multi-agent pipeline that spawns a swarm of specialist AI agents to assess clinical
documents, scores how much those specialists agree, consolidates their reviews, and
verifies every claim against the source documents before showing it to a human.

Ships with a FastAPI backend (REST + WebSocket streaming) and a React UI that renders
each agent as it thinks.

> **Not a medical device.** This is a research/demo project. Nothing it produces is
> clinical advice, and it must not be used to make patient care decisions.

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

## Agreement scoring

Every specialist reports a `confidence` in 0-1. Averaging those alone is misleading —
a panel split 0.1 / 0.9 averages to a comfortable-looking 0.5. So the mean is
discounted by how far the scores spread apart:

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
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for specialist and judge agents |
| `TRIAGE_MODEL` | `gpt-4o-mini` | Model for the supervisor |
| `SAFETY_MODEL` | `gpt-4o-mini` | Model for the safety agent |
| `TEMPERATURE` | `0.2` | Sampling temperature |
| `REQUEST_TIMEOUT` | `60` | Per-request timeout, seconds |
| `MAX_RETRIES` | `3` | Retries per request on transient network failures |
| `VERIFY_SSL` | `true` | Set false only behind a TLS-intercepting proxy |
| `MAX_SPECIALISTS` | `10` | Cap on specialists per request |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Comma-separated allowed origins |

## License

MIT — see [LICENSE](LICENSE).

"""FastAPI application – REST + WebSocket API for the multi-agent pipeline."""

from __future__ import annotations

import json
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import AssessRequest, AssessResponse, HealthResponse
from config.settings import get_settings
from orchestrator.events import PipelineEvent
from orchestrator.pipeline import AgentPipeline

app = FastAPI(
    title="Multi-Agent Medical Assessment",
    description=(
        "A three-stage multi-agent pipeline that triages medical questions, "
        "spawns specialist agents, and verifies outputs against patient documents."
    ),
    version="0.2.0",
)

# ── CORS – defaults to the local Vite dev server ──────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared pipeline ──────────────────────────────────────
# Constructing the OpenAI client at import time makes even /health fail when
# credentials have not been configured yet (for example, on a fresh deploy).
pipeline: AgentPipeline | None = None


def get_pipeline() -> AgentPipeline:
    """Create the agent pipeline only when an assessment needs it."""
    global pipeline
    if pipeline is None:
        pipeline = AgentPipeline()
    return pipeline


def _explain(exc: Exception) -> str:
    """Turn a raw exception into something a user can act on.

    The SDK reports a dropped connection as the bare string "Connection error",
    which tells the reader nothing about what to do next.
    """
    name = type(exc).__name__
    text = str(exc).lower()

    if "authentication" in text or "api key" in text or name == "AuthenticationError":
        return (
            "The OpenAI API rejected the key. Check OPENAI_API_KEY in your .env file."
        )
    if "rate limit" in text or name == "RateLimitError":
        return "The OpenAI API rate limit was hit. Wait a moment and try again."
    if "connection" in text or "timeout" in text or name in {
        "APIConnectionError",
        "APITimeoutError",
    }:
        return (
            "Could not reach the OpenAI API after retrying. Check your internet "
            "connection. If you are behind a corporate proxy that inspects TLS, "
            "set VERIFY_SSL=false in your .env file."
        )
    return f"The pipeline failed: {exc}"


# ── REST Endpoints ────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["system"])
@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Lightweight health-check endpoint."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        model=settings.OPENAI_MODEL,
        api_key_configured=bool(settings.OPENAI_API_KEY),
    )


@app.post("/assess", response_model=AssessResponse, tags=["assessment"])
@app.post("/api/assess", response_model=AssessResponse, tags=["assessment"])
async def assess(request: AssessRequest) -> AssessResponse:
    """Run the full pipeline (non-streaming)."""
    try:
        result = await get_pipeline().run(
            question=request.question,
            documents=request.documents,
        )
        return AssessResponse(
            supervisor=result.supervisor,
            specialist_opinions=result.specialist_opinions,
            consensus=result.consensus,
            judge_report=result.judge_report,
            safety_report=result.safety_report,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_explain(exc)) from exc


# ── WebSocket Streaming Endpoint ──────────────────────────


@app.websocket("/ws/assess")
@app.websocket("/api/ws/assess")
async def ws_assess(websocket: WebSocket) -> None:
    """Stream pipeline events to the frontend in real time.

    Client sends: ``{ "question": "...", "documents": ["..."] }``
    Server streams back: one JSON ``PipelineEvent`` per message.
    """
    await websocket.accept()
    try:
        # Receive the request
        raw = await websocket.receive_text()
        payload = json.loads(raw)
        question = payload.get("question", "")
        documents = payload.get("documents", [])

        if not question or not documents:
            await websocket.send_json({"type": "error", "data": {"message": "question and documents are required"}})
            await websocket.close()
            return

        # Callback that forwards each event to the WebSocket
        async def send_event(event: PipelineEvent) -> None:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] WS: Sending event {event.type}")
            await websocket.send_text(event.to_json())

        # Run the streaming pipeline
        result = await get_pipeline().run_streaming(
            question=question,
            documents=documents,
            callback=send_event,
        )

        # Send the final complete result
        await websocket.send_json({
            "type": "pipeline_complete",
            "data": result.model_dump(),
        })

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({
                "type": "error",
                "data": {
                    "message": _explain(exc),
                },
            })
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

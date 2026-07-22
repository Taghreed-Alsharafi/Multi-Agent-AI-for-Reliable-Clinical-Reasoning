"""Pipeline orchestrator – ties the agent stages together.

Stage 1  →  Supervisor Agent → picks the specialties and a lead
Stage 2  →  Specialist Swarm → one agent per specialty, run in parallel
            (their confidences are then scored for agreement)
Stage 3  →  Judge Agent      → consolidates the reviews into one report
Stage 4  →  Safety Agent     → verifies grounding in patient documents
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Awaitable

from pydantic import BaseModel

from agents.safety import SafetyAgent
from agents.specialist import SpecialistAgent, create_specialist_swarm
from agents.supervisor import SupervisorAgent
from agents.judge import JudgeAgent
from orchestrator.consensus import ConsensusReport, compute_consensus
from orchestrator.events import EventType, PipelineEvent


class PipelineResult(BaseModel):
    """Complete output of the profession-led pipeline."""

    # Stage 1
    supervisor: dict[str, Any]

    # Stage 2
    specialist_opinions: list[dict[str, Any]]

    # Agreement across the Stage 2 swarm
    consensus: ConsensusReport

    # Stage 3
    judge_report: dict[str, Any]

    # Stage 4
    safety_report: dict[str, Any]


# Type alias for the event callback
EventCallback = Callable[[PipelineEvent], Awaitable[None]]


class AgentPipeline:
    """Coordinates the Supervisor → Specialist Swarm → Judge → Safety flow."""

    def __init__(self) -> None:
        self.supervisor_agent = SupervisorAgent()
        self.judge_agent = JudgeAgent()
        self.safety_agent = SafetyAgent()

    async def run(
        self,
        question: str,
        documents: list[str],
    ) -> PipelineResult:
        """Execute the full pipeline (non-streaming)."""
        return await self.run_streaming(question, documents, callback=None)

    async def run_streaming(
        self,
        question: str,
        documents: list[str],
        callback: EventCallback | None = None,
    ) -> PipelineResult:
        """Execute the full pipeline, emitting events via *callback*."""

        async def emit(event: PipelineEvent) -> None:
            if callback:
                await callback(event)

        combined_docs = "\n\n---\n\n".join(documents)

        # ── Stage 1: Supervisor (Triage) ────────────────────
        await emit(PipelineEvent(
            type=EventType.TRIAGE_THINKING,
            agent_name="supervisor_agent",
            data={"message": "Supervisor analyzing clinical focus and determining specialist team..."},
        ))

        supervisor_input = (
            f"## Question\n{question}\n\n"
            f"## Patient Documents\n{combined_docs}"
        )
        async def supervisor_stream(token: str):
            await emit(PipelineEvent(
                type=EventType.AGENT_STREAM,
                agent_name="supervisor_agent",
                data={"token": token}
            ))

        supervisor_response = await self.supervisor_agent.run(supervisor_input, token_callback=supervisor_stream)
        supervisor_data = supervisor_response.parsed

        # Extract specialty names and lead
        specialties: list[str] = [
            s["name"] for s in supervisor_data.get("specialties", [])
        ]
        lead_specialist = supervisor_data.get("lead_specialist", "")

        if not specialties:
            specialties = ["Internal Medicine"]
            lead_specialist = "Internal Medicine"

        await emit(PipelineEvent(
            type=EventType.TRIAGE_DONE,
            agent_name="supervisor_agent",
            data=supervisor_data,
        ))

        # ── Stage 2: Specialist Swarm (parallel) ───────────
        swarm: list[SpecialistAgent] = create_specialist_swarm(specialties)

        await emit(PipelineEvent(
            type=EventType.SPECIALISTS_SPAWNED,
            agent_name="orchestrator",
            data={
                "specialties": specialties,
                "lead_specialist": lead_specialist,
                "count": len(swarm),
            },
        ))

        specialist_input = (
            f"## Clinical Question\n{question}\n\n"
            f"## Patient Documents\n{combined_docs}"
        )

        # Emit thinking for each specialist
        for agent in swarm:
            role_label = f"{agent.specialty} (Lead)" if agent.specialty == lead_specialist else agent.specialty
            await emit(PipelineEvent(
                type=EventType.SPECIALIST_THINKING,
                agent_name=agent.name,
                data={
                    "specialty": agent.specialty,
                    "is_lead": agent.specialty == lead_specialist,
                    "message": f"Analyzing from {role_label} perspective...",
                },
            ))

        # Run all concurrently, but emit done as each finishes
        async def run_specialist(agent: SpecialistAgent) -> dict[str, Any]:
            async def spec_stream(token: str):
                await emit(PipelineEvent(
                    type=EventType.AGENT_STREAM,
                    agent_name=agent.name,
                    data={"token": token}
                ))
            
            response = await agent.run(specialist_input, token_callback=spec_stream)
            await emit(PipelineEvent(
                type=EventType.SPECIALIST_DONE,
                agent_name=agent.name,
                data=response.parsed,
            ))
            return response.parsed

        specialist_opinions = await asyncio.gather(
            *(run_specialist(agent) for agent in swarm)
        )

        # ── Agreement scoring across the swarm ─────────────
        consensus = compute_consensus(list(specialist_opinions))

        await emit(PipelineEvent(
            type=EventType.CONSENSUS_DONE,
            agent_name="orchestrator",
            data=consensus.model_dump(),
        ))

        # ── Stage 3: Consolidation (Judge) ──────────────────
        await emit(PipelineEvent(
            type=EventType.DISCUSSION_SUMMARY,
            agent_name="orchestrator",
            data={
                "message": f"Summarizing reviews. Lead Specialist: {lead_specialist}.",
                "total_specialists": len(specialist_opinions),
            },
        ))

        await emit(PipelineEvent(
            type=EventType.JUDGE_THINKING,
            agent_name="judge_agent",
            data={"message": "Medical Judge synthesizing reviews and resolving conflicts..."},
        ))

        judge_input = (
            f"## Original Question\n{question}\n\n"
            f"## Original Document\n{combined_docs}\n\n"
            f"## Specialist Reviews\n{json.dumps(list(specialist_opinions), indent=2)}\n\n"
            f"## Lead Specialist\n{lead_specialist}\n\n"
            f"## Panel Agreement\n{json.dumps(consensus.model_dump(), indent=2)}\n"
            f"Weigh the reviews accordingly: where agreement is weak, say so in "
            f"the summary and explain where the specialists diverge."
        )
        async def judge_stream(token: str):
            await emit(PipelineEvent(
                type=EventType.AGENT_STREAM,
                agent_name="judge_agent",
                data={"token": token}
            ))

        judge_response = await self.judge_agent.run(judge_input, token_callback=judge_stream)

        await emit(PipelineEvent(
            type=EventType.JUDGE_DONE,
            agent_name="judge_agent",
            data=judge_response.parsed,
        ))

        # ── Stage 4: Safety Verification ───────────────────
        await emit(PipelineEvent(
            type=EventType.SAFETY_THINKING,
            agent_name="safety_agent",
            data={"message": "Final safety agent verifying Judge's report against documents..."},
        ))

        safety_input = (
            f"## Original Patient Documents\n{combined_docs}\n\n"
            f"## Draft Report for Verification\n{json.dumps(judge_response.parsed, indent=2)}"
        )
        async def safety_stream(token: str):
            await emit(PipelineEvent(
                type=EventType.AGENT_STREAM,
                agent_name="safety_agent",
                data={"token": token}
            ))

        safety_response = await self.safety_agent.run(safety_input, token_callback=safety_stream)

        await emit(PipelineEvent(
            type=EventType.SAFETY_DONE,
            agent_name="safety_agent",
            data=safety_response.parsed,
        ))

        return PipelineResult(
            supervisor=supervisor_data,
            specialist_opinions=list(specialist_opinions),
            consensus=consensus,
            judge_report=judge_response.parsed,
            safety_report=safety_response.parsed,
        )

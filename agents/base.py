"""Base agent abstraction used by all agents in the framework."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel

from config.settings import get_settings


@lru_cache
def get_client() -> AsyncOpenAI:
    """Return the shared OpenAI client.

    One client (and one connection pool) is reused by every agent — building
    one per agent leaks a socket pool for each specialist the swarm spawns.
    """
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        http_client=httpx.AsyncClient(verify=settings.VERIFY_SSL),
        # A dropped connection mid-swarm otherwise fails the whole assessment.
        max_retries=settings.MAX_RETRIES,
    )


class AgentResponse(BaseModel):
    """Standard wrapper returned by every agent."""

    agent_name: str
    raw_content: str
    parsed: dict[str, Any] = {}


class BaseAgent(ABC):
    """Abstract base for all agents in the pipeline.

    Subclasses must set ``name`` and ``system_prompt`` and implement
    ``parse_response`` to convert the raw LLM text into a structured dict.
    """

    name: str = "base_agent"
    system_prompt: str = ""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = get_client()
        self._model = settings.OPENAI_MODEL
        self._temperature = settings.TEMPERATURE
        self._timeout = settings.REQUEST_TIMEOUT
        self.project_root = Path(__file__).parent.parent

    # ── public API ──────────────────────────────────────────

    async def run(
        self, 
        user_message: str, 
        token_callback: Callable[[str], Awaitable[None]] | None = None
    ) -> AgentResponse:
        """Send *user_message* to the LLM and return a parsed response."""
        
        start_time = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.name}: Request started (model={self._model})")

        try:
            # If no callback, we can use structured format directly. 
            # But if we want to stream AND have JSON, we usually have to parse the full string at the end.
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.name}: Creating completion promise...")
            completion_promise = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                stream=bool(token_callback),
                timeout=self._timeout,  # guard against infinite hangs
            )
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.name}: Awaiting completion promise...")
            completion = await completion_promise
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.name}: Connection established, streaming={bool(token_callback)}")

            full_content = ""
            if token_callback:
                first_token = True
                async for chunk in completion:
                    if first_token:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.name}: First token received")
                        first_token = False
                    
                    token = chunk.choices[0].delta.content or ""
                    full_content += token
                    await token_callback(token)
            else:
                full_content = completion.choices[0].message.content or ""
            
            end_time = time.time()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.name}: Generation complete in {end_time - start_time:.2f}s")

            parsed = self.parse_response(full_content)
            return AgentResponse(agent_name=self.name, raw_content=full_content, parsed=parsed)
            
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.name}: ERROR during run: {e}")
            raise

    # ── helpers ─────────────────────────────────────────────

    def load_skill(self, skill_name: str) -> str:
        """Load internal instructions and reference materials from a skill folder."""
        skill_path = self.project_root / "skills" / skill_name
        instructions = ""

        # 1. Load primary instructions
        skill_file = skill_path / "SKILL.md"
        if skill_file.exists():
            instructions += f"### {skill_name} Core Instructions\n"
            instructions += skill_file.read_text(encoding="utf-8") + "\n\n"

        # 2. Load reference materials
        ref_dir = skill_path / "references"
        if ref_dir.exists() and ref_dir.is_dir():
            instructions += "### Professional Reference Materials\n"
            for ref_file in ref_dir.iterdir():
                if ref_file.is_file():
                    name = ref_file.stem.replace("-", " ").title()
                    instructions += f"#### {name}\n"
                    instructions += ref_file.read_text(encoding="utf-8") + "\n\n"

        print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.name}: Loaded skill '{skill_name}' ({len(instructions)} chars)")
        return instructions

    @abstractmethod
    def parse_response(self, raw: str) -> dict[str, Any]:
        """Convert raw LLM text into a structured dict."""

    @staticmethod
    def safe_json_loads(text: str) -> dict[str, Any]:
        """Parse a JSON string, returning an empty dict on failure."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return {"raw": text}

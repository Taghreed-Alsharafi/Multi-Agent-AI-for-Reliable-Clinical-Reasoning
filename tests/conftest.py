"""Shared test fixtures and helpers."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def make_mock_completion(content: str) -> MagicMock:
    """Build a fake OpenAI ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    completion = MagicMock()
    completion.choices = [choice]
    return completion


@pytest.fixture
def mock_openai():
    """Patch AsyncOpenAI so no real API calls are made.

    Yields a helper dict: ``{"set_response": callable}`` so each test can
    configure what the mocked LLM will return.
    """
    from agents.base import get_client

    # The client is cached process-wide; drop it so the patch below is the one
    # every agent in this test picks up.
    get_client.cache_clear()

    with patch("agents.base.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        create_mock = AsyncMock()
        instance.chat.completions.create = create_mock

        def set_response(content: dict[str, Any] | str) -> None:
            if isinstance(content, dict):
                content = json.dumps(content)
            create_mock.return_value = make_mock_completion(content)

        yield {"set_response": set_response, "create_mock": create_mock}

    # Don't leave the mock cached for the next test.
    get_client.cache_clear()

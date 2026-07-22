"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration for the multi-agent framework."""

    # ── OpenAI ──────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    TRIAGE_MODEL: str = "gpt-4o-mini"
    SAFETY_MODEL: str = "gpt-4o-mini"
    TEMPERATURE: float = 0.2
    REQUEST_TIMEOUT: float = 60.0
    #: Retries per request for transient network failures.
    MAX_RETRIES: int = 3

    # Set to false only behind a TLS-intercepting corporate proxy.
    VERIFY_SSL: bool = True

    # ── Agent limits ────────────────────────────────────────
    MAX_SPECIALISTS: int = 10

    # ── API ─────────────────────────────────────────────────
    #: Comma-separated browser origins allowed to call the API.
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def cors_origins(self) -> list[str]:
        """``CORS_ORIGINS`` split into a list."""
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()

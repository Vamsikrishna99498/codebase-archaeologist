"""Application settings, loaded from environment variables and an optional `.env` file.

Every setting can be overridden with an environment variable of the same name
(case-insensitive), e.g. ``MAX_FILES=500``.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Secrets (never logged; SecretStr hides them in repr) ---
    openrouter_api_key: SecretStr | None = None
    github_token: SecretStr | None = None
    supabase_db_url: SecretStr | None = None

    # --- LLM ---
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- Ingestion ---
    allowed_extensions: frozenset[str] = Field(
        # TODO(phase-D): add .java .go .rs .cpp etc. once tree-sitter chunking lands.
        default=frozenset({".py", ".js", ".ts", ".tsx", ".md", ".ipynb"})
    )
    # Phase A "small repo" guard.
    # TODO(phase-D): raise these limits together with incremental/streamed scale-up work.
    max_files: int = 2_000
    max_total_bytes: int = 50 * 1024 * 1024
    max_file_bytes: int = 500 * 1024

    # --- Misc ---
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance (cached)."""
    return Settings()

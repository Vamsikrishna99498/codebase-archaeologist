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
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_batch_size: int = 32

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
    # Notebooks are mostly embedded images/HTML that the reader strips in memory,
    # so they get a separate, larger cap.
    max_notebook_bytes: int = 10 * 1024 * 1024

    # --- Fetching (safety limits; keep memory bounded and failures contained) ---
    fetch_concurrency: int = 8  # parallel downloads == max files buffered in memory
    fetch_timeout_s: float = 30.0
    fetch_max_retries: int = 3
    # Above this many files to fetch, one tarball beats N small requests
    # (measured: 125 files took 6.9s raw vs 1.4s via tarball).
    tarball_threshold_files: int = 50
    # TODO(phase-D): revisit together with max_total_bytes.
    max_tarball_bytes: int = 200 * 1024 * 1024
    # Abort the whole fetch if more than this fraction of files fail.
    max_failed_ratio: float = 0.2

    # --- Chunking ---
    chunk_tokens: int = 400  # bge-small reads at most 512 tokens incl. the header
    chunk_overlap_tokens: int = 50
    # Small-to-big: max tokens of parent context sent to the LLM per retrieved chunk.
    parent_max_tokens: int = 1_200

    # --- Misc ---
    log_level: str = "INFO"

    def max_bytes_for(self, path: str) -> int:
        """Per-file byte cap for `path`."""
        return self.max_notebook_bytes if path.lower().endswith(".ipynb") else self.max_file_bytes


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance (cached)."""
    return Settings()

"""Pick the storage backend from settings."""

from __future__ import annotations

from codebase_archaeologist.config import Settings
from codebase_archaeologist.log import get_logger
from codebase_archaeologist.storage.base import MetadataStore, VectorStore

log = get_logger(__name__)


def open_stores(settings: Settings) -> tuple[VectorStore, MetadataStore]:
    """Supabase when SUPABASE_DB_URL is set; otherwise a non-persistent in-memory store."""
    url = settings.supabase_db_url
    if url is not None and url.get_secret_value().strip():
        from codebase_archaeologist.storage.postgres import (
            PostgresMetadataStore,
            PostgresVectorStore,
            create_pool,
        )

        pool = create_pool(url.get_secret_value())
        return PostgresVectorStore(pool), PostgresMetadataStore(pool)

    from codebase_archaeologist.storage.memory import InMemoryMetadataStore, InMemoryVectorStore

    log.warning("SUPABASE_DB_URL not set: using in-memory storage (data is lost on exit)")
    return InMemoryVectorStore(), InMemoryMetadataStore()

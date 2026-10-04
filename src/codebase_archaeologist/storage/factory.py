"""Pick the storage backend from settings."""

from __future__ import annotations

import socket
from urllib.parse import urlsplit

from codebase_archaeologist.config import Settings
from codebase_archaeologist.errors import ArchaeologistError
from codebase_archaeologist.log import get_logger
from codebase_archaeologist.storage.base import MetadataStore, VectorStore

log = get_logger(__name__)


def _set(secret) -> bool:
    return secret is not None and bool(secret.get_secret_value().strip())


def postgres_reachable(url: str, timeout: float = 3.0) -> bool:
    """Quick TCP check: some networks block outbound Postgres ports."""
    parts = urlsplit(url)
    try:
        socket.create_connection((parts.hostname, parts.port or 5432), timeout=timeout).close()
        return True
    except OSError:
        return False


def choose_backend(settings: Settings) -> str:
    if settings.storage_backend != "auto":
        return settings.storage_backend
    rest_ok = bool(settings.supabase_url) and _set(settings.supabase_service_key)
    if _set(settings.supabase_db_url):
        if postgres_reachable(settings.supabase_db_url.get_secret_value()):
            return "postgres"
        if rest_ok:
            log.info("Postgres port unreachable on this network; using Supabase over HTTPS")
            return "rest"
        raise ArchaeologistError(
            "Supabase Postgres is unreachable (port blocked?) and HTTPS access is not "
            "configured. Set SUPABASE_URL and SUPABASE_SERVICE_KEY, or try another network."
        )
    return "rest" if rest_ok else "memory"


def open_stores(settings: Settings) -> tuple[VectorStore, MetadataStore]:
    backend = choose_backend(settings)
    if backend == "postgres":
        from codebase_archaeologist.storage.postgres import (
            PostgresMetadataStore,
            PostgresVectorStore,
            create_pool,
        )

        pool = create_pool(settings.supabase_db_url.get_secret_value())
        return PostgresVectorStore(pool), PostgresMetadataStore(pool)

    if backend == "rest":
        from codebase_archaeologist.storage.rest import (
            RestClient,
            RestMetadataStore,
            RestVectorStore,
        )

        if not (settings.supabase_url and _set(settings.supabase_service_key)):
            raise ArchaeologistError("REST storage needs SUPABASE_URL and SUPABASE_SERVICE_KEY")
        client = RestClient(settings.supabase_url, settings.supabase_service_key.get_secret_value())
        return RestVectorStore(client), RestMetadataStore(client)

    from codebase_archaeologist.storage.memory import InMemoryMetadataStore, InMemoryVectorStore

    log.warning("No Supabase configured: using in-memory storage (data is lost on exit)")
    return InMemoryVectorStore(), InMemoryMetadataStore()

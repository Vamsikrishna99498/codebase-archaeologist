import pytest

from codebase_archaeologist.config import Settings
from codebase_archaeologist.storage.memory import InMemoryMetadataStore, InMemoryVectorStore

STORAGE_BACKENDS = ["memory", "postgres", "rest"]


def _secret(value) -> str | None:
    return value.get_secret_value().strip() or None if value is not None else None


@pytest.fixture(scope="session")
def pg_pool():
    """Pool to the configured Supabase database; skips when unset or unreachable."""
    url = _secret(Settings().supabase_db_url)
    if not url:
        pytest.skip("SUPABASE_DB_URL not configured")
    from codebase_archaeologist.storage.factory import postgres_reachable
    from codebase_archaeologist.storage.postgres import create_pool

    if not postgres_reachable(url):
        pytest.skip("Supabase Postgres port unreachable on this network")
    try:
        pool = create_pool(url)
    except Exception as exc:  # never print the message: it may contain URL fragments
        pytest.skip(f"Supabase unreachable ({type(exc).__name__})")
    yield pool
    pool.close()


@pytest.fixture(scope="session")
def rest_client():
    """REST client for Supabase over HTTPS; skips when unset or unreachable."""
    s = Settings()
    key = _secret(s.supabase_service_key)
    if not (s.supabase_url and key):
        pytest.skip("SUPABASE_URL / SUPABASE_SERVICE_KEY not configured")
    from codebase_archaeologist.storage.rest import RestClient

    client = RestClient(s.supabase_url, key)
    try:
        client.count("repos", {})
    except Exception as exc:
        client.close()
        pytest.skip(f"Supabase REST unavailable ({type(exc).__name__})")
    yield client
    client.close()


@pytest.fixture(params=STORAGE_BACKENDS)
def backend(request):
    """(vector_store, metadata_store) pair for one backend."""
    if request.param == "memory":
        return InMemoryVectorStore(), InMemoryMetadataStore()
    if request.param == "postgres":
        from codebase_archaeologist.storage.postgres import (
            PostgresMetadataStore,
            PostgresVectorStore,
        )

        pool = request.getfixturevalue("pg_pool")
        return PostgresVectorStore(pool), PostgresMetadataStore(pool)
    from codebase_archaeologist.storage.rest import RestMetadataStore, RestVectorStore

    client = request.getfixturevalue("rest_client")
    return RestVectorStore(client), RestMetadataStore(client)


@pytest.fixture
def vector_store(backend):
    return backend[0]


@pytest.fixture
def metadata_store(backend):
    return backend[1]

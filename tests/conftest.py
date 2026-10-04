import pytest

from codebase_archaeologist.config import Settings
from codebase_archaeologist.storage.memory import InMemoryMetadataStore, InMemoryVectorStore

STORAGE_BACKENDS = ["memory", "postgres"]


@pytest.fixture(scope="session")
def pg_pool():
    """Pool to the configured Supabase database; skips when unset or unreachable."""
    url = Settings().supabase_db_url
    if url is None or not url.get_secret_value().strip():
        pytest.skip("SUPABASE_DB_URL not configured")
    from codebase_archaeologist.storage.postgres import create_pool

    try:
        pool = create_pool(url.get_secret_value())
    except Exception as exc:  # never print the message: it may contain URL fragments
        pytest.skip(f"Supabase unreachable ({type(exc).__name__})")
    yield pool
    pool.close()


@pytest.fixture(params=STORAGE_BACKENDS)
def backend(request):
    """(vector_store, metadata_store) pair for one backend."""
    if request.param == "memory":
        return InMemoryVectorStore(), InMemoryMetadataStore()
    from codebase_archaeologist.storage.postgres import PostgresMetadataStore, PostgresVectorStore

    pool = request.getfixturevalue("pg_pool")
    return PostgresVectorStore(pool), PostgresMetadataStore(pool)


@pytest.fixture
def vector_store(backend):
    return backend[0]


@pytest.fixture
def metadata_store(backend):
    return backend[1]

import pytest

from codebase_archaeologist.storage.memory import InMemoryMetadataStore, InMemoryVectorStore

STORAGE_BACKENDS = ["memory"]


@pytest.fixture(params=STORAGE_BACKENDS)
def backend(request):
    """(vector_store, metadata_store) pair for one backend."""
    if request.param == "memory":
        return InMemoryVectorStore(), InMemoryMetadataStore()
    raise ValueError(request.param)


@pytest.fixture
def vector_store(backend):
    return backend[0]


@pytest.fixture
def metadata_store(backend):
    return backend[1]

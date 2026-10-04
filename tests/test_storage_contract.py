"""Behaviour every storage backend must satisfy.

Backends are listed in conftest.py; the Postgres backend joins automatically when
SUPABASE_DB_URL is set. Each test uses a unique repo slug and cleans up, so the
suite is safe to run against a real database.
"""

import uuid

import pytest

from codebase_archaeologist.schemas import Chunk, RepoRecord


@pytest.fixture
def repo(vector_store, metadata_store):
    slug = f"test__{uuid.uuid4().hex[:12]}"
    yield slug
    vector_store.delete_repo(slug)
    metadata_store.delete_repo(slug)


def chunk(repo: str, path: str, idx: int, start: int, end: int, text: str = "x") -> Chunk:
    return Chunk(
        id=f"{repo}-{path}-{idx}",
        repo=repo,
        path=path,
        blob_sha="sha",
        language="python",
        chunk_index=idx,
        start_line=start,
        end_line=end,
        parent_start_line=start,
        parent_end_line=end,
        scope=None if idx % 2 else f"def f{idx}",
        text=text,
        embed_text=f"File: {path}\n\n{text}",
        token_count=3,
    )


def unit(dim: int, i: int) -> list[float]:
    v = [0.0] * dim
    v[i] = 1.0
    return v


DIM = 384


# --- VectorStore ---


def test_upsert_search_returns_best_first(vector_store, repo):
    chunks = [chunk(repo, "a.py", i, i * 10 + 1, i * 10 + 10, f"text {i}") for i in range(3)]
    vector_store.upsert(chunks, [unit(DIM, i) for i in range(3)])
    query = [0.0] * DIM
    query[1], query[2] = 0.9, 0.1
    hits = vector_store.search(repo, query, k=2)
    assert [h.chunk.id for h in hits] == [chunks[1].id, chunks[2].id]
    assert hits[0].score > hits[1].score
    assert hits[0].chunk == chunks[1]  # round-trips every field
    assert hits[0].vector is None


def test_search_with_vectors(vector_store, repo):
    vector_store.upsert([chunk(repo, "a.py", 0, 1, 5)], [unit(DIM, 3)])
    (hit,) = vector_store.search(repo, unit(DIM, 3), k=1, with_vectors=True)
    assert hit.vector is not None and len(hit.vector) == DIM
    assert hit.vector[3] == pytest.approx(1.0)
    assert hit.score == pytest.approx(1.0, abs=1e-5)


def test_upsert_replaces_by_id(vector_store, repo):
    vector_store.upsert([chunk(repo, "a.py", 0, 1, 5, "old")], [unit(DIM, 0)])
    vector_store.upsert([chunk(repo, "a.py", 0, 1, 5, "new")], [unit(DIM, 0)])
    assert vector_store.count(repo) == 1
    assert vector_store.search(repo, unit(DIM, 0), k=1)[0].chunk.text == "new"


def test_search_is_scoped_to_repo(vector_store, repo):
    other = f"{repo}-other"
    try:
        vector_store.upsert([chunk(other, "a.py", 0, 1, 5)], [unit(DIM, 0)])
        assert vector_store.search(repo, unit(DIM, 0), k=5) == []
    finally:
        vector_store.delete_repo(other)


def test_delete_files(vector_store, repo):
    vector_store.upsert(
        [chunk(repo, "a.py", 0, 1, 5), chunk(repo, "a.py", 1, 6, 9), chunk(repo, "b.py", 0, 1, 5)],
        [unit(DIM, 0), unit(DIM, 1), unit(DIM, 2)],
    )
    assert vector_store.delete_files(repo, ["a.py"]) == 2
    assert vector_store.count(repo) == 1
    assert vector_store.delete_files(repo, ["missing.py"]) == 0


def test_get_overlapping_orders_by_line(vector_store, repo):
    chunks = [chunk(repo, "a.py", i, i * 10 + 1, i * 10 + 12) for i in range(5)]  # overlap 2
    vector_store.upsert(list(reversed(chunks)), [unit(DIM, i) for i in range(5)])
    got = vector_store.get_overlapping(repo, "a.py", 15, 31)
    assert [c.chunk_index for c in got] == [1, 2, 3]
    assert vector_store.get_overlapping(repo, "b.py", 1, 100) == []


# --- MetadataStore ---


def test_repo_records(metadata_store, repo):
    assert metadata_store.get_repo(repo) is None
    rec = RepoRecord(slug=repo, full_name="octo/demo", ref="main", embedding_model="bge")
    metadata_store.upsert_repo(rec)
    metadata_store.upsert_repo(rec.model_copy(update={"indexed_sha": "abc", "chunk_count": 7}))
    got = metadata_store.get_repo(repo)
    assert got.indexed_sha == "abc" and got.chunk_count == 7
    assert repo in [r.slug for r in metadata_store.list_repos()]


def test_file_shas(metadata_store, repo):
    metadata_store.upsert_files(repo, {"a.py": "s1", "b.py": "s2"})
    metadata_store.upsert_files(repo, {"a.py": "s3"})
    assert metadata_store.get_files(repo) == {"a.py": "s3", "b.py": "s2"}
    metadata_store.delete_files(repo, ["b.py"])
    assert metadata_store.get_files(repo) == {"a.py": "s3"}


def test_jobs_lifecycle(metadata_store, repo):
    first = metadata_store.create_job(repo, "c1")
    assert first.status == "queued"
    metadata_store.update_job(first.model_copy(update={"status": "failed", "error": "boom"}))
    second = metadata_store.create_job(repo, "c2")
    metadata_store.update_job(
        second.model_copy(update={"status": "running", "files_total": 10, "files_done": 4})
    )
    got = metadata_store.get_job(second.id)
    assert (got.status, got.files_done, got.files_total) == ("running", 4, 10)
    assert got.updated_at >= got.started_at
    assert metadata_store.latest_job(repo).id == second.id
    assert metadata_store.get_job(first.id).error == "boom"
    assert metadata_store.get_job("nope") is None


def test_factory_falls_back_to_memory_without_url():
    from codebase_archaeologist.config import Settings
    from codebase_archaeologist.storage.factory import open_stores
    from codebase_archaeologist.storage.memory import InMemoryVectorStore

    vs, _ = open_stores(Settings(_env_file=None, supabase_db_url=None))
    assert isinstance(vs, InMemoryVectorStore)

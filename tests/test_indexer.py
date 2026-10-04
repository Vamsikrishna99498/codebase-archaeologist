"""End-to-end indexing tests against a mocked GitHub repo (offline)."""

import pytest
import respx

from codebase_archaeologist.config import Settings
from codebase_archaeologist.embeddings import HashEmbedder
from codebase_archaeologist.errors import ArchaeologistError
from codebase_archaeologist.ingestion.chunking import Chunker
from codebase_archaeologist.ingestion.jobs import Indexer
from codebase_archaeologist.ingestion.repo_map import REPO_MAP_PATH
from codebase_archaeologist.sources.fetch import RAW_URL, git_blob_sha
from codebase_archaeologist.sources.github import API_URL, GitHubSource
from codebase_archaeologist.storage.memory import InMemoryMetadataStore, InMemoryVectorStore

URL = "https://github.com/octo/demo"
SLUG = "octo__demo"
SECRET = "gh" + "p_" + "Q7wK2mZp9XvR4tLb8NcY3hJd6FsG5aEeQ7wK"


class FakeGitHub:
    """Serves one repo whose files can change between commits."""

    def __init__(self, router: respx.MockRouter) -> None:
        self.router = router
        self.raw_calls: list[str] = []

    def publish(self, commit: str, files: dict[str, str], fail: set[str] = frozenset()) -> None:
        self.router.get(f"{API_URL}/repos/octo/demo").respond(json={"default_branch": "main"})
        self.router.get(f"{API_URL}/repos/octo/demo/commits/main").respond(json={"sha": commit})
        tree = [
            {"path": p, "type": "blob", "sha": git_blob_sha(t.encode()), "size": len(t.encode())}
            for p, t in files.items()
        ]
        self.router.get(f"{API_URL}/repos/octo/demo/git/trees/{commit}").respond(
            json={"truncated": False, "tree": tree}
        )
        for path, text in files.items():
            route = self.router.get(f"{RAW_URL}/octo/demo/{commit}/{path}")
            if path in fail:
                route.respond(404)
            else:
                route.mock(side_effect=self._serve(path, text))

    def _serve(self, path, text):
        def handler(request):
            self.raw_calls.append(path)
            return respx.MockResponse(200, content=text.encode())

        return handler


def py(name: str, lines: int = 3) -> str:
    body = "\n".join(f"    x_{i} = step({i})" for i in range(lines))
    return f"def {name}():\n{body}\n    return x_0\n"


FILES_V1 = {
    "README.md": "# Demo\nA demo project.\n",
    "app.py": py("serve") + f'\nTOKEN = "{SECRET}"\n',
    "model.py": py("train", 40),
    "util.py": py("helper"),
    "logo.png": "binary-ish",  # filtered by extension
}


def make_indexer(**overrides):
    s = Settings(
        _env_file=None,
        github_token=None,
        embedding_dim=64,
        chunk_tokens=40,
        chunk_overlap_tokens=5,
        index_batch_chunks=3,
        **overrides,
    )
    vectors, metadata = InMemoryVectorStore(), InMemoryMetadataStore()
    indexer = Indexer(
        s,
        GitHubSource(),
        Chunker(s, lambda t: len(t.split())),
        HashEmbedder(dim=64),
        vectors,
        metadata,
    )
    return indexer, vectors, metadata


@pytest.fixture
def gh():
    with respx.mock(assert_all_called=False) as router:
        yield FakeGitHub(router)


async def test_first_run_indexes_everything(gh):
    gh.publish("c1", FILES_V1)
    indexer, vectors, metadata = make_indexer()
    events = []
    result = await indexer.index(URL, on_progress=lambda j: events.append(j.status))

    assert result.files_selected == 4
    assert result.files_fetched == 4 and result.files_unchanged == 0
    assert result.redactions == 1
    assert metadata.get_files(SLUG).keys() == {"README.md", "app.py", "model.py", "util.py"}
    repo = metadata.get_repo(SLUG)
    assert repo.indexed_sha == "c1" and repo.file_count == 4
    assert repo.chunk_count == vectors.count(SLUG) == result.chunk_count
    assert vectors.get_overlapping(SLUG, REPO_MAP_PATH, 1, 10_000)  # repo map indexed
    job = metadata.latest_job(SLUG)
    assert (job.status, job.files_done, job.files_total) == ("succeeded", 4, 4)
    assert events[0] == "running" and events[-1] == "succeeded"


async def test_secrets_never_reach_storage(gh):
    gh.publish("c1", FILES_V1)
    indexer, vectors, _ = make_indexer()
    await indexer.index(URL)
    stored = vectors.get_overlapping(SLUG, "app.py", 1, 10_000)
    assert stored and all(SECRET not in c.text and SECRET not in c.embed_text for c in stored)
    assert any("[REDACTED:github_token]" in c.text for c in stored)


async def test_rerun_on_same_commit_is_a_no_op(gh):
    gh.publish("c1", FILES_V1)
    indexer, _, _ = make_indexer()
    await indexer.index(URL)
    gh.raw_calls.clear()

    result = await indexer.index(URL)
    assert result.up_to_date and result.job_id is None
    assert gh.raw_calls == []


async def test_new_commit_reindexes_only_changes(gh):
    gh.publish("c1", FILES_V1)
    indexer, vectors, metadata = make_indexer()
    await indexer.index(URL)
    old_util = vectors.get_overlapping(SLUG, "util.py", 1, 10_000)

    v2 = {k: v for k, v in FILES_V1.items() if k != "model.py"}  # removed
    v2["app.py"] = py("serve_v2")  # changed
    v2["new.py"] = py("fresh")  # added
    gh.publish("c2", v2)
    gh.raw_calls.clear()
    result = await indexer.index(URL)

    # README is re-fetched for the repo map but not re-indexed.
    assert sorted(gh.raw_calls) == ["README.md", "app.py", "new.py"]
    assert (result.files_fetched, result.files_unchanged, result.files_removed) == (2, 2, 1)
    assert vectors.get_overlapping(SLUG, "model.py", 1, 10_000) == []
    app = vectors.get_overlapping(SLUG, "app.py", 1, 10_000)
    assert app and all("serve_v2" in c.text for c in app if "def " in c.text)
    assert all("def serve()" not in c.text for c in app)  # old version gone
    assert vectors.get_overlapping(SLUG, "util.py", 1, 10_000) == old_util  # untouched
    assert metadata.get_files(SLUG).keys() == {"README.md", "app.py", "util.py", "new.py"}
    assert metadata.get_repo(SLUG).indexed_sha == "c2"
    assert vectors.count(SLUG) == metadata.get_repo(SLUG).chunk_count


async def test_interrupted_job_resumes_where_it_stopped(gh):
    gh.publish("c1", FILES_V1)
    indexer, _, metadata = make_indexer()
    real_embed = indexer.embedder.embed_documents
    calls = {"n": 0}

    def flaky(texts):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("embedding service crashed")
        return real_embed(texts)

    indexer.embedder.embed_documents = flaky
    with pytest.raises(RuntimeError):
        await indexer.index(URL)
    job = metadata.latest_job(SLUG)
    assert job.status == "failed" and "embedding service crashed" in job.error
    done_first = set(metadata.get_files(SLUG))
    assert 0 < len(done_first) < 4  # first batch committed, rest not
    assert metadata.get_repo(SLUG) is None

    indexer.embedder.embed_documents = real_embed
    gh.raw_calls.clear()
    result = await indexer.index(URL)
    all_files = {"README.md", "app.py", "model.py", "util.py"}
    # Only unfinished files are fetched again (plus the README for the repo map).
    assert set(gh.raw_calls) - {"README.md"} == all_files - done_first - {"README.md"}
    assert result.files_unchanged == len(done_first)
    assert metadata.get_repo(SLUG).indexed_sha == "c1"


async def test_failed_file_is_retried_next_run(gh):
    gh.publish("c1", FILES_V1, fail={"util.py"})
    indexer, _, metadata = make_indexer(max_failed_ratio=0.5)
    result = await indexer.index(URL)
    assert result.files_failed == {"util.py": "http_404"}
    assert "util.py" not in metadata.get_files(SLUG)
    assert metadata.get_repo(SLUG).indexed_sha is None  # so the next run is not skipped

    gh.publish("c1", FILES_V1)
    gh.raw_calls.clear()
    result = await indexer.index(URL)
    assert set(gh.raw_calls) == {"util.py", "README.md"}
    assert metadata.get_repo(SLUG).indexed_sha == "c1"


async def test_force_reindexes_everything(gh):
    gh.publish("c1", FILES_V1)
    indexer, vectors, _ = make_indexer()
    await indexer.index(URL)
    count = vectors.count(SLUG)
    gh.raw_calls.clear()
    result = await indexer.index(URL, force=True)
    assert result.files_fetched == 4
    assert vectors.count(SLUG) == count  # replaced, not duplicated


def test_embedding_dimension_mismatch_is_rejected():
    s = Settings(_env_file=None, embedding_dim=384)
    with pytest.raises(ArchaeologistError, match="64-d"):
        Indexer(
            s,
            GitHubSource(),
            Chunker(s, len),
            HashEmbedder(dim=64),
            InMemoryVectorStore(),
            InMemoryMetadataStore(),
        )

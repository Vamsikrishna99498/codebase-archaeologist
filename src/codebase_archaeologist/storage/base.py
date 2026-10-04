"""Storage interfaces. Supabase Postgres (pgvector) is the real backend; an
in-memory implementation backs the offline test suite. Both must pass the same
contract tests (tests/test_storage_contract.py)."""

from __future__ import annotations

from typing import Protocol

from codebase_archaeologist.schemas import Chunk, IndexJob, RepoRecord, SearchHit


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or replace chunks by id. `Chunk.embed_text` must equal
        chunking.build_embed_text(...) of the chunk's fields; backends may store
        only the fields and rebuild it."""
        ...

    def delete_files(self, repo: str, paths: list[str]) -> int:
        """Delete all chunks of `paths`; returns the number deleted."""
        ...

    def delete_repo(self, repo: str) -> int: ...

    def search(
        self, repo: str, vector: list[float], k: int, *, with_vectors: bool = False
    ) -> list[SearchHit]:
        """Top-k chunks of `repo` by cosine similarity, best first."""
        ...

    def get_overlapping(self, repo: str, path: str, start_line: int, end_line: int) -> list[Chunk]:
        """Chunks of `path` overlapping the line range, ordered by start line.
        Used to rebuild small-to-big parent context."""
        ...

    def count(self, repo: str) -> int: ...


class MetadataStore(Protocol):
    # Repos
    def upsert_repo(self, record: RepoRecord) -> None: ...

    def get_repo(self, slug: str) -> RepoRecord | None: ...

    def list_repos(self) -> list[RepoRecord]: ...

    def delete_repo(self, slug: str) -> None: ...

    # Indexed files: path -> blob SHA (drives incremental re-indexing)
    def get_files(self, repo: str) -> dict[str, str]: ...

    def upsert_files(self, repo: str, files: dict[str, str]) -> None: ...

    def delete_files(self, repo: str, paths: list[str]) -> None: ...

    # Index jobs
    def create_job(self, repo: str, commit_sha: str) -> IndexJob: ...

    def update_job(self, job: IndexJob) -> None: ...

    def get_job(self, job_id: str) -> IndexJob | None: ...

    def latest_job(self, repo: str) -> IndexJob | None: ...

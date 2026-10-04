"""In-memory storage for tests and offline experiments. Not persistent."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import numpy as np

from codebase_archaeologist.schemas import Chunk, IndexJob, RepoRecord, SearchHit


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, np.ndarray] = {}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for chunk, vec in zip(chunks, vectors, strict=True):
            v = np.asarray(vec, dtype=np.float32)
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = v / (np.linalg.norm(v) or 1.0)

    def delete_files(self, repo: str, paths: list[str]) -> int:
        wanted = set(paths)
        return self._delete(lambda c: c.repo == repo and c.path in wanted)

    def delete_repo(self, repo: str) -> int:
        return self._delete(lambda c: c.repo == repo)

    def search(
        self, repo: str, vector: list[float], k: int, *, with_vectors: bool = False
    ) -> list[SearchHit]:
        q = np.asarray(vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        scored = [
            (float(self._vectors[cid] @ q), c) for cid, c in self._chunks.items() if c.repo == repo
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            SearchHit(
                chunk=c,
                score=score,
                vector=self._vectors[c.id].tolist() if with_vectors else None,
            )
            for score, c in scored[:k]
        ]

    def get_overlapping(self, repo: str, path: str, start_line: int, end_line: int) -> list[Chunk]:
        hits = [
            c
            for c in self._chunks.values()
            if c.repo == repo
            and c.path == path
            and c.start_line <= end_line
            and c.end_line >= start_line
        ]
        return sorted(hits, key=lambda c: (c.start_line, c.chunk_index))

    def count(self, repo: str) -> int:
        return sum(c.repo == repo for c in self._chunks.values())

    def _delete(self, predicate) -> int:
        ids = [cid for cid, c in self._chunks.items() if predicate(c)]
        for cid in ids:
            del self._chunks[cid], self._vectors[cid]
        return len(ids)


class InMemoryMetadataStore:
    def __init__(self) -> None:
        self._repos: dict[str, RepoRecord] = {}
        self._files: dict[str, dict[str, str]] = {}
        self._jobs: dict[str, IndexJob] = {}

    def upsert_repo(self, record: RepoRecord) -> None:
        self._repos[record.slug] = record.model_copy()

    def get_repo(self, slug: str) -> RepoRecord | None:
        r = self._repos.get(slug)
        return r.model_copy() if r else None

    def list_repos(self) -> list[RepoRecord]:
        return [r.model_copy() for _, r in sorted(self._repos.items())]

    def delete_repo(self, slug: str) -> None:
        self._repos.pop(slug, None)
        self._files.pop(slug, None)
        self._jobs = {k: j for k, j in self._jobs.items() if j.repo != slug}

    def get_files(self, repo: str) -> dict[str, str]:
        return dict(self._files.get(repo, {}))

    def upsert_files(self, repo: str, files: dict[str, str]) -> None:
        self._files.setdefault(repo, {}).update(files)

    def delete_files(self, repo: str, paths: list[str]) -> None:
        for p in paths:
            self._files.get(repo, {}).pop(p, None)

    def create_job(self, repo: str, commit_sha: str) -> IndexJob:
        now = datetime.now(UTC)
        job = IndexJob(
            id=uuid.uuid4().hex, repo=repo, commit_sha=commit_sha, started_at=now, updated_at=now
        )
        self._jobs[job.id] = job
        return job.model_copy()

    def update_job(self, job: IndexJob) -> None:
        self._jobs[job.id] = job.model_copy(update={"updated_at": datetime.now(UTC)})

    def get_job(self, job_id: str) -> IndexJob | None:
        j = self._jobs.get(job_id)
        return j.model_copy() if j else None

    def latest_job(self, repo: str) -> IndexJob | None:
        jobs = [j for j in self._jobs.values() if j.repo == repo]
        return max(jobs, key=lambda j: j.started_at).model_copy() if jobs else None

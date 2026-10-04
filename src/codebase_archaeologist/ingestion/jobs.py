"""The indexing job: GitHub repo -> searchable chunks, incrementally and resumably.

1. List the repo tree (no clone) and select indexable files.
2. Diff blob SHAs with what is already indexed: fetch only new/changed files,
   delete chunks of removed files, skip unchanged ones.
3. Stream files: read -> redact -> chunk; embed + store in batches.
4. A file's SHA is recorded only after its chunks are stored, so an interrupted
   job resumes by simply running again.
5. Rebuild the repo map; record the repo and job outcome.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import PurePosixPath

from codebase_archaeologist.config import Settings
from codebase_archaeologist.embeddings import Embedder
from codebase_archaeologist.errors import ArchaeologistError
from codebase_archaeologist.guardrails.redaction import redact
from codebase_archaeologist.ingestion.chunking import Chunker
from codebase_archaeologist.ingestion.readers import SourceDocument, read_document
from codebase_archaeologist.ingestion.repo_map import REPO_MAP_PATH, build_repo_map
from codebase_archaeologist.log import get_logger
from codebase_archaeologist.schemas import (
    Chunk,
    FetchedFile,
    IndexJob,
    RemoteFile,
    RepoRecord,
    RepoSnapshot,
)
from codebase_archaeologist.sources.base import Source
from codebase_archaeologist.sources.fetch import GitHubFetcher
from codebase_archaeologist.sources.filters import select_files
from codebase_archaeologist.sources.github import parse_repo_url
from codebase_archaeologist.storage.base import MetadataStore, VectorStore

log = get_logger(__name__)

ProgressCallback = Callable[[IndexJob], None]


@dataclass
class IndexResult:
    repo: str
    commit_sha: str
    job_id: str | None
    up_to_date: bool = False
    files_selected: int = 0
    files_fetched: int = 0
    files_unchanged: int = 0
    files_removed: int = 0
    files_failed: dict[str, str] = field(default_factory=dict)
    redactions: int = 0
    chunks_written: int = 0
    chunk_count: int = 0
    seconds: float = 0.0


class Indexer:
    def __init__(
        self,
        settings: Settings,
        source: Source,
        chunker: Chunker,
        embedder: Embedder,
        vectors: VectorStore,
        metadata: MetadataStore,
        *,
        fetcher_factory: Callable[[], GitHubFetcher] | None = None,
    ) -> None:
        if embedder.dim != settings.embedding_dim:
            raise ArchaeologistError(
                f"Embedder produces {embedder.dim}-d vectors but storage expects "
                f"{settings.embedding_dim}-d (EMBEDDING_DIM)."
            )
        self.s = settings
        self.source = source
        self.chunker = chunker
        self.embedder = embedder
        self.vectors = vectors
        self.metadata = metadata
        token = settings.github_token.get_secret_value() if settings.github_token else None
        self._new_fetcher = fetcher_factory or (lambda: GitHubFetcher(settings, token))

    async def index(
        self,
        url: str,
        *,
        ref: str | None = None,
        force: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> IndexResult:
        started = time.monotonic()
        repo = parse_repo_url(url)
        snapshot = await self.source.snapshot(repo, ref)
        slug = repo.slug
        existing = self.metadata.get_repo(slug)
        model_changed = (
            existing is not None and existing.embedding_model != self.embedder.model_name
        )

        if (
            existing
            and not force
            and not model_changed
            and existing.indexed_sha == snapshot.commit_sha
        ):
            log.info("%s is already indexed at %s", repo.full_name, snapshot.commit_sha[:7])
            return IndexResult(
                repo=slug,
                commit_sha=snapshot.commit_sha,
                job_id=None,
                up_to_date=True,
                files_selected=existing.file_count,
                files_unchanged=existing.file_count,
                chunk_count=self.vectors.count(slug),
                seconds=time.monotonic() - started,
            )

        selection = select_files(snapshot.files, self.s)  # raises RepoTooLargeError
        if force or model_changed:
            if model_changed:
                log.info("Embedding model changed; re-indexing %s from scratch", repo.full_name)
            self.vectors.delete_repo(slug)
            self.metadata.delete_files(slug, list(self.metadata.get_files(slug)))

        known = self.metadata.get_files(slug)
        current = {f.path: f.blob_sha for f in selection.selected}
        changed = [f for f in selection.selected if known.get(f.path) != f.blob_sha]
        removed = sorted(known.keys() - current.keys())
        readme = _find_readme(selection.selected)

        job = self.metadata.create_job(slug, snapshot.commit_sha)
        job = job.model_copy(update={"status": "running", "files_total": len(changed)})
        self._save(job, on_progress)
        result = IndexResult(
            repo=slug,
            commit_sha=snapshot.commit_sha,
            job_id=job.id,
            files_selected=len(current),
            files_unchanged=len(current) - len(changed),
            files_removed=len(removed),
        )

        try:
            if removed:
                self.vectors.delete_files(slug, removed)
                self.metadata.delete_files(slug, removed)

            # The README feeds the repo map, so fetch it even when unchanged.
            to_fetch = list(changed)
            if readme and readme not in changed:
                to_fetch.append(readme)
            readme_text = await self._ingest(
                snapshot,
                to_fetch,
                index_paths={f.path for f in changed},
                readme_path=readme.path if readme else None,
                job=job,
                result=result,
                on_progress=on_progress,
            )

            await asyncio.to_thread(self._index_repo_map, snapshot, list(current), readme_text)

            result.chunk_count = self.vectors.count(slug)
            indexed_files = len(current) - len(result.files_failed)
            self.metadata.upsert_repo(
                RepoRecord(
                    slug=slug,
                    full_name=repo.full_name,
                    ref=snapshot.ref,
                    # A failed file is retried on the next run only if the SHA differs,
                    # so keep the commit unset until every selected file is indexed.
                    indexed_sha=None if result.files_failed else snapshot.commit_sha,
                    embedding_model=self.embedder.model_name,
                    file_count=indexed_files,
                    chunk_count=result.chunk_count,
                    indexed_at=datetime.now(UTC),
                )
            )
            job = self.metadata.get_job(job.id) or job
            self._save(job.model_copy(update={"status": "succeeded"}), on_progress)
        except Exception as exc:
            job = self.metadata.get_job(job.id) or job
            self._save(
                job.model_copy(
                    update={"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                ),
                on_progress,
            )
            raise
        finally:
            result.seconds = time.monotonic() - started

        log.info(
            "Indexed %s@%s: %d fetched, %d unchanged, %d removed, %d failed, %d chunks (%.1fs)",
            repo.full_name,
            snapshot.commit_sha[:7],
            result.files_fetched,
            result.files_unchanged,
            result.files_removed,
            len(result.files_failed),
            result.chunk_count,
            result.seconds,
        )
        return result

    # --- internals ---

    async def _ingest(
        self,
        snapshot: RepoSnapshot,
        files: list[RemoteFile],
        *,
        index_paths: set[str],
        readme_path: str | None,
        job: IndexJob,
        result: IndexResult,
        on_progress: ProgressCallback | None,
    ) -> str | None:
        """Fetch `files`, index those in `index_paths`; return the redacted README text."""
        readme_text: str | None = None
        batch: list[tuple[FetchedFile, list[Chunk]]] = []
        batch_chunks = 0
        fetcher = self._new_fetcher()

        async for fetched in fetcher.fetch(snapshot, files):
            doc = read_document(fetched)
            redaction = redact(doc.text)
            doc.text = redaction.text
            if fetched.path == readme_path:
                readme_text = doc.text
            if fetched.path not in index_paths:
                continue  # unchanged README, fetched only for the repo map
            result.redactions += sum(redaction.findings.values())
            chunks = self.chunker.chunk(snapshot.repo, doc)
            batch.append((fetched, chunks))
            batch_chunks += len(chunks)
            if batch_chunks >= self.s.index_batch_chunks:
                job = await asyncio.to_thread(self._flush, snapshot, batch, job, result)
                self._notify(job, on_progress)
                batch, batch_chunks = [], 0

        if batch:
            job = await asyncio.to_thread(self._flush, snapshot, batch, job, result)
            self._notify(job, on_progress)

        result.files_failed = {
            p: why for p, why in fetcher.report.failed.items() if p in index_paths
        }
        return readme_text

    def _flush(
        self,
        snapshot: RepoSnapshot,
        batch: list[tuple[FetchedFile, list[Chunk]]],
        job: IndexJob,
        result: IndexResult,
    ) -> IndexJob:
        slug = snapshot.repo.slug
        all_chunks = [c for _, chunks in batch for c in chunks]
        vectors = self.embedder.embed_documents([c.embed_text for c in all_chunks])
        # Replace old versions of these files, then store the new chunks...
        self.vectors.delete_files(slug, [f.path for f, _ in batch])
        self.vectors.upsert(all_chunks, vectors)
        # ...and only then mark the files as indexed (resumability).
        self.metadata.upsert_files(slug, {f.path: f.blob_sha for f, _ in batch})
        result.files_fetched += len(batch)
        result.chunks_written += len(all_chunks)
        job = job.model_copy(
            update={
                "files_done": job.files_done + len(batch),
                "chunks_written": job.chunks_written + len(all_chunks),
            }
        )
        self.metadata.update_job(job)
        return job

    def _index_repo_map(
        self, snapshot: RepoSnapshot, paths: list[str], readme_text: str | None
    ) -> None:
        doc: SourceDocument = build_repo_map(snapshot, paths, readme_text)
        doc.text = redact(doc.text).text
        chunks = self.chunker.chunk(snapshot.repo, doc)
        vectors = self.embedder.embed_documents([c.embed_text for c in chunks])
        self.vectors.delete_files(snapshot.repo.slug, [REPO_MAP_PATH])
        self.vectors.upsert(chunks, vectors)

    def _save(self, job: IndexJob, on_progress: ProgressCallback | None) -> None:
        self.metadata.update_job(job)
        self._notify(job, on_progress)

    @staticmethod
    def _notify(job: IndexJob, on_progress: ProgressCallback | None) -> None:
        if on_progress:
            on_progress(job)


def _find_readme(files: list[RemoteFile]) -> RemoteFile | None:
    """Top-level README, if indexed."""
    for f in files:
        p = PurePosixPath(f.path)
        if len(p.parts) == 1 and p.stem.lower() == "readme":
            return f
    return None

"""Download file contents from GitHub into memory, with hard safety limits.

Two strategies, picked automatically:

* **raw**: one request per file to ``raw.githubusercontent.com`` (CDN, does not use
  the REST API quota), with bounded concurrency. Best for few files, e.g. an
  incremental re-index.
* **tarball**: one streamed download of the repo archive; only selected files are
  kept, everything else is read past and discarded. Best for many files.

Safety guarantees:

* per-file and total byte caps are enforced on the bytes actually received,
* the tarball download is aborted past ``max_tarball_bytes`` (falls back to raw),
* at most ``fetch_concurrency`` downloads run and are buffered at once,
* every file is verified against its git blob SHA,
* per-file problems are recorded, never raised; the fetch only fails when more
  than ``max_failed_ratio`` of files fail,
* nothing is written to disk.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import tarfile
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from urllib.parse import quote

import httpx

from codebase_archaeologist.config import Settings
from codebase_archaeologist.errors import FetchError, RepoTooLargeError
from codebase_archaeologist.log import get_logger
from codebase_archaeologist.schemas import FetchedFile, RemoteFile, RepoSnapshot
from codebase_archaeologist.sources.github import API_URL

log = get_logger(__name__)

RAW_URL = "https://raw.githubusercontent.com"
_RETRYABLE = {429, 500, 502, 503, 504}


class _FileTooLarge(Exception):
    pass


class _ArchiveTooLarge(Exception):
    pass


def git_blob_sha(data: bytes) -> str:
    """The SHA git assigns to a blob with these contents."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def decode_text(data: bytes) -> str | None:
    """Decode UTF-8 text; None for binary content."""
    if b"\x00" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


@dataclass
class FetchReport:
    method: str = ""
    requested: int = 0
    fetched: int = 0
    bytes: int = 0
    failed: dict[str, str] = field(default_factory=dict)  # path -> reason


class GitHubFetcher:
    def __init__(
        self,
        settings: Settings,
        token: str | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.s = settings
        self._headers = {"User-Agent": "codebase-archaeologist"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        self._sleep = sleep
        self.report = FetchReport()

    async def fetch(
        self, snapshot: RepoSnapshot, files: list[RemoteFile]
    ) -> AsyncIterator[FetchedFile]:
        """Yield the contents of `files`. Inspect `self.report` afterwards."""
        self.report = FetchReport(requested=len(files))
        if not files:
            return

        pending = files
        if len(files) > self.s.tarball_threshold_files:
            done: set[str] = set()
            try:
                async for f in self._fetch_tarball(snapshot, files):
                    done.add(f.path)
                    yield f
            except (_ArchiveTooLarge, tarfile.TarError, httpx.HTTPError) as exc:
                log.warning("Tarball fetch failed (%s); falling back to raw downloads", exc)
                self.report.failed.clear()
            pending = [f for f in files if f.path not in done and f.path not in self.report.failed]

        if pending:
            self.report.method = "raw" if not self.report.method else "tarball+raw"
            async for f in self._fetch_raw(snapshot, pending):
                yield f

        self._check_failures()

    # --- accounting shared by both strategies ---

    def _accept(self, file: RemoteFile, data: bytes) -> FetchedFile | None:
        if len(data) > self.s.max_file_bytes:
            self.report.failed[file.path] = "too_large"
            return None
        if git_blob_sha(data) != file.blob_sha:
            self.report.failed[file.path] = "integrity_mismatch"
            return None
        text = decode_text(data)
        if text is None:
            self.report.failed[file.path] = "binary"
            return None
        self.report.bytes += len(data)
        if self.report.bytes > self.s.max_total_bytes:
            raise RepoTooLargeError(
                f"Fetched content exceeded {self.s.max_total_bytes / 1_048_576:.0f} MB; aborting."
            )
        self.report.fetched += 1
        return FetchedFile(path=file.path, blob_sha=file.blob_sha, text=text, size=len(data))

    def _check_failures(self) -> None:
        r = self.report
        if r.failed:
            log.warning("%d/%d files failed to fetch: %s", len(r.failed), r.requested, r.failed)
        if r.requested and len(r.failed) / r.requested > self.s.max_failed_ratio:
            raise FetchError(
                f"{len(r.failed)} of {r.requested} files failed to download "
                f"(limit {self.s.max_failed_ratio:.0%}). First errors: "
                + ", ".join(f"{p}: {why}" for p, why in list(r.failed.items())[:5])
            )

    # --- raw CDN strategy ---

    async def _fetch_raw(
        self, snapshot: RepoSnapshot, files: list[RemoteFile]
    ) -> AsyncIterator[FetchedFile]:
        todo: asyncio.Queue[RemoteFile] = asyncio.Queue()
        for f in files:
            todo.put_nowait(f)
        # Bounded result queue: workers pause when the consumer falls behind.
        results: asyncio.Queue[FetchedFile] = asyncio.Queue(self.s.fetch_concurrency)
        timeout = httpx.Timeout(self.s.fetch_timeout_s)
        base = f"{RAW_URL}/{snapshot.repo.full_name}/{snapshot.commit_sha}/"

        async with httpx.AsyncClient(headers=self._headers, timeout=timeout) as client:

            async def worker() -> None:
                while True:
                    try:
                        file = todo.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    data = await self._download_one(client, base, file)
                    if data is not None and (fetched := self._accept(file, data)):
                        await results.put(fetched)

            async def run_workers() -> None:
                # TaskGroup cancels sibling workers if one raises.
                try:
                    async with asyncio.TaskGroup() as tg:
                        for _ in range(min(self.s.fetch_concurrency, len(files))):
                            tg.create_task(worker())
                except BaseExceptionGroup as eg:
                    raise eg.exceptions[0] from None

            runner = asyncio.create_task(run_workers())
            try:
                while True:
                    getter = asyncio.ensure_future(results.get())
                    done, _ = await asyncio.wait(
                        {getter, runner}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if getter in done:
                        yield getter.result()
                        continue
                    getter.cancel()
                    while not results.empty():
                        yield results.get_nowait()
                    runner.result()  # surface worker errors (e.g. RepoTooLargeError)
                    break
            finally:
                runner.cancel()

    async def _download_one(
        self, client: httpx.AsyncClient, base: str, file: RemoteFile
    ) -> bytes | None:
        url = base + quote(file.path)
        for attempt in range(self.s.fetch_max_retries + 1):
            try:
                async with client.stream("GET", url) as resp:
                    if resp.status_code in _RETRYABLE and attempt < self.s.fetch_max_retries:
                        await self._backoff(attempt, resp.headers.get("retry-after"))
                        continue
                    if not resp.is_success:
                        self.report.failed[file.path] = f"http_{resp.status_code}"
                        return None
                    return await self._read_capped(resp)
            except _FileTooLarge:
                self.report.failed[file.path] = "too_large"
                return None
            except httpx.TransportError as exc:
                if attempt < self.s.fetch_max_retries:
                    await self._backoff(attempt, None)
                    continue
                self.report.failed[file.path] = f"network: {type(exc).__name__}"
                return None
        return None

    async def _read_capped(self, resp: httpx.Response) -> bytes:
        buf = bytearray()
        async for chunk in resp.aiter_bytes():
            buf += chunk
            if len(buf) > self.s.max_file_bytes:
                raise _FileTooLarge
        return bytes(buf)

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        delay = 2**attempt + random.uniform(0, 1)
        if retry_after and retry_after.isdigit():
            delay = max(delay, int(retry_after))
        await self._sleep(min(delay, 60))

    # --- tarball strategy ---

    async def _fetch_tarball(
        self, snapshot: RepoSnapshot, files: list[RemoteFile]
    ) -> AsyncIterator[FetchedFile]:
        self.report.method = "tarball"
        wanted = {f.path: f for f in files}
        url = f"{API_URL}/repos/{snapshot.repo.full_name}/tarball/{snapshot.commit_sha}"
        # tarfile is synchronous; run the streamed read in a worker thread.
        members = await asyncio.to_thread(self._read_tarball, url, set(wanted))
        for path, data in members:
            if fetched := self._accept(wanted[path], data):
                yield fetched
        for path in wanted.keys() - {p for p, _ in members}:
            self.report.failed.setdefault(path, "missing_in_archive")

    def _read_tarball(self, url: str, wanted: set[str]) -> list[tuple[str, bytes]]:
        out: list[tuple[str, bytes]] = []
        kept_bytes = 0
        timeout = httpx.Timeout(self.s.fetch_timeout_s)
        with (
            httpx.Client(headers=self._headers, timeout=timeout, follow_redirects=True) as client,
            client.stream("GET", url) as resp,
        ):
            resp.raise_for_status()
            length = resp.headers.get("content-length")
            if length and int(length) > self.s.max_tarball_bytes:
                raise _ArchiveTooLarge(f"archive is {int(length) / 1_048_576:.0f} MB")
            stream = _CappedStream(resp.iter_bytes(), self.s.max_tarball_bytes)
            with tarfile.open(fileobj=stream, mode="r|gz") as tar:
                for member in tar:
                    # Only regular files; never symlinks/devices. Nothing touches disk.
                    if not member.isfile():
                        continue
                    # Archive entries are prefixed with "<owner>-<repo>-<sha>/".
                    path = member.name.split("/", 1)[-1]
                    if path not in wanted or member.size > self.s.max_file_bytes:
                        continue
                    if (fh := tar.extractfile(member)) is not None:
                        data = fh.read()
                        kept_bytes += len(data)
                        if kept_bytes > self.s.max_total_bytes:
                            raise RepoTooLargeError(
                                "Selected files exceed "
                                f"{self.s.max_total_bytes / 1_048_576:.0f} MB; aborting."
                            )
                        out.append((path, data))
        return out


class _CappedStream:
    """Minimal read-only file object over a byte iterator, with a hard size cap."""

    def __init__(self, chunks: Iterator[bytes], limit: int) -> None:
        self._chunks = chunks
        self._buf = b""
        self._read = 0
        self._limit = limit

    def read(self, size: int = -1) -> bytes:
        while size < 0 or len(self._buf) < size:
            try:
                chunk = next(self._chunks)
            except StopIteration:
                break
            self._read += len(chunk)
            if self._read > self._limit:
                raise _ArchiveTooLarge(f"archive exceeded {self._limit / 1_048_576:.0f} MB")
            self._buf += chunk
        if size < 0:
            data, self._buf = self._buf, b""
        else:
            data, self._buf = self._buf[:size], self._buf[size:]
        return data

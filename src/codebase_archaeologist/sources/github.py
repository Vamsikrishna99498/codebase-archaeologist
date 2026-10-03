"""GitHub source: list a repository's files through the REST API, without cloning.

One recursive tree call returns every file's path, size and blob SHA. If GitHub
truncates that response (>100k entries or >7 MB), we fall back to walking the tree
one directory at a time.
"""

from __future__ import annotations

import re
from collections import deque

import httpx

from codebase_archaeologist.errors import (
    GitHubAuthError,
    InvalidRepoURLError,
    RateLimitedError,
    RepoNotFoundError,
)
from codebase_archaeologist.log import get_logger
from codebase_archaeologist.schemas import RemoteFile, RepoRef, RepoSnapshot

log = get_logger(__name__)

API_URL = "https://api.github.com"
_NAME = r"[A-Za-z0-9_.-]+"
_URL_PATTERNS = [
    re.compile(rf"^https?://(?:www\.)?github\.com/(?P<owner>{_NAME})/(?P<name>{_NAME})(?:/.*)?$"),
    re.compile(rf"^git@github\.com:(?P<owner>{_NAME})/(?P<name>{_NAME})$"),
    re.compile(rf"^(?P<owner>{_NAME})/(?P<name>{_NAME})$"),
]


def parse_repo_url(url: str) -> RepoRef:
    """Parse ``https://github.com/o/r``, ``git@github.com:o/r.git`` or ``o/r``."""
    cleaned = url.strip().rstrip("/")
    for pattern in _URL_PATTERNS:
        if m := pattern.match(cleaned):
            name = m["name"].removesuffix(".git")
            if name and m["owner"] not in {".", ".."} and name not in {".", ".."}:
                return RepoRef(owner=m["owner"], name=name)
    raise InvalidRepoURLError(f"Not a GitHub repository URL: {url!r}")


class GitHubSource:
    """Async GitHub REST client for repository listings."""

    def __init__(
        self,
        token: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str = API_URL,
    ) -> None:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "codebase-archaeologist",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = client or httpx.AsyncClient(base_url=base_url, timeout=30.0)
        self._client.headers.update(headers)
        self._owns_client = client is None

    async def __aenter__(self) -> GitHubSource:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # --- public API ---

    async def snapshot(self, repo: RepoRef, ref: str | None = None) -> RepoSnapshot:
        if ref is None:
            ref = await self.default_branch(repo)
        commit_sha = await self.resolve_commit(repo, ref)
        files = await self.list_files(repo, commit_sha)
        log.info("Listed %d files in %s@%s", len(files), repo.full_name, commit_sha[:7])
        return RepoSnapshot(repo=repo, ref=ref, commit_sha=commit_sha, files=files)

    async def default_branch(self, repo: RepoRef) -> str:
        data = await self._get_json(f"/repos/{repo.full_name}")
        return data["default_branch"]

    async def resolve_commit(self, repo: RepoRef, ref: str) -> str:
        data = await self._get_json(f"/repos/{repo.full_name}/commits/{ref}")
        return data["sha"]

    async def list_files(self, repo: RepoRef, tree_sha: str) -> list[RemoteFile]:
        data = await self._get_json(
            f"/repos/{repo.full_name}/git/trees/{tree_sha}", params={"recursive": "1"}
        )
        if not data.get("truncated"):
            return _blobs(data["tree"])
        log.warning("Tree for %s is truncated; walking directories individually", repo.full_name)
        return await self._walk_tree(repo, tree_sha)

    # --- internals ---

    async def _walk_tree(self, repo: RepoRef, root_sha: str) -> list[RemoteFile]:
        files: list[RemoteFile] = []
        queue: deque[tuple[str, str]] = deque([("", root_sha)])
        while queue:
            prefix, sha = queue.popleft()
            data = await self._get_json(f"/repos/{repo.full_name}/git/trees/{sha}")
            for entry in data["tree"]:
                path = f"{prefix}{entry['path']}"
                if entry["type"] == "tree":
                    queue.append((f"{path}/", entry["sha"]))
                elif entry["type"] == "blob":
                    files.append(_to_file({**entry, "path": path}))
        return files

    async def _get_json(self, url: str, params: dict[str, str] | None = None) -> dict:
        resp = await self._client.get(url, params=params)
        _raise_for_status(resp)
        return resp.json()


def _blobs(entries: list[dict]) -> list[RemoteFile]:
    # "commit" entries are submodules; "tree" entries are directories.
    return [_to_file(e) for e in entries if e["type"] == "blob"]


def _to_file(entry: dict) -> RemoteFile:
    return RemoteFile(path=entry["path"], size=entry.get("size", 0), blob_sha=entry["sha"])


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.is_success:
        return
    status = resp.status_code
    if status == 401:
        raise GitHubAuthError("GitHub rejected the token (401). Check GITHUB_TOKEN.")
    if status == 429 or (status == 403 and resp.headers.get("x-ratelimit-remaining") == "0"):
        reset = resp.headers.get("x-ratelimit-reset")
        raise RateLimitedError(
            "GitHub API rate limit exceeded. Set GITHUB_TOKEN for 5,000 requests/hour.",
            reset_at=int(reset) if reset else None,
        )
    if status in (404, 409, 422):
        # 409: empty repository; 422: unknown ref.
        raise RepoNotFoundError(f"Repository or ref not found or not public ({status}): {resp.url}")
    resp.raise_for_status()

"""Source interface. GitHub is the first implementation; GitLab etc. can follow."""

from __future__ import annotations

from typing import Protocol

from codebase_archaeologist.schemas import RepoRef, RepoSnapshot


class Source(Protocol):
    async def snapshot(self, repo: RepoRef, ref: str | None = None) -> RepoSnapshot:
        """List all files of `repo` at `ref` (default branch if None)."""
        ...

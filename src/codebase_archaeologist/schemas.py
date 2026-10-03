"""Core data models shared across ingestion, storage and answering."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class RepoRef(BaseModel):
    """A GitHub repository identifier, e.g. ``owner/name``."""

    model_config = ConfigDict(frozen=True)

    owner: str
    name: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def slug(self) -> str:
        """Filesystem/collection-safe identifier."""
        return f"{self.owner}__{self.name}".lower()


class RemoteFile(BaseModel):
    """A file listed in a repo tree; contents are fetched separately."""

    model_config = ConfigDict(frozen=True)

    path: str
    size: int
    blob_sha: str  # git blob SHA = hash of the file contents; used for incremental indexing


class RepoSnapshot(BaseModel):
    """The file listing of a repo at one commit."""

    repo: RepoRef
    ref: str
    commit_sha: str
    files: list[RemoteFile]

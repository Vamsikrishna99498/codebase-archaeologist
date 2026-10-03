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


class FetchedFile(BaseModel):
    """A file's decoded contents, held in memory only."""

    model_config = ConfigDict(frozen=True)

    path: str
    blob_sha: str
    text: str
    size: int


class Chunk(BaseModel):
    """A retrieval unit. `text` is shown to the LLM; `embed_text` adds a header."""

    model_config = ConfigDict(frozen=True)

    id: str
    repo: str  # RepoRef.slug
    path: str
    blob_sha: str
    language: str
    chunk_index: int
    start_line: int
    end_line: int
    # Small-to-big: line span of the larger context sent to the LLM.
    parent_start_line: int
    parent_end_line: int
    scope: str | None = None
    text: str
    embed_text: str
    token_count: int

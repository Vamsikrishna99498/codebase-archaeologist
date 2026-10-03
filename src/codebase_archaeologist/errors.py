"""Exception hierarchy. Clients (CLI, Streamlit, API) catch `ArchaeologistError`."""

from __future__ import annotations


class ArchaeologistError(Exception):
    """Base class for all expected, user-facing errors."""


class InvalidRepoURLError(ArchaeologistError):
    pass


class RepoNotFoundError(ArchaeologistError):
    pass


class GitHubAuthError(ArchaeologistError):
    pass


class RateLimitedError(ArchaeologistError):
    def __init__(self, message: str, reset_at: int | None = None) -> None:
        super().__init__(message)
        self.reset_at = reset_at  # unix timestamp when the limit resets, if known


class RepoTooLargeError(ArchaeologistError):
    pass

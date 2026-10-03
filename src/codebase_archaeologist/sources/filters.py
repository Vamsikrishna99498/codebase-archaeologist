"""Decide which listed files are worth indexing, before any content is downloaded."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from codebase_archaeologist.config import Settings
from codebase_archaeologist.errors import RepoTooLargeError
from codebase_archaeologist.schemas import RemoteFile

# Directory names that hold vendored, generated or environment files.
EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".github",
        ".next",
        ".nuxt",
        ".tox",
        ".venv",
        "__pycache__",
        "bower_components",
        "build",
        "coverage",
        "dist",
        "env",
        "node_modules",
        "site-packages",
        "third_party",
        "vendor",
        "venv",
    }
)
# Filename suffixes of generated or minified code.
EXCLUDED_SUFFIXES = (".min.js", ".bundle.js", ".chunk.js", "_pb2.py", "_pb2_grpc.py", ".d.ts")


@dataclass
class FilterResult:
    selected: list[RemoteFile]
    skipped: Counter[str] = field(default_factory=Counter)  # reason -> count

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.selected)


def skip_reason(file: RemoteFile, settings: Settings) -> str | None:
    """Return why `file` should be skipped, or None to keep it."""
    path = PurePosixPath(file.path)
    if path.suffix.lower() not in settings.allowed_extensions:
        return "extension"
    if any(part in EXCLUDED_DIRS for part in path.parts[:-1]):
        return "excluded_dir"
    if path.name.lower().endswith(EXCLUDED_SUFFIXES):
        return "generated"
    if file.size > settings.max_bytes_for(file.path):
        return "too_large"
    if file.size == 0:
        return "empty"
    return None


def select_files(files: list[RemoteFile], settings: Settings) -> FilterResult:
    """Filter `files` and enforce the repo-size guard.

    Raises RepoTooLargeError if the selection exceeds the configured limits.
    """
    result = FilterResult(selected=[])
    for f in files:
        if reason := skip_reason(f, settings):
            result.skipped[reason] += 1
        else:
            result.selected.append(f)

    n, size = len(result.selected), result.total_bytes
    if n > settings.max_files or size > settings.max_total_bytes:
        raise RepoTooLargeError(
            f"Repository has {n} indexable files ({size / 1_048_576:.1f} MB); "
            f"current limits are {settings.max_files} files / "
            f"{settings.max_total_bytes / 1_048_576:.0f} MB."
        )
    return result

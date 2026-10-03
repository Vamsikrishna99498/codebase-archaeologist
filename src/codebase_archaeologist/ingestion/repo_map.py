"""A synthetic "repo map" document: structure + README intro.

Answers questions no single file can ("what is the project layout?", "where is
the entry point?") and is indexed like any other document.
"""

from __future__ import annotations

from collections import Counter
from pathlib import PurePosixPath

from codebase_archaeologist.ingestion.readers import LANGUAGES, SourceDocument, Unit
from codebase_archaeologist.schemas import RepoSnapshot
from codebase_archaeologist.sources.filters import EXCLUDED_DIRS

REPO_MAP_PATH = "<repo-map>"
FULL_TREE_MAX_FILES = 150
README_EXCERPT_CHARS = 1_500


def build_repo_map(
    snapshot: RepoSnapshot, paths: list[str], readme: str | None = None
) -> SourceDocument:
    """`paths` are the indexed files; the tree shows every non-vendored repo file
    (templates, configs, data) since layout questions need them too."""
    tree_paths = [
        f.path
        for f in snapshot.files
        if not any(part in EXCLUDED_DIRS for part in PurePosixPath(f.path).parts[:-1])
    ] or paths
    langs = Counter(LANGUAGES.get(PurePosixPath(p).suffix.lower(), "other") for p in paths)
    lang_summary = ", ".join(f"{lang} {n}" for lang, n in langs.most_common())

    sections: list[tuple[str, list[str]]] = [
        (
            "Overview",
            [
                f"Repository: {snapshot.repo.full_name} @ {snapshot.ref} "
                f"(commit {snapshot.commit_sha[:7]})",
                f"Indexed files: {len(paths)} ({lang_summary})",
            ],
        ),
        (f"Directory structure ({len(tree_paths)} files)", _tree(tree_paths)),
    ]
    if readme and readme.strip():
        excerpt = readme.strip()[:README_EXCERPT_CHARS]
        if len(readme.strip()) > README_EXCERPT_CHARS:
            excerpt += "\n..."
        sections.append(("README excerpt", excerpt.splitlines()))

    lines: list[str] = []
    units: list[Unit] = []
    for title, body in sections:
        start = len(lines) + 1
        lines.append(f"## {title}")
        lines.extend(body)
        units.append(Unit(start, len(lines), f"Repo map > {title}", 0))
        lines.append("")
    return SourceDocument(
        path=REPO_MAP_PATH,
        blob_sha=snapshot.commit_sha,
        language="text",
        text="\n".join(lines),
        units=units,
    )


def _tree(paths: list[str]) -> list[str]:
    if len(paths) <= FULL_TREE_MAX_FILES:
        return _full_tree(sorted(paths))
    return _summary_tree(paths)


def _full_tree(paths: list[str]) -> list[str]:
    out: list[str] = []
    seen_dirs: set[tuple[str, ...]] = set()
    for p in paths:
        parts = PurePosixPath(p).parts
        for depth in range(len(parts) - 1):
            d = parts[: depth + 1]
            if d not in seen_dirs:
                seen_dirs.add(d)
                out.append(f"{'  ' * depth}{d[-1]}/")
        out.append(f"{'  ' * (len(parts) - 1)}{parts[-1]}")
    return out


def _summary_tree(paths: list[str], depth: int = 2) -> list[str]:
    """Large repos: directories up to `depth` with file counts, plus top-level files."""
    counts: Counter[str] = Counter()
    top_files: list[str] = []
    for p in paths:
        parts = PurePosixPath(p).parts
        if len(parts) == 1:
            top_files.append(p)
        else:
            counts["/".join(parts[: min(depth, len(parts) - 1)]) + "/"] += 1
    out = [f"{d} ({n} files)" for d, n in sorted(counts.items())]
    out.extend(sorted(top_files))
    return out

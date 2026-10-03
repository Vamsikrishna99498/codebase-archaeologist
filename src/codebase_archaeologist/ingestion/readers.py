"""Turn fetched files into plain text plus structural units.

Units (Python functions/classes, Markdown sections, notebook cells) give chunks a
human-readable scope ("class Model > def train") and define the parent span used
for small-to-big retrieval.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from codebase_archaeologist.log import get_logger
from codebase_archaeologist.schemas import FetchedFile

log = get_logger(__name__)

LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".md": "markdown",
    ".ipynb": "notebook",
}

# Notebook output limits.
OUTPUT_HEAD_LINES = 15
OUTPUT_TAIL_LINES = 15  # final metrics usually sit at the end of training logs
OUTPUT_MAX_LINE_CHARS = 300

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_HTML_TAG = re.compile(r"<[^>]+>")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True)
class Unit:
    """A structural region of a document; lines are 1-based and inclusive."""

    start_line: int
    end_line: int
    name: str
    depth: int  # 0 = top level


@dataclass
class SourceDocument:
    path: str
    blob_sha: str
    language: str
    text: str
    units: list[Unit] = field(default_factory=list)

    @property
    def line_count(self) -> int:
        return self.text.count("\n") + (0 if self.text.endswith("\n") else 1)


def read_document(file: FetchedFile) -> SourceDocument:
    language = LANGUAGES.get(PurePosixPath(file.path).suffix.lower(), "text")
    if language == "notebook":
        text, units = _read_notebook(file.text, file.path)
    else:
        text, units = file.text, []
        if language == "python":
            units = python_units(text)
        elif language == "markdown":
            units = markdown_units(text)
    return SourceDocument(file.path, file.blob_sha, language, text, units)


# --- Python ---


def python_units(source: str) -> list[Unit]:
    """Functions and classes at every nesting level, with qualified names."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []  # e.g. Python 2 code; chunking still works, just without scopes
    units: list[Unit] = []

    def visit(node: ast.AST, prefix: str, depth: int) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                kind = "class" if isinstance(child, ast.ClassDef) else "def"
                name = f"{prefix}{kind} {child.name}"
                # Include decorators in the span.
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                units.append(Unit(start, child.end_lineno or child.lineno, name, depth))
                visit(child, f"{name} > ", depth + 1)

    visit(tree, "", 0)
    return units


# --- Markdown ---


def markdown_units(text: str) -> list[Unit]:
    """One unit per heading section, named by its heading path."""
    lines = text.splitlines()
    headings: list[tuple[int, int, str]] = []  # (line, level, title)
    in_fence = False
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        elif not in_fence and (m := _MD_HEADING.match(line)):
            headings.append((i, len(m[1]), m[2]))

    units: list[Unit] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    for idx, (line, level, title) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        # A section ends right before the next heading of the same or higher level.
        end = len(lines)
        for next_line, next_level, _ in headings[idx + 1 :]:
            if next_level <= level:
                end = next_line - 1
                break
        units.append(Unit(line, end, " > ".join(t for _, t in stack), len(stack) - 1))
    return units


# --- Notebooks ---


def _read_notebook(raw: str, path: str) -> tuple[str, list[Unit]]:
    try:
        nb = json.loads(raw)
        cells = nb.get("cells") or nb.get("worksheets", [{}])[0].get("cells", [])
    except (json.JSONDecodeError, AttributeError, IndexError, TypeError):
        log.warning("Could not parse notebook %s; indexing it as plain text", path)
        return raw, []

    out: list[str] = []
    units: list[Unit] = []
    for i, cell in enumerate(cells, 1):
        kind = cell.get("cell_type", "code")
        source = _joined(cell.get("source", cell.get("input", ""))).rstrip()
        rendered = [_outputs_text(cell.get("outputs", []))] if kind == "code" else []
        if not source and not any(rendered):
            continue
        start = len(out) + 1
        out.append(f"# [cell {i} · {kind}]")
        out.extend(source.splitlines())
        if rendered and rendered[0]:
            out.append(f"# [cell {i} · output]")
            out.extend(rendered[0].splitlines())
        units.append(Unit(start, len(out), f"cell {i}", 0))
        out.append("")
    return "\n".join(out), units


def _outputs_text(outputs: list[dict]) -> str:
    parts: list[str] = []
    for output in outputs:
        otype = output.get("output_type")
        if otype == "stream":
            if output.get("name") == "stdout":  # stderr is mostly warnings/progress bars
                parts.append(_joined(output.get("text", "")))
        elif otype in ("execute_result", "display_data", "pyout"):
            data = output.get("data", output)
            if "text/plain" in data:
                parts.append(_joined(data["text/plain"]))
            elif "text/html" in data:
                parts.append(_HTML_TAG.sub(" ", _joined(data["text/html"])))
            if images := [k for k in data if k.startswith("image/")]:
                parts.append(f"[image output: {images[0].split('/')[1]}]")
        elif otype in ("error", "pyerr"):
            parts.append(f"{output.get('ename', 'Error')}: {output.get('evalue', '')}")
    return _clean_output("\n".join(p for p in parts if p.strip()))


def _clean_output(text: str) -> str:
    text = _ANSI.sub("", text)
    lines = []
    # split("\n"), not splitlines(): splitlines() would also break on "\r".
    for line in text.split("\n"):
        line = line.rsplit("\r", 1)[-1].rstrip()  # progress bars: keep the final state
        if len(line) > OUTPUT_MAX_LINE_CHARS:
            line = line[:OUTPUT_MAX_LINE_CHARS] + " ..."
        if line.strip():
            lines.append(line)
    if len(lines) > OUTPUT_HEAD_LINES + OUTPUT_TAIL_LINES:
        omitted = len(lines) - OUTPUT_HEAD_LINES - OUTPUT_TAIL_LINES
        lines = [
            *lines[:OUTPUT_HEAD_LINES],
            f"... [{omitted} lines omitted]",
            *lines[-OUTPUT_TAIL_LINES:],
        ]
    return "\n".join(lines)


def _joined(value: str | list[str]) -> str:
    return "".join(value) if isinstance(value, list) else value

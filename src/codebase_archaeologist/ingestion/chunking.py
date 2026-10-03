"""Split documents into retrieval chunks with line ranges, scopes and parent spans.

Small-to-big retrieval: chunks are small (precise search), and each records the
line span of a larger *parent* (the whole file if small, else the enclosing
top-level units, else neighbouring chunks) that is sent to the LLM instead.
"""

from __future__ import annotations

import hashlib

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from codebase_archaeologist.config import Settings
from codebase_archaeologist.ingestion.readers import SourceDocument, Unit
from codebase_archaeologist.ingestion.tokens import TokenCounter
from codebase_archaeologist.schemas import Chunk, RepoRef

_SPLITTER_LANGUAGES = {
    "python": Language.PYTHON,
    "javascript": Language.JS,
    "typescript": Language.TS,
    "markdown": Language.MARKDOWN,
}


def chunk_id(repo: RepoRef, path: str, blob_sha: str, index: int) -> str:
    """Stable id: unchanged files keep their chunk ids across re-indexes."""
    return hashlib.sha1(f"{repo.slug}|{path}|{blob_sha}|{index}".encode()).hexdigest()


class Chunker:
    def __init__(self, settings: Settings, count_tokens: TokenCounter) -> None:
        self.s = settings
        self.count = count_tokens

    def chunk(self, repo: RepoRef, doc: SourceDocument) -> list[Chunk]:
        if not doc.text.strip():
            return []
        lines = doc.text.splitlines()
        spans = self._split(doc)
        file_fits = self.count(doc.text) <= self.s.parent_max_tokens

        chunks = []
        for i, (piece, start, end) in enumerate(spans):
            whole_file = (1, len(lines))
            parent = whole_file if file_fits else self._parent_span(doc.units, spans, i, lines)
            scope = _scope(doc.units, start)
            header = f"File: {doc.path} (lines {start}-{end})\nLanguage: {doc.language}"
            if scope:
                header += f"\nScope: {scope}"
            embed_text = f"{header}\n\n{piece}"
            chunks.append(
                Chunk(
                    id=chunk_id(repo, doc.path, doc.blob_sha, i),
                    repo=repo.slug,
                    path=doc.path,
                    blob_sha=doc.blob_sha,
                    language=doc.language,
                    chunk_index=i,
                    start_line=start,
                    end_line=end,
                    parent_start_line=parent[0],
                    parent_end_line=parent[1],
                    scope=scope,
                    text=piece,
                    embed_text=embed_text,
                    token_count=self.count(embed_text),
                )
            )
        return chunks

    def _split(self, doc: SourceDocument) -> list[tuple[str, int, int]]:
        """Return (text, start_line, end_line) pieces covering the document."""
        text = doc.text
        if self.count(text) <= self.s.chunk_tokens:
            stripped = text.strip("\n")
            first = len(text) - len(text.lstrip("\n")) + 1
            return [(stripped, first, first + stripped.count("\n"))]

        # Locate pieces ourselves, scanning forward: pieces come out in document
        # order, so searching from the previous start maps repeated code (identical
        # functions, boilerplate) to the right place. LangChain's add_start_index
        # can match an earlier identical block and produce wrong line numbers.
        pieces = []
        cursor = 0
        for piece in self._splitter(doc.language).split_text(text):
            start_char = text.find(piece, cursor)
            if start_char < 0:  # should not happen; fall back to a global search
                start_char = max(text.find(piece), 0)
            cursor = start_char + 1
            start = text.count("\n", 0, start_char) + 1
            pieces.append((piece, start, start + piece.count("\n")))
        return pieces

    def _splitter(self, language: str) -> RecursiveCharacterTextSplitter:
        kwargs = {
            "chunk_size": self.s.chunk_tokens,
            "chunk_overlap": self.s.chunk_overlap_tokens,
            "length_function": self.count,
        }
        if language == "notebook":
            # Prefer cell boundaries, then Python structure inside a cell.
            seps = RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)
            return RecursiveCharacterTextSplitter(separators=["\n# [cell ", *seps], **kwargs)
        if lang := _SPLITTER_LANGUAGES.get(language):
            return RecursiveCharacterTextSplitter.from_language(lang, **kwargs)
        return RecursiveCharacterTextSplitter(**kwargs)

    def _parent_span(
        self,
        units: list[Unit],
        spans: list[tuple[str, int, int]],
        i: int,
        lines: list[str],
    ) -> tuple[int, int]:
        _, start, end = spans[i]
        # 1. Enclosing top-level units (function/class, section, cell) overlapping the chunk.
        top = [u for u in units if u.depth == 0 and u.start_line <= end and u.end_line >= start]
        if top:
            candidate = (
                min(start, *(u.start_line for u in top)),
                max(end, *(u.end_line for u in top)),
            )
            if self._fits(candidate, lines):
                return candidate
        # 2. The chunk plus its neighbours.
        prev_start = spans[i - 1][1] if i > 0 else start
        next_end = spans[i + 1][2] if i + 1 < len(spans) else end
        if self._fits((prev_start, next_end), lines):
            return prev_start, next_end
        # 3. Just the chunk itself.
        return start, end

    def _fits(self, span: tuple[int, int], lines: list[str]) -> bool:
        text = "\n".join(lines[span[0] - 1 : span[1]])
        return self.count(text) <= self.s.parent_max_tokens


def _scope(units: list[Unit], line: int) -> str | None:
    """Name of the deepest unit containing `line`."""
    containing = [u for u in units if u.start_line <= line <= u.end_line]
    return max(containing, key=lambda u: u.depth).name if containing else None

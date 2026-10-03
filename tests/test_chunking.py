import pytest

from codebase_archaeologist.config import Settings
from codebase_archaeologist.ingestion.chunking import Chunker, chunk_id
from codebase_archaeologist.ingestion.readers import read_document
from codebase_archaeologist.schemas import FetchedFile, RepoRef

REPO = RepoRef(owner="octo", name="demo")


def words(text: str) -> int:
    """Offline stand-in for the model tokenizer."""
    return len(text.split())


@pytest.fixture
def chunker():
    s = Settings(_env_file=None, chunk_tokens=40, chunk_overlap_tokens=5, parent_max_tokens=150)
    return Chunker(s, words)


def doc(path: str, text: str, sha: str = "sha1"):
    return read_document(FetchedFile(path=path, blob_sha=sha, text=text, size=len(text)))


def func(name: str, body_lines: int) -> str:
    body = "\n".join(f"    value_{i} = compute(value_{i - 1}, factor)" for i in range(body_lines))
    return f"def {name}():\n{body}\n    return value_0\n"


def big_module(n_funcs: int = 6, body_lines: int = 6) -> str:
    return "import os\n\n\n" + "\n\n".join(func(f"step_{i}", body_lines) for i in range(n_funcs))


def assert_line_ranges_match(chunks, text):
    # Chunks come in document order; identical code blocks must not be mapped to
    # an earlier occurrence (regression: LangChain's add_start_index did this).
    starts = [c.start_line for c in chunks]
    assert starts == sorted(starts)
    lines = text.splitlines()
    for c in chunks:
        assert c.text in "\n".join(lines[c.start_line - 1 : c.end_line]), c


def test_small_file_is_one_chunk_with_whole_file_parent(chunker):
    text = "\n\nx = 1\ny = 2\n"
    (c,) = chunker.chunk(REPO, doc("tiny.py", text))
    assert (c.start_line, c.end_line) == (3, 4)
    assert (c.parent_start_line, c.parent_end_line) == (1, 4)
    assert c.text == "x = 1\ny = 2"
    assert c.embed_text.startswith("File: tiny.py (lines 3-4)\nLanguage: python\n\nx = 1")


def test_large_python_file_line_ranges_scopes_and_sizes(chunker):
    text = big_module()
    chunks = chunker.chunk(REPO, doc("pkg/steps.py", text))
    assert len(chunks) > 3
    assert_line_ranges_match(chunks, text)
    assert all(words(c.text) <= 40 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Every function's definition line lands in some chunk.
    for i in range(6):
        assert any(f"def step_{i}():" in c.text for c in chunks)
    in_step3 = next(c for c in chunks if c.text.startswith("def step_3"))
    assert in_step3.scope == "def step_3"
    assert "Scope: def step_3" in in_step3.embed_text


def test_parent_is_enclosing_function_when_file_is_large(chunker):
    text = big_module(n_funcs=4, body_lines=12)  # functions bigger than one chunk
    d = doc("big.py", text)
    chunks = chunker.chunk(REPO, d)
    unit = next(u for u in d.units if u.name == "def step_2")
    inside = [c for c in chunks if unit.start_line <= c.start_line <= unit.end_line]
    assert len(inside) >= 2  # function was split...
    for c in inside:  # ...but every piece points back at the whole function
        assert c.parent_start_line <= unit.start_line
        assert c.parent_end_line >= unit.end_line
        assert c.parent_end_line - c.parent_start_line < len(text.splitlines()) - 1


def test_parent_falls_back_to_neighbours_when_unit_too_big():
    # ~330-word function: too big as a parent; three neighbouring chunks (~115) fit.
    s = Settings(_env_file=None, chunk_tokens=40, chunk_overlap_tokens=5, parent_max_tokens=130)
    text = func("huge", 80)  # one function far larger than parent_max_tokens
    chunks = Chunker(s, words).chunk(REPO, doc("huge.py", text))
    mid = chunks[len(chunks) // 2]
    prev, nxt = chunks[mid.chunk_index - 1], chunks[mid.chunk_index + 1]
    assert (mid.parent_start_line, mid.parent_end_line) == (prev.start_line, nxt.end_line)


def test_parent_is_chunk_itself_when_nothing_larger_fits():
    s = Settings(_env_file=None, chunk_tokens=40, chunk_overlap_tokens=5, parent_max_tokens=60)
    chunks = Chunker(s, words).chunk(REPO, doc("huge.py", func("huge", 80)))
    mid = chunks[len(chunks) // 2]
    assert (mid.parent_start_line, mid.parent_end_line) == (mid.start_line, mid.end_line)


def test_notebook_chunks_split_on_cell_boundaries(chunker):
    import json

    cells = [
        {"cell_type": "code", "source": f"x_{i} = load({i})\n" * 6, "outputs": []} for i in range(8)
    ]
    text = json.dumps({"cells": cells})
    d = doc("nb.ipynb", text)
    chunks = chunker.chunk(REPO, d)
    assert len(chunks) > 1
    assert all(c.text.startswith("# [cell ") for c in chunks)
    assert chunks[0].scope == "cell 1"
    assert_line_ranges_match(chunks, d.text)


def test_markdown_scope_uses_heading_path(chunker):
    text = (
        "# Guide\n\n## Install\n\n" + "Run the installer with care. " * 30 + "\n\n## Usage\nRun.\n"
    )
    chunks = chunker.chunk(REPO, doc("README.md", text))
    assert any(c.scope == "Guide > Install" for c in chunks)
    assert chunks[-1].scope == "Guide > Usage"


def test_chunk_ids_stable_and_content_addressed(chunker):
    a = chunker.chunk(REPO, doc("a.py", big_module(), sha="v1"))
    again = chunker.chunk(REPO, doc("a.py", big_module(), sha="v1"))
    changed = chunker.chunk(REPO, doc("a.py", big_module(), sha="v2"))
    assert [c.id for c in a] == [c.id for c in again]
    assert a[0].id != changed[0].id
    assert a[0].id == chunk_id(REPO, "a.py", "v1", 0)
    assert len({c.id for c in a}) == len(a)


def test_empty_document_yields_no_chunks(chunker):
    assert chunker.chunk(REPO, doc("empty.md", "\n\n")) == []


def test_notebook_output_stays_with_its_code(chunker):
    import json

    cells = [
        {
            "cell_type": "code",
            "source": f"score_{i} = evaluate(model_{i})\nprint(score_{i})",
            "outputs": [{"output_type": "stream", "name": "stdout", "text": f"MAE {i}.5\n" * 5}],
        }
        for i in range(10)
    ]
    chunks = chunker.chunk(REPO, doc("eval.ipynb", json.dumps({"cells": cells})))
    assert len(chunks) > 1
    for c in chunks:
        for line in c.text.splitlines():
            if line.startswith("# [output of cell "):
                n = line.removeprefix("# [output of cell ").rstrip("]")
                assert f"# [cell {n} · code]" in c.text

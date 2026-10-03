import json

from codebase_archaeologist.ingestion.readers import (
    Unit,
    markdown_units,
    python_units,
    read_document,
)
from codebase_archaeologist.schemas import FetchedFile


def fetched(path: str, text: str) -> FetchedFile:
    return FetchedFile(path=path, blob_sha="sha", text=text, size=len(text))


PY = """\
import os


@decorator
def top(x):
    return x


class Model:
    def __init__(self):
        self.a = 1

    async def train(self):
        def inner():
            pass
        return inner
"""


def test_python_units_qualified_names_and_spans():
    units = python_units(PY)
    assert units == [
        Unit(4, 6, "def top", 0),  # span starts at the decorator
        Unit(9, 16, "class Model", 0),
        Unit(10, 11, "class Model > def __init__", 1),
        Unit(13, 16, "class Model > def train", 1),
        Unit(14, 15, "class Model > def train > def inner", 2),
    ]


def test_python_units_tolerate_syntax_errors():
    assert python_units("print 'python 2'\n") == []


MD = """\
# Project
intro
## Install
pip install x
```bash
# not a heading
```
### Docker
docker run
## Usage
run it
"""


def test_markdown_units_heading_paths_and_fences():
    assert markdown_units(MD) == [
        Unit(1, 11, "Project", 0),
        Unit(3, 9, "Project > Install", 1),
        Unit(8, 9, "Project > Install > Docker", 2),
        Unit(10, 11, "Project > Usage", 1),
    ]


def make_notebook(cells):
    return json.dumps({"cells": cells, "nbformat": 4, "nbformat_minor": 5})


def code(source, outputs=()):
    return {"cell_type": "code", "source": source, "outputs": list(outputs)}


def test_notebook_keeps_text_outputs_with_their_cells():
    nb = make_notebook(
        [
            {"cell_type": "markdown", "source": ["# Training\n", "We train an LSTM."]},
            code(
                ["acc = evaluate()\n", "print(acc)"],
                [{"output_type": "stream", "name": "stdout", "text": ["Accuracy: 0.93\n"]}],
            ),
            code(
                "df.head()",
                [
                    {
                        "output_type": "execute_result",
                        "data": {"text/plain": "   close\n0  101.2", "text/html": "<table>"},
                    },
                    {"output_type": "display_data", "data": {"image/png": "iVBORw0KGgo" * 1000}},
                ],
            ),
            code(
                "boom()",
                [
                    {
                        "output_type": "error",
                        "ename": "KeyError",
                        "evalue": "'Close'",
                        "traceback": [],
                    }
                ],
            ),
            code("warn()", [{"output_type": "stream", "name": "stderr", "text": "Warning!"}]),
            code("", []),  # empty cells are skipped
        ]
    )
    doc = read_document(fetched("nb/train.ipynb", nb))
    assert doc.language == "notebook"
    assert doc.text.splitlines() == [
        "# [cell 1 · markdown]",
        "# Training",
        "We train an LSTM.",
        "",
        "# [cell 2 · code]",
        "acc = evaluate()",
        "print(acc)",
        "# [cell 2 · output]",
        "Accuracy: 0.93",
        "",
        "# [cell 3 · code]",
        "df.head()",
        "# [cell 3 · output]",
        "   close",
        "0  101.2",
        "[image output: png]",
        "",
        "# [cell 4 · code]",
        "boom()",
        "# [cell 4 · output]",
        "KeyError: 'Close'",
        "",
        "# [cell 5 · code]",
        "warn()",
    ]
    assert [u.name for u in doc.units] == ["cell 1", "cell 2", "cell 3", "cell 4", "cell 5"]
    assert doc.units[1] == Unit(5, 9, "cell 2", 0)


def test_notebook_long_output_keeps_head_and_tail_and_strips_noise():
    log_lines = [f"\x1b[32mEpoch {i}/100\x1b[0m loss={1 / i:.3f}" for i in range(1, 101)]
    progress = "10%|#         |\r50%|#####     |\r100%|##########|\n"
    nb = make_notebook(
        [
            code(
                "model.fit()",
                [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": progress + "\n".join(log_lines),
                    }
                ],
            )
        ]
    )
    text = read_document(fetched("fit.ipynb", nb)).text
    assert "\x1b" not in text
    assert "100%|##########|" in text and "10%|" not in text
    assert "Epoch 100/100 loss=0.010" in text  # the tail (final metrics) survives
    assert "lines omitted]" in text
    assert "Epoch 50/100" not in text


def test_invalid_notebook_falls_back_to_plain_text():
    doc = read_document(fetched("bad.ipynb", "{not json"))
    assert doc.text == "{not json"
    assert doc.units == []


def test_plain_files_pass_through():
    doc = read_document(fetched("web/app.ts", "export const x = 1;\n"))
    assert doc.language == "typescript"
    assert doc.text == "export const x = 1;\n"
    assert doc.line_count == 1

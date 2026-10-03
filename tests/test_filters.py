import pytest

from codebase_archaeologist.config import Settings
from codebase_archaeologist.errors import RepoTooLargeError
from codebase_archaeologist.schemas import RemoteFile
from codebase_archaeologist.sources.filters import select_files, skip_reason


@pytest.fixture
def settings():
    return Settings(_env_file=None, max_files=5, max_total_bytes=1_000, max_file_bytes=500)


def f(path, size=100):
    return RemoteFile(path=path, size=size, blob_sha="x")


@pytest.mark.parametrize(
    ("path", "size", "reason"),
    [
        ("src/app.py", 100, None),
        ("README.md", 100, None),
        ("web/App.TSX", 100, None),
        ("notebooks/eda.ipynb", 100, None),
        ("logo.png", 100, "extension"),
        ("package-lock.json", 100, "extension"),
        ("node_modules/react/index.js", 100, "excluded_dir"),
        ("a/venv/lib/x.py", 100, "excluded_dir"),
        ("static/app.min.js", 100, "generated"),
        ("proto/api_pb2.py", 100, "generated"),
        ("types/index.d.ts", 100, "generated"),
        ("data/huge.py", 501, "too_large"),
        ("notebooks/plots.ipynb", 5_000, None),  # notebooks have their own, larger cap
        ("pkg/__init__.py", 0, "empty"),
    ],
)
def test_skip_reason(settings, path, size, reason):
    assert skip_reason(f(path, size), settings) == reason


def test_dir_named_like_excluded_file_is_kept(settings):
    # Only parent directories are matched, not the filename itself.
    assert skip_reason(f("docs/build.md"), settings) is None


def test_select_files_counts_skips(settings):
    result = select_files([f("a.py"), f("b.md"), f("c.png"), f("node_modules/d.js")], settings)
    assert [x.path for x in result.selected] == ["a.py", "b.md"]
    assert result.skipped == {"extension": 1, "excluded_dir": 1}
    assert result.total_bytes == 200


def test_too_many_files(settings):
    with pytest.raises(RepoTooLargeError, match="6 indexable files"):
        select_files([f(f"m{i}.py") for i in range(6)], settings)


def test_too_many_bytes(settings):
    with pytest.raises(RepoTooLargeError):
        select_files([f(f"m{i}.py", 400) for i in range(3)], settings)

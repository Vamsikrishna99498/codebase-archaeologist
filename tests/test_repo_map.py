from codebase_archaeologist.config import Settings
from codebase_archaeologist.ingestion.chunking import Chunker
from codebase_archaeologist.ingestion.repo_map import REPO_MAP_PATH, build_repo_map
from codebase_archaeologist.schemas import RemoteFile, RepoRef, RepoSnapshot

SNAP = RepoSnapshot(
    repo=RepoRef(owner="octo", name="demo"), ref="main", commit_sha="abcdef1234" * 4, files=[]
)


def test_small_repo_map_has_full_tree_and_readme():
    paths = ["app.py", "src/pkg/core.py", "src/pkg/util.py", "README.md", "nb/eda.ipynb"]
    doc = build_repo_map(SNAP, paths, readme="# Demo\nPredicts prices.")
    assert doc.path == REPO_MAP_PATH
    assert "Repository: octo/demo @ main (commit abcdef1)" in doc.text
    assert "Indexed files: 5 (python 3, markdown 1, notebook 1)" in doc.text
    tree = doc.text.split("## Directory structure (5 files)\n")[1].split("\n\n")[0].splitlines()
    assert tree == [
        "README.md",
        "app.py",
        "nb/",
        "  eda.ipynb",
        "src/",
        "  pkg/",
        "    core.py",
        "    util.py",
    ]
    assert "Predicts prices." in doc.text
    assert [u.name for u in doc.units] == [
        "Repo map > Overview",
        "Repo map > Directory structure (5 files)",
        "Repo map > README excerpt",
    ]


def test_large_repo_map_summarises_directories():
    paths = [f"src/mod{i}/f{j}.py" for i in range(20) for j in range(10)] + ["setup.py"]
    doc = build_repo_map(SNAP, paths)
    assert "src/mod0/ (10 files)" in doc.text
    assert "src/mod0/f0.py" not in doc.text
    assert "setup.py" in doc.text
    assert "README excerpt" not in doc.text


def test_long_readme_is_truncated():
    doc = build_repo_map(SNAP, ["a.py"], readme="word " * 2_000)
    assert doc.text.rstrip().endswith("...")
    assert len(doc.text) < 2_500


def test_repo_map_chunks_like_any_document():
    doc = build_repo_map(SNAP, ["a.py", "b.py"], readme="# Demo")
    s = Settings(_env_file=None)
    chunks = Chunker(s, lambda t: len(t.split())).chunk(SNAP.repo, doc)
    assert len(chunks) == 1
    assert chunks[0].path == REPO_MAP_PATH
    assert chunks[0].embed_text.startswith("File: <repo-map>")


def test_tree_shows_all_non_vendored_repo_files():
    files = ["app.py", "templates/index.html", "static/logo.png", "node_modules/x/i.js"]
    snap = SNAP.model_copy(
        update={"files": [RemoteFile(path=p, size=1, blob_sha="s") for p in files]}
    )
    doc = build_repo_map(snap, ["app.py"])
    assert "Indexed files: 1 (python 1)" in doc.text
    assert "  index.html" in doc.text and "  logo.png" in doc.text
    assert "node_modules" not in doc.text

import httpx
import pytest
import respx

from codebase_archaeologist.errors import (
    GitHubAuthError,
    InvalidRepoURLError,
    RateLimitedError,
    RepoNotFoundError,
)
from codebase_archaeologist.schemas import RepoRef
from codebase_archaeologist.sources.github import API_URL, GitHubSource, parse_repo_url

REPO = RepoRef(owner="octo", name="demo")


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/octo/demo",
        "https://github.com/octo/demo/",
        "https://github.com/octo/demo.git",
        "http://www.github.com/octo/demo/tree/main/src",
        "git@github.com:octo/demo.git",
        "octo/demo",
        "  octo/demo  ",
    ],
)
def test_parse_repo_url_valid(url):
    assert parse_repo_url(url) == REPO


@pytest.mark.parametrize(
    "url", ["", "octo", "https://gitlab.com/octo/demo", "https://github.com/octo", "../x"]
)
def test_parse_repo_url_invalid(url):
    with pytest.raises(InvalidRepoURLError):
        parse_repo_url(url)


def _mock_repo(router, tree: dict):
    router.get("/repos/octo/demo").respond(json={"default_branch": "main"})
    router.get("/repos/octo/demo/commits/main").respond(json={"sha": "c0ffee" * 6 + "abcd"})
    router.get(url__regex=r"/repos/octo/demo/git/trees/c0ffee.*").respond(json=tree)


@respx.mock(base_url=API_URL)
async def test_snapshot_lists_blobs_only(respx_mock):
    _mock_repo(
        respx_mock,
        {
            "truncated": False,
            "tree": [
                {"path": "src", "type": "tree", "sha": "t1"},
                {"path": "src/app.py", "type": "blob", "sha": "b1", "size": 120},
                {"path": "README.md", "type": "blob", "sha": "b2", "size": 40},
                {"path": "vendor/lib", "type": "commit", "sha": "s1"},
            ],
        },
    )
    async with GitHubSource() as gh:
        snap = await gh.snapshot(REPO)

    assert snap.ref == "main"
    assert snap.commit_sha.startswith("c0ffee")
    assert {f.path for f in snap.files} == {"src/app.py", "README.md"}
    assert next(f for f in snap.files if f.path == "src/app.py").blob_sha == "b1"


@respx.mock(base_url=API_URL)
async def test_truncated_tree_falls_back_to_walking(respx_mock):
    respx_mock.get("/repos/octo/demo/commits/main").respond(json={"sha": "root"})
    respx_mock.get("/repos/octo/demo/git/trees/root", params={"recursive": "1"}).respond(
        json={"truncated": True, "tree": []}
    )
    respx_mock.get("/repos/octo/demo/git/trees/root").respond(
        json={
            "tree": [
                {"path": "pkg", "type": "tree", "sha": "pkg-sha"},
                {"path": "setup.py", "type": "blob", "sha": "b0", "size": 10},
            ]
        }
    )
    respx_mock.get("/repos/octo/demo/git/trees/pkg-sha").respond(
        json={"tree": [{"path": "core.py", "type": "blob", "sha": "b1", "size": 20}]}
    )
    async with GitHubSource() as gh:
        snap = await gh.snapshot(REPO, ref="main")

    assert sorted(f.path for f in snap.files) == ["pkg/core.py", "setup.py"]


@respx.mock(base_url=API_URL)
async def test_token_sent_as_bearer(respx_mock):
    route = respx_mock.get("/repos/octo/demo").respond(json={"default_branch": "dev"})
    async with GitHubSource(token="tok123") as gh:
        assert await gh.default_branch(REPO) == "dev"
    assert route.calls.last.request.headers["Authorization"] == "Bearer tok123"


@pytest.mark.parametrize(
    ("status", "headers", "exc"),
    [
        (404, {}, RepoNotFoundError),
        (409, {}, RepoNotFoundError),
        (401, {}, GitHubAuthError),
        (403, {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1700000000"}, RateLimitedError),
        (429, {}, RateLimitedError),
    ],
)
@respx.mock(base_url=API_URL)
async def test_error_mapping(respx_mock, status, headers, exc):
    respx_mock.get("/repos/octo/demo").respond(status_code=status, headers=headers)
    async with GitHubSource() as gh:
        with pytest.raises(exc) as info:
            await gh.default_branch(REPO)
    if status == 403:
        assert info.value.reset_at == 1700000000


@respx.mock(base_url=API_URL)
async def test_unexpected_status_raises_http_error(respx_mock):
    respx_mock.get("/repos/octo/demo").respond(status_code=500)
    async with GitHubSource() as gh:
        with pytest.raises(httpx.HTTPStatusError):
            await gh.default_branch(REPO)

import io
import tarfile

import pytest
import respx

from codebase_archaeologist.config import Settings
from codebase_archaeologist.errors import FetchError, RepoTooLargeError
from codebase_archaeologist.schemas import RemoteFile, RepoRef, RepoSnapshot
from codebase_archaeologist.sources.fetch import (
    API_URL,
    RAW_URL,
    GitHubFetcher,
    decode_text,
    git_blob_sha,
)

SHA = "a" * 40
SNAP = RepoSnapshot(repo=RepoRef(owner="octo", name="demo"), ref="main", commit_sha=SHA, files=[])
RAW = f"{RAW_URL}/octo/demo/{SHA}/"
TARBALL = f"{API_URL}/repos/octo/demo/tarball/{SHA}"


def rf(path: str, data: bytes, size: int | None = None) -> RemoteFile:
    return RemoteFile(
        path=path, size=len(data) if size is None else size, blob_sha=git_blob_sha(data)
    )


def settings(**kw) -> Settings:
    base = {"max_file_bytes": 1_000, "max_total_bytes": 10_000, "fetch_concurrency": 2}
    return Settings(_env_file=None, **{**base, **kw})


async def no_sleep(_):
    return None


async def collect(fetcher, files):
    return {f.path: f.text async for f in fetcher.fetch(SNAP, files)}


def make_tarball(members: dict[str, bytes], prefix="octo-demo-aaaaaaa/") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(prefix + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo(prefix + "evil-link")
        link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tar.addfile(link)
    return buf.getvalue()


def test_git_blob_sha_matches_git():
    # `printf 'hello\n' | git hash-object --stdin`
    assert git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_decode_text_rejects_binary():
    assert decode_text(b"print('hi')") == "print('hi')"
    assert decode_text(b"\x89PNG\x00\x00") is None


@respx.mock
async def test_raw_fetch_success_with_space_in_path():
    a, b = b"print('a')\n", b"# Title\n"
    respx.get(RAW + "app.py").respond(content=a)
    respx.get(RAW + "my%20notes.md").respond(content=b)
    fetcher = GitHubFetcher(settings())
    got = await collect(fetcher, [rf("app.py", a), rf("my notes.md", b)])
    assert got == {"app.py": "print('a')\n", "my notes.md": "# Title\n"}
    assert fetcher.report.method == "raw"
    assert fetcher.report.fetched == 2
    assert fetcher.report.bytes == len(a) + len(b)


@respx.mock
async def test_retries_on_429_then_succeeds():
    data = b"x = 1\n"
    respx.get(RAW + "a.py").mock(
        side_effect=[
            respx.MockResponse(429, headers={"retry-after": "1"}),
            respx.MockResponse(503),
            respx.MockResponse(200, content=data),
        ]
    )
    fetcher = GitHubFetcher(settings(), sleep=no_sleep)
    assert await collect(fetcher, [rf("a.py", data)]) == {"a.py": "x = 1\n"}


@respx.mock
async def test_per_file_problems_are_recorded_not_raised():
    ok = b"ok = True\n"
    big = b"y" * 2_000
    files = [rf("ok.py", ok)] * 1 + [
        rf("lied_about_size.py", big, size=10),  # listing said small; stream says otherwise
        rf("tampered.py", b"original"),
        rf("image.py", b"\x00\x01binary"),
        rf("gone.py", b"whatever"),
    ]
    # Five files, four bad -> only allowed with a permissive failure ratio.
    respx.get(RAW + "ok.py").respond(content=ok)
    respx.get(RAW + "lied_about_size.py").respond(content=big)
    respx.get(RAW + "tampered.py").respond(content=b"modified")
    respx.get(RAW + "image.py").respond(content=b"\x00\x01binary")
    respx.get(RAW + "gone.py").respond(404)
    fetcher = GitHubFetcher(settings(max_failed_ratio=1.0), sleep=no_sleep)
    assert await collect(fetcher, files) == {"ok.py": "ok = True\n"}
    assert fetcher.report.failed == {
        "lied_about_size.py": "too_large",
        "tampered.py": "integrity_mismatch",
        "image.py": "binary",
        "gone.py": "http_404",
    }


@respx.mock
async def test_too_many_failures_aborts():
    respx.get(url__startswith=RAW).respond(404)
    fetcher = GitHubFetcher(settings(), sleep=no_sleep)
    with pytest.raises(FetchError, match="3 of 3 files failed"):
        await collect(fetcher, [rf(f"{i}.py", b"x") for i in range(3)])


@respx.mock
async def test_total_byte_cap_aborts_fetch():
    data = b"z" * 900
    respx.get(url__startswith=RAW).respond(content=data)
    fetcher = GitHubFetcher(settings(max_total_bytes=2_000))
    with pytest.raises(RepoTooLargeError):
        await collect(fetcher, [rf(f"f{i}.py", data) for i in range(5)])


@respx.mock
async def test_tarball_used_above_threshold_and_skips_unwanted_and_symlinks():
    members = {"a.py": b"a = 1\n", "b.md": b"# b\n", "logo.png": b"\x89PNG"}
    respx.get(TARBALL).respond(content=make_tarball(members))
    fetcher = GitHubFetcher(settings(tarball_threshold_files=1))
    got = await collect(fetcher, [rf("a.py", members["a.py"]), rf("b.md", members["b.md"])])
    assert got == {"a.py": "a = 1\n", "b.md": "# b\n"}
    assert fetcher.report.method == "tarball"
    assert not fetcher.report.failed


@respx.mock
async def test_oversized_tarball_falls_back_to_raw():
    a, b = b"a = 1\n", b"b = 2\n"
    respx.get(TARBALL).respond(content=b"x" * 5_000, headers={"content-length": "5000"})
    respx.get(RAW + "a.py").respond(content=a)
    respx.get(RAW + "b.py").respond(content=b)
    fetcher = GitHubFetcher(settings(tarball_threshold_files=1, max_tarball_bytes=1_000))
    got = await collect(fetcher, [rf("a.py", a), rf("b.py", b)])
    assert got == {"a.py": "a = 1\n", "b.py": "b = 2\n"}
    assert fetcher.report.method == "tarball+raw"


@respx.mock
async def test_corrupt_tarball_falls_back_to_raw():
    a, b = b"a = 1\n", b"b = 2\n"
    respx.get(TARBALL).respond(content=b"not a gzip stream")
    respx.get(RAW + "a.py").respond(content=a)
    respx.get(RAW + "b.py").respond(content=b)
    fetcher = GitHubFetcher(settings(tarball_threshold_files=1))
    assert len(await collect(fetcher, [rf("a.py", a), rf("b.py", b)])) == 2


async def test_empty_file_list_yields_nothing():
    fetcher = GitHubFetcher(settings())
    assert await collect(fetcher, []) == {}

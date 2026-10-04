"""Offline unit tests for the REST client helpers and backend selection."""

import httpx
import pytest
import respx

from codebase_archaeologist.config import Settings
from codebase_archaeologist.errors import ArchaeologistError
from codebase_archaeologist.storage import factory
from codebase_archaeologist.storage.rest import RestClient, _in, _total, _vector_literal

BASE = "https://proj.supabase.co"


def test_in_filter_quotes_and_escapes():
    assert _in(["a.py", 'we"ird,(x).py', "back\\slash"]) == (
        'in.("a.py","we\\"ird,(x).py","back\\\\slash")'
    )


def test_total_parses_content_range():
    assert _total(httpx.Response(200, headers={"content-range": "0-0/42"})) == 42
    assert _total(httpx.Response(200, headers={"content-range": "*/7"})) == 7
    assert _total(httpx.Response(200)) == 0


def test_vector_literal():
    assert _vector_literal([0.5, -1.0, 1e-9]) == "[0.5,-1,1e-09]"


def test_client_headers_target_private_schema():
    c = RestClient(BASE, "sb_secret_abc")
    assert c.http.headers["Accept-Profile"] == "archaeologist"
    assert c.http.headers["apikey"] == "sb_secret_abc"
    assert "Authorization" not in c.http.headers
    assert RestClient(BASE, "eyJhbGciOi.x.y").http.headers["Authorization"].startswith("Bearer ")


@respx.mock
def test_select_all_paginates():
    calls = []

    def handler(request):
        start = int(request.headers["Range"].split("-")[0])
        calls.append(start)
        size = 1000 if start == 0 else 3
        return httpx.Response(200, json=[{"i": start + n} for n in range(size)])

    respx.get(f"{BASE}/rest/v1/files").mock(side_effect=handler)
    rows = RestClient(BASE, "k").select_all("files", {"repo": "eq.x"})
    assert len(rows) == 1003 and calls == [0, 1000]


@respx.mock
def test_errors_are_raised_with_status():
    respx.get(f"{BASE}/rest/v1/repos").respond(404, json={"message": "schema not exposed"})
    with pytest.raises(ArchaeologistError, match="404"):
        RestClient(BASE, "k").select_all("repos", {})


def settings(**kw):
    base = {"supabase_db_url": None, "supabase_url": None, "supabase_service_key": None}
    return Settings(_env_file=None, **{**base, **kw})


def test_choose_backend(monkeypatch):
    db = "postgresql://u:p@host:5432/postgres"
    rest = {"supabase_url": BASE, "supabase_service_key": "k"}
    monkeypatch.setattr(factory, "postgres_reachable", lambda url, timeout=3.0: True)
    assert factory.choose_backend(settings(supabase_db_url=db, **rest)) == "postgres"
    monkeypatch.setattr(factory, "postgres_reachable", lambda url, timeout=3.0: False)
    assert factory.choose_backend(settings(supabase_db_url=db, **rest)) == "rest"
    with pytest.raises(ArchaeologistError, match="unreachable"):
        factory.choose_backend(settings(supabase_db_url=db))
    assert factory.choose_backend(settings(**rest)) == "rest"
    assert factory.choose_backend(settings()) == "memory"
    assert factory.choose_backend(settings(storage_backend="memory", **rest)) == "memory"

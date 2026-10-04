"""Check local credentials without ever printing secret values.

Usage: .venv/bin/python scripts/check_env.py
"""

from __future__ import annotations

from codebase_archaeologist.config import Settings


def _is_set(value) -> bool:
    return value is not None and bool(value.get_secret_value().strip())


def main() -> None:
    s = Settings()
    for name in ("supabase_db_url", "supabase_service_key", "github_token", "openrouter_api_key"):
        print(f"{name.upper():22} {'set' if _is_set(getattr(s, name)) else 'missing'}")
    print(f"{'SUPABASE_URL':22} {'set' if s.supabase_url else 'missing'}")

    if s.supabase_url and _is_set(s.supabase_service_key):
        _check_rest(s)
    if not _is_set(s.supabase_db_url):
        return
    import psycopg

    url = s.supabase_db_url.get_secret_value()
    if url.count("@") != 1:
        print("SUPABASE_DB_URL: the password seems to contain '@'; use a letters/digits password")
        return
    try:
        with psycopg.connect(url, connect_timeout=15) as conn:
            version = conn.execute("show server_version").fetchone()[0]
            row = conn.execute(
                "select installed_version from pg_available_extensions where name = 'vector'"
            ).fetchone()
        print(f"Supabase: connected (Postgres {version}); pgvector installed: {row and row[0]}")
    except Exception as exc:  # never echo driver messages: they can contain URL fragments
        print(f"Supabase: connection failed ({type(exc).__name__}); check host, password, port")


def _check_rest(s: Settings) -> None:
    from codebase_archaeologist.storage.rest import RestClient

    client = RestClient(s.supabase_url, s.supabase_service_key.get_secret_value())
    try:
        n = client.count("repos", {})
        print(f"Supabase HTTPS: connected; {n} indexed repo(s) visible")
    except Exception as exc:
        # REST error bodies carry no credentials; show the status-code part only.
        print(f"Supabase HTTPS: failed ({type(exc).__name__}: {str(exc)[:120]})")
    finally:
        client.close()


if __name__ == "__main__":
    main()

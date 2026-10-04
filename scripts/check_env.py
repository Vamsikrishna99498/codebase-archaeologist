"""Check local credentials without ever printing secret values.

Usage: .venv/bin/python scripts/check_env.py
"""

from __future__ import annotations

from codebase_archaeologist.config import Settings


def _is_set(value) -> bool:
    return value is not None and bool(value.get_secret_value().strip())


def main() -> None:
    s = Settings()
    for name in ("supabase_db_url", "github_token", "openrouter_api_key"):
        print(f"{name.upper():20} {'set' if _is_set(getattr(s, name)) else 'missing'}")

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


if __name__ == "__main__":
    main()

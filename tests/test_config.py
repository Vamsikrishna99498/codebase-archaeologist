from codebase_archaeologist.config import Settings


def test_defaults(monkeypatch):
    for var in ("OPENROUTER_API_KEY", "GITHUB_TOKEN", "SUPABASE_DB_URL", "MAX_FILES"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.openrouter_api_key is None
    assert s.max_files == 2_000
    assert ".py" in s.allowed_extensions
    assert s.embedding_model == "BAAI/bge-small-en-v1.5"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("MAX_FILES", "10")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret_value")
    s = Settings(_env_file=None)
    assert s.max_files == 10
    assert s.github_token is not None
    assert s.github_token.get_secret_value() == "ghp_secret_value"


def test_secrets_hidden_in_repr(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-very-secret")
    s = Settings(_env_file=None)
    assert "sk-or-very-secret" not in repr(s)

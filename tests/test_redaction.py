"""Fake credentials are assembled at runtime so no secret-looking literal sits in the
repository (keeps GitHub push protection and secret scanners quiet)."""

import pytest

from codebase_archaeologist.guardrails.redaction import redact, redact_file, shannon_entropy
from codebase_archaeologist.schemas import FetchedFile

ALNUM = "Q7wK2mZp9XvR4tLb8NcY3hJd6FsG5aEe"  # 32 chars, high entropy


def fake(prefix: str, body_len: int, alphabet: str = ALNUM) -> str:
    return prefix + (alphabet * 3)[:body_len]


@pytest.mark.parametrize(
    ("secret", "kind"),
    [
        (fake("AK" + "IA", 16, "ABCDEFGHJKLMNPQRSTUVWXYZ234567"), "aws_access_key"),
        (fake("gh" + "p_", 36), "github_token"),
        (fake("github" + "_pat_", 40), "github_token"),
        (fake("sk-" + "or-v1-", 48), "openrouter_key"),
        (fake("sk-" + "ant-api03-", 40), "anthropic_key"),
        (fake("sk-" + "proj-", 40), "openai_key"),
        (fake("sk_" + "live_", 24), "stripe_key"),
        (fake("xo" + "xb-", 30), "slack_token"),
        (fake("AI" + "za", 35), "google_api_key"),
        (fake("h" + "f_", 34), "huggingface_token"),
        (fake("ey" + "J", 20) + "." + fake("ey" + "J", 20) + "." + fake("", 20), "jwt"),
    ],
)
def test_known_secret_formats(secret, kind):
    result = redact(f'KEY = "{secret}"  # do not commit')
    assert secret not in result.text
    assert f"[REDACTED:{kind}]" in result.text
    assert result.findings[kind] == 1


def test_private_key_block_preserves_line_count():
    begin, end = "-----BEGIN RSA PRI" + "VATE KEY-----", "-----END RSA PRI" + "VATE KEY-----"
    text = f"line1\n{begin}\nMIIEow\nIBAAKCAQEA\n{end}\nline6\n"
    result = redact(text)
    assert "MIIEow" not in result.text
    assert result.text.count("\n") == text.count("\n")
    assert result.text.splitlines()[5] == "line6"
    assert result.findings["private_key"] == 1


def test_connection_string_password_only():
    pw = "Tr0ub4dor" + "&3xyz"
    result = redact(f'DB = "postgresql://admin:{pw}@db.internal:5432/app"')
    assert pw not in result.text
    assert "postgresql://admin:[REDACTED:password]@db.internal:5432/app" in result.text


@pytest.mark.parametrize(
    "line",
    [
        'password = "' + "kX9#mQ2$vL7p" + '"',
        'API_KEY: "' + "f3a9c1e7b2d84f60" + '"',
        '"client_secret": "' + "Zr8vN2qLx5Tw" + '"',
    ],
)
def test_credential_assignments(line):
    result = redact(line)
    assert "[REDACTED:secret]" in result.text
    assert result.findings["credential_assignment"] == 1


@pytest.mark.parametrize(
    "line",
    [
        'token = os.getenv("GITHUB_TOKEN")',
        'password = "changeme123"',
        'api_key = "your-api-key-here"',
        'password = "aaaaaaaaaaaa"',  # low entropy
        'secret_name = "short"',
        'PASSWORD_FIELD = "${DB_PASSWORD}"',
        "def get_token(self): return self._token",
        "max_tokens = 800",
        'url = "https://github.com/octo/demo"',
        'remote = "git@github.com:octo/demo.git"',
        'proxy = "http://user:pass@localhost:8080"',  # test-fixture credentials
        '("hTTp://u:p@some.host/path", "http://proxy")',
        'url = "http://{}:{}@{}:9000/path".format(user, pw, host)',
        'f"http://{ENCODED_USER}:{ENCODED_PASSWORD}@request.com/"',
    ],
)
def test_benign_code_untouched(line):
    result = redact(line)
    assert result.text == line
    assert not result.redacted


def test_emails_redacted_except_safe_domains():
    text = "# Author: jane.doe@company.io\n# Contact: support@example.com\n"
    result = redact(text)
    assert "jane.doe@company.io" not in result.text
    assert "support@example.com" in result.text
    assert result.findings == {"email": 1}


def test_shannon_entropy():
    assert shannon_entropy("aaaa") == 0
    assert shannon_entropy("abcd") == 2


def test_redact_file_keeps_identity():
    key = fake("gh" + "p_", 36)
    original = FetchedFile(path="cfg.py", blob_sha="abc", text=f'T = "{key}"\n', size=50)
    redacted, findings = redact_file(original)
    assert redacted.blob_sha == "abc" and redacted.path == "cfg.py"
    assert key not in redacted.text
    assert findings["github_token"] == 1

    clean = FetchedFile(path="a.py", blob_sha="d", text="x = 1\n", size=6)
    same, findings = redact_file(clean)
    assert same is clean and not findings

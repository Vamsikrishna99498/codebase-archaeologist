"""Ingest guardrail: scrub secrets and email addresses from code before it is embedded,
stored in the cloud, or sent to an LLM.

Matches are replaced with ``[REDACTED:<kind>]`` so the code stays readable and the
model can still say "an API key is configured here". Line numbers are preserved
(multi-line matches keep their newlines) so citations stay accurate.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from codebase_archaeologist.schemas import FetchedFile

# Order matters: more specific patterns first (e.g. sk-or- before generic sk-).
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "private_key",
        re.compile(
            r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY(?: BLOCK)?-----"
            r"[\s\S]+?-----END (?:[A-Z]+ )?PRIVATE KEY(?: BLOCK)?-----"
        ),
    ),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})")),
    ("openrouter_key", re.compile(r"\bsk-or-[A-Za-z0-9-]{20,}")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("stripe_key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
]

# scheme://user:PASSWORD@host -> only the password is redacted.
_CONN_STRING = re.compile(r"(?P<pre>\b[a-z][a-z0-9+.-]*://[^:@\s/]+:)(?P<pw>[^@\s/]+)(?=@)")

# key = "value" style assignments of credential-looking names to string literals.
_ASSIGNMENT = re.compile(
    r"""(?ix)
    (?P<pre>\b[\w.-]*(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|
        credential|auth[_-]?key)[\w.-]*["']?\s*[:=]\s*)
    (?P<q>["'])(?P<val>[^"'\s]{8,})(?P=q)
    """
)
_PLACEHOLDER_HINTS = ("xxx", "your", "example", "changeme", "placeholder", "dummy", "<", "${", "{{")
_MIN_ENTROPY = 3.0  # bits per character

_EMAIL = re.compile(
    # Not preceded by ":" or "/" so URL userinfo (scheme://user:pw@host) is not an email.
    r"(?<![\w.%+:/-])[A-Za-z0-9._%+-]+@(?P<domain>[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})\b"
)
_SAFE_EMAIL_DOMAINS = ("example.com", "example.org", "example.net", "localhost")


def _looks_like_real_secret(value: str) -> bool:
    """Filter out placeholders and test fixtures such as "pass" or "changeme"."""
    if "[REDACTED:" in value or len(value) < 8:
        return False
    if value.startswith(("{", "%", "$")):  # format/template placeholders
        return False
    if any(h in value.lower() for h in _PLACEHOLDER_HINTS):
        return False
    return shannon_entropy(value) >= _MIN_ENTROPY


def _tag(kind: str, matched: str = "") -> str:
    # Keep the original line count so line-range citations remain valid.
    return f"[REDACTED:{kind}]" + "\n" * matched.count("\n")


def shannon_entropy(value: str) -> float:
    counts = Counter(value)
    n = len(value)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


@dataclass
class RedactionResult:
    text: str
    findings: Counter[str] = field(default_factory=Counter)  # kind -> count

    @property
    def redacted(self) -> bool:
        return bool(self.findings)


def redact(text: str) -> RedactionResult:
    findings: Counter[str] = Counter()

    def sub(kind: str, pattern: re.Pattern[str], s: str) -> str:
        def repl(m: re.Match[str]) -> str:
            findings[kind] += 1
            return _tag(kind, m.group(0))

        return pattern.sub(repl, s)

    for kind, pattern in SECRET_PATTERNS:
        text = sub(kind, pattern, text)

    def conn_repl(m: re.Match[str]) -> str:
        if not _looks_like_real_secret(m["pw"]):
            return m.group(0)
        findings["connection_string_password"] += 1
        return m["pre"] + _tag("password")

    text = _CONN_STRING.sub(conn_repl, text)

    def assign_repl(m: re.Match[str]) -> str:
        if not _looks_like_real_secret(m["val"]):
            return m.group(0)
        findings["credential_assignment"] += 1
        return f"{m['pre']}{m['q']}{_tag('secret')}{m['q']}"

    text = _ASSIGNMENT.sub(assign_repl, text)

    def email_repl(m: re.Match[str]) -> str:
        if m.group(0).lower().startswith("git@") or m["domain"].lower().endswith(
            _SAFE_EMAIL_DOMAINS
        ):
            return m.group(0)
        findings["email"] += 1
        return _tag("email")

    text = _EMAIL.sub(email_repl, text)
    return RedactionResult(text=text, findings=findings)


def redact_file(file: FetchedFile) -> tuple[FetchedFile, Counter[str]]:
    """Return a redacted copy of `file` (blob SHA kept: it identifies the source)."""
    result = redact(file.text)
    if not result.redacted:
        return file, result.findings
    return file.model_copy(update={"text": result.text}), result.findings

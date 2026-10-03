"""Logging setup shared by the CLI, notebook and (later) the web apps."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once; safe to call repeatedly."""
    logging.basicConfig(level=level.upper(), format=_FORMAT, force=True)
    # Third-party libraries are noisy at INFO.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

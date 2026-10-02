"""Structured JSON logging for the analysis pipeline.

Every record carries the fields required for auditing: ``analysis_id``,
``stage``, ``duration_ms``, ``status``, ``error`` and ``model_version``.

REDACTION: ``SENSITIVE_KEYS`` is scrubbed from every log record. Cookies,
``Authorization`` headers, tokens and raw page bodies must never reach a log
file. The redactor works on the key name, which is the only reliable signal we
have, plus a value-level guard for obvious bearer/JWT shapes.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

import structlog

SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "cookie",
        "cookies",
        "set-cookie",
        "set_cookie",
        "authorization",
        "auth",
        "token",
        "access_token",
        "refresh_token",
        "api_key",
        "apikey",
        "password",
        "passwd",
        "secret",
        "session",
        "sessionid",
        "session_id",
        "jsessionid",
        "phpsessid",
        "credential",
        "credentials",
        "body",
        "html",
        "page_body",
        "response_body",
        "screenshot",
        "raw_html",
    }
)

# Value-level guard: strip anything that looks like a JWT or a Bearer token.
_BEARER_RE = re.compile(r"(?i)\bbearer\s+\S+")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")

STAGES: tuple[str, ...] = (
    "validate",
    "url_model",
    "fetch_html",
    "html_model",
    "screenshot",
    "vision_model",
    "fusion",
    "xai",
    "done",
)


def redact(value: Any, _depth: int = 0) -> Any:
    """Recursively drop sensitive keys and scrub token-shaped values."""
    if _depth > 8:
        return "<truncated:max-depth>"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if str(k).strip().lower() in SENSITIVE_KEYS:
                out[str(k)] = "<redacted>"
            else:
                out[str(k)] = redact(v, _depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(v, _depth + 1) for v in value][:200]
    if isinstance(value, str):
        value = _BEARER_RE.sub("Bearer <redacted>", value)
        return _JWT_RE.sub("<redacted-jwt>", value)
    return value


def _add_context(
    _logger: Any, _name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """structlog processor: apply redaction to every record."""
    return redact(event_dict)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Install the structlog + stdlib logging configuration.

    ``fmt="json"`` (default) produces one machine-readable object per line.
    ``fmt="console"`` is the human-readable variant for interactive work.
    """
    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if fmt == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=[*shared, _add_context, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        # stdlib.LoggerFactory (not PrintLoggerFactory) is required because
        # structlog.stdlib.add_logger_name reads ``logger.name``, which
        # PrintLogger does not provide.
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper(), force=True)
    # Keep third-party loggers from leaking request URLs or bodies.
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "playwright"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> Any:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)


@contextmanager
def stage_timer(
    analysis_id: str,
    stage: str,
    model_version: str | None = None,
    **extra: Any,
) -> Iterator[dict[str, Any]]:
    """Time a pipeline stage and emit exactly one structured record.

    Yields a mutable dict; setting ``ctx["error"] = "..."`` marks the stage as
    failed in the emitted log. The context is also returned so the caller can
    reuse ``duration_ms`` in the API response.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    log = get_logger("pipeline")
    ctx: dict[str, Any] = {"error": None, **extra}
    started = time.perf_counter()
    try:
        yield ctx
    except Exception as exc:  # noqa: BLE001 - re-raised after logging
        # Keep the caller's own diagnosis (e.g. "connection reset") and append the
        # raised exception, so the record explains the failure rather than only
        # the fact that one happened.
        raised = f"{type(exc).__name__}: {exc}"
        prior = ctx.get("error")
        ctx["error"] = f"{prior} | {raised}" if prior else raised
        raise
    finally:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        ctx["duration_ms"] = elapsed_ms
        log.info(
            "stage_complete",
            analysis_id=analysis_id,
            stage=stage,
            status="failed" if ctx.get("error") else "ok",
            duration_ms=elapsed_ms,
            error=ctx.get("error"),
            model_version=model_version,
            **{k: v for k, v in ctx.items() if k not in {"error", "duration_ms"}},
        )


def dumps(record: dict[str, Any]) -> str:
    """Serialise a record to a JSON string (used by the API's inline logs)."""
    return json.dumps(redact(record), default=str)

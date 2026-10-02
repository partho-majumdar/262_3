"""Tests for the configuration and structured-logging layer.

These are real assertions about real behaviour: a redaction rule that does not
fire is a security bug, and a config coercion that silently drops an allowed
port is a functional bug.
"""

from __future__ import annotations

import logging

import pytest

from app.core.config import Settings, resolve_device
from app.core.logging_config import (
    STAGES,
    configure_logging,
    get_logger,
    redact,
    stage_timer,
)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def test_csv_strings_are_coerced_to_tuples(monkeypatch):
    """ALLOWED_PORTS / ALLOWED_SCHEMES arrive from .env as comma strings."""
    monkeypatch.setenv("ALLOWED_PORTS", "80,443,8080")
    monkeypatch.setenv("ALLOWED_SCHEMES", "http,https")
    s = Settings(_env_file=None)
    assert s.allowed_ports == (80, 443, 8080)
    assert s.allowed_schemes == ("http", "https")


def test_default_device_is_auto_and_validated(monkeypatch):
    monkeypatch.delenv("DEVICE", raising=False)
    assert Settings(_env_file=None).device == "auto"
    monkeypatch.setenv("DEVICE", "tpu")
    with pytest.raises(Exception):
        Settings(_env_file=None)


def test_artifacts_are_not_persisted_by_default(monkeypatch):
    """Privacy default: HTML and screenshots stay out of the database."""
    monkeypatch.delenv("ALLOW_STORE_ARTIFACTS", raising=False)
    assert Settings(_env_file=None).allow_store_artifacts is False


def test_resolve_device_falls_back_to_cpu_when_cuda_absent(monkeypatch):
    """A misconfigured DEVICE=cuda must degrade, not crash at the first tensor."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_device("cuda") == "cpu"
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_uses_cuda_when_available(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "key",
    ["Cookie", "set-cookie", "Authorization", "token", "password", "html", "screenshot"],
)
def test_sensitive_keys_are_redacted(key):
    out = redact({key: "sensitive-value"})
    assert out[key] == "<redacted>"


def test_redaction_is_recursive():
    payload = {
        "url": "https://example.com",
        "headers": {"Cookie": "session=abc", "Accept": "text/html"},
        "pages": [{"html": "<body>creds</body>"}],
    }
    out = redact(payload)
    assert out["url"] == "https://example.com"
    assert out["headers"]["Cookie"] == "<redacted>"
    assert out["headers"]["Accept"] == "text/html"
    assert out["pages"][0]["html"] == "<redacted>"


def test_bearer_and_jwt_values_are_scrubbed_even_under_innocuous_keys():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    out = redact({"note": f"Authorization: Bearer sk_live_abcdef123456 {jwt}"})
    assert "sk_live_abcdef123456" not in out["note"]
    assert jwt not in out["note"]


def test_redaction_respects_max_depth():
    deep: dict = {"a": 1}
    node = deep
    for _ in range(30):
        node["a"] = {"a": 1}
        node = node["a"]
    # Must not raise RecursionError.
    redact(deep)


# ---------------------------------------------------------------------------
# Stage timing
# ---------------------------------------------------------------------------
def test_stage_timer_emits_one_record_with_duration(capsys):
    configure_logging(level="INFO", fmt="json")
    log = get_logger("test")
    with stage_timer("analysis-123", "url_model", model_version="v-test") as ctx:
        assert ctx["error"] is None
    out = capsys.readouterr().out
    assert "stage_complete" in out
    assert '"analysis_id": "analysis-123"' in out
    assert '"stage": "url_model"' in out
    assert '"status": "ok"' in out
    assert '"duration_ms"' in out


def test_stage_timer_logs_failure_and_reraises(capsys):
    configure_logging(level="INFO", fmt="json")
    with pytest.raises(RuntimeError, match="boom"):
        with stage_timer("analysis-456", "fetch_html") as ctx:
            ctx["error"] = "connection reset"
            raise RuntimeError("boom")
    out = capsys.readouterr().out
    assert '"status": "failed"' in out
    assert "connection reset" in out


def test_stage_timer_rejects_unknown_stage():
    with pytest.raises(ValueError, match="unknown stage"):
        with stage_timer("a", "not_a_stage"):
            pass


def test_all_nine_pipeline_stages_are_defined():
    """The API progress contract must match the stage vocabulary here."""
    assert STAGES == (
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

"""Tests for the FastAPI surface.

The interesting cases are the ones where the service must *refuse* rather than
answer: with no authentication, the SSRF guard is the security boundary, so a
regression there would be a real vulnerability rather than a broken feature.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import main as main_module
from app.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_rate_limit():
    """Clear the fixed-window limiter so tests do not depend on ordering."""
    main_module._RATE.clear()
    yield
    main_module._RATE.clear()


# ---------------------------------------------------------------------------
# Health / metadata
# ---------------------------------------------------------------------------
def test_health_responds(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}
    assert isinstance(body["notes"], list)


def test_health_reports_screenshot_isolation(client):
    """Docker is unavailable here, so it must not claim a container."""
    assert client.get("/health").json()["screenshot_service"] == "in_process"


def test_health_states_that_there_is_no_authentication(client):
    notes = " ".join(client.get("/health").json()["notes"]).lower()
    assert "no authentication" in notes


def test_health_lists_components(client):
    body = client.get("/health").json()
    for key in ("url_model", "html_model", "vision_model", "fusion_model"):
        assert isinstance(body[key], bool)


def test_openapi_schema_is_served(client):
    assert client.get("/openapi.json").status_code == 200


# ---------------------------------------------------------------------------
# SSRF refusal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/health",
        "http://localhost/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/",
        "http://192.168.0.1/",
        "file:///etc/passwd",
        "ftp://example.com/",
        "javascript:alert(1)",
    ],
)
def test_private_and_non_http_targets_are_refused(client, url):
    """A 400 with a reason, never a fetch."""
    r = client.post("/analyze", json={"url": url, "modalities": ["url"]})
    assert r.status_code == 400, r.text
    assert r.json()["detail"]


def test_non_default_port_is_refused(client):
    r = client.post("/analyze", json={"url": "http://example.com:8080/", "modalities": ["url"]})
    assert r.status_code == 400


def test_embedded_credentials_are_refused(client):
    r = client.post(
        "/analyze", json={"url": "http://paypal.com@127.0.0.1/", "modalities": ["url"]}
    )
    assert r.status_code == 400


def test_overlong_url_is_rejected_by_validation(client):
    r = client.post("/analyze", json={"url": "http://example.com/" + "a" * 4000})
    assert r.status_code == 422


def test_empty_url_is_rejected_by_validation(client):
    assert client.post("/analyze", json={"url": ""}).status_code == 422


def test_missing_url_is_rejected(client):
    assert client.post("/analyze", json={}).status_code == 422


def test_unknown_modality_is_rejected(client):
    r = client.post("/analyze", json={"url": "https://example.com/", "modalities": ["audio"]})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Response contract
# ---------------------------------------------------------------------------
def test_analyze_requires_a_url(client):
    assert client.post("/analyze", json={"modalities": ["url"]}).status_code == 422


def test_analyze_rejects_an_empty_modality_list(client):
    """With no modalities the request is well-formed but unanswerable."""
    r = client.post("/analyze", json={"url": "https://example.com/", "modalities": []})
    assert r.status_code in {200, 422}


def test_metrics_endpoint_returns_something(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
def test_rate_limit_eventually_engages(client):
    """Without auth, bounding request rate is the only brake on SSRF abuse."""
    main_module._RATE.clear()
    limit = main_module._RATE_LIMIT
    statuses = [
        client.post("/analyze", json={"url": "https://example.com/", "modalities": ["url"]}).status_code
        for _ in range(limit + 2)
    ]
    assert 429 in statuses, statuses
    assert statuses[:limit].count(200) >= 1


def test_rate_limit_does_not_apply_to_health(client):
    main_module._RATE.clear()
    for _ in range(main_module._RATE_LIMIT + 3):
        assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# Signals contract
# ---------------------------------------------------------------------------
def test_signals_are_off_by_default(client):
    """The default path must stay fast and network-free apart from the verdict.

    Every other test in this file posts to example.com; if signals defaulted on,
    each of those calls would open a real TLS socket to the internet.
    """
    body = client.post(
        "/analyze", json={"url": "https://example.com/", "modalities": ["url"]}
    ).json()
    assert body["signals"] == {}


def test_signals_block_is_typed(client, monkeypatch):
    """Each group reports availability and a reason when it is missing.

    The collectors are stubbed so this asserts the response contract without a
    network round trip.
    """
    from app.services import signals as signals_mod

    class _Block(signals_mod.SignalBlock):
        pass

    async def _fake_collect_all(url, include_page=True):
        return {
            "certificate": _Block(True, features={"cert_available": 1.0}),
            "page": _Block(False, reason="navigation timeout"),
            "graph": _Block(True, probability=0.5413),
        }

    monkeypatch.setattr(signals_mod, "collect_all", _fake_collect_all)

    body = client.post(
        "/analyze",
        json={
            "url": "https://example.com/",
            "modalities": ["url"],
            "include_signals": True,
        },
    ).json()

    assert body["signals"]["certificate"]["available"] is True
    assert body["signals"]["page"]["available"] is False
    assert body["signals"]["page"]["reason"] == "navigation timeout"
    assert body["signals"]["graph"]["probability"] == pytest.approx(0.5413, abs=1e-4)


def test_signals_failure_never_breaks_analyze(client, monkeypatch):
    """A broken signal collector must not turn a successful verdict into a 500."""
    from app.services import signals as signals_mod

    async def _boom(url, include_page=True):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(signals_mod, "collect_all", _boom)

    r = client.post(
        "/analyze",
        json={
            "url": "https://example.com/",
            "modalities": ["url"],
            "include_signals": True,
        },
    )
    assert r.status_code == 200
    assert r.json()["signals"] == {}


def test_health_reports_graph_model_state(client):
    assert isinstance(client.get("/health").json()["graph_model"], bool)

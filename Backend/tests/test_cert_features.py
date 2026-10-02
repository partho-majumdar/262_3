"""Tests for the TLS certificate feature extractor.

None of these touch the network. The module's only seam to the outside world is
:func:`cert_features._tls_handshake`, so most tests script that function's
outcomes; a second group goes one level lower and fakes the socket plus the
:class:`ssl.SSLContext` to exercise the handshake wrapper itself. The last group
feeds a **real** DER certificate through the fallback parser, because that
parser is the one piece of this module with enough moving parts to be wrong in
a way a hand-written fake would happily agree with.

What these tests are really guarding: the vector must never change width, and a
dead host must never raise. A phishing URL that cannot be reached is the normal
case, not the exception.
"""

from __future__ import annotations

import base64
import socket
import ssl
from datetime import datetime, timedelta

import pytest

from app.preprocessing import cert_features as cf
from app.preprocessing.cert_features import (
    CERT_FEATURE_NAMES,
    N_CERT_FEATURES,
    CertTarget,
    cert_context_for,
    describe_certificate,
    extract_cert_features,
    extract_cert_features_batch,
    extract_cert_features_for_url,
    is_ip_literal,
    probe_certificate,
    ssl_error_reason,
)

# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------
#: A real self-signed certificate, DER-encoded, minted once with
#: ``openssl req -x509 -subj "/CN=Login Secure Bank/O=Acme"`` and carrying
#: ``DNS:*.secure-login.example, DNS:secure-login.example, IP:10.1.2.3``.
#: Baking in actual bytes (rather than a mock) is what makes the DER walker
#: test meaningful; a mock would only prove the mock agrees with itself.
_REAL_SAN_CERT_DER = base64.b64decode(
    "MIIDJTCCAg2gAwIBAgIUTV9dkqPzYGa7q0TP9fiNagrXUlcwDQYJKoZIhvcNAQELBQAwKzEaMBgG"
    "A1UEAwwRTG9naW4gU2VjdXJlIEJhbmsxDTALBgNVBAoMBEFjbWUwHhcNMjYxMDAxMjAwMjU3WhcN"
    "MjcxMTA1MjAwMjU3WjArMRowGAYDVQQDDBFMb2dpbiBTZWN1cmUgQmFuazENMAsGA1UECgwEQWNt"
    "ZTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBANDQyo0k/x66SkJvYcXinVZUkBK1M3el"
    "N/sgyxyUtXUZEF6J62o7U5n8FhCWcTrMLDCrBosGIxWVgsNUokJYvzEoD33iVi6/em81piLY0f8o"
    "2Hp/NDSbLarVcg3qIMtVrl9UM8bNVbNpTyWgkaL72TBtgRIHQ7Y+WkVfwls8Sb+C46RNkL1IjJj9"
    "UMK26XXX1wzoVLB5K5/Ra1Y8gVbyO7l/OQvFhEdrfUpAmvU3HYzc6QmSPPLe2MlWG4UtfrDHSQ2Y"
    "l5XuGSEG5lrLOkMnc3aj17LEvdbbIacjGI7NK4gnHymA8F2pzEOdwn0d0XO+/6V7uHj9Ns3ydI9q"
    "V+lNS1sCAwEAAaNBMD8wPQYDVR0RBDYwNIIWKi5zZWN1cmUtbG9naW4uZXhhbXBsZYIUc2VjdXJl"
    "LWxvZ2luLmV4YW1wbGWHBAoBAgMwDQYJKoZIhvcNAQELBQADggEBAJh3Hw0Ntm7zoCNNIlqMrTw7"
    "QHa9ewc5FlyZSGQdxsdGUsT8FP1hnaqGgmPOJpS++DIQwdp7AgmZZk+3am+v7fn6HUF41j5XpTT8"
    "olnF3FuPXIMCt2WD1tld54dSn6yM5uPnEvmF2jXs9eRDaOWTXcspOrI7Lr821PZLGXKipf/mJM5L"
    "HA/j6NWJJ+ltWTOZxpNP3ujmU8BL3ciGtDXVPaBU2/4GMz+AgoAUKPW2l5C21vhQkilF+3nri8X3"
    "N53oDHs+vD0yq3m5/JOSgoKK9ZGiK6kJYXtLOXNfvF6D65dHMmjf8aOWndcukf6cJ2uS4A5wRnI5"
    "rHPe9CpiYuKe6gY="
)


def _verified_cert_dict(
    *,
    subject_cn: str = "secure-login.example",
    issuer_cn: str = "Example Root CA",
    san: tuple = (("DNS", "secure-login.example"),),
    serial: str = "0a1b2c3d4e5f60718293a4b5c6d7e8f9",
    days_valid_for: int = 365,
    days_since_issue: int = 10,
) -> dict:
    """The dict shape CPython's ``getpeercert()`` returns on a trusted chain."""
    now = cf._now_utc()
    return {
        "subject": ((("commonName", subject_cn),),),
        "issuer": ((("organizationName", "Example Inc"),), (("commonName", issuer_cn),)),
        "notBefore": now - timedelta(days=days_since_issue),
        "notAfter": now + timedelta(days=days_valid_for),
        "serialNumber": serial,
        "subjectAltName": san,
        "version": 3,
    }


def _script_handshake(monkeypatch, *outcomes):
    """Replace the network seam with a scripted sequence of handshake results.

    Each outcome is either a return tuple ``(cert_dict, der, version, cipher)``
    or an exception instance to raise. The last outcome repeats if the module
    makes more calls than were scripted, and the returned list records every
    call so tests can assert on retry behaviour.
    """
    calls = []

    def fake(host, port, *, timeout, context):
        calls.append((host, port, context))
        outcome = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(cf, "_tls_handshake", fake)
    return calls


def _verification_error(detail: str = "self signed certificate") -> Exception:
    return ssl.SSLCertVerificationError(1, f"[SSL: CERTIFICATE_VERIFY_FAILED] {detail}")


# ---------------------------------------------------------------------------
# Shape of the feature contract
# ---------------------------------------------------------------------------
def test_declared_count_matches_the_name_list():
    assert N_CERT_FEATURES == len(CERT_FEATURE_NAMES)
    assert len(set(CERT_FEATURE_NAMES)) == N_CERT_FEATURES
    assert all(isinstance(n, str) and n for n in CERT_FEATURE_NAMES)


def test_every_feature_is_actually_written_by_the_vector_body():
    """A misspelled name would raise inside the extractor and be swallowed by
    its own ``except``, leaving a silently all-zero column. Assert the set of
    names the body writes equals the set the module declares."""
    import inspect
    import re as _re

    source = inspect.getsource(cf._vector_from_probe)
    written = set(_re.findall(r'put\(\s*"([^"]+)"', source))
    assert written == set(CERT_FEATURE_NAMES)
    assert len(written) == N_CERT_FEATURES  # no column written twice


def test_vector_width_is_stable_across_every_outcome(monkeypatch):
    """Width must not depend on how far the extraction got.

    A short vector would either crash the fusion layer or, worse, get padded
    silently and shift every later column.
    """
    outcomes = [
        (None, None, "", ""),                                     # nothing presented
        _verification_error(),                                    # unverified fallback
        (_verified_cert_dict(), b"\x30\x03\x02\x01\x01", "TLSv1.3", "X"),
        ConnectionRefusedError(10061, "Connection refused"),
    ]
    for outcome in outcomes:
        _script_handshake(monkeypatch, outcome)
        assert len(extract_cert_features("a.test")) == N_CERT_FEATURES


def test_feature_order_is_the_declared_order():
    """Guard the checkpoint contract: names and indices are locked together."""
    assert CERT_FEATURE_NAMES[:5] == [
        "cert_available",
        "cert_verified",
        "verify_failed",
        "cert_from_der_parse",
        "self_signed_guess",
    ]
    assert CERT_FEATURE_NAMES == sorted(set(CERT_FEATURE_NAMES), key=CERT_FEATURE_NAMES.index)
    assert CERT_FEATURE_NAMES[-1] == "tls_version_id"


# ---------------------------------------------------------------------------
# Failure paths: total by construction
# ---------------------------------------------------------------------------
def test_connection_refused_returns_zeros_with_a_reason(monkeypatch):
    _script_handshake(monkeypatch, ConnectionRefusedError(10061, "Connection refused"))
    errors: dict = {}
    vec = extract_cert_features("dead.test", errors=errors)

    assert vec == [0.0] * N_CERT_FEATURES
    assert errors["reason"] == "connection refused"
    assert errors["available"] is False


def test_dns_failure_is_reported_rather_than_raised(monkeypatch):
    _script_handshake(monkeypatch, socket.gaierror(-2, "Name or service not known"))
    assert extract_cert_features("nx.test") == [0.0] * N_CERT_FEATURES
    assert "dns lookup failed" in cf.ssl_error_reason("nx.test")


def test_timeout_is_reported_rather_than_raised(monkeypatch):
    _script_handshake(monkeypatch, TimeoutError("timed out"))
    assert cf.ssl_error_reason("slow.test") == "timeout"


def test_blank_host_never_reaches_the_socket(monkeypatch):
    calls = _script_handshake(monkeypatch, (_verified_cert_dict(), b"der", "TLSv1.3", "X"))
    assert extract_cert_features("") == [0.0] * N_CERT_FEATURES
    assert calls == []


def test_handshake_failure_on_both_paths_is_survivable(monkeypatch):
    _script_handshake(monkeypatch, ssl.SSLError("wrong version number"), ssl.SSLError("nope"))
    errors: dict = {}
    assert len(extract_cert_features("a.test", errors=errors)) == N_CERT_FEATURES
    assert errors["available"] is False
    assert errors["reason"]


def test_socket_error_is_not_retried_unverified(monkeypatch):
    """Retrying a refused connection would double the latency of every dead URL.

    There is no certificate to observe on the second attempt, so the module
    must give up after the first.
    """
    calls = _script_handshake(monkeypatch, ConnectionResetError("reset by peer"))
    extract_cert_features("a.test")
    assert len(calls) == 1


def test_empty_host_placeholder_error_dict_is_still_returned(monkeypatch):
    """``errors`` must be usable even when the probe could not be built at all."""
    monkeypatch.setattr(
        cf, "probe_certificate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    errors: dict = {}
    vec = extract_cert_features("a.test", errors=errors)
    assert vec == [0.0] * N_CERT_FEATURES
    assert "boom" in errors["reason"]


# ---------------------------------------------------------------------------
# Self-signed handling: a feature, not a failure
# ---------------------------------------------------------------------------
def test_self_signed_certificate_is_still_observed(monkeypatch):
    """A self-signed cert is exactly the phishing case, so it must not be lost.

    The verifying handshake fails, the unverified retry succeeds, and the
    certificate is read from the DER the second time round.
    """
    calls = _script_handshake(
        monkeypatch, _verification_error(), (None, _REAL_SAN_CERT_DER, "TLSv1.3", "X")
    )
    errors: dict = {}
    vec = extract_cert_features("login.example", errors=errors)
    as_dict = dict(zip(CERT_FEATURE_NAMES, vec))

    assert len(calls) == 2
    assert as_dict["cert_available"] == 1.0
    assert as_dict["verify_failed"] == 1.0
    assert as_dict["cert_verified"] == 0.0
    assert as_dict["cert_from_der_parse"] == 1.0
    assert as_dict["self_signed_guess"] == 1.0
    assert as_dict["issuer_equals_subject"] == 1.0
    assert errors["reason"] == "self-signed certificate"


@pytest.mark.parametrize(
    "detail,expected",
    [
        ("self signed certificate", "self-signed certificate"),
        ("certificate has expired", "expired"),
        ("hostname mismatch", "hostname mismatch"),
        ("unable to get local issuer certificate", "unknown authority"),
        ("some other problem", "certificate verify failed"),
    ],
)
def test_verification_failures_are_classified_for_humans(detail, expected):
    """The XAI panel shows this phrase, so it must be short and specific."""
    reason = cf._classify_error(_verification_error(detail))
    assert expected in reason
    assert "SSLCertVerificationError" not in reason


def test_verified_certificate_reports_no_reason(monkeypatch):
    _script_handshake(monkeypatch, (_verified_cert_dict(), b"der", "TLSv1.3", "X"))
    as_dict = dict(zip(CERT_FEATURE_NAMES, extract_cert_features("secure-login.example")))

    assert as_dict["cert_verified"] == 1.0
    assert as_dict["verify_failed"] == 0.0
    assert cf.ssl_error_reason("secure-login.example") is None


def test_no_certificate_presented_is_reported(monkeypatch):
    _script_handshake(monkeypatch, (None, None, "", ""))
    errors: dict = {}
    assert extract_cert_features("a.test", errors=errors) == [0.0] * N_CERT_FEATURES
    assert errors["reason"] == "no certificate presented"


# ---------------------------------------------------------------------------
# Validity arithmetic
# ---------------------------------------------------------------------------
def test_days_to_expiry_is_negative_and_clipped_when_already_expired(monkeypatch):
    """A cert dead for 15 years must not push days_to_expiry to -5000.

    The clip keeps the feature informative (expired) while leaving the linear
    model's input range sane.
    """
    _script_handshake(
        monkeypatch, (_verified_cert_dict(days_valid_for=-5000, days_since_issue=6000), b"d", "T", "C")
    )
    as_dict = dict(zip(CERT_FEATURE_NAMES, extract_cert_features("a.test")))

    assert as_dict["days_to_expiry"] == cf._EXPIRY_CLIP[0] == -365.0
    assert as_dict["is_expired"] == 1.0
    assert as_dict["cert_age_days"] == cf._AGE_CLIP[1] == 3650.0


def test_long_lived_expiry_is_clipped_at_the_top(monkeypatch):
    _script_handshake(
        monkeypatch, (_verified_cert_dict(days_valid_for=36500, days_since_issue=0), b"d", "T", "C")
    )
    as_dict = dict(zip(CERT_FEATURE_NAMES, extract_cert_features("a.test")))
    assert as_dict["days_to_expiry"] == cf._EXPIRY_CLIP[1] == 3650.0
    assert as_dict["is_expired"] == 0.0


def test_certificate_not_yet_valid_is_flagged(monkeypatch):
    """notBefore in the future is a real, observed signal -- clone sites are
    frequently issued with a clock that runs ahead."""
    now = cf._now_utc()
    cert = _verified_cert_dict()
    cert["notBefore"] = now + timedelta(days=3)
    cert["notAfter"] = now + timedelta(days=400)
    _script_handshake(monkeypatch, (cert, b"d", "T", "C"))

    as_dict = dict(zip(CERT_FEATURE_NAMES, extract_cert_features("a.test")))
    assert as_dict["not_before_in_future"] == 1.0
    assert as_dict["cert_age_days"] == 0.0  # clipped, never negative


def test_validity_span_is_reported(monkeypatch):
    # days_since_issue=0 makes notBefore "now", so the span is exactly the
    # days_valid_for window rather than that window plus the issue offset.
    _script_handshake(monkeypatch, (_verified_cert_dict(days_valid_for=90, days_since_issue=0), b"d", "T", "C"))
    as_dict = dict(zip(CERT_FEATURE_NAMES, extract_cert_features("a.test")))
    assert 89.9 <= as_dict["validity_span_days"] <= 90.1


# ---------------------------------------------------------------------------
# Name matching
# ---------------------------------------------------------------------------
def test_wildcard_san_matches_exactly_one_label():
    assert cf._name_matches_host("*.secure-login.example", "a.secure-login.example")
    assert not cf._name_matches_host("*.secure-login.example", "a.b.secure-login.example")
    assert not cf._name_matches_host("*.secure-login.example", "secure-login.example")


def test_hostname_mismatch_uses_san_and_wildcards(monkeypatch):
    cert = _verified_cert_dict(san=(("DNS", "*.secure-login.example"),))
    _script_handshake(monkeypatch, (cert, b"d", "TLSv1.3", "C"))

    good = dict(zip(CERT_FEATURE_NAMES, extract_cert_features("a.secure-login.example")))
    bad = dict(zip(CERT_FEATURE_NAMES, extract_cert_features("a.b.secure-login.example")))

    assert good["san_matches_host"] == 1.0
    assert good["hostname_mismatch"] == 0.0
    assert good["has_wildcard"] == 1.0
    assert bad["san_matches_host"] == 0.0
    assert bad["hostname_mismatch"] == 1.0


def test_mismatch_is_not_claimed_when_the_certificate_names_nothing(monkeypatch):
    """With no SAN and no CN there is nothing to compare, so the honest answer
    is "unevaluable" (0.0), not "mismatch"."""
    cert = _verified_cert_dict(subject_cn="", san=())
    cert["subject"] = ()
    _script_handshake(monkeypatch, (cert, b"d", "T", "C"))

    as_dict = dict(zip(CERT_FEATURE_NAMES, extract_cert_features("paypal-secure.tld")))
    assert as_dict["hostname_mismatch"] == 0.0
    assert as_dict["san_matches_host"] == 0.0


def test_ip_literal_host_matches_an_ip_san(monkeypatch):
    cert = _verified_cert_dict(san=(("IP Address", "203.0.113.7"),), subject_cn="")
    cert["subject"] = ()
    _script_handshake(monkeypatch, (cert, b"d", "T", "C"))

    hit = dict(zip(CERT_FEATURE_NAMES, extract_cert_features("203.0.113.7")))
    miss = dict(zip(CERT_FEATURE_NAMES, extract_cert_features("203.0.113.8")))

    assert hit["url_host_is_ip_literal"] == 1.0
    assert hit["san_matches_host"] == 1.0
    assert hit["san_has_ip_literal"] == 1.0
    assert hit["hostname_mismatch"] == 0.0
    assert miss["san_matches_host"] == 0.0
    assert miss["hostname_mismatch"] == 1.0


# ---------------------------------------------------------------------------
# Issuer heuristics and serial
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "issuer,expected",
    [
        ("C=US, O=Let's Encrypt, CN=R3", True),
        ("CN=DigiCert Global Root G2", True),
        ("CN=Bob", False),
        ("CN=America Location Scan Ltd", False),  # a bare "ca" must not fire
        ("O=Localisation Bureau, CN=Holdings", False),
    ],
)
def test_known_ca_heuristic(issuer, expected):
    assert bool(cf._CA_LOOKING_RE.search(issuer)) is expected


def test_known_ca_looking_issuer_sets_the_feature(monkeypatch):
    _script_handshake(monkeypatch, (_verified_cert_dict(issuer_cn="DigiCert Global Root G2"), b"d", "T", "C"))
    assert dict(zip(CERT_FEATURE_NAMES, extract_cert_features("a.test")))["issuer_cn_is_known_ca_looking"] == 1.0


def test_serial_entropy_is_bounded_by_the_byte_width(monkeypatch):
    """A serial of one repeated byte carries no information; a random one does."""
    _script_handshake(monkeypatch, (_verified_cert_dict(serial="ffffffffffffffffffffffffffffffff"), b"d", "T", "C"))
    flat = dict(zip(CERT_FEATURE_NAMES, extract_cert_features("a.test")))["serial_entropy_bits"]
    assert flat == 0.0

    _script_handshake(monkeypatch, (_verified_cert_dict(serial="0a1b2c3d4e5f60718293a4b5c6d7e8f9"), b"d", "T", "C"))
    rich = dict(zip(CERT_FEATURE_NAMES, extract_cert_features("a.test")))["serial_entropy_bits"]
    assert 0.0 < rich <= 8.0


# ---------------------------------------------------------------------------
# URL-derived features
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "url,host,port,scheme",
    [
        ("https://paypal.com/login", "paypal.com", 443, "https"),
        ("https://paypal.com:8443/x?a=1", "paypal.com", 8443, "https"),
        ("http://paypal.com/login", "paypal.com", 80, "http"),
        ("paypal.com/login", "paypal.com", 443, "https"),
        ("", "", 443, "https"),
    ],
)
def test_cert_context_resolves_host_port_and_scheme(url, host, port, scheme):
    ctx = cert_context_for(url)
    assert (ctx.host, ctx.port, ctx.scheme) == (host, port, scheme)


@pytest.mark.parametrize(
    "host,expected",
    [
        ("192.0.2.1", True),
        ("[2001:db8::1]", True),
        ("2001:db8::1", True),
        ("paypal.com", False),
        ("deadbeef", False),
        ("", False),
    ],
)
def test_ip_literal_detection(host, expected):
    assert is_ip_literal(host) is expected


def test_http_scheme_is_only_set_when_the_url_really_used_http(monkeypatch):
    _script_handshake(monkeypatch, (_verified_cert_dict(), b"d", "T", "C"))
    https = dict(zip(CERT_FEATURE_NAMES, extract_cert_features("a.test")))
    http = dict(zip(CERT_FEATURE_NAMES, extract_cert_features("a.test", scheme="http")))
    via_url = dict(zip(CERT_FEATURE_NAMES, extract_cert_features_for_url("http://a.test")))

    assert https["url_uses_http_scheme"] == 0.0
    assert http["url_uses_http_scheme"] == 1.0
    assert via_url["url_uses_http_scheme"] == 1.0


# ---------------------------------------------------------------------------
# The DER fallback parser, on real bytes
# ---------------------------------------------------------------------------
def test_der_parser_recovers_names_and_serial():
    parsed = cf._parse_certificate_der(_REAL_SAN_CERT_DER)
    assert parsed["subject"][0][0] == ("commonName", "Login Secure Bank")
    assert parsed["subject"][1][0] == ("organizationName", "Acme")
    assert parsed["issuer"] == parsed["subject"]
    assert parsed["version"] == 3  # X.509 v3
    assert len(parsed["serialNumber"]) == 40
    assert isinstance(parsed["notBefore"], datetime)


def test_der_parser_recovers_subject_alt_names():
    parsed = cf._parse_certificate_der(_REAL_SAN_CERT_DER)
    assert parsed["subjectAltName"] == (
        ("DNS", "*.secure-login.example"),
        ("DNS", "secure-login.example"),
        ("IP Address", "10.1.2.3"),
    )


def test_der_parser_never_raises_on_garbage():
    for junk in (b"", b"\x30", b"\x30\x82\xff\xff" + b"x" * 8, bytes(range(64))):
        assert isinstance(cf._parse_certificate_der(junk), dict)


# ---------------------------------------------------------------------------
# Handshake wrapper, with a faked socket layer
# ---------------------------------------------------------------------------
class _FakeRawSocket:
    """Stands in for the TCP socket; ``_tls_handshake`` uses it as a CM."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeTLSSocket:
    def __init__(self, cert, der):
        self._cert, self._der = cert, der

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def getpeercert(self, binary_form=False):
        return self._der if binary_form else self._cert

    def version(self):
        return "TLSv1.2"

    def cipher(self):
        return ("ECDHE-RSA-AES256-GCM-SHA384", "TLSv1.2", 256)


class _FakeContext:
    def __init__(self, cert, der):
        self.cert, self.der, self.server_hostname = cert, der, None

    def wrap_socket(self, raw, server_hostname=None):
        self.server_hostname = server_hostname
        return _FakeTLSSocket(self.cert, self.der)


@pytest.mark.parametrize("verify", [True, False])
def test_handshake_wrapper_reads_cert_and_sends_sni(monkeypatch, verify):
    cert = _verified_cert_dict() if verify else {}
    ctx = _FakeContext(cert, _REAL_SAN_CERT_DER)
    seen = {}

    def fake_create_connection(address, timeout=None):
        seen["address"] = address
        seen["timeout"] = timeout
        return _FakeRawSocket()

    monkeypatch.setattr(cf.socket, "create_connection", fake_create_connection)
    cert_dict, der, version, cipher = cf._tls_handshake(
        "a.test", 443, timeout=1.5, context=ctx
    )

    assert seen["address"] == ("a.test", 443)
    assert seen["timeout"] == 1.5
    assert ctx.server_hostname == "a.test"  # SNI is required for vhosts
    assert der == _REAL_SAN_CERT_DER
    assert version == "TLSv1.2"
    # An unverified context makes getpeercert() an empty dict; that empty dict
    # is exactly what tells the caller to read the DER instead.
    assert (cert_dict is not None) is verify


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def test_describe_certificate_exposes_the_documented_keys(monkeypatch):
    _script_handshake(monkeypatch, (_verified_cert_dict(), b"d", "TLSv1.3", "C"))
    described = describe_certificate("secure-login.example")

    for key in (
        "issuer_cn", "subject_cn", "not_before", "not_after",
        "days_to_expiry", "san_count", "self_signed", "hostname_match",
    ):
        assert key in described
    assert described["subject_cn"] == "secure-login.example"
    assert described["issuer_cn"] == "Example Root CA"
    assert described["hostname_match"] is True
    assert described["self_signed"] is False
    assert 360 <= described["days_to_expiry"] <= 366


def test_describe_certificate_uses_none_for_unknowns(monkeypatch):
    """``None``, not ``0``, so a frontend cannot render "expires today"."""
    _script_handshake(monkeypatch, ConnectionRefusedError(10061, "Connection refused"))
    described = describe_certificate("dead.test")

    assert described["available"] is False
    assert described["reason"] == "connection refused"
    assert described["days_to_expiry"] is None
    assert described["subject_cn"] is None


def test_batch_extraction_preserves_input_order(monkeypatch):
    _script_handshake(monkeypatch, (_verified_cert_dict(), b"d", "T", "C"))
    targets = [CertTarget("a.test"), "b.test", CertTarget("c.test", port=8443)]
    rows = extract_cert_features_batch(targets, timeout=1.0)

    assert len(rows) == 3
    assert all(len(r) == N_CERT_FEATURES for r in rows)
    # Compared on a clock-independent column: ``days_to_expiry`` legitimately
    # drifts by microseconds between two extractions, which is not an ordering
    # bug and should not be papered over with a tolerance here.
    col = CERT_FEATURE_NAMES.index("subject_cn_len")
    assert [r[col] for r in rows] == [20.0, 20.0, 20.0]  # len("secure-login.example")


def test_probe_certificate_accepts_bare_hosts_and_targets(monkeypatch):
    _script_handshake(monkeypatch, (_verified_cert_dict(), b"d", "T", "C"))
    assert probe_certificate("A.TEST").host == "a.test"
    assert probe_certificate(CertTarget(host="a.test", port=8443)).port == 8443


def test_module_exports_its_public_surface():
    for name in cf.__all__:
        assert hasattr(cf, name), name

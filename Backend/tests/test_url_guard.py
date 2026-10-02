"""Tests for the SSRF guard.

This is the security boundary for a service with no authentication, so the
blocklist is tested adversarially: every technique below is a real way to reach
a private address from a URL that *looks* external.
"""

from __future__ import annotations

import ipaddress

import pytest

from app.security.url_guard import (
    BlockedTarget,
    is_public_address,
    validate_target,
)


def resolver_for(*addrs: str):
    return lambda host, port: list(addrs)


PUBLIC = resolver_for("93.184.216.34")

# ---------------------------------------------------------------------------
# Things that must be refused
# ---------------------------------------------------------------------------
def test_localhost_is_refused_even_if_it_resolves_publicly():
    """A resolver override must not be able to wave an internal name through."""
    with pytest.raises(BlockedTarget, match="metadata network|not publicly routable"):
        validate_target("http://localhost/", resolver=PUBLIC)


def test_cloud_metadata_hostname_is_refused_even_if_it_resolves_publicly():
    with pytest.raises(BlockedTarget, match="metadata network|not publicly routable"):
        validate_target("http://metadata.google.internal/", resolver=PUBLIC)


def test_private_literals_are_refused_with_the_real_resolver():
    """No resolver override: these must be blocked on their own merits."""
    for url in ("http://127.0.0.1/", "http://[::1]/", "http://169.254.169.254/"):
        with pytest.raises(BlockedTarget):
            validate_target(url)


def test_localhost_is_refused_without_any_resolver_override():
    """Blocked by name before DNS is even consulted."""
    with pytest.raises(BlockedTarget):
        validate_target("http://localhost/")


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/admin",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://0.0.0.0/",
    ],
)
def test_literal_private_addresses_are_refused(url):
    """A literal private address is blocked before any DNS lookup happens."""
    with pytest.raises(BlockedTarget):
        validate_target(url)


def test_hostname_resolving_to_a_private_address_is_refused():
    """A public-looking name that resolves inward is the classic DNS bypass."""
    with pytest.raises(BlockedTarget, match="not publicly routable"):
        validate_target(
            "http://evil.example/", resolver=resolver_for("127.0.0.1")
        )


def test_a_single_private_answer_among_public_ones_is_enough_to_refuse():
    with pytest.raises(BlockedTarget):
        validate_target(
            "http://mixed.example/",
            resolver=resolver_for("93.184.216.34", "10.1.2.3"),
        )


@pytest.mark.parametrize(
    "url,reason",
    [
        ("ftp://example.com/x", "scheme"),
        ("file:///etc/passwd", "scheme"),
        ("gopher://example.com/", "scheme"),
        ("javascript:alert(1)", "scheme"),
        ("data:text/html,<h1>x", "scheme"),
    ],
)
def test_non_http_schemes_are_refused(url, reason):
    with pytest.raises(BlockedTarget, match=reason):
        validate_target(url, resolver=PUBLIC)


def test_non_default_ports_are_refused():
    with pytest.raises(BlockedTarget, match="port"):
        validate_target("http://example.com:8080/", resolver=PUBLIC)


def test_ssh_port_is_refused():
    with pytest.raises(BlockedTarget, match="port"):
        validate_target("http://example.com:22/", resolver=PUBLIC)


def test_embedded_credentials_are_refused():
    """Credentials both mark phishing and disguise the real host."""
    with pytest.raises(BlockedTarget, match="credentials"):
        validate_target("http://paypal.com@127.0.0.1/", resolver=PUBLIC)


def test_control_characters_are_refused():
    with pytest.raises(BlockedTarget, match="control characters"):
        validate_target("http://example.com/\r\nHost: evil", resolver=PUBLIC)


def test_null_byte_is_refused():
    with pytest.raises(BlockedTarget, match="control characters"):
        validate_target("http://example.com/\x00", resolver=PUBLIC)


def test_empty_url_is_refused():
    with pytest.raises(BlockedTarget, match="empty"):
        validate_target("   ", resolver=PUBLIC)


def test_overlong_url_is_refused():
    with pytest.raises(BlockedTarget, match="exceeds"):
        validate_target("http://example.com/" + "a" * 5000, resolver=PUBLIC)


def test_missing_host_is_refused():
    with pytest.raises(BlockedTarget):
        validate_target("http://", resolver=PUBLIC)


def test_dns_failure_is_reported_not_crashed():
    with pytest.raises(BlockedTarget, match="DNS resolution failed"):
        validate_target("http://does-not-exist.invalid/")


def test_non_string_input_is_refused():
    with pytest.raises(BlockedTarget, match="must be a string"):
        validate_target(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Things that must be allowed
# ---------------------------------------------------------------------------
def test_ordinary_public_https_url_is_allowed():
    t = validate_target("https://example.com/path?q=1", resolver=PUBLIC)
    assert t.scheme == "https"
    assert t.hostname == "example.com"
    assert t.port == 443
    assert t.ip == "93.184.216.34"


def test_http_default_port_is_80():
    assert validate_target("http://example.com/", resolver=PUBLIC).port == 80


def test_explicit_default_port_is_allowed():
    t = validate_target("https://example.com:443/", resolver=PUBLIC)
    assert t.port == 443


def test_trailing_dot_in_hostname_is_stripped():
    t = validate_target("http://example.com./", resolver=PUBLIC)
    assert t.hostname == "example.com"


def test_uppercase_scheme_and_host_are_normalised():
    t = validate_target("HTTP://EXAMPLE.COM/Path", resolver=PUBLIC)
    assert t.scheme == "http"
    assert t.hostname == "example.com"
    assert t.url.startswith("http://example.com/")


def test_fragment_is_stripped_from_the_request_url():
    t = validate_target("http://example.com/a#frag", resolver=PUBLIC)
    assert "#frag" not in t.url


def test_public_literal_ip_is_allowed():
    t = validate_target("https://93.184.216.34/", resolver=PUBLIC)
    assert t.ip == "93.184.216.34"


def test_ipv6_literal_keeps_its_brackets_in_the_url():
    t = validate_target("http://[2001:4860:4860::8888]/", resolver=PUBLIC)
    assert "[" in t.url and "]" in t.url


def test_netloc_omits_the_default_port():
    assert validate_target("https://example.com/", resolver=PUBLIC).netloc == "example.com"


def test_netloc_includes_a_non_default_port():
    t = validate_target("http://example.com:8080/", allowed_ports=(8080,), resolver=PUBLIC)
    assert t.netloc == "example.com:8080"


# ---------------------------------------------------------------------------
# Address classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "addr",
    ["127.0.0.1", "10.0.0.1", "192.168.0.1", "172.20.0.1", "169.254.1.1",
     "0.0.0.0", "224.0.0.1", "100.64.0.1", "::1", "fe80::1", "fc00::1"],
)
def test_non_public_addresses_are_classified_correctly(addr):
    assert not is_public_address(ipaddress.ip_address(addr))


@pytest.mark.parametrize("addr", ["93.184.216.34", "8.8.8.8", "2001:4860:4860::8888"])
def test_public_addresses_are_classified_correctly(addr):
    assert is_public_address(ipaddress.ip_address(addr))


# ---------------------------------------------------------------------------
# Unresolvable hosts
#
# A dead domain is not an SSRF risk -- there is no address to connect to -- and
# it is a high-signal phishing case, because takedowns and short-lived
# campaigns are exactly what a detector should still score from the string.
# These tests pin both halves: the relaxed path, and that SSRF blocking is
# completely unchanged.
# ---------------------------------------------------------------------------
def _nxdomain(host: str, port: int):
    raise BlockedTarget(f"DNS resolution failed for {host!r}: [Errno 11001] getaddrinfo failed")


def test_unresolvable_host_raises_by_default():
    """The default stays strict for any caller that wants a hard refusal."""
    with pytest.raises(BlockedTarget, match="DNS resolution failed"):
        validate_target("http://dead-domain.invalid/", resolver=_nxdomain)


def test_unresolvable_host_is_reported_when_allowed():
    t = validate_target(
        "http://dead-domain.invalid/signin?session=9f2",
        resolver=_nxdomain,
        allow_unresolved=True,
    )
    assert t.resolved is False
    assert t.ip == ""
    assert t.dns_error and "does not resolve" in t.dns_error


def test_unresolvable_host_keeps_the_path_and_query():
    t = validate_target(
        "http://dead.invalid/signin?session=9f2", resolver=_nxdomain, allow_unresolved=True
    )
    assert t.url == "http://dead.invalid/signin?session=9f2"
    assert t.hostname == "dead.invalid"
    assert t.port == 80


def test_resolved_host_is_unaffected_by_the_new_flag():
    t = validate_target("https://example.com/", resolver=PUBLIC, allow_unresolved=True)
    assert t.resolved is True
    assert t.ip
    assert t.dns_error is None


@pytest.mark.parametrize(
    "resolver",
    [
        lambda host, port: ["127.0.0.1"],
        lambda host, port: ["10.0.0.5"],
        lambda host, port: ["169.254.169.254"],
    ],
)
def test_private_resolution_is_still_blocked_when_unresolved_is_allowed(resolver):
    """The relaxation must not weaken SSRF protection in any way.

    allow_unresolved only covers the NXDOMAIN path. A name that resolves into a
    private address is the actual attack and stays a hard refusal.
    """
    with pytest.raises(BlockedTarget, match="not publicly routable"):
        validate_target(
            "http://sneaky.example/", resolver=resolver, allow_unresolved=True
        )


def test_empty_resolver_result_still_raises_even_when_unresolved_allowed():
    """No addresses at all is a resolver failure, not NXDOMAIN, so it blocks."""
    with pytest.raises(BlockedTarget, match="no addresses resolved"):
        validate_target(
            "http://empty.example/", resolver=lambda h, p: [], allow_unresolved=True
        )


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:8000/", "http://169.254.169.254/latest/meta-data/"]
)
def test_literal_private_addresses_remain_blocked(url):
    """An IP literal never goes through DNS, so the flag must not apply."""
    with pytest.raises(BlockedTarget):
        validate_target(url, resolver=_nxdomain, allow_unresolved=True)


def test_scheme_and_port_checks_still_apply_to_unresolvable_hosts():
    """Relaxing DNS must not become a way to smuggle a bad scheme or port."""
    with pytest.raises(BlockedTarget, match="scheme"):
        validate_target("ftp://dead.invalid/", resolver=_nxdomain, allow_unresolved=True)
    with pytest.raises(BlockedTarget, match="port"):
        validate_target("http://dead.invalid:8080/", resolver=_nxdomain, allow_unresolved=True)

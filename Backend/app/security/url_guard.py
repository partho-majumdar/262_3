"""SSRF / open-redirect guards for any user-supplied URL.

Every outbound request in this project goes through :func:`validate_target`
first. The threat is a detector that fetches whatever it is handed: an attacker
submits ``http://127.0.0.1:8080/admin`` or ``http://169.254.169.254/`` and the
server dutifully retrieves it, turning the detector into a proxy for the private
network it sits in.

Design notes
------------
* Hostnames are **resolved** here, and the resolved address is returned, so the
  caller can pin the connection to the IP that was actually checked. Validating
  the hostname string alone is not enough (DNS rebinding).
* Blocking is by *address property*, not by a blocklist of names, so an attacker
  cannot bypass it with an unfamiliar hostname that happens to resolve inward.
* Everything is deny-by-default: an address that cannot be classified is
  rejected.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

__all__ = [
    "BlockedTarget",
    "ValidatedTarget",
    "validate_target",
    "is_public_address",
    "BLOCKED_HOSTNAMES",
]


class BlockedTarget(ValueError):
    """Raised when a URL may not be requested.

    Subclasses :class:`ValueError` so a plain validation handler can catch it.
    """


#: Hostnames that denote the local or cloud-internal network. Refused by name
#: even when they currently resolve to a public address, because the whole point
#: of these names is "reach something inside", and a resolver override or a
#: hosts-file entry should not be able to wave them through.
BLOCKED_HOSTNAMES: frozenset[str] = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata",
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
        "instance-data.ec2.internal",
    }
)


@dataclass(frozen=True)
class ValidatedTarget:
    """A URL that passed validation, with the address it resolved to."""

    url: str
    scheme: str
    hostname: str
    port: int
    ip: str

    #: Empty when the hostname did not resolve. That is not an SSRF risk --
    #: there is no address to connect to -- so the caller may still analyse the
    #: URL string, it just cannot acquire the page. ``ip`` is "" in that case.
    resolved: bool = True
    dns_error: str | None = None

    @property
    def netloc(self) -> str:
        """Hostname (and port when non-default) for a ``Host`` header."""
        default = 443 if self.scheme == "https" else 80
        return self.hostname if self.port == default else f"{self.hostname}:{self.port}"


def is_public_address(ip: ipaddress._BaseAddress) -> bool:
    """True only for globally routable unicast addresses.

    Anything inside a private, loopback, link-local, multicast, reserved, or
    otherwise special-purpose range is refused, as is an address that is
    unspecified or a broadcast address.
    """
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False
    # ``is_global`` is the single authoritative answer for IPv4/IPv6 and already
    # excludes the ranges above; kept as an explicit belt-and-braces check.
    return bool(getattr(ip, "is_global", False))


def _resolve(hostname: str, port: int) -> list[ipaddress._BaseAddress]:
    """Resolve ``hostname`` to every address it answers with."""
    try:
        infos = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise BlockedTarget(f"DNS resolution failed for {hostname!r}: {exc}") from exc

    addresses: list[ipaddress._BaseAddress] = []
    for family, _type, _proto, _canon, sockaddr in infos:
        try:
            addresses.append(ipaddress.ip_address(sockaddr[0]))
        except ValueError:  # pragma: no cover - defensive
            continue
    if not addresses:
        raise BlockedTarget(f"no addresses resolved for {hostname!r}")
    return addresses


def validate_target(
    url: str,
    *,
    allowed_schemes: tuple[str, ...] = ("http", "https"),
    allowed_ports: tuple[int, ...] = (80, 443),
    max_length: int = 2048,
    resolver: object | None = None,
    allow_unresolved: bool = False,
) -> ValidatedTarget:
    """Validate a user-supplied URL for safe fetching.

    Parameters
    ----------
    url:
        The candidate URL. Normalised before validation.
    allowed_schemes / allowed_ports:
        Permit lists. Ports other than these are refused outright rather than
        being probed, which keeps an attacker's port scan out of reach.
    max_length:
        Upper bound on the normalised URL.
    resolver:
        Optional ``(hostname, port) -> list[str]`` override returning IP
        strings. Exists so tests can exercise blocking without touching DNS.
    allow_unresolved:
        When True, a hostname that does not resolve is reported as
        ``resolved=False`` instead of raising.

        Why this is safe and why it matters: this guard exists to stop the
        service being used to reach private or link-local addresses. A name
        with no address cannot be connected to at all, so a resolution failure
        is not an SSRF risk -- it is a dead domain. And a dead domain is a
        *high-signal* phishing case: takedowns and short-lived campaigns are
        exactly what a detector should still be able to score from the URL
        string alone. Refusing outright would make the detector blind to the
        URLs that are already gone.

        Resolution that *succeeds* into a private address is still a hard
        block; only the NXDOMAIN path is relaxed.

    Raises
    ------
    BlockedTarget
        With a reason suitable for showing to the caller.
    """
    if not isinstance(url, str):
        raise BlockedTarget("URL must be a string")

    candidate = url.strip()
    if not candidate:
        raise BlockedTarget("URL is empty")
    if len(candidate) > max_length:
        raise BlockedTarget(f"URL exceeds {max_length} characters")
    # Control characters can smuggle a second request line or hide a host.
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in candidate):
        raise BlockedTarget("URL contains control characters")

    parts = urlsplit(candidate)
    scheme = parts.scheme.lower()
    if not scheme:
        raise BlockedTarget("URL has no scheme")
    if scheme not in allowed_schemes:
        raise BlockedTarget(
            f"scheme {scheme!r} is not allowed (permitted: {', '.join(allowed_schemes)})"
        )

    # Credentials in a submitted URL are a phishing pattern *and* a way to
    # disguise the real host, so they are refused rather than stripped.
    if parts.username or parts.password:
        raise BlockedTarget("URL must not embed credentials")

    hostname = (parts.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        raise BlockedTarget("URL has no host")

    try:
        port = parts.port
    except ValueError as exc:
        raise BlockedTarget(f"invalid port: {exc}") from exc
    if port is None:
        port = 443 if scheme == "https" else 80
    if port not in allowed_ports:
        raise BlockedTarget(
            f"port {port} is not allowed (permitted: {', '.join(str(p) for p in allowed_ports)})"
        )

    # A bare IPv6 literal arrives from urlsplit without brackets in .hostname.
    literal = hostname.strip("[]")

    if not literal and hostname in BLOCKED_HOSTNAMES:
        raise BlockedTarget(f"host {hostname!r} denotes the local or cloud metadata network")
    if literal in BLOCKED_HOSTNAMES:
        raise BlockedTarget(f"host {literal!r} denotes the local or cloud metadata network")

    try:
        direct = ipaddress.ip_address(literal)
    except ValueError:
        direct = None

    resolved = True
    dns_error: str | None = None

    if direct is not None:
        if not is_public_address(direct):
            raise BlockedTarget(
                f"{direct} is not a publicly routable address "
                "(private, loopback, link-local, or reserved)"
            )
        addresses = [direct]
    else:
        try:
            if resolver is not None:
                addresses = [
                    ipaddress.ip_address(a) for a in resolver(hostname, port)  # type: ignore[operator]
                ]
            else:
                addresses = _resolve(hostname, port)
        except BlockedTarget:
            if not allow_unresolved:
                raise
            # No address exists, so there is nothing to connect to and nothing
            # to protect against. Fall through with an empty address list.
            resolved = False
            addresses = []
            dns_error = f"host {hostname!r} does not resolve"
        if not addresses and resolved:
            raise BlockedTarget(f"no addresses resolved for {hostname!r}")

    for addr in addresses:
        if not is_public_address(addr):
            raise BlockedTarget(
                f"{hostname} resolves to {addr}, which is not publicly routable "
                "(private, loopback, link-local, or reserved)"
            )

    # Keep the bracket form for IPv6 literals so the reconstructed URL is valid.
    netloc_host = f"[{literal}]" if ":" in literal else literal
    default_port = 443 if scheme == "https" else 80
    netloc = netloc_host if port == default_port else f"{netloc_host}:{port}"

    normalised = urlunsplit(
        (scheme, netloc, parts.path or "/", parts.query, "")
    )

    return ValidatedTarget(
        url=normalised,
        scheme=scheme,
        hostname=literal,
        port=port,
        ip=str(addresses[0]) if addresses else "",
        resolved=resolved,
        dns_error=dns_error,
    )

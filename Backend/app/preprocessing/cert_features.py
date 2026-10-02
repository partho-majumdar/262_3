"""TLS certificate features for the certificate modality.

Why this module exists
----------------------
The two existing handcrafted branches (``url_features`` and ``html_features``)
are pure functions of the URL string and the fetched page respectively. Neither
can see the *transport*: whether the host even speaks TLS, whether the
certificate chains to a public CA, whether the name on the certificate is the
name we asked for, how freshly it was minted. For phishing specifically those
are unusually informative, because cloning a convincing login page does not
require obtaining a valid certificate.

Constraints that shaped the design
----------------------------------
* **Standard library only.** The project ships no ``cryptography`` dependency,
  and adding one is not on the table. That is not as limiting as it sounds:
  :meth:`ssl.SSLSocket.getpeercert` hands back an OpenSSL-parsed dict
  (validity dates, issuer/subject RDN sequences, SAN list, serial) for free.
* **Total function.** A phishing URL is, by definition, likely to be dead,
  self-signed, expired, or plain HTTP. :func:`extract_cert_features` therefore
  never raises and never returns a short vector: a failure yields zeros of
  exactly :data:`N_CERT_FEATURES` width, so the fusion layer can mask the
  modality instead of crashing or silently shifting feature alignment.
* **A bad certificate is a feature, not an error.** Phishing hosts routinely
  present self-signed, expired, or wrong-name certificates. The module first
  attempts a *verifying* handshake, and on a certificate failure falls back to
  ``ssl._create_unverified_context()`` purely so the certificate can still be
  *observed*. ``verify_failed`` / ``self_signed_guess`` then carry that
  observation to the model.

The :mod:`ssl` getpeercert() gap
--------------------------------
With ``verify_mode == CERT_NONE`` OpenSSL has no reason to parse the peer
certificate, and CPython's :meth:`SSLSocket.getpeercert` consequently returns
an **empty dict** -- verified on CPython 3.10/OpenSSL 1.1.1. Since the
unverified fallback is exactly the path most phishing sites take, a dict-only
implementation would return all-zero vectors for the cases this module exists to
study. So the module carries a small, defensive DER walker
(:func:`_parse_certificate_der`) that recovers the same field shapes directly
from ``getpeercert(binary_form=True)``. Every value it produces is a real
observation from the wire bytes; nothing is defaulted or guessed. The
``cert_from_der_parse`` feature records which of the two sources was used so a
consumer can audit provenance.

Deliberate omissions
--------------------
* ``issuer_is_self`` is not a feature. It would be bit-for-bit identical to
  ``issuer_equals_subject`` (both are "issuer DN == subject DN"); shipping both
  would hand the model a duplicate column and flatter its importance.
  ``self_signed_guess`` is the stricter version, since it additionally requires
  that verification actually failed.
* No signature-algorithm, key-size, or SHA-256-fingerprint feature. These are
  obtainable from the DER, but they are *certificate-hardening* signals, they
  are near-constant across the modern web, and half of them are not derivable
  from the fields this module already carries. They belong to a future
  iteration with a real X.509 parser, not to a hand-rolled one.
"""

from __future__ import annotations

import math
import re
import socket
import ssl
import struct
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit

__all__ = [
    "CERT_FEATURE_NAMES",
    "N_CERT_FEATURES",
    "CertTarget",
    "CertProbe",
    "cert_context_for",
    "is_ip_literal",
    "probe_certificate",
    "extract_cert_features",
    "extract_cert_features_for_url",
    "extract_cert_features_batch",
    "describe_certificate",
    "ssl_error_reason",
]

#: Order is load-bearing: the scaler is fitted against this exact sequence, so a
#: reordering is a checkpoint-breaking change rather than a cosmetic edit.
CERT_FEATURE_NAMES: list[str] = [
    "cert_available",
    "cert_verified",
    "verify_failed",
    "cert_from_der_parse",
    "self_signed_guess",
    "issuer_equals_subject",
    "days_to_expiry",
    "is_expired",
    "not_before_in_future",
    "cert_age_days",
    "validity_span_days",
    "subject_cn_len",
    "san_count",
    "san_matches_host",
    "hostname_mismatch",
    "has_wildcard",
    "san_has_ip_literal",
    "issuer_cn_len",
    "issuer_cn_is_known_ca_looking",
    "serial_entropy_bits",
    "url_host_is_ip_literal",
    "url_uses_http_scheme",
    "tls_version_id",
]

N_CERT_FEATURES = len(CERT_FEATURE_NAMES)

#: Name -> index, so the feature body below reads by name while the vector it
#: fills stays in the declared order. Tests assert specific names land on
#: specific indices, so a rename cannot silently repoint a column.
_FEATURE_INDEX = {name: i for i, name in enumerate(CERT_FEATURE_NAMES)}

#: ``days_to_expiry`` is clipped symmetrically-ish so that a certificate issued
#: for 30 years (some IoT fleets) cannot dominate a linear model, while an
#: already-expired certificate keeps a usable negative signal.
_EXPIRY_CLIP = (-365.0, 3650.0)
_AGE_CLIP = (0.0, 3650.0)
_SPAN_CLIP = (0.0, 36500.0)

_IPV4_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

#: Word-boundary matched, because a bare ``ca`` substring would fire on
#: "America", "location" and "scan" and make the heuristic useless.
_CA_LOOKING_RE = re.compile(
    r"\b(?:"
    r"ca|certificate authority|root ca|root|trust|anchors?|"
    r"certification|authority|authorities|issuer|signing|"
    r"digicert|comodo|sectigo|entrust|geotrust|globalsign|verisign|"
    r"thawte|let'?s encrypt|letsencrypt|zerossl|buypass|actalis|"
    r"usertrust|starfield|rapidssl|amazon|google|apple|microsoft|"
    r"microsoft|cloudflare|ssl|certificate|security"
    r")\b",
    re.IGNORECASE,
)

#: DN attribute short names, so the DER walker can emit the same tuple shape
#: that ``getpeercert()`` produces for the verified path.
_DN_OID_NAMES = {
    "2.5.4.3": "commonName",
    "2.5.4.4": "surname",
    "2.5.4.5": "serialNumber",
    "2.5.4.6": "countryName",
    "2.5.4.7": "localityName",
    "2.5.4.8": "stateOrProvinceName",
    "2.5.4.9": "streetAddress",
    "2.5.4.10": "organizationName",
    "2.5.4.11": "organizationalUnitName",
    "2.5.4.12": "title",
    "2.5.4.15": "businessCategory",
    "2.5.4.17": "postalCode",
    "2.5.4.42": "givenName",
    "0.9.2342.19200300.100.1.25": "domainComponent",
    "1.2.840.113549.1.9.1": "emailAddress",
    "1.3.6.1.4.1.311.60.2.1.1": "jurisdictionL",
    "1.3.6.1.4.1.311.60.2.1.2": "jurisdictionST",
    "1.3.6.1.4.1.311.60.2.1.3": "jurisdictionC",
}

_DER_STRING_ENCODINGS = {
    0x0C: "utf-8",     # UTF8String
    0x12: "ascii",     # NumericString
    0x13: "ascii",     # PrintableString
    0x14: "latin-1",   # T61String
    0x16: "ascii",     # IA5String
    0x1A: "ascii",     # VisibleString
    0x1E: "utf-16-be",  # BMPString
    0x1C: "utf-32-be",  # UniversalString
}

_TLS_VERSION_IDS = {
    "SSLv3": 3.0,
    "TLSv1": 10.0,
    "TLSv1.1": 11.0,
    "TLSv1.2": 12.0,
    "TLSv1.3": 13.0,
}

_SAN_DNS_OID = "2.5.29.17"


# ---------------------------------------------------------------------------
# Small value objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CertTarget:
    """Where to look for a certificate, and how the URL got there.

    ``scheme`` is carried alongside the host because "no certificate at all"
    and "no certificate because the URL was plain http" are different findings
    for a phishing classifier, and the host alone cannot tell them apart.
    """

    host: str
    port: int = 443
    scheme: str = "https"

    @property
    def is_ip_literal(self) -> bool:
        return is_ip_literal(self.host)


@dataclass
class CertProbe:
    """Everything one handshake attempt could observe about a certificate.

    Every attribute defaults to a value that means *not observed*, so a
    connection that never completed produces a probe whose every feature slot
    is honestly empty rather than conveniently plausible.
    """

    host: str = ""
    port: int = 443
    scheme: str = "https"
    available: bool = False
    verified: bool = False
    from_der: bool = False
    reason: str | None = None
    verify_error: str | None = None
    subject: tuple = ()
    issuer: tuple = ()
    not_before: datetime | None = None
    not_after: datetime | None = None
    serial_hex: str = ""
    san: tuple = ()
    tls_version: str = ""
    cipher: str = ""

    def error_dict(self) -> dict:
        """Machine-readable failure record for logs, the API layer, or XAI."""
        return {
            "host": self.host,
            "port": self.port,
            "scheme": self.scheme,
            "available": self.available,
            "verified": self.verified,
            "from_der": self.from_der,
            "reason": self.reason,
            "verify_error": self.verify_error,
            "tls_version": self.tls_version,
        }


# ---------------------------------------------------------------------------
# Host / URL resolution
# ---------------------------------------------------------------------------
def is_ip_literal(host: str) -> bool:
    """True for IPv4, IPv6 (bare or bracketed) and bare-hex IPv6 hostnames."""
    h = (host or "").strip().strip("[]").lower()
    if not h:
        return False
    if _IPV4_RE.match(h):
        return True
    if ":" in h:
        return True
    # A bare hex run with at least one digit group is an IPv6 address that
    # arrived without its colons; pure-letter hostnames such as "deadbeef" are
    # legal DNS labels, so require a digit somewhere to claim it is an address.
    if re.fullmatch(r"[0-9a-f:]+", h) and any(c.isdigit() for c in h):
        return True
    return False


def cert_context_for(url: str) -> CertTarget:
    """Resolve a URL (or a bare host) into a :class:`CertTarget`.

    Default ports follow the scheme so that an ``http://`` URL is probed on 80
    rather than silently demanding TLS on 443: the failure mode it produces
    ("plain http, nothing to inspect") is the finding, not a bug.
    """
    raw = (url or "").strip()
    scheme = "https"
    host = raw
    port = 443
    if raw:
        try:
            parts = urlsplit(raw if "://" in raw else f"//{raw}")
        except ValueError:
            host = raw.lower()
        else:
            scheme = (parts.scheme or "").lower() or "https"
            host = (parts.hostname or "").lower()
            if parts.port:
                port = int(parts.port)
            elif scheme == "http":
                # Probing http:// on 443 would manufacture a "handshake failed"
                # finding instead of the real one: the site has no certificate
                # to look at because it does not speak TLS.
                port = 80
    host = (host or "").strip().strip("[]")
    return CertTarget(host=host, port=port if _valid_port(port) else 443, scheme=scheme)


def _valid_port(port: Any) -> bool:
    return isinstance(port, int) and not isinstance(port, bool) and 0 < port <= 65535


# ---------------------------------------------------------------------------
# DER walking (only used when OpenSSL declines to parse the certificate)
# ---------------------------------------------------------------------------
def _read_tlv(buf: bytes, i: int) -> tuple[int, int, int]:
    """Return ``(tag, content_start, content_end)`` for the TLV at ``i``.

    Raises ``ValueError`` on anything malformed: the caller treats that as
    "this certificate is not parseable" and reports zeros.
    """
    if i + 2 > len(buf):
        raise ValueError("truncated DER")
    tag = buf[i]
    length = buf[i + 1]
    i += 2
    if length & 0x80:
        n = length & 0x7F
        if n == 0 or n > 4 or i + n > len(buf):
            raise ValueError("unsupported DER length")
        length = int.from_bytes(buf[i:i + n], "big")
        i += n
    end = i + length
    if end > len(buf):
        raise ValueError("DER length exceeds buffer")
    return tag, i, end


def _der_oid(buf: bytes, start: int, end: int) -> str:
    if start >= end:
        raise ValueError("empty OID")
    first = buf[start]
    parts = [str(first // 40), str(first % 40)]
    value = 0
    for b in buf[start + 1:end]:
        value = (value << 7) | (b & 0x7F)
        if not b & 0x80:
            parts.append(str(value))
            value = 0
    return ".".join(parts)


def _der_text(buf: bytes, tag: int, start: int, end: int) -> str:
    encoding = _DER_STRING_ENCODINGS.get(tag)
    if encoding is None:
        return buf[start:end].hex()
    return buf[start:end].decode(encoding, errors="replace")


def _der_name(buf: bytes, start: int, end: int) -> tuple:
    """Parse a ``Name`` into the nested-tuple shape ``getpeercert()`` uses."""
    rdns = []
    p = start
    while p < end:
        try:
            tag, s, e = _read_tlv(buf, p)
        except ValueError:
            break
        p = e
        if tag != 0x31:  # not a RelativeDistinguishedName SET
            continue
        attrs = []
        q = s
        while q < e:
            try:
                atag, a_s, a_e = _read_tlv(buf, q)
            except ValueError:
                break
            q = a_e
            if atag != 0x30:
                continue
            # AttributeTypeAndValue ::= SEQUENCE { type OID, value ANY }
            try:
                _, t_s, t_e = _read_tlv(buf, a_s)
                oid = _der_oid(buf, t_s, t_e)
                vtag, v_s, v_e = _read_tlv(buf, t_e)
            except ValueError:
                continue
            attrs.append(
                (_DN_OID_NAMES.get(oid, oid), _der_text(buf, vtag, v_s, v_e))
            )
        if attrs:
            rdns.append(tuple(attrs))
    return tuple(rdns)


def _der_time(buf: bytes, tag: int, start: int, end: int) -> datetime:
    """Decode UTCTime / GeneralizedTime into a naive UTC datetime."""
    raw = buf[start:end].decode("ascii", errors="replace").strip()
    if raw.endswith("Z"):
        raw = raw[:-1]
    if tag == 0x17:  # UTCTime: YYMMDDHHMMSS, 1950-2049 pivot
        if len(raw) < 10:
            raise ValueError("short UTCTime")
        yy = int(raw[0:2])
        year = 2000 + yy if yy < 50 else 1900 + yy
        rest = raw[2:]
    else:  # GeneralizedTime: YYYYMMDDHHMMSS
        if len(raw) < 12:
            raise ValueError("short GeneralizedTime")
        year = int(raw[0:4])
        rest = raw[4:]
    month, day = int(rest[0:2]), int(rest[2:4])
    hour, minute = int(rest[4:6]), int(rest[6:8])
    second = int(rest[8:10]) if len(rest) >= 10 else 0
    return datetime(year, month, day, hour, minute, second)


def _der_validity(buf: bytes, start: int, end: int) -> tuple[datetime, datetime]:
    tag, s, e = _read_tlv(buf, start)
    not_before = _der_time(buf, tag, s, e)
    tag, s, e = _read_tlv(buf, e)
    not_after = _der_time(buf, tag, s, e)
    return not_before, not_after


def _der_san(octets: bytes) -> tuple:
    """Parse a SubjectAltName payload into ``getpeercert``-shaped tuples.

    ``octets`` is the content of the extension's OCTET STRING, i.e. a
    GeneralNames SEQUENCE. Every offset in here is relative to that slice --
    mixing it up with offsets into the whole certificate is an easy and silent
    way to "parse" the wrong bytes.
    """
    entries: list[tuple[str, str]] = []
    _, start, end = _read_tlv(octets, 0)
    p = start
    while p < end:
        try:
            tag, s, e = _read_tlv(octets, p)
        except ValueError:
            break
        p = e
        kind = tag & 0x1F
        if kind == 2:  # dNSName, IA5String
            entries.append(("DNS", octets[s:e].decode("ascii", errors="replace")))
        elif kind == 7:  # iPAddress
            raw = octets[s:e]
            if len(raw) == 4:
                entries.append(("IP Address", socket.inet_ntop(socket.AF_INET, raw)))
            elif len(raw) == 16:
                entries.append(("IP Address", socket.inet_ntop(socket.AF_INET6, raw)))
        elif kind == 1:  # rfc822Name
            entries.append(("email", octets[s:e].decode("ascii", errors="replace")))
        elif kind == 6:  # uniformResourceIdentifier
            entries.append(("URI", octets[s:e].decode("ascii", errors="replace")))
    return tuple(entries)


def _parse_certificate_der(der: bytes) -> dict:
    """Recover cert fields straight from the DER bytes.

    The verified path already gets all of this from OpenSSL for free; this
    exists because the *unverified* path does not (``getpeercert()`` returns an
    empty dict when ``verify_mode`` is ``CERT_NONE``), and the unverified path
    is where the interesting phishing certificates live.
    """
    out: dict[str, Any] = {}
    if not der:
        return out
    try:
        _, c_start, c_end = _read_tlv(der, 0)
        tag, p, _ = _read_tlv(der, c_start)
        if tag != 0x30:
            return out
        i = p

        # version [0] EXPLICIT INTEGER, default v1
        tag, s, e = _read_tlv(der, i)
        if tag == 0xA0:
            _, v_s, v_e = _read_tlv(der, s)
            out["version"] = int.from_bytes(der[v_s:v_e], "big") + 1
            i = e
        else:
            out["version"] = 0

        # serialNumber INTEGER
        tag, s, e = _read_tlv(der, i)
        serial = der[s:e]
        stripped = serial.lstrip(b"\x00") or serial
        out["serialNumber"] = stripped.hex()
        i = e

        # signature AlgorithmIdentifier
        tag, s, e = _read_tlv(der, i)
        try:
            _, a_s, a_e = _read_tlv(der, s)
            out["signature_algorithm_oid"] = _der_oid(der, a_s, a_e)
        except ValueError:
            pass
        i = e

        tag, s, e = _read_tlv(der, i)
        out["issuer"] = _der_name(der, s, e)
        i = e

        tag, s, e = _read_tlv(der, i)
        not_before, not_after = _der_validity(der, s, e)
        out["notBefore"] = not_before
        out["notAfter"] = not_after
        i = e

        tag, s, e = _read_tlv(der, i)
        out["subject"] = _der_name(der, s, e)
        i = e

        # Skip subjectPublicKeyInfo, then look for extensions [3].
        tag, s, e = _read_tlv(der, i)
        i = e
        while i < c_end:
            try:
                tag, s, e = _read_tlv(der, i)
            except ValueError:
                break
            if tag == 0xA3:
                # [3] EXPLICIT Extensions. ``x_s`` is where the Extensions
                # SEQUENCE *starts*, so iteration starts there too; walking
                # from its content offset would land on the first extension's
                # OID and silently find nothing.
                _, x_s, x_e = _read_tlv(der, s)
                p = x_s
                while p < x_e:
                    try:
                        etag, e_s, e_e = _read_tlv(der, p)
                    except ValueError:
                        break
                    p = e_e
                    if etag != 0x30:
                        continue
                    _, o_s, o_e = _read_tlv(der, e_s)
                    if _der_oid(der, o_s, o_e) != _SAN_DNS_OID:
                        continue
                    # Extension ::= SEQUENCE { extnID, critical BOOL DEFAULT
                    # FALSE, extnValue OCTET STRING }; the BOOLEAN is usually
                    # absent because SAN is almost never marked critical.
                    vtag, v_s, v_e = _read_tlv(der, o_e)
                    if vtag == 0x01:
                        _, v_s, v_e = _read_tlv(der, v_e)
                    if v_e > e_e:
                        v_e = e_e
                    out["subjectAltName"] = _der_san(der[v_s:v_e])
                    break
                break
            i = e
    except (ValueError, IndexError, struct.error):
        return out
    return out


# ---------------------------------------------------------------------------
# Handshake
# ---------------------------------------------------------------------------
def _verifying_context() -> ssl.SSLContext:
    """System trust store, hostname checking on -- the 'honest' handshake."""
    return ssl.create_default_context()


def _unverified_context() -> ssl.SSLContext:
    """No chain validation, used *only* to observe an untrusted certificate.

    ``ssl._create_unverified_context`` is private, so fall back to building the
    equivalent by hand (and set ``check_hostname`` first, because assigning
    ``CERT_NONE`` while hostname checking is on raises).
    """
    factory = getattr(ssl, "_create_unverified_context", None)
    if factory is not None:
        return factory()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _tls_handshake(
    host: str, port: int, *, timeout: float, context: ssl.SSLContext
) -> tuple[dict | None, bytes | None, str, str]:
    """One TCP + TLS handshake. Returns ``(cert_dict, der, version, cipher)``.

    This is the single seam between the module and the network, which is what
    lets the test-suite drive the whole pipeline with fakes and no network.

    ``cert_dict`` is ``None`` when OpenSSL declined to parse the certificate
    (i.e. ``CERT_NONE``), which is the caller's cue to parse the DER instead.
    """
    with socket.create_connection((host, port), timeout=timeout) as raw:
        with context.wrap_socket(raw, server_hostname=host or None) as tls:
            der = tls.getpeercert(binary_form=True)
            try:
                info = tls.getpeercert()
            except (ValueError, ssl.SSLError):
                info = None
            version = tls.version() or ""
            cipher = tls.cipher()
            return (
                info if isinstance(info, dict) and info else None,
                der if isinstance(der, (bytes, bytearray)) else None,
                version,
                (cipher[0] if cipher else ""),
            )


def _classify_error(exc: BaseException) -> str:
    """Turn an exception into the short human phrase the API layer shows.

    The phrases are deliberately coarse and stable: they are read by humans in
    an XAI panel, so "connection refused" is more useful than
    ``ConnectionRefusedError(10061, 'Connection refused')``.
    """
    if isinstance(exc, socket.gaierror):
        return f"dns lookup failed: {exc}"
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return "timeout"
    if isinstance(exc, ssl.SSLCertVerificationError):
        detail = getattr(exc, "verify_message", "") or str(exc)
        if "self signed" in detail.lower() or "self-signed" in detail.lower():
            return "self-signed certificate"
        if "certificate has expired" in detail.lower() or "has expired" in detail.lower():
            return "certificate verify failed (expired)"
        if "hostname mismatch" in detail.lower() or "doesn't match" in detail.lower():
            return "certificate verify failed (hostname mismatch)"
        if "unable to get local issuer" in detail.lower():
            return "certificate verify failed (unknown authority)"
        return "certificate verify failed"
    if isinstance(exc, ssl.SSLError):
        return f"tls handshake failed: {getattr(exc, 'reason', None) or exc}"
    if isinstance(exc, ConnectionRefusedError):
        return "connection refused"
    if isinstance(exc, ConnectionResetError):
        return "connection reset"
    if isinstance(exc, OSError):
        return f"connection failed: {exc}"
    return f"unexpected error: {type(exc).__name__}: {exc}"


def _is_certificate_failure(exc: BaseException) -> bool:
    """Whether an unverified retry could plausibly still yield a certificate.

    Only TLS-layer failures qualify. Retrying a refused connection or a timeout
    just doubles the worst-case latency of every dead URL in the dataset, and
    there is no certificate to observe either way.
    """
    return isinstance(exc, ssl.SSLError)


def probe_certificate(
    target: CertTarget | str, *, port: int = 443, timeout: float = 5.0, scheme: str | None = None
) -> CertProbe:
    """Connect, inspect the peer certificate, and report what was observed.

    Never raises. Attempts a verifying handshake first; on a certificate-level
    TLS failure it retries unverified so that the certificate can be *read*,
    recording the verification failure as a feature.
    """
    tgt = target if isinstance(target, CertTarget) else CertTarget(
        host=(target or "").strip().strip("[]").lower(),
        port=port if _valid_port(port) else 443,
        scheme=(scheme or "https").lower(),
    )
    probe = CertProbe(host=tgt.host, port=tgt.port, scheme=tgt.scheme)

    if not tgt.host:
        probe.reason = "no host to connect to"
        return probe

    try:
        cert_dict, der, version, cipher = _tls_handshake(
            tgt.host, tgt.port, timeout=timeout, context=_verifying_context()
        )
        probe.verified = True
    except Exception as exc:  # noqa: BLE001 - every failure is a feature
        if not _is_certificate_failure(exc):
            probe.reason = _classify_error(exc)
            return probe
        verify_error = _classify_error(exc)
        probe.verify_error = verify_error
        try:
            cert_dict, der, version, cipher = _tls_handshake(
                tgt.host, tgt.port, timeout=timeout, context=_unverified_context()
            )
        except Exception as exc2:  # noqa: BLE001
            probe.reason = verify_error or _classify_error(exc2)
            probe.verify_error = probe.verify_error or _classify_error(exc2)
            return probe

    if not der:
        probe.reason = "no certificate presented"
        return probe

    probe.available = True
    probe.tls_version = version or ""
    probe.cipher = cipher or ""

    info: dict = dict(cert_dict) if cert_dict else {}
    if info:
        probe.from_der = False
    else:
        info = _parse_certificate_der(bytes(der))
        probe.from_der = bool(info)

    probe.subject = tuple(tuple(r) for r in info.get("subject", ()) or ())
    probe.issuer = tuple(tuple(r) for r in info.get("issuer", ()) or ())
    probe.not_before = _as_datetime(info.get("notBefore"))
    probe.not_after = _as_datetime(info.get("notAfter"))
    probe.serial_hex = str(info.get("serialNumber", "") or "")
    probe.san = tuple(tuple(x) for x in info.get("subjectAltName", ()) or ())
    if not probe.verify_error and not probe.verified:
        probe.verify_error = "certificate unavailable for verification"
    probe.reason = probe.verify_error if not probe.verified else None
    return probe


def _as_datetime(value: Any) -> datetime | None:
    """Normalise a parsed validity stamp to naive UTC.

    ``getpeercert()`` yields naive datetimes already; the DER walker does too.
    Anything timezone-aware is stripped rather than mixed in, because mixing
    aware and naive stamps is how off-by-hours expiry features appear.
    """
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, str):
        # Not produced by CPython, which always hands back datetime objects,
        # but a proxy or a test double may pass the OpenSSL text form through.
        for fmt in ("%b %d %H:%M:%S %Y %Z", "%b %d %H:%M:%S %Y"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _cn_of(name: Sequence) -> str:
    """The subject/issuer commonName, or ``""`` if the DN is absent or odd.

    Written defensively on purpose: a DN arrives as nested tuples from two
    different parsers (OpenSSL's dict and our DER walk), and a malformed one
    must not take down a feature vector that is contractually total.
    """
    for rdn in name or ():
        try:
            attrs = tuple(rdn or ())
        except TypeError:
            continue
        for pair in attrs:
            try:
                key, value = pair
            except (TypeError, ValueError):
                continue
            if str(key).lower() in {"commonname", "cn"}:
                return str(value)
    return ""


def _dn_to_string(name: Sequence) -> str:
    """``"C=US, O=Example Inc, CN=Root CA"`` -- the form the CA regex reads."""
    parts = []
    for rdn in name or ():
        try:
            attrs = tuple(rdn or ())
        except TypeError:
            continue
        for pair in attrs:
            try:
                key, value = pair
            except (TypeError, ValueError):
                continue
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _dns_sans(san: Sequence) -> list[str]:
    return [str(v[1]).lower().rstrip(".") for v in san or () if len(v) == 2 and v[0] == "DNS"]


def _ip_equals(left: str, right: str) -> bool:
    """Compare two addresses by their packed bytes, not their spelling.

    Needed because a certificate may carry ``2001:db8::1`` while the URL says
    ``[2001:0db8:0000::1]``; a string comparison would call that a mismatch and
    flag a correctly-issued certificate as impersonation.
    """
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            return socket.inet_pton(family, left) == socket.inet_pton(family, right)
        except (OSError, ValueError):
            continue
    return False


def _ip_sans(san: Sequence) -> list[str]:
    return [str(v[1]) for v in san or () if len(v) == 2 and v[0] == "IP Address"]


def _san_has_ip(san: Sequence) -> bool:
    return bool(_ip_sans(san))


def _san_matches(san: Sequence, host: str) -> bool:
    """True when any SAN entry legitimately covers ``host``.

    A wildcard covers exactly one label; an iPAddress entry only ever matches an
    IP-literal host, and then only when the packed addresses agree.
    """
    for entry in _dns_sans(san):
        if _name_matches_host(entry, host):
            return True
    if is_ip_literal(host):
        bare = (host or "").strip().strip("[]")
        if any(_ip_equals(entry, bare) for entry in _ip_sans(san)):
            return True
    return False


def _name_matches_host(name: str, host: str) -> bool:
    """RFC 6125 name match, including a single left-most wildcard label.

    A wildcard is required to cover exactly one label: ``*.example.com`` covers
    ``a.example.com`` but not ``a.b.example.com``. Comparing with a bare
    ``endswith`` is the classic way to invent hostname matches that were never
    issued.
    """
    n = (name or "").lower().strip().rstrip(".")
    h = (host or "").lower().strip().rstrip(".")
    if not n or not h:
        return False
    if n == h:
        return True
    if n.startswith("*."):
        suffix = n[2:]
        if not suffix or not h.endswith("." + suffix):
            return False
        return "." not in h[: -(len(suffix) + 1)]
    return False


def _shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    n = len(data)
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))


def _clip(value: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    if value != value:  # NaN guard
        return 0.0
    return float(min(max(value, lo), hi))


def _vector_from_probe(probe: CertProbe, scheme: str) -> list[float]:
    """Turn one probe into the ordered vector, in :data:`CERT_FEATURE_NAMES` order.

    Anything unobserved stays 0.0. That is a real limitation and is stated
    rather than papered over: ``cert_available`` is the model's only way to tell
    "zero because absent" from "zero because the observation was zero", which is
    exactly why it is feature 0.
    """
    vec = [0.0] * N_CERT_FEATURES
    host = probe.host
    subject_cn = _cn_of(probe.subject)
    issuer_cn = _cn_of(probe.issuer)
    issuer_str = _dn_to_string(probe.issuer)
    dns_sans = _dns_sans(probe.san)
    now = _now_utc()

    issuer_equals_subject = bool(probe.issuer) and probe.issuer == probe.subject

    days_to_expiry = 0.0
    is_expired = 0.0
    not_before_future = 0.0
    age_days = 0.0
    span_days = 0.0
    if probe.not_after is not None:
        days_to_expiry = _clip((probe.not_after - now).total_seconds() / 86400.0, _EXPIRY_CLIP)
        is_expired = 1.0 if probe.not_after < now else 0.0
    if probe.not_before is not None:
        not_before_future = 1.0 if probe.not_before > now else 0.0
        age_days = _clip((now - probe.not_before).total_seconds() / 86400.0, _AGE_CLIP)
    if probe.not_before is not None and probe.not_after is not None:
        span_days = _clip(
            (probe.not_after - probe.not_before).total_seconds() / 86400.0, _SPAN_CLIP
        )

    san_matches_host = 1.0 if _san_matches(probe.san, host) else 0.0
    cn_matches_host = 1.0 if _name_matches_host(subject_cn, host) else 0.0
    # A mismatch is only claimable when there was at least one name on the
    # certificate to compare against. No SAN and no CN means "unevaluable",
    # which is 0.0 -- asserting a mismatch we did not observe would be a lie
    # the model would happily learn from. An IP SAN counts as a name: for an
    # IP-literal host it is the only identity the certificate can possibly
    # assert, so failing to match it is a real mismatch rather than silence.
    has_any_name = bool(dns_sans) or bool(subject_cn) or bool(_ip_sans(probe.san))
    hostname_mismatch = 1.0 if (probe.available and has_any_name and not san_matches_host and not cn_matches_host) else 0.0

    serial_bytes = b""
    try:
        serial_bytes = bytes.fromhex(probe.serial_hex) if probe.serial_hex else b""
    except ValueError:
        serial_bytes = probe.serial_hex.encode("ascii", errors="ignore")

    put = lambda name, value: vec.__setitem__(_FEATURE_INDEX[name], float(value))
    put("cert_available", 1.0 if probe.available else 0.0)
    # Gated on availability on purpose: a "verified" flag next to a zero vector
    # would claim we hold a trustworthy certificate when we hold none at all.
    put("cert_verified", 1.0 if (probe.available and probe.verified) else 0.0)
    put("verify_failed", 1.0 if (probe.available and not probe.verified) else 0.0)
    put("cert_from_der_parse", 1.0 if probe.from_der else 0.0)
    put("self_signed_guess", 1.0 if (issuer_equals_subject and not probe.verified) else 0.0)
    put("issuer_equals_subject", 1.0 if issuer_equals_subject else 0.0)
    put("days_to_expiry", days_to_expiry)
    put("is_expired", is_expired)
    put("not_before_in_future", not_before_future)
    put("cert_age_days", age_days)
    put("validity_span_days", span_days)
    put("subject_cn_len", len(subject_cn))
    put("san_count", len(probe.san))
    put("san_matches_host", san_matches_host)
    put("hostname_mismatch", hostname_mismatch)
    put("has_wildcard", 1.0 if any(s.startswith("*.") for s in dns_sans) else 0.0)
    put("san_has_ip_literal", 1.0 if _san_has_ip(probe.san) else 0.0)
    put("issuer_cn_len", len(issuer_cn))
    put("issuer_cn_is_known_ca_looking", 1.0 if _CA_LOOKING_RE.search(issuer_str) else 0.0)
    put("serial_entropy_bits", _shannon_entropy(serial_bytes))
    put("url_host_is_ip_literal", 1.0 if is_ip_literal(host) else 0.0)
    put("url_uses_http_scheme", 1.0 if (scheme or probe.scheme or "").lower() == "http" else 0.0)
    put("tls_version_id", _TLS_VERSION_IDS.get(probe.tls_version, 0.0))
    return vec


def extract_cert_features(
    host: str,
    *,
    port: int = 443,
    timeout: float = 5.0,
    scheme: str | None = None,
    errors: dict | None = None,
) -> list[float]:
    """Fixed-length TLS certificate feature vector for one host.

    Returns exactly :data:`N_CERT_FEATURES` floats in :data:`CERT_FEATURE_NAMES`
    order. **Never raises**: a dead host, a plaintext port, a self-signed
    certificate, or an unparseable certificate all yield a same-width vector so
    downstream shapes stay stable.

    ``scheme`` and ``errors`` are optional extensions to the minimal
    ``(host, port, timeout)`` call: ``scheme`` feeds ``url_uses_http_scheme``,
    and ``errors`` is filled in-place with the reason so a caller can log or
    display *why* a vector is mostly zeros.
    """
    try:
        probe = probe_certificate(
            CertTarget(
                host=(host or "").strip().strip("[]").lower(),
                port=port if _valid_port(port) else 443,
                scheme=(scheme or "https").lower(),
            ),
            timeout=timeout,
        )
        vec = _vector_from_probe(probe, (scheme or probe.scheme or "").lower())
    except Exception as exc:  # noqa: BLE001 - the last line of defence
        probe = CertProbe(host=str(host), port=port if _valid_port(port) else 443)
        probe.reason = _classify_error(exc)
        vec = [0.0] * N_CERT_FEATURES
    if errors is not None:
        try:
            errors.update(probe.error_dict())
        except Exception:  # noqa: BLE001 - never fail because of a log dict
            pass
    return vec


def extract_cert_features_for_url(
    url: str, *, timeout: float = 5.0, errors: dict | None = None
) -> list[float]:
    """URL-first entry point: resolves host, port and scheme, then extracts.

    Preferred over :func:`extract_cert_features` for anything that arrives as a
    URL, because only this path can populate ``url_uses_http_scheme`` honestly.
    """
    target = cert_context_for(url)
    return extract_cert_features(
        target.host, port=target.port, timeout=timeout, scheme=target.scheme, errors=errors
    )


def extract_cert_features_batch(
    targets: Iterable[CertTarget | str], *, timeout: float = 5.0
) -> list[list[float]]:
    """Extract one vector per target, preserving input order.

    Order preservation is the whole point: the caller zips this against its
    label array, and a reordering here would silently misalign training data.
    """
    out = []
    for t in targets:
        if isinstance(t, CertTarget):
            out.append(
                extract_cert_features(
                    t.host, port=t.port, timeout=timeout, scheme=t.scheme
                )
            )
        else:
            out.append(extract_cert_features(t, timeout=timeout))
    return out


def describe_certificate(
    host: str,
    *,
    port: int = 443,
    timeout: float = 5.0,
    scheme: str | None = None,
) -> dict:
    """Human-readable certificate description for the API / XAI layer.

    Same probe, same honesty guarantees as the vector, but keyed by name so a
    frontend can render "Self-signed, expires in 4 days" without reverse-
    engineering indices. Values are ``None`` rather than ``0`` when unknown,
    because ``days_to_expiry: 0`` would read as "expires today".
    """
    probe = probe_certificate(
        CertTarget(
            host=(host or "").strip().strip("[]").lower(),
            port=port if _valid_port(port) else 443,
            scheme=(scheme or "https").lower(),
        ),
        timeout=timeout,
    )
    now = _now_utc()
    subject_cn = _cn_of(probe.subject)
    san_match = _san_matches(probe.san, probe.host)
    cn_match = _name_matches_host(subject_cn, probe.host)
    issuer_equals_subject = bool(probe.issuer) and probe.issuer == probe.subject
    days_to_expiry = (
        (probe.not_after - now).days if probe.not_after is not None else None
    )
    return {
        "host": probe.host,
        "port": probe.port,
        "scheme": probe.scheme,
        "available": probe.available,
        # Gated on availability, matching the ``cert_verified`` feature: a
        # frontend must not render "verified" beside an empty certificate.
        "verified": bool(probe.available and probe.verified),
        "parsed_from_der": probe.from_der,
        "reason": probe.reason,
        "verify_error": probe.verify_error,
        "subject_cn": subject_cn or None,
        "issuer_cn": _cn_of(probe.issuer) or None,
        "subject": probe.subject,
        "issuer": probe.issuer,
        "not_before": probe.not_before.isoformat() if probe.not_before else None,
        "not_after": probe.not_after.isoformat() if probe.not_after else None,
        "days_to_expiry": days_to_expiry,
        "expired": (
            None if probe.not_after is None else bool(probe.not_after < now)
        ),
        "san_count": len(probe.san),
        "san": list(probe.san),
        "self_signed": bool(issuer_equals_subject and not probe.verified),
        "issuer_equals_subject": issuer_equals_subject,
        "hostname_match": bool(san_match or cn_match),
        "serial_number": probe.serial_hex or None,
        "tls_version": probe.tls_version or None,
        "cipher": probe.cipher or None,
    }


def ssl_error_reason(
    host: str, *, port: int = 443, timeout: float = 5.0, scheme: str | None = None
) -> str | None:
    """Why no trustworthy certificate could be obtained, or ``None`` if fine.

    A verified certificate returns ``None``; an unverified-but-readable one
    returns the verification reason (e.g. ``"self-signed certificate"``), since
    "we have a certificate but we do not trust it" is exactly the message the
    API wants to surface.
    """
    probe = probe_certificate(
        CertTarget(
            host=(host or "").strip().strip("[]").lower(),
            port=port if _valid_port(port) else 443,
            scheme=(scheme or "https").lower(),
        ),
        timeout=timeout,
    )
    return probe.reason

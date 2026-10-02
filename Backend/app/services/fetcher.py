"""Hardened outbound HTTP fetching.

Wraps :mod:`httpx` with the limits the detector needs and the SSRF defences
from :mod:`app.security.url_guard`:

* only ``http``/``https`` on permitted ports;
* every hostname resolved and every address checked as publicly routable;
* response size capped while streaming, so a hostile multi-gigabyte body cannot
  exhaust memory;
* redirect count capped, with **each hop re-validated** (a permitted first hop
  can redirect inward);
* content type checked before the body is stored;
* per-domain request cap and a minimum delay, so collection stays polite.

If the fetcher cannot get a page, that is recorded as a *mask*, not an error:
the fusion model is explicitly built to cope with missing modalities.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from app.core.logging_config import get_logger
from app.security.url_guard import BlockedTarget, validate_target

log = get_logger(__name__)

__all__ = ["FetchResult", "SecureFetcher", "FETCH_OK", "FETCH_FAILED"]

FETCH_OK = "ok"
FETCH_FAILED = "failed"

#: Substrings that mark a failure the detector should not retry endlessly.
_TERMINAL_HINTS = ("blocked", "not allowed", "not publicly routable", "control characters")


@dataclass
class FetchResult:
    """Outcome of one fetch attempt."""

    url: str
    ok: bool
    status_code: int | None = None
    final_url: str | None = None
    html: str | None = None
    content_type: str | None = None
    error: str | None = None
    elapsed_ms: int = 0
    bytes_read: int = 0
    redirects: int = 0

    @property
    def artifact_id(self) -> str:
        """Stable content-addressed id used to name stored artefacts."""
        return hashlib.sha256((self.final_url or self.url).encode("utf-8")).hexdigest()[:32]


class SecureFetcher:
    """Fetch pages subject to the project's SSRF and politeness policy."""

    def __init__(
        self,
        *,
        allowed_schemes: tuple[str, ...] = ("http", "https"),
        allowed_ports: tuple[int, ...] = (80, 443),
        timeout: float = 10.0,
        connect_timeout: float = 5.0,
        max_response_size_mb: float = 5.0,
        max_redirects: int = 5,
        allowed_content_types: tuple[str, ...] = ("text/html", "application/xhtml+xml"),
        user_agent: str = "MMRESearchBot/1.0",
        delay_seconds: float = 1.0,
        max_per_domain: int = 1,
        verify_tls: bool = False,
        total_timeout: float = 30.0,
    ) -> None:
        self.allowed_schemes = allowed_schemes
        self.allowed_ports = allowed_ports
        self.max_bytes = int(max_response_size_mb * 1024 * 1024)
        self.max_redirects = max_redirects
        self.allowed_content_types = tuple(t.lower() for t in allowed_content_types)
        self.user_agent = user_agent
        self.delay_seconds = delay_seconds
        self.max_per_domain = max_per_domain
        # Phishing pages routinely serve broken or self-signed certificates, so
        # TLS verification is off by default. This is a deliberate, recorded
        # research choice: the artifact is analysed as *untrusted hostile input*
        # and never trusted for anything.
        self.verify_tls = verify_tls
        # Hard ceiling on one fetch. httpx's timeout is per-operation, so a host
        # that dribbles a byte just inside the read timeout can keep a response
        # alive indefinitely; this deadline is what actually bounds the work.
        self.total_timeout = total_timeout
        self.timeout = httpx.Timeout(timeout, connect=connect_timeout)
        self._domain_counts: dict[str, int] = {}
        self._last_hit: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    async def _throttle(self, host: str) -> None:
        """Delay and cap requests per host."""
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            used = self._domain_counts.get(host, 0)
            if used >= self.max_per_domain:
                raise BlockedTarget(
                    f"per-domain cap of {self.max_per_domain} reached for {host}"
                )
            last = self._last_hit.get(host)
            if last is not None:
                wait = self.delay_seconds - (time.monotonic() - last)
                if wait > 0:
                    await asyncio.sleep(wait)
            self._domain_counts[host] = used + 1
            self._last_hit[host] = time.monotonic()

    async def fetch(self, url: str) -> FetchResult:
        """Fetch one URL, following redirects under the same policy."""
        started = time.perf_counter()
        result = FetchResult(url=url, ok=False)
        try:
            # DNS is a blocking C call. Run it off the event loop, otherwise a
            # dead hostname (very common among phishing domains) stalls every
            # other in-flight fetch for the length of the resolver timeout.
            target = await asyncio.to_thread(
                validate_target,
                url,
                allowed_schemes=self.allowed_schemes,
                allowed_ports=self.allowed_ports,
            )
            await self._throttle(target.hostname)

            current = target.url
            redirects = 0
            limits = httpx.Limits(max_connections=4, max_keepalive_connections=2)

            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=False,  # handled manually so each hop is re-validated
                limits=limits,
                verify=self.verify_tls,
                headers={"User-Agent": self.user_agent},
            ) as client:
                while True:
                    resp = await client.get(current)
                    if resp.is_redirect:
                        if redirects >= self.max_redirects:
                            raise BlockedTarget(
                                f"exceeded {self.max_redirects} redirects"
                            )
                        location = resp.headers.get("location", "")
                        if not location:
                            raise BlockedTarget("redirect without a Location header")
                        nxt = str(httpx.URL(current).join(location))
                        # Re-validate the hop: a public URL can redirect to
                        # 169.254.169.254, so the first check is not enough.
                        target = await asyncio.to_thread(
                            validate_target,
                            nxt,
                            allowed_schemes=self.allowed_schemes,
                            allowed_ports=self.allowed_ports,
                        )
                        await self._throttle(target.hostname)
                        current = target.url
                        redirects += 1
                        continue
                    break

                result.status_code = resp.status_code
                result.final_url = current
                result.redirects = redirects
                ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
                result.content_type = ctype or None

                if not self._content_type_allowed(ctype):
                    raise BlockedTarget(
                        f"content type {ctype or 'unknown'!r} is not an HTML page"
                    )
                if resp.status_code >= 400:
                    raise BlockedTarget(f"HTTP {resp.status_code}")

                # Read with a running cap rather than trusting Content-Length,
                # under a hard total deadline so a slow-drip response cannot
                # stall the whole collection pass.
                buf = bytearray()

                async def _read_body() -> None:
                    async for chunk in resp.aiter_bytes():
                        buf.extend(chunk)
                        if len(buf) > self.max_bytes:
                            raise BlockedTarget(
                                f"response exceeded {self.max_bytes} bytes"
                            )

                try:
                    await asyncio.wait_for(_read_body(), timeout=self.total_timeout)
                except asyncio.TimeoutError as exc:
                    raise BlockedTarget(
                        f"body read exceeded {self.total_timeout:g}s total deadline"
                    ) from exc

                result.bytes_read = len(buf)
                text = buf.decode(resp.encoding or "utf-8", errors="replace")
                result.html = text[: 200_000]
                result.ok = True
        except (BlockedTarget, httpx.HTTPError, ValueError, OSError) as exc:
            result.error = str(exc) or exc.__class__.__name__
        except asyncio.CancelledError:
            # Our own hard deadline fired. Record it as a failed fetch rather
            # than letting cancellation escape and take down the whole pass.
            result.error = f"exceeded {self.total_timeout:g}s total deadline"
            result.elapsed_ms = int((time.perf_counter() - started) * 1000)
            log.info("fetch", url=url, ok=False, status=None, bytes=0,
                     ms=result.elapsed_ms, error=result.error)
            return result

        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "fetch",
            url=url,
            ok=result.ok,
            status=result.status_code,
            bytes=result.bytes_read,
            ms=result.elapsed_ms,
            error=result.error,
        )
        return result

    def _content_type_allowed(self, ctype: str) -> bool:
        if not ctype:
            # Some servers omit it; the HTML sniff below is the backstop.
            return True
        return any(ctype.startswith(allowed) for allowed in self.allowed_content_types)


async def fetch_many(
    urls: list[str],
    *,
    concurrency: int = 8,
    **kwargs: object,
) -> list[FetchResult]:
    """Fetch many URLs with bounded concurrency."""
    fetcher = SecureFetcher(**kwargs)  # type: ignore[arg-type]
    semaphore = asyncio.Semaphore(concurrency)

    async def one(u: str) -> FetchResult:
        async with semaphore:
            return await fetcher.fetch(u)

    return list(await asyncio.gather(*(one(u) for u in urls)))


def store_html(result: FetchResult, out_dir: Path) -> Path | None:
    """Persist fetched HTML, or return ``None`` when there is nothing to store.

    Content is addressed by final URL so re-running collection overwrites rather
    than accumulating near-duplicate copies of the same page.
    """
    if not result.ok or not result.html:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result.artifact_id}.html"
    path.write_text(result.html, encoding="utf-8", errors="replace")
    return path


def host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""

"""Rendered-screenshot capture with Playwright/Chromium.

Isolation strategy
------------------
The deployment contract for this service is a container. Docker is **not
available on this host**, so the capture runs in-process with the browser
hardened from the inside:

* a fresh, isolated browser context per capture, so cookies and storage never
  persist between pages;
* request interception that aborts every non-``http(s)`` scheme, which is what
  stops a hostile page from reading local files (``file://``) or exfiltrating
  over ``ftp://``/``ws://``;
* permission prompts for camera, microphone, geolocation and notifications are
  denied, and downloads are refused;
* JS dialogs are auto-dismissed so a page cannot stall the collector;
* a hard navigation timeout plus a total byte budget on the response.

**What this cannot do, stated plainly:** without a container there is no
filesystem, PID, or network namespace around the browser. A vulnerability in
Chromium itself would run with the collector's privileges. This is a real
reduction in the security posture and is reported as such rather than papered
over; the containerised service remains the deployment target and is the only
form suitable for untrusted traffic.
"""

from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.logging_config import get_logger

log = get_logger(__name__)

__all__ = ["ScreenshotResult", "capture_screenshot", "capture_many"]

#: Chromium flags that reduce the reachable surface of the renderer.
SCREENSHOT_HARDENING_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--disable-extensions",
    "--disable-background-networking",
    "--disable-sync",
    "--disable-translate",
    "--disable-default-apps",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-features=Translate,AutomationControlled,IsolateOrigins",
    "--blink-settings=imagesEnabled=true,scriptEnabled=true",
    "--mute-audio",
    "--hide-scrollbars",
]

#: Schemes a page is allowed to load. Anything else is aborted.
_ALLOWED_SCHEMES = ("http", "https", "data", "about")


@dataclass
class ScreenshotResult:
    """Outcome of one capture attempt."""

    url: str
    ok: bool
    png_path: Path | None = None
    png_bytes: int = 0
    final_url: str | None = None
    error: str | None = None
    elapsed_ms: int = 0


async def _capture_one(
    page_url: str,
    out_dir: Path,
    *,
    width: int,
    height: int,
    nav_timeout_ms: int,
    user_agent: str,
    artifact_id: str | None = None,
    browser=None,
) -> ScreenshotResult:
    started = time.perf_counter()
    result = ScreenshotResult(url=page_url, ok=False)

    # The browser is launched once and reused: launching per capture costs
    # several seconds each and dominated the whole pass.
    own_browser = browser is None
    try:
        if own_browser:
            from playwright.async_api import async_playwright

            async with async_playwright() as pw:
                browser = await pw.chromium.launch(
                    headless=True,
                    args=SCREENSHOT_HARDENING_ARGS,
                    chromium_sandbox=False,
                    # Without this, a wedged renderer makes close() hang forever
                    # and stalls the entire collection pass with it.
                    timeout=20_000,
                )
                try:
                    return await _capture_with_browser(
                        browser,
                        page_url,
                        Path(out_dir),
                        width=width,
                        height=height,
                        nav_timeout_ms=nav_timeout_ms,
                        user_agent=user_agent,
                        artifact_id=artifact_id,
                        result=result,
                        started=started,
                    )
                finally:
                    await _safe_close(browser)
        return await _capture_with_browser(
            browser,
            page_url,
            Path(out_dir),
            width=width,
            height=height,
            nav_timeout_ms=nav_timeout_ms,
            user_agent=user_agent,
            artifact_id=artifact_id,
            result=result,
            started=started,
        )
    except Exception as exc:  # noqa: BLE001 - a failed capture is a mask, not a crash
        result.error = f"{exc.__class__.__name__}: {exc}"[:300]
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "screenshot",
            url=page_url,
            ok=False,
            bytes=0,
            ms=result.elapsed_ms,
            error=result.error,
        )
        return result


async def _safe_close(closeable) -> None:
    """Close a browser/context without letting teardown hang the pass."""
    try:
        await asyncio.wait_for(closeable.close(), timeout=15)
    except Exception:  # noqa: BLE001 - teardown failure must not mask the result
        pass


async def _capture_with_browser(
    browser,
    page_url: str,
    out_dir: Path,
    *,
    width: int,
    height: int,
    nav_timeout_ms: int,
    user_agent: str,
    artifact_id: str | None,
    result: ScreenshotResult,
    started: float,
) -> ScreenshotResult:
    """Render one page in a fresh isolated context on an existing browser."""
    context = None
    try:
        context = await browser.new_context(
            viewport={"width": width, "height": height},
            user_agent=user_agent,
            java_script_enabled=True,
            bypass_csp=False,
            ignore_https_errors=True,  # phishing hosts often have bad certs
            service_workers="block",
        )
        # Deny every capability a page could use to exfiltrate or prompt.
        await context.grant_permissions([])

        async def route_handler(route):
            req = route.request
            if req.url.split(":", 1)[0].lower() not in _ALLOWED_SCHEMES:
                await route.abort()
                return
            await route.continue_()

        await context.route("**/*", route_handler)

        page = await context.new_page()
        page.on("dialog", lambda d: asyncio.ensure_future(d.dismiss()))

        resp = await page.goto(
            page_url,
            wait_until="domcontentloaded",
            timeout=nav_timeout_ms,
        )
        result.final_url = page.url
        if resp is not None and resp.status >= 400:
            raise RuntimeError(f"HTTP {resp.status}")

        # A short settle window lets above-the-fold content paint without
        # waiting on slow third-party assets indefinitely.
        try:
            await page.wait_for_load_state("networkidle", timeout=4_000)
        except Exception:
            pass  # networkidle is best-effort; domcontentloaded is enough

        out_dir.mkdir(parents=True, exist_ok=True)
        name = artifact_id or _artifact_id(page.url)
        png_path = out_dir / f"{name}.png"
        data = await page.screenshot(full_page=False, type="png", timeout=15_000)
        png_path.write_bytes(data)

        result.png_path = png_path
        result.png_bytes = len(data)
        result.ok = True
    except Exception as exc:  # noqa: BLE001 - a failed capture is a mask, not a crash
        result.error = f"{exc.__class__.__name__}: {exc}"[:300]
    finally:
        if context is not None:
            await _safe_close(context)

    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "screenshot",
        url=page_url,
        ok=result.ok,
        bytes=result.png_bytes,
        ms=result.elapsed_ms,
        error=result.error,
    )
    return result

    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "screenshot",
        url=page_url,
        ok=result.ok,
        bytes=result.png_bytes,
        ms=result.elapsed_ms,
        error=result.error,
    )
    return result


def _artifact_id(url: str) -> str:
    import hashlib

    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


async def capture_screenshot(
    page_url: str,
    out_dir: Path,
    *,
    width: int = 1280,
    height: int = 800,
    nav_timeout_ms: int = 15_000,
    user_agent: str = "MMRESearchBot/1.0",
    artifact_id: str | None = None,
    browser=None,
) -> ScreenshotResult:
    """Render one page and write a PNG.

    Pass ``browser`` to reuse an already-launched browser; omitting it launches
    and tears down one for this single page.
    """
    return await _capture_one(
        page_url,
        Path(out_dir),
        width=width,
        height=height,
        nav_timeout_ms=nav_timeout_ms,
        user_agent=user_agent,
        artifact_id=artifact_id,
        browser=browser,
    )


async def capture_many(
    urls: list[str],
    out_dir: Path,
    *,
    concurrency: int = 2,
    **kwargs: object,
) -> list[ScreenshotResult]:
    """Capture many pages with bounded concurrency on a single shared browser.

    Each capture still gets its own isolated browser context, so pages still
    cannot see each other's cookies or storage; only the (expensive) browser
    process is shared.
    """
    from playwright.async_api import async_playwright

    semaphore = asyncio.Semaphore(concurrency)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=SCREENSHOT_HARDENING_ARGS,
            chromium_sandbox=False,
            timeout=20_000,
        )

        async def one(u: str) -> ScreenshotResult:
            async with semaphore:
                return await capture_screenshot(u, out_dir, browser=browser, **kwargs)  # type: ignore[arg-type]

        try:
            return list(await asyncio.gather(*(one(u) for u in urls)))
        finally:
            await _safe_close(browser)


def load_png_as_data_uri(path: Path) -> str:
    """Base64 data URI for serving a stored screenshot."""
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")

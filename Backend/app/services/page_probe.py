"""Live page probing with Playwright/Chromium: layout, behaviour, brand images.

Relationship to :mod:`app.services.screenshot`
----------------------------------------------
The browser bootstrap is **not** reinvented here. This module imports the
hardening argument list, the bounded-teardown helper and the artifact-id scheme
from ``screenshot.py`` and reuses the same context policy: a fresh isolated
context per probe, non-``http(s)`` schemes aborted at the route layer, all
permissions denied, dialogs auto-dismissed, hard navigation timeout plus a
best-effort settle window. Launching the browser still costs seconds, so a
:class:`PageProbe` launches lazily once and reuses the process across probes
while keeping one context per page, exactly as ``capture_many`` does.

What makes this *behavioural* rather than static
------------------------------------------------
A markup read can only see the scripts a page ships. The counters that matter
here are what the page *does*, so :data:`_INIT_SCRIPT` is installed with
``context.add_init_script`` **before** navigation and therefore runs in every
frame at document-start, ahead of any page script. It wraps ``window.open``,
``eval``, ``Function``, ``atob``, ``unescape``, ``String.fromCharCode``,
``history.pushState``, ``HTMLFormElement.submit``, ``EventTarget.
addEventListener``, ``Event.stopImmediatePropagation``, the timer functions,
the clipboard and the permission/fullscreen APIs, and listens for ``error`` and
``unhandledrejection``. Counter state is installed non-configurable so a hostile
page cannot trivially delete it.

Geometry and image attributes are gathered afterwards by a single in-page
function (:data:`_GEOMETRY_JS`) that returns rects in **document** coordinates.
Overlapping-control pairs are computed there because an O(n^2) sweep in the
browser is far cheaper to ship across the CDP boundary than to reconstruct from
rectangles in Python.

Failure policy
--------------
This never raises. A DNS failure, a navigation timeout, an HTTP 500, a wedged
renderer or a browser that refuses to launch all produce
``available=False`` with a short human-readable ``reason``, because in this
pipeline an unprobeable page is a masking signal, not a crash.

Known limitation, stated plainly: external script *bodies* are not fetched, so
``obfuscated_script_hits`` covers inline scripts only. Fetching them would add
an unbounded number of third-party requests to an untrusted page.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.logging_config import get_logger
from app.services.screenshot import (
    SCREENSHOT_HARDENING_ARGS,
    _ALLOWED_SCHEMES,
    _artifact_id,
    _safe_close,
)

log = get_logger(__name__)

__all__ = ["PageProbeResult", "PageProbe", "probe_page", "probe_many"]


@dataclass
class PageProbeResult:
    """Outcome of one page probe.

    ``available`` is mandatory, so a result can never read as a success by
    accident: when it is ``False`` the three payload containers are empty and
    ``reason`` says why in a form an operator can act on.
    """

    url: str
    available: bool
    layout: dict = field(default_factory=dict)
    behavior: dict = field(default_factory=dict)
    images: list = field(default_factory=list)
    reason: str | None = None
    final_url: str | None = None
    elapsed_ms: int = 0


#: Counters installed at document-start. Any key absent from a frame means that
#: frame did not run the init script (or tore down before we read it); the
#: collector sums whatever it finds rather than trusting a fixed shape.
_INIT_SCRIPT = r"""
(() => {
  const W = window;
  if (W.__probe) { return; }
  // The native references are captured before anything is wrapped: once
  // ``window.Function`` is replaced, ``Function.prototype.toString`` inside this
  // script would resolve to the wrapper and throw.
  const NativeFunction = W.Function;
  const NativeSetTimeout = W.setTimeout;
  const NativeSetInterval = W.setInterval;
  const p = {
    js_errors: 0,
    unhandled_rejections: 0,
    window_open_calls: 0,
    push_state_calls: 0,
    replace_state_calls: 0,
    location_assign_calls: 0,
    eval_calls: 0,
    function_ctor_calls: 0,
    atob_calls: 0,
    unescape_calls: 0,
    from_char_code_calls: 0,
    long_blob_literals: 0,
    form_submits_total: 0,
    form_submits_without_gesture: 0,
    keyboard_event_intercepts: 0,
    keystroke_suppressions: 0,
    paste_event_intercepts: 0,
    clipboard_access_attempts: 0,
    permission_queries: 0,
    fullscreen_requests: 0,
    notification_permission_requests: 0,
    timers_registered: 0,
    timer_redirects: 0
  };
  Object.defineProperty(W, '__probe', {
    value: p, enumerable: false, configurable: false, writable: false
  });

  const bump = (k) => { p[k] = (p[k] || 0) + 1; };

  // A payload handed to a decoder that is long and base64-shaped is an encoded
  // blob; short/printable arguments are ordinary string work.
  const noteBlob = (s) => {
    try {
      if (typeof s !== 'string') { return; }
      const t = s.trim();
      if (t.length < 80 || t.length > 400000) { return; }
      if (!/^[A-Za-z0-9+/=\s]+$/.test(t)) { return; }
      if (!/[A-Za-z]/.test(t) || !/[0-9]/.test(t)) { return; }
      p.long_blob_literals++;
    } catch (e) {}
  };

  const wrap = (obj, name, key, after) => {
    try {
      const orig = obj ? obj[name] : null;
      if (typeof orig !== 'function') { return; }
      obj[name] = function () {
        const args = arguments;
        try { bump(key); } catch (e) {}
        if (after) { try { after(args); } catch (e) {} }
        return orig.apply(this, args);
      };
    } catch (e) {}
  };

  wrap(W, 'open', 'window_open_calls');
  wrap(W, 'eval', 'eval_calls');
  wrap(W, 'atob', 'atob_calls', (a) => noteBlob(a[0]));
  wrap(W, 'unescape', 'unescape_calls');
  wrap(W, 'Function', 'function_ctor_calls');
  wrap(String, 'fromCharCode', 'from_char_code_calls', (a) => noteBlob(a[0]));
  wrap(W.history, 'pushState', 'push_state_calls');
  wrap(W.history, 'replaceState', 'replace_state_calls');
  wrap(W.location, 'assign', 'location_assign_calls');
  wrap(W.location, 'replace', 'location_assign_calls');

  // Runtime errors only: a failed <img>/<script> load also fires 'error', but
  // it arrives with a DOM target rather than as the window itself.
  W.addEventListener('error', (ev) => {
    try { if (ev && ev.target && ev.target !== W) { return; } } catch (e) {}
    p.js_errors++;
  }, true);
  W.addEventListener('unhandledrejection', () => { p.unhandled_rejections++; }, true);

  // Capture-phase listeners see a keystroke before the page's own handlers and
  // can swallow it, which is the mechanism of a keystroke-stealing kit.
  try {
    const origAEL = EventTarget.prototype.addEventListener;
    EventTarget.prototype.addEventListener = function (type) {
      try {
        const capture = arguments[2] === true ||
          (arguments[2] && typeof arguments[2] === 'object' && arguments[2].capture);
        const t = String(type);
        if (capture && /^key/.test(t)) { p.keyboard_event_intercepts++; }
        if (capture && t === 'paste') { p.paste_event_intercepts++; }
      } catch (e) {}
      return origAEL.apply(this, arguments);
    };
    const origSIP = Event.prototype.stopImmediatePropagation;
    Event.prototype.stopImmediatePropagation = function () {
      try { if (this && /^key/.test(String(this.type || ''))) { p.keystroke_suppressions++; } } catch (e) {}
      return origSIP.apply(this, arguments);
    };
  } catch (e) {}

  const userGesture = () => {
    try { return !!(navigator.userActivation && navigator.userActivation.isActive); }
    catch (e) { return false; }
  };
  try {
    ['submit', 'requestSubmit'].forEach((name) => {
      const orig = HTMLFormElement.prototype[name];
      if (typeof orig !== 'function') { return; }
      HTMLFormElement.prototype[name] = function () {
        p.form_submits_total++;
        if (!userGesture()) { p.form_submits_without_gesture++; }
        return orig.apply(this, arguments);
      };
    });
  } catch (e) {}

  try {
    const clip = navigator.clipboard;
    if (clip) {
      ['writeText', 'readText', 'write', 'read'].forEach((m) => {
        const orig = clip[m];
        if (typeof orig !== 'function') { return; }
        clip[m] = function () {
          p.clipboard_access_attempts++;
          return orig.apply(this, arguments);
        };
      });
    }
  } catch (e) {}
  try {
    if (navigator.permissions && typeof navigator.permissions.query === 'function') {
      const oq = navigator.permissions.query.bind(navigator.permissions);
      navigator.permissions.query = function () {
        p.permission_queries++;
        return oq.apply(null, arguments);
      };
    }
  } catch (e) {}
  try {
    if (window.Notification && typeof window.Notification.requestPermission === 'function') {
      const on = window.Notification.requestPermission.bind(window.Notification);
      window.Notification.requestPermission = function () {
        p.notification_permission_requests++;
        return on.apply(null, arguments);
      };
    }
  } catch (e) {}
  try {
    ['requestFullscreen', 'webkitRequestFullscreen', 'mozRequestFullScreen',
     'msRequestFullscreen'].forEach((m) => {
      wrap(Element.prototype, m, 'fullscreen_requests');
    });
  } catch (e) {}

  // A timer whose callback body assigns to location is a scheduled redirect.
  const looksLikeRedirect = (fn) => {
    try {
      if (typeof fn !== 'function') { return false; }
      const src = NativeFunction.prototype.toString.call(fn);
      return /(window|document|self|top)\.location\s*[.=]|location\.(href|replace|assign)\s*=/.test(src);
    } catch (e) { return false; }
  };
  ['setTimeout', 'setInterval'].forEach((name) => {
    const orig = name === 'setTimeout' ? NativeSetTimeout : NativeSetInterval;
    if (typeof orig !== 'function') { return; }
    W[name] = function (fn) {
      p.timers_registered++;
      if (looksLikeRedirect(fn)) { p.timer_redirects++; }
      return orig.apply(this, arguments);
    };
  });
})();
"""


#: One-shot DOM inspection: rendered geometry, image attributes, cursor
#: overlays, form action and the static obfuscation scan of inline scripts.
#: Returns ``{"layout": ..., "behavior": ..., "images": [...]}``.
_GEOMETRY_JS = r"""
() => {
  const sx = window.scrollX || 0;
  const sy = window.scrollY || 0;
  const vw = window.innerWidth || document.documentElement.clientWidth || 0;
  const vh = window.innerHeight || document.documentElement.clientHeight || 0;
  const de = document.documentElement;
  const bodyH = document.body ? (document.body.scrollHeight || 0) : 0;
  const scrollHeight = Math.max(de.scrollHeight || 0, bodyH, vh);
  const docWidth = Math.max(de.scrollWidth || 0, de.offsetWidth || 0, vw);

  const rectOf = (el) => {
    const r = el.getBoundingClientRect();
    return { x: r.left + sx, y: r.top + sy, w: r.width, h: r.height };
  };
  const isVisible = (el, r) => {
    if (!(r.w > 0.5 && r.h > 0.5)) { return false; }
    try {
      const cs = getComputedStyle(el);
      if (cs) {
        if (cs.display === 'none') { return false; }
        if (cs.visibility === 'hidden' || cs.visibility === 'collapse') { return false; }
        if (parseFloat(cs.opacity || '1') <= 0.01) { return false; }
      }
    } catch (e) {}
    return true;
  };
  const namey = (el) => [el.name, el.id, el.placeholder, el.getAttribute('aria-label'),
                         el.getAttribute('autocomplete')].join(' ');
  const PWD_RE = /pass|pwd|secret|credential|otp|pin/i;

  const MAX_ELEMENTS = 60;
  const inputs = Array.prototype.slice.call(
    document.querySelectorAll('input, textarea, select')
  ).slice(0, MAX_ELEMENTS);
  const inputRecs = inputs.map((el) => {
    const r = rectOf(el);
    const type = String(el.type || el.tagName || '').toLowerCase();
    let isPwd = type === 'password';
    if (!isPwd && type !== 'checkbox' && type !== 'radio' && type !== 'hidden') {
      isPwd = PWD_RE.test(namey(el));
    }
    return {
      x: r.x, y: r.y, w: r.w, h: r.h, type: type,
      visible: isVisible(el, r), is_password: isPwd
    };
  });
  const passwords = inputRecs.filter(
    (rec) => rec.is_password && rec.type !== 'hidden' && rec.visible
  );

  const forms = Array.prototype.slice.call(document.forms || []).slice(0, 20);
  let mainForm = null;
  forms.forEach((f) => {
    const r = rectOf(f);
    if (!(r.w > 0 && r.h > 0)) { return; }
    if (!mainForm || r.w * r.h > mainForm.w * mainForm.h) { mainForm = r; }
  });

  const SEL = 'input, button, select, textarea, a[href], [role="button"], [onclick]';
  const targets = Array.prototype.slice.call(document.querySelectorAll(SEL)).slice(0, MAX_ELEMENTS);
  const targetEls = [];
  const seen = [];
  targets.forEach((el) => { if (seen.indexOf(el) === -1) { seen.push(el); targetEls.push(el); } });
  const clickTargets = targetEls.map((el) => {
    const r = rectOf(el);
    return { x: r.x, y: r.y, w: r.w, h: r.h, visible: isVisible(el, r) };
  });

  // Overlap sweep over field + control elements, capped so the page cannot make
  // us do unbounded work.
  const overlapEls = [];
  inputs.concat(targetEls).forEach((el) => {
    if (overlapEls.length < 40 && overlapEls.indexOf(el) === -1) { overlapEls.push(el); }
  });
  const boxes = overlapEls.map((el) => {
    const r = rectOf(el);
    return { x: r.x, y: r.y, w: r.w, h: r.h };
  });
  let overlappingPairs = 0;
  let maxOverlapRatio = 0;
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i];
      const b = boxes[j];
      const ox = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const oy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      if (ox <= 0 || oy <= 0) { continue; }
      const inter = ox * oy;
      const smaller = Math.min(a.w * a.h, b.w * b.h);
      if (!(smaller > 0)) { continue; }
      const ratio = inter / smaller;
      if (ratio > 0.2) { overlappingPairs++; }
      if (ratio > maxOverlapRatio) { maxOverlapRatio = ratio; }
    }
  }

  const flowSource = mainForm
    ? clickTargets.filter((c) => {
        const cx = c.x + c.w / 2;
        const cy = c.y + c.h / 2;
        return cx >= mainForm.x && cx <= mainForm.x + mainForm.w &&
               cy >= mainForm.y && cy <= mainForm.y + mainForm.h;
      })
    : clickTargets;
  const flow = flowSource.slice().sort((a, b) => (a.y + a.h) - (b.y + b.h));
  const verticalGaps = [];
  for (let i = 1; i < flow.length; i++) {
    const gap = flow[i].y - (flow[i - 1].y + flow[i - 1].h);
    if (gap > -4 && gap < 4000) { verticalGaps.push(gap); }
  }

  // A custom or suppressed cursor over a large element is how a kit hides the
  // real pointer and directs the victim to its own target.
  let fakeCursors = 0;
  try {
    const all = document.querySelectorAll('body *');
    const lim = Math.min(all.length, 300);
    for (let i = 0; i < lim; i++) {
      const el = all[i];
      const cs = getComputedStyle(el);
      if (!cs) { continue; }
      const cur = String(cs.cursor || '');
      if (cur !== 'none' && cur.indexOf('url(') !== 0) { continue; }
      const r = rectOf(el);
      if (r.w > 40 && r.h > 40 && isVisible(el, r)) { fakeCursors++; }
    }
  } catch (e) {}

  let scriptCount = 0;
  let externalCount = 0;
  let obfuscatedHits = 0;
  const origins = {};
  try {
    const scripts = document.querySelectorAll('script');
    scriptCount = scripts.length;
    for (let i = 0; i < scripts.length && i < 120; i++) {
      const el = scripts[i];
      const src = el.getAttribute('src') || '';
      if (src.trim()) {
        externalCount++;
        let origin = '';
        try { origin = new URL(el.src, location.href).origin; } catch (e) {}
        if (origin && origin !== 'null') { origins[origin] = true; }
        continue;
      }
      const text = el.textContent || '';
      if (!text || text.length > 200000) { continue; }
      if (/\beval\s*\(/.test(text)) { obfuscatedHits++; }
      if (/\batob\s*\(|unescape\s*\(/.test(text)) { obfuscatedHits++; }
      if (/String\s*\.\s*fromCharCode/.test(text)) { obfuscatedHits++; }
      if (/(\\x[0-9a-f]{2}){6,}/i.test(text)) { obfuscatedHits++; }
      if (/[A-Za-z0-9+/]{120,}={0,2}/.test(text)) { obfuscatedHits++; }
      if (/\[\s*['"][^'"]{1,4}['"]\s*(\+\s*['"][^'"]{1,4}['"]\s*){6,}/.test(text)) { obfuscatedHits++; }
      if (/\\u00[0-9a-f]{2}/i.test(text) && text.length > 200) { obfuscatedHits++; }
    }
  } catch (e) {}

  let formActionCrossOrigin = 0;
  let formAction = '';
  try {
    const f = document.forms && document.forms.length ? document.forms[0] : null;
    if (f) {
      const raw = f.getAttribute('action');
      formAction = String(raw === null ? '' : raw).slice(0, 512);
      if (/^\s*javascript:/i.test(formAction)) {
        formActionCrossOrigin = 1;
      } else {
        let target = null;
        try { target = new URL(formAction || location.href, location.href); } catch (e) { target = null; }
        if (!target || target.origin !== location.origin) { formActionCrossOrigin = 1; }
      }
    }
  } catch (e) {}

  const pageHost = String(location.hostname || '').toLowerCase();
  const images = Array.prototype.slice.call(document.images || []).slice(0, 80).map((el) => {
    const r = rectOf(el);
    let src = '';
    try { src = el.currentSrc || el.src || ''; } catch (e) {}
    return {
      src: String(src).slice(0, 512),
      alt: String(el.getAttribute('alt') || '').slice(0, 200),
      title: String(el.getAttribute('title') || '').slice(0, 200),
      natural_width: Number(el.naturalWidth) || 0,
      natural_height: Number(el.naturalHeight) || 0,
      rendered_width: Math.round(r.w),
      rendered_height: Math.round(r.h),
      visible: isVisible(el, r),
      page_host: pageHost
    };
  });

  return {
    layout: {
      viewport_width: vw,
      viewport_height: vh,
      scroll_height: scrollHeight,
      doc_width: docWidth,
      doc_height: scrollHeight,
      n_forms: forms.length,
      form: mainForm,
      inputs: inputRecs,
      password_fields: passwords,
      click_targets: clickTargets,
      overlapping_pairs: overlappingPairs,
      max_overlap_ratio: maxOverlapRatio,
      vertical_gaps: verticalGaps
    },
    behavior: {
      script_count: scriptCount,
      external_script_count: externalCount,
      external_script_origins: Object.keys(origins).length,
      obfuscated_script_hits: obfuscatedHits,
      fake_cursor_overlays: fakeCursors,
      form_action_cross_origin: formActionCrossOrigin,
      form_action: formAction
    },
    images: images
  };
}
"""


def _classify_error(exc: BaseException, stage: str) -> str:
    """Turn a Playwright/driver exception into a short operator-readable reason."""
    message = str(exc)
    for marker in ("ERR_NAME_NOT_RESOLVED", "ERR_CONNECTION_REFUSED", "ERR_CONNECTION_RESET",
                   "ERR_INTERNET_DISCONNECTED", "ERR_CONNECTION_CLOSED", "ERR_ADDRESS_UNREACHABLE",
                   "ERR_SSL_PROTOCOL_ERROR", "ERR_CERT", "ERR_BLOCKED_BY_CLIENT",
                   "ERR_PROXY_CONNECTION_FAILED", "ERR_TOO_MANY_REDIRECTS",
                   "net::ERR_ABORTED"):
        if marker in message:
            return marker.replace("net::", "")
    if "Timeout" in exc.__class__.__name__ or "timeout" in message.lower():
        return f"{stage} timeout"
    return f"{exc.__class__.__name__}: {message}"[:200]


class PageProbe:
    """Reusable probe over a single Chromium process.

    The browser is launched lazily on the first probe and kept until
    :meth:`aclose` (or the ``async with`` block) exits; every probe still gets a
    brand-new isolated context, so no cookies or storage survive between pages.
    Pass an already-launched ``browser`` to share a process with the screenshot
    service.
    """

    def __init__(
        self,
        *,
        width: int = 1280,
        height: int = 800,
        nav_timeout_ms: int = 15_000,
        settle_ms: int = 4_000,
        user_agent: str = "MMRESearchBot/1.0",
        evaluate_timeout_ms: int = 10_000,
        browser: Any = None,
    ) -> None:
        self.width = width
        self.height = height
        self.nav_timeout_ms = nav_timeout_ms
        self.settle_ms = settle_ms
        self.user_agent = user_agent
        self.evaluate_timeout_ms = evaluate_timeout_ms
        self._browser = browser
        self._owns_browser = browser is None
        self._playwright: Any = None

    # -- lifecycle ---------------------------------------------------------- #

    async def _ensure_browser(self) -> Any:
        if self._browser is not None:
            return self._browser
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        try:
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=SCREENSHOT_HARDENING_ARGS,
                chromium_sandbox=False,
                timeout=20_000,
            )
        except Exception:
            # Leave nothing half-started behind a failed launch.
            await self.aclose()
            raise
        return self._browser

    async def aclose(self) -> None:
        """Tear down the owned browser without letting teardown raise."""
        if self._browser is not None and self._owns_browser:
            await _safe_close(self._browser)
        self._browser = None
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:  # pragma: no cover - driver already gone
                pass
            self._playwright = None

    async def __aenter__(self) -> "PageProbe":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # -- probing ------------------------------------------------------------ #

    async def probe(
        self,
        url: str,
        *,
        timeout_ms: int | None = None,
        shot: Path | None = None,
        artifact_id: str | None = None,
    ) -> PageProbeResult:
        """Load ``url`` in an isolated context and collect all three signals.

        ``timeout_ms`` overrides the navigation timeout for this call.
        ``shot`` is an optional artifact directory: when given, a viewport PNG is
        written to ``<shot>/<artifact_id or sha256(url)[:32]>.png``, matching the
        screenshot service's naming so a page's screenshot and probe land under
        the same key.

        Never raises; a failure yields ``available=False`` plus ``reason``.
        """
        started = time.perf_counter()
        result = PageProbeResult(url=url, available=False)
        nav_timeout = int(timeout_ms or self.nav_timeout_ms)

        try:
            browser = await self._ensure_browser()
        except Exception as exc:  # noqa: BLE001 - no browser is a mask, not a crash
            result.reason = f"browser launch failed: {_classify_error(exc, 'launch')}"
            result.elapsed_ms = int((time.perf_counter() - started) * 1000)
            log.info("page_probe", url=url, available=False, ms=result.elapsed_ms,
                     reason=result.reason)
            return result

        try:
            await self._probe_with_browser(
                browser, url, nav_timeout, shot, artifact_id, result, started
            )
        except Exception as exc:  # noqa: BLE001 - a failed probe is a mask, not a crash
            result.available = False
            result.reason = f"{_classify_error(exc, 'probe')}"
            result.elapsed_ms = int((time.perf_counter() - started) * 1000)
            log.info("page_probe", url=url, available=False, ms=result.elapsed_ms,
                     reason=result.reason)
        return result

    async def probe_many(
        self,
        urls: list[str],
        *,
        concurrency: int = 2,
        **kwargs: Any,
    ) -> list[PageProbeResult]:
        """Probe many pages with bounded concurrency on one shared browser."""
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def one(target: str) -> PageProbeResult:
            async with semaphore:
                return await self.probe(target, **kwargs)

        return list(await asyncio.gather(*(one(u) for u in urls)))

    # -- internals ---------------------------------------------------------- #

    async def _probe_with_browser(
        self,
        browser: Any,
        url: str,
        nav_timeout: int,
        shot: Path | None,
        artifact_id: str | None,
        result: PageProbeResult,
        started: float,
    ) -> None:
        context = None
        dialogs = {"count": 0}
        # Real navigations (redirect chains, meta refresh, scripted redirects)
        # are only counted once the document has loaded, so the initial
        # navigation itself is never mistaken for post-load movement.
        navs = {"count": 0, "armed": False}

        async def route_handler(route: Any) -> None:
            req = route.request
            if req.url.split(":", 1)[0].lower() not in _ALLOWED_SCHEMES:
                await route.abort()
                return
            await route.continue_()

        try:
            context = await browser.new_context(
                viewport={"width": self.width, "height": self.height},
                user_agent=self.user_agent,
                java_script_enabled=True,
                bypass_csp=False,
                ignore_https_errors=True,  # phishing hosts often have bad certs
                service_workers="block",
            )
            # Deny every capability a page could use to prompt or exfiltrate.
            await context.grant_permissions([])
            await context.route("**/*", route_handler)

            # Document-start instrumentation: this is what makes the behaviour
            # branch behavioural. It must be installed before the first
            # navigation or it would miss everything the page does on load.
            await context.add_init_script(_INIT_SCRIPT)

            page = await context.new_page()
            dialog_handler = self._dialog_handler(dialogs)
            page.on("dialog", lambda d: asyncio.ensure_future(dialog_handler(d)))
            page.on(
                "framenavigated",
                lambda frame: navs.__setitem__("count", navs["count"] + 1)
                if navs["armed"]
                else None,
            )

            response = await page.goto(url, wait_until="domcontentloaded", timeout=nav_timeout)
            result.final_url = page.url
            if response is not None and response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}")

            # A short settle window lets deferred JS (timers, injected frames,
            # lazy images) run so the counters mean something. Best-effort.
            if self.settle_ms > 0:
                try:
                    await page.wait_for_load_state("networkidle", timeout=self.settle_ms)
                except Exception:
                    pass  # a page that never idles is still worth probing

            navs["armed"] = True
            geometry = await asyncio.wait_for(
                page.evaluate(_GEOMETRY_JS), timeout=self.evaluate_timeout_ms / 1000
            )
            if not isinstance(geometry, dict):
                raise RuntimeError("geometry collector returned no data")

            behavior = await self._collect_behavior(page, dialogs, navs)
            dom_extras = geometry.get("behavior")
            if isinstance(dom_extras, dict):
                behavior.update(dom_extras)

            layout = geometry.get("layout")
            images = geometry.get("images")
            result.layout = layout if isinstance(layout, dict) else {}
            result.images = [i for i in images if isinstance(i, dict)] if isinstance(images, list) else []
            result.behavior = behavior
            result.available = True
            result.reason = None

            if shot is not None:
                await self._write_shot(page, shot, artifact_id, result.final_url or url)
        except Exception as exc:  # noqa: BLE001 - a failed probe is a mask, not a crash
            result.available = False
            result.reason = _classify_error(exc, "navigation")
            result.layout = {}
            result.behavior = {}
            result.images = []
        finally:
            if context is not None:
                await _safe_close(context)

        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "page_probe",
            url=url,
            available=result.available,
            images=len(result.images),
            ms=result.elapsed_ms,
            reason=result.reason,
        )

    @staticmethod
    def _dialog_handler(dialogs: dict[str, int]):
        """Auto-dismiss dialogs so a page cannot stall the pass, and count them.

        An alert used to rush the victim is a social-engineering tell, so the
        count is kept rather than discarded.
        """

        async def handler(dialog: Any) -> None:
            dialogs["count"] += 1
            try:
                await dialog.dismiss()
            except Exception:  # pragma: no cover - dialog already gone
                pass

        return handler

    async def _collect_behavior(
        self, page: Any, dialogs: dict[str, int], navs: dict[str, Any]
    ) -> dict:
        """Sum the per-frame counters plus the driver-observed signals.

        Frames are summed rather than reading only the top document: a
        credential form rendered inside an iframe is precisely the case where
        the top-level window looks benign.

        ``post_load_navigations`` merges same-document movement seen by the
        instrumentation (history API, ``location`` assignment) with the real
        frame navigations Playwright observed after load.
        """
        merged: dict[str, float] = {}
        frames = 0
        try:
            frame_list = list(page.frames)
        except Exception:  # pragma: no cover - page already torn down
            frame_list = []

        for frame in frame_list:
            try:
                counters = await frame.evaluate("() => window.__probe || null")
            except Exception:
                continue  # cross-origin or detached frame; its counters are lost
            if not isinstance(counters, dict):
                continue
            frames += 1
            for key, value in counters.items():
                try:
                    merged[key] = merged.get(key, 0.0) + float(value)
                except (TypeError, ValueError):
                    continue

        real_navigations = float(navs.get("count", 0))
        return {
            **merged,
            "dialogs_dismissed": float(dialogs.get("count", 0)),
            "frames_probed": float(frames),
            "navigations": real_navigations,
            "post_load_navigations": real_navigations
            + merged.get("push_state_calls", 0.0)
            + merged.get("replace_state_calls", 0.0)
            + merged.get("location_assign_calls", 0.0),
        }

    async def _write_shot(
        self, page: Any, out_dir: Path, artifact_id: str | None, final_url: str
    ) -> None:
        """Optional companion screenshot, named with the shared artifact scheme."""
        try:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            data = await page.screenshot(full_page=False, type="png", timeout=15_000)
            (out_dir / f"{artifact_id or _artifact_id(final_url)}.png").write_bytes(data)
        except Exception as exc:  # noqa: BLE001 - a missing PNG must not void the probe
            log.info("page_probe_shot_failed", url=final_url, error=str(exc)[:200])


async def probe_page(url: str, **kwargs: Any) -> PageProbeResult:
    """Probe one URL with a throwaway browser (convenience over :class:`PageProbe`)."""
    probe_kwargs = {
        k: kwargs.pop(k)
        for k in ("timeout_ms", "shot", "artifact_id")
        if k in kwargs
    }
    async with PageProbe(**kwargs) as probe:
        return await probe.probe(url, **probe_kwargs)


async def probe_many(urls: list[str], *, concurrency: int = 2, **kwargs: Any) -> list[PageProbeResult]:
    """Batch helper: probe many URLs on one shared browser, bounded concurrency."""
    probe_kwargs = {
        k: kwargs.pop(k)
        for k in ("width", "height", "nav_timeout_ms", "settle_ms", "user_agent")
        if k in kwargs
    }
    call_kwargs = {
        k: kwargs.pop(k)
        for k in ("timeout_ms", "shot", "artifact_id")
        if k in kwargs
    }
    async with PageProbe(**probe_kwargs) as probe:
        return await probe.probe_many(urls, concurrency=concurrency, **call_kwargs)
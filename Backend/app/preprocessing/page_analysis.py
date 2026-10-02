"""Feature extraction for *live* page analysis: layout, behaviour, brand images.

Why this module exists
----------------------
:mod:`app.preprocessing.html_features` reads the fetched markup. That answers
"what did the author write down", never "what does the page actually do once a
script engine has touched it" and never "where did the credential field end up
on screen". Three signals in this project were missing for exactly that reason,
and they are the three implemented here:

``extract_layout_features``
    Login-page geometry. Credential kits are re-laid-out clones: the form is
    dropped into an absolute pixel grid, the password box is the only
    correctly-labelled control, decoy inputs are stacked invisibly on top of
    real ones, and the spacing is not drawn from any vertical scale. All of that
    is invisible in the markup and obvious from rectangles.

``extract_behavior_features``
    Runtime instrumentation. The counters are produced by a script injected via
    ``add_init_script`` *before* navigation (see
    :mod:`app.services.page_probe`), so they describe what the page executed,
    not what it shipped: dynamic code assembly, capture-phase key listeners,
    clipboard writes, form submissions with no user gesture, popups, timer
    driven redirects.

``extract_brand_image_features``
    Brand impersonation from ``<img>`` elements: a PayPal logo served from a
    host that is not PayPal, or a logo whose filename names a bank the page has
    nothing to do with.

The split between collection and extraction is deliberate. This module takes
**plain dicts** and contains no browser code at all, so every feature is unit
testable without a page, and the collector can be re-run against cached probe
output without re-navigating.

Everything here is total: a missing key, a ``None``, a string where a number was
expected, or an entirely malformed record yields zeros rather than an exception,
because the collector hands us hostile and frequently broken pages.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Sequence

__all__ = [
    "LAYOUT_FEATURE_NAMES",
    "N_LAYOUT_FEATURES",
    "extract_layout_features",
    "BEHAVIOR_FEATURE_NAMES",
    "N_BEHAVIOR_FEATURES",
    "extract_behavior_features",
    "BRANDIMG_FEATURE_NAMES",
    "N_BRANDIMG_FEATURES",
    "extract_brand_image_features",
    "DEFAULT_BRANDS",
]

# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

#: Order is load-bearing: the scaler is fitted against this sequence.
LAYOUT_FEATURE_NAMES: tuple[str, ...] = (
    "n_input_fields",
    "n_password_fields",
    "has_password_field",
    "password_vertical_position",
    "form_center_deviation",
    "password_offset_from_viewport_center",
    "n_overlapping_pairs",
    "max_overlap_ratio",
    "n_inputs_below_fold",
    "form_width_fraction",
    "form_aspect_anomaly",
    "page_aspect_anomaly",
    "n_offscreen_inputs",
    "n_hidden_inputs",
    "n_forms",
    "n_form_controls",
    "n_zero_size_controls",
    "n_small_click_targets",
    "mean_control_width",
    "vertical_rhythm_consistency",
)

N_LAYOUT_FEATURES = len(LAYOUT_FEATURE_NAMES)

#: WCAG 2.2 minimum target size. A control smaller than this cannot be aimed at
#: comfortably and is usually either a decoy or a mis-assembled clone.
_MIN_TARGET_PX = 24.0
#: Aspect ratio (w/h) of a conventional centred login card, roughly 360x480.
_REFERENCE_FORM_ASPECT = 0.75
#: Reference content height / width for a page that fits a single screen.
_REFERENCE_PAGE_ASPECT = 1.78

#: Two boxes count as overlapping once the shared area covers this fraction of
#: the smaller one. Click-jacking and invisible decoy fields sit near 1.0.
_OVERLAP_THRESHOLD = 0.2


def extract_layout_features(layout: dict | None) -> list[float]:
    """Return the fixed-length login-layout feature vector for one page.

    ``layout`` is the geometry dictionary produced by the in-page collector in
    :mod:`app.services.page_probe`; it is a plain dict of numbers and rects so
    this function can be exercised with no browser at all. Recognised keys:

    ``viewport_width``, ``viewport_height``, ``scroll_height``, ``doc_width``
        Rendered page geometry, in CSS pixels.
    ``inputs``
        ``[{"x","y","w","h","visible"}, ...]`` in **document** coordinates.
    ``password_fields``
        The subset of ``inputs`` the page presents as a password field.
    ``form``
        ``{"x","y","w","h"}`` of the largest form, or ``None``.
    ``click_targets``
        ``[{"w","h"}, ...]`` rendered sizes of interactive controls.
    ``overlapping_pairs``, ``max_overlap_ratio``
        Pre-computed by the page (O(n^2) in JS is cheaper to ship than to
        reconstruct from rectangles here).
    ``vertical_gaps``
        Spacing between consecutive form controls, used for rhythm regularity.
    ``n_forms``
        Number of ``<form>`` elements.

    Anything absent or non-numeric contributes zero.
    """
    vec = [0.0] * N_LAYOUT_FEATURES
    if not isinstance(layout, dict) or not layout:
        return vec

    try:
        put = lambda i, v: vec.__setitem__(i, _clip(_fnum(v), -1e6, 1e6))

        vw = _fnum(layout.get("viewport_width"))
        vh = _fnum(layout.get("viewport_height"))
        doc_w = _fnum(layout.get("doc_width")) or vw
        scroll_h = _fnum(layout.get("scroll_height")) or vh
        # Guard the division below against a probe that reported no geometry.
        scroll_h = max(scroll_h, vh, 1.0)

        inputs = _rect_list(layout.get("inputs"))
        passwords = _rect_list(layout.get("password_fields"))
        if not passwords:
            # A probe that only stamped ``is_password`` on the input records is
            # still usable; recover the subset rather than losing the signal.
            passwords = [
                item
                for item, raw in zip(inputs, _dict_list(layout.get("inputs")))
                if _truthy(raw.get("is_password"))
            ]

        form = _rect(layout.get("form"))

        put(0, len(inputs))
        put(1, len(passwords))
        put(2, 1.0 if passwords else 0.0)

        # Vertical position of the credential box, 0 = top of page, 1 = bottom.
        # A hand-cloned login page usually places it near the fold.
        if passwords:
            top_pwd = min(passwords, key=lambda r: r["y"] + r["h"] / 2.0)
            put(3, (top_pwd["y"] + top_pwd["h"] / 2.0) / scroll_h)

        # Horizontal centring deviation of the form: 0 when dead centre.
        if form is not None and vw > 0:
            centre = form["x"] + form["w"] / 2.0
            put(4, abs(centre - vw / 2.0) / vw)

        # Distance of the password box from the viewport centre, normalised by
        # the viewport diagonal so the value is comparable across viewports.
        if passwords and vw > 0 and vh > 0:
            pw = passwords[0]
            dx = (pw["x"] + pw["w"] / 2.0) - vw / 2.0
            dy = (pw["y"] + pw["h"] / 2.0) - vh / 2.0
            put(5, math.hypot(dx, dy) / math.hypot(vw, vh))

        put(6, layout.get("overlapping_pairs"))
        put(7, layout.get("max_overlap_ratio"))

        below_fold = 0
        offscreen = 0
        hidden = 0
        raw_inputs = _dict_list(layout.get("inputs"))
        for rect, raw in zip(inputs, raw_inputs):
            if not _truthy(raw.get("visible", True)):
                hidden += 1
            if vh > 0 and rect["y"] + rect["h"] > vh and rect["h"] > 0:
                below_fold += 1
            if (
                rect["y"] + rect["h"] <= 0
                or rect["y"] >= scroll_h
                or rect["x"] + rect["w"] <= 0
                or (doc_w > 0 and rect["x"] >= doc_w)
            ):
                offscreen += 1
        put(8, below_fold)

        if form is not None:
            put(9, form["w"] / vw if vw > 0 else 0.0)
            if form["h"] > 0:
                put(10, abs(form["w"] / form["h"] - _REFERENCE_FORM_ASPECT))
        if doc_w > 0:
            put(11, abs(scroll_h / doc_w - _REFERENCE_PAGE_ASPECT))

        put(12, offscreen)
        put(13, hidden)
        put(14, layout.get("n_forms"))

        controls = _rect_list(layout.get("click_targets"))
        if form is not None:
            inside = 0
            for c in controls:
                cx = c["x"] + c["w"] / 2.0
                cy = c["y"] + c["h"] / 2.0
                if form["x"] <= cx <= form["x"] + form["w"] and form["y"] <= cy <= form["y"] + form["h"]:
                    inside += 1
            put(15, inside)

        zero_size = 0
        small = 0
        widths: list[float] = []
        for c in controls:
            if c["w"] < 1.0 or c["h"] < 1.0:
                zero_size += 1
            else:
                widths.append(c["w"])
            if min(c["w"], c["h"]) < _MIN_TARGET_PX:
                small += 1
        put(16, zero_size)
        put(17, small)
        put(18, sum(widths) / len(widths) if widths else 0.0)

        put(19, _rhythm_consistency(layout.get("vertical_gaps")))
    except Exception:  # pragma: no cover - the contract is totality
        return [0.0] * N_LAYOUT_FEATURES
    return vec


def _rhythm_consistency(gaps: Any) -> float:
    """1.0 for perfectly uniform vertical spacing, decaying as variation grows.

    Legitimate templates draw spacing from a scale (8/16/24px multiples); a
    hand-assembled credential kit usually stacks boxes at arbitrary offsets.
    Fewer than two gaps yields 0.0 because there is no rhythm to measure.
    """
    values = [g for g in (_fnum(x) for x in _seq(gaps)) if g > 0]
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    cv = math.sqrt(var) / mean
    return _clip(1.0 - cv / 2.0, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #

#: Order is load-bearing: the scaler is fitted against this sequence.
BEHAVIOR_FEATURE_NAMES: tuple[str, ...] = (
    "js_error_count",
    "unhandled_rejection_count",
    "window_open_calls",
    "post_load_navigations",
    "form_submits_without_gesture",
    "keyboard_event_intercepts",
    "keystroke_suppressions",
    "paste_event_intercepts",
    "clipboard_access_attempts",
    "permission_requests",
    "fullscreen_requests",
    "eval_calls",
    "function_constructor_calls",
    "dynamic_code_decoders",
    "long_blob_literals",
    "obfuscated_script_hits",
    "dynamic_code_any",
    "script_count",
    "external_script_origins",
    "external_script_ratio",
    "form_action_cross_origin",
    "timer_redirects",
    "total_timers_registered",
    "dialogs_auto_dismissed",
    "frames_with_scripts",
)

N_BEHAVIOR_FEATURES = len(BEHAVIOR_FEATURE_NAMES)

#: Raw counter keys summed into ``dynamic_code_decoders``. Runtime string
#: assembly (base64 decode, escape unescaping, char-code assembly) is the
#: mechanism every obfuscated kit uses to hide its payload from a static read.
_DECODER_KEYS = ("atob_calls", "unescape_calls", "from_char_code_calls")


def extract_behavior_features(behavior: dict | None) -> list[float]:
    """Return the fixed-length runtime-behaviour vector for one page.

    ``behavior`` is the merged counter dictionary built by
    :meth:`app.services.page_probe.PageProbe.probe` from the counters installed
    by the pre-navigation instrumentation script plus the Playwright-level
    dialog and navigation listeners. Counters from every frame are summed,
    because a credential form rendered inside a cross-origin iframe is exactly
    the case where looking only at the top document would miss the behaviour.

    Missing counters contribute zero; a non-dict contributes the whole vector of
    zeros so the fusion layer can mask the modality.
    """
    vec = [0.0] * N_BEHAVIOR_FEATURES
    if not isinstance(behavior, dict) or not behavior:
        return vec

    try:
        put = lambda i, v: vec.__setitem__(i, _clip(_fnum(v), -1e6, 1e6))

        put(0, behavior.get("js_errors"))
        put(1, behavior.get("unhandled_rejections"))
        put(2, behavior.get("window_open_calls"))
        # Same-document navigations seen after load (history API + location
        # assignment) plus real frame navigations observed by the driver.
        if "post_load_navigations" in behavior:
            put(3, behavior["post_load_navigations"])
        else:
            put(3, behavior.get("navigations"))
        put(4, behavior.get("form_submits_without_gesture"))
        put(5, behavior.get("keyboard_event_intercepts"))
        put(6, behavior.get("keystroke_suppressions"))
        put(7, behavior.get("paste_event_intercepts"))
        put(8, behavior.get("clipboard_access_attempts"))

        put(9, _fnum(behavior.get("permission_queries"))
            + _fnum(behavior.get("notification_permission_requests")))
        put(10, behavior.get("fullscreen_requests"))

        put(11, behavior.get("eval_calls"))
        put(12, behavior.get("function_ctor_calls"))
        decoders = sum(_fnum(behavior.get(k)) for k in _DECODER_KEYS)
        put(13, decoders)
        put(14, behavior.get("long_blob_literals"))
        put(15, behavior.get("obfuscated_script_hits"))

        dynamic_any = any(
            _fnum(behavior.get(k)) > 0
            for k in ("eval_calls", "function_ctor_calls", * _DECODER_KEYS,
                      "long_blob_literals", "obfuscated_script_hits")
        )
        put(16, 1.0 if dynamic_any else 0.0)

        script_count = _fnum(behavior.get("script_count"))
        external_origins = _fnum(behavior.get("external_script_origins"))
        put(17, script_count)
        put(18, external_origins)
        external_count = _fnum(behavior.get("external_script_count"))
        put(19, external_count / script_count if script_count > 0 else 0.0)

        put(20, behavior.get("form_action_cross_origin"))
        put(21, behavior.get("timer_redirects"))
        put(22, behavior.get("timers_registered"))
        put(23, behavior.get("dialogs_dismissed"))
        put(24, behavior.get("frames_probed"))
    except Exception:  # pragma: no cover - the contract is totality
        return [0.0] * N_BEHAVIOR_FEATURES
    return vec


# --------------------------------------------------------------------------- #
# Brand images
# --------------------------------------------------------------------------- #

#: Order is load-bearing: the scaler is fitted against this sequence.
BRANDIMG_FEATURE_NAMES: tuple[str, ...] = (
    "n_images",
    "n_images_with_alt",
    "n_images_without_alt",
    "n_brand_token_images",
    "n_brand_host_mismatch_images",
    "brand_impersonation_any",
    "n_brand_in_filename",
    "n_brand_in_alt_text",
    "n_brand_token_hits",
    "n_distinct_brand_impersonations",
    "n_foreign_domain_images",
    "foreign_domain_ratio",
    "n_suspicious_ext_images",
    "n_base64_images",
    "n_tiny_images",
    "n_zero_rendered_images",
    "n_logoish_images",
)

N_BRANDIMG_FEATURES = len(BRANDIMG_FEATURE_NAMES)

#: Brands checked for by default. Overridable per call via ``brands=`` so the
#: model that owns the decision is not hard-coded into the collector.
DEFAULT_BRANDS: tuple[str, ...] = (
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook", "google",
    "gmail", "netflix", "amazon", "facebook", "instagram", "whatsapp",
    "linkedin", "telegram", "dropbox", "coinbase", "binance", "metamask",
    "kraken", "blockchain", "bank", "chase", "wellsfargo", "citibank", "hsbc",
    "barclays", "santander", "revolut", "wise", "dhl", "fedex", "ups", "visa",
    "mastercard", "amex",
)

#: File extensions that have no business being the source of an ``<img>``.
#: ``.svg`` is deliberately absent: it is the normal format for a real logo.
_SUSPICIOUS_IMAGE_EXTS = frozenset(
    {"php", "phtml", "php3", "php4", "php5", "php7", "pht", "phar",
     "html", "htm", "shtml", "asp", "aspx", "jsp", "jspx", "cfm", "cgi",
     "exe", "scr", "com", "bat", "cmd", "dll", "jar", "hta", "vbs"}
)
#: Words in a filename or ``alt`` that mark an image as branding rather than
#: decoration. A ``logo.png``/``verified-badge`` is the impersonation vector.
_LOGOISH_TOKENS = (
    "logo", "brand", "icon", "badge", "seal", "verified", "trust", "secure",
    "security", "official", "support", "helpdesk", "signin", "login",
)
#: Natural size below which an ``<img>`` is a spacer, a tracking pixel, or a
#: loader whose payload never arrived; a real logo is never 2x2.
_TINY_NATURAL_PX = 2

#: A small public-suffix approximation. The point is only to tell
#: ``cdn.evil.tld`` from ``cdn.paypal.com``; resolving every real
#: multi-label suffix is not worth a dependency here.
_MULTI_LABEL_SUFFIXES = frozenset(
    {
        "co.uk", "org.uk", "me.uk", "ac.uk", "gov.uk", "co.jp", "or.jp",
        "ne.jp", "com.au", "net.au", "org.au", "co.nz", "com.br", "com.mx",
        "co.in", "co.za", "com.tr", "com.cn", "com.sg", "com.hk", "com.tw",
        "co.kr", "com.ar", "com.pl", "co.il",
    }
)
_IPV4_RE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")


def extract_brand_image_features(
    images: list[dict] | None,
    *,
    brands: Iterable[str] | None = None,
) -> list[float]:
    """Return the fixed-length logo/brand-impersonation vector for one page.

    ``images`` is the list of image records collected in-page by
    :mod:`app.services.page_probe`. Each record is a plain dict with ``src``,
    ``alt``, ``title``, ``natural_width``, ``natural_height``,
    ``rendered_width``, ``rendered_height`` and ``page_host`` (the host of the
    document, stamped by the collector so this function needs no extra
    argument).

    The headline signal is ``brand_impersonation_any``: an image whose
    ``src``/``alt``/``title`` names a brand that the page's own host does not.
    Everything else is the supporting evidence for that claim.
    """
    vec = [0.0] * N_BRANDIMG_FEATURES
    records = _dict_list(images)
    if not records:
        return vec

    try:
        tokens = tuple(
            str(b).strip().lower()
            for b in (DEFAULT_BRANDS if brands is None else brands)
            if str(b).strip()
        )

        page_host = ""
        for rec in records:
            host = str(rec.get("page_host") or "").strip().lower()
            if host:
                page_host = host
                break
        page_domain = _registrable_domain(page_host)
        page_compact = _compact(page_host)

        put = lambda i, v: vec.__setitem__(i, _clip(_fnum(v), -1e6, 1e6))

        n_images = len(records)
        with_alt = 0
        brand_token_images = 0
        mismatch_images = 0
        brand_in_filename = 0
        brand_in_alt = 0
        brand_hits = 0
        foreign_domain = 0
        suspicious_ext = 0
        base64_images = 0
        tiny_images = 0
        zero_rendered = 0
        logoish = 0
        distinct_impersonations: set[str] = set()

        for rec in records:
            src = str(rec.get("src") or "").strip()
            alt = str(rec.get("alt") or "").strip()
            title = str(rec.get("title") or "").strip()
            haystack = f"{src} {alt} {title}".lower()
            compact = _compact(haystack)
            filename = src.rsplit("/", 1)[-1].split("?", 1)[0].split("#", 1)[0].lower()
            file_compact = _compact(filename)
            alt_text = f"{alt} {title}".lower()
            alt_compact = _compact(alt_text)

            if alt or title:
                with_alt += 1

            hits = 0
            for token in tokens:
                if _token_in(token, haystack, compact):
                    hits += 1
                    if page_host and not _token_in(token, page_host, page_compact):
                        mismatch_images += 1
                        distinct_impersonations.add(token)
            if hits:
                brand_token_images += 1
                brand_hits += hits
                if any(_token_in(token, filename, file_compact) for token in tokens):
                    brand_in_filename += 1
                alt_text = f"{alt} {title}".lower()
                if any(_token_in(token, alt_text, alt_compact) for token in tokens):
                    brand_in_alt += 1

            if any(token in haystack for token in _LOGOISH_TOKENS):
                logoish += 1

            lowered = src.lower()
            if lowered.startswith("data:") or lowered.startswith("blob:"):
                base64_images += 1
            else:
                ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
                if ext in _SUSPICIOUS_IMAGE_EXTS:
                    suspicious_ext += 1
                img_domain = _registrable_domain(_host_of(lowered))
                if img_domain and page_domain and img_domain != page_domain:
                    foreign_domain += 1

            nat_w = _fnum(rec.get("natural_width"))
            nat_h = _fnum(rec.get("natural_height"))
            if 0 < nat_w <= _TINY_NATURAL_PX or 0 < nat_h <= _TINY_NATURAL_PX:
                tiny_images += 1
            # Only meaningful when the collector actually reported a rendered
            # box; an absent key is unknown, not a zero-sized image.
            if "rendered_width" in rec or "rendered_height" in rec:
                if _fnum(rec.get("rendered_width")) < 1.0 or _fnum(rec.get("rendered_height")) < 1.0:
                    zero_rendered += 1

        put(0, n_images)
        put(1, with_alt)
        put(2, n_images - with_alt)
        put(3, brand_token_images)
        put(4, mismatch_images)
        put(5, 1.0 if mismatch_images else 0.0)
        put(6, brand_in_filename)
        put(7, brand_in_alt)
        put(8, brand_hits)
        put(9, len(distinct_impersonations))
        put(10, foreign_domain)
        put(11, foreign_domain / n_images if n_images else 0.0)
        put(12, suspicious_ext)
        put(13, base64_images)
        put(14, tiny_images)
        put(15, zero_rendered)
        put(16, logoish)
    except Exception:  # pragma: no cover - the contract is totality
        return [0.0] * N_BRANDIMG_FEATURES
    return vec


# --------------------------------------------------------------------------- #
# Small total helpers
# --------------------------------------------------------------------------- #


def _fnum(value: Any, default: float = 0.0) -> float:
    """``float(value)`` that returns ``default`` for anything non-finite."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _clip(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _truthy(value: Any) -> bool:
    """JS-style truthiness: the empty string and ``0`` are false."""
    if isinstance(value, str):
        return value.strip() not in {"", "false", "0", "null", "undefined"}
    return bool(value)


def _seq(value: Any) -> Sequence[Any]:
    if isinstance(value, (list, tuple)):
        return value
    return ()


def _dict_list(value: Any) -> list[dict]:
    return [item for item in _seq(value) if isinstance(item, dict)]


def _compact(text: str) -> str:
    """Lowercase text with every non-alphanumeric character dropped.

    Brand names reach us hyphenated and spaced far more often than verbatim
    (``wells-fargo-logo.png``, ``alt="Bank of America"``), so the raw substring
    test alone would miss the exact case this module exists to catch.
    """
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _token_in(token: str, text: str, compact_text: str) -> bool:
    """Brand token present either verbatim or with separators removed."""
    if token in text:
        return True
    flat = re.sub(r"[^a-z0-9]+", "", token)
    return bool(flat) and flat in compact_text


def _rect(value: Any) -> dict[str, float] | None:
    """Normalise one rectangle, or ``None`` if it is not usable."""
    if not isinstance(value, dict):
        return None
    return {
        "x": _fnum(value.get("x")),
        "y": _fnum(value.get("y")),
        "w": _fnum(value.get("w")),
        "h": _fnum(value.get("h")),
    }


def _rect_list(value: Any) -> list[dict[str, float]]:
    out = []
    for item in _seq(value):
        rect = _rect(item)
        if rect is not None:
            out.append(rect)
    return out


def _host_of(url: str) -> str:
    """Hostname of an absolute or protocol-relative URL, ``""`` if unparseable."""
    try:
        from urllib.parse import urlsplit

        candidate = url if "://" in url else f"//{url}" if url.startswith("//") else url
        return (urlsplit(candidate).hostname or "").lower()
    except Exception:
        return ""


def _registrable_domain(host: str) -> str:
    """Best-effort registrable domain, used to compare against the page host.

    Subdomains collapse (``cdn.paypal.com`` -> ``paypal.com``) so a legitimate
    CDN-hosted logo is not mistaken for a foreign one, while
    ``paypal-secure.tk`` collapses to ``paypal-secure.tk`` and does not match.
    """
    host = (host or "").strip().lower().rstrip(".")
    if not host or host.startswith("[") or ":" in host or _IPV4_RE.fullmatch(host):
        return host
    labels = [label for label in host.split(".") if label]
    if len(labels) <= 2:
        return ".".join(labels)
    last_two = ".".join(labels[-2:])
    if last_two in _MULTI_LABEL_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return last_two
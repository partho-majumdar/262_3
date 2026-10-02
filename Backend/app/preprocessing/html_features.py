"""HTML/DOM feature extraction for the page-content modality.

These features are computed from the **fetched page only**. They are deliberately
separate from :mod:`app.preprocessing.url_features`, which is computed from the
URL string: keeping the two apart is what makes the per-modality attribution in
the UI meaningful, and it is what prevents page information from silently
leaking into the URL branch.

Everything here is total: malformed markup, a truncated body, or an empty
document yields zeros rather than an exception, because the collector will
happily hand us hostile and frequently broken HTML.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

__all__ = [
    "HTML_FEATURE_NAMES",
    "N_HTML_FEATURES",
    "extract_html_features",
    "extract_html_features_batch",
    "visible_text_of",
]

_TAG_RE = re.compile(r"<\s*/?\s*([a-zA-Z][a-zA-Z0-9-]*)")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")
_URL_IN_TEXT_RE = re.compile(r"https?://[^\s\"'<>]{4,}", re.IGNORECASE)

#: Keywords that legitimately appear on credential-harvesting pages.
_LOGIN_TOKENS = (
    "login", "log in", "signin", "sign in", "signon", "username", "user name",
    "password", "passwd", "credential", "verify your identity", "session expired",
    "confirm your password", "otp", "one-time", "2fa", "mfa",
)
#: Brand names used for the brand-vs-host mismatch feature.
_BRANDS = (
    "paypal", "apple", "icloud", "microsoft", "google", "gmail", "netflix",
    "amazon", "facebook", "instagram", "whatsapp", "linkedin", "coinbase",
    "binance", "metamask", "kraken", "blockchain", "bank", "chase", "wellsfargo",
    "citibank", "hsbc", "barclays", "santander", "revolut", "wise", "dhl",
    "fedex", "ups", "dropbox", "office365", "outlook", "yahoo", "telegram",
)

#: Order is load-bearing: the scaler is fitted against this sequence.
HTML_FEATURE_NAMES: tuple[str, ...] = (
    "html_length",
    "visible_text_length",
    "n_tags",
    "n_inputs",
    "n_password_inputs",
    "n_forms",
    "n_form_actions_external",
    "n_iframes",
    "n_scripts",
    "n_external_scripts",
    "n_inline_scripts",
    "n_styles",
    "n_links",
    "n_external_links",
    "n_images",
    "n_hidden_inputs",
    "n_submit_buttons",
    "has_meta_refresh",
    "title_length",
    "title_login_keyword",
    "text_login_keyword_hits",
    "n_password_like_strings",
    "brand_in_title",
    "brand_in_text",
    "brand_host_mismatch",
    "n_brand_tokens_in_text",
    "n_external_form_submit",
    "n_social_links",
    "n_obfuscated_elements",
    "n_meta_tags",
    "n_base_tags",
    "text_urls_count",
    "text_external_urls_count",
    "text_digit_ratio",
    "text_upper_ratio",
    "text_non_ascii_ratio",
    "avg_input_maxlength",
    "n_select_elements",
    "n_textareas",
    "n_email_inputs",
    "n_card_like_inputs",
    "n_checkbox_inputs",
    "n_images_without_alt",
    "charset_declared",
    "doc_height_ratio",
)

N_HTML_FEATURES = len(HTML_FEATURE_NAMES)


def _safe(soup, name: str) -> list:
    """``find_all`` that never raises on malformed markup."""
    try:
        return soup.find_all(name)
    except Exception:  # pragma: no cover - BeautifulSoup is lenient already
        return []


def visible_text_of(soup) -> str:
    """Page text with script and style bodies removed."""
    for tag in _safe(soup, "script") + _safe(soup, "style"):
        tag.decompose()
    return _WS_RE.sub(" ", soup.get_text(" ", strip=True))


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for ch in text.lower():
        counts[ch] = counts.get(ch, 0) + 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract_html_features(html: str | None) -> list[float]:
    """Return the fixed-length HTML feature vector for one document.

    A ``None`` or empty body yields an all-zero vector of the correct width, so
    downstream shapes stay stable and the fusion layer can mask the modality.
    """
    vec = [0.0] * N_HTML_FEATURES
    if not html or not html.strip():
        return vec

    from bs4 import BeautifulSoup

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:  # pragma: no cover - fall back to the stdlib parser
        soup = BeautifulSoup(html, "html.parser")

    put = lambda i, v: vec.__setitem__(i, float(v))

    text = visible_text_of(soup)
    low = text.lower()
    title_tag = soup.title.string if (soup.title and soup.title.string) else ""
    title = str(title_tag).strip()
    title_low = title.lower()

    # Resolve the host once; brand mismatch needs it.
    import re as _re
    from urllib.parse import urlsplit
    _m = _re.search(r'https?://[^\s"\'<>]+', html)
    host = ""
    if _m:
        host = (urlsplit(_m.group(0)).hostname or "").lower()

    inputs = _safe(soup, "input")
    forms = _safe(soup, "form")
    scripts = _safe(soup, "script")
    links = _safe(soup, "a")
    images = _safe(soup, "img")
    iframes = _safe(soup, "iframe")
    metas = _safe(soup, "meta")

    def _type_of(inp) -> str:
        return str(inp.get("type") or "text").strip().lower()

    pwd_inputs = [i for i in inputs if _type_of(i) in {"password", "pwd"}]
    email_inputs = [i for i in inputs if _type_of(i) == "email"]
    hidden_inputs = [i for i in inputs if _type_of(i) == "hidden"]
    checkboxes = [i for i in inputs if _type_of(i) in {"checkbox", "radio"}]
    submits = [i for i in inputs if _type_of(i) in {"submit", "button", "image"}]
    selects = _safe(soup, "select")
    textareas = _safe(soup, "textarea")

    external_form_actions = 0
    external_form_submit = 0
    for form in forms:
        action = str(form.get("action") or "").strip()
        if action:
            low_action = action.lower()
            if low_action.startswith(("http://", "https://", "//")):
                external_form_actions += 1
            if not low_action.startswith(("javascript:", "#")):
                external_form_submit += 1

    n_ext_scripts = sum(
        1 for s in scripts if str(s.get("src") or "").strip()
    )
    n_inline_scripts = sum(
        1 for s in scripts if not str(s.get("src") or "").strip() and s.string
    )

    n_ext_links = 0
    social = 0
    _social_hosts = ("facebook.", "twitter.", "x.com", "instagram.", "linkedin.",
                     "youtube.", "telegram.", "whatsapp.", "tiktok.")
    for a in links:
        href = str(a.get("href") or "").strip().lower()
        if href.startswith(("http://", "https://", "//")):
            n_ext_links += 1
        if any(tok in href for tok in _social_hosts):
            social += 1

    # "Obfuscated" proxy: inline handlers / obfuscated attribute names.
    n_obf = 0
    for tag in soup.find_all(True):
        for attr in tag.attrs:
            name = str(attr).lower()
            if "on" == name or name.startswith("on"):
                n_obf += 1
            elif name not in {"class", "id", "style", "href", "src", "alt", "type",
                              "name", "value", "width", "height", "title",
                              "action", "method", "placeholder", "target",
                              "rel", "charset", "content", "lang", "dir",
                              "disabled", "required", "checked", "selected",
                              "maxlength", "minlength", "pattern", "for",
                              "async", "defer", "role", "tabindex", "data"}:
                n_obf += 1

    meta_refresh = 0
    charset_declared = 0
    for m in metas:
        http_equiv = str(m.get("http-equiv") or "").strip().lower()
        if http_equiv == "refresh":
            meta_refresh = 1
        if str(m.get("charset") or "").strip():
            charset_declared = 1

    login_hits = sum(low.count(tok) for tok in _LOGIN_TOKENS)
    title_login = 1.0 if any(tok in title_low for tok in _LOGIN_TOKENS) else 0.0

    # Count password-ish strings anywhere in visible text.
    pwd_like = sum(low.count(tok) for tok in ("password", "passwd", "credential", "otp"))

    host_labels = set(re.split(r"[^a-z0-9]+", host)) - {""}
    brand_in_title = 0.0
    brand_in_text = 0.0
    brand_mismatch = 0.0
    n_brand_tokens = 0
    host_str = host
    for brand in _BRANDS:
        if brand in title_low:
            brand_in_title = 1.0
            if brand not in host_str:
                brand_mismatch = 1.0
        if brand in low:
            brand_in_text = 1.0
            n_brand_tokens += low.count(brand)

    n_tags = len(_TAG_RE.findall(html))
    n_images_no_alt = sum(1 for i in images if not str(i.get("alt") or "").strip())

    maxlengths = [
        float(i.get("maxlength"))
        for i in inputs
        if str(i.get("maxlength") or "").strip().isdigit()
    ]
    avg_maxlength = sum(maxlengths) / len(maxlengths) if maxlengths else 0.0

    n_text = len(text)
    urls_in_text = _URL_IN_TEXT_RE.findall(text)
    n_ext_text_urls = sum(1 for u in urls_in_text if host and host not in u.lower())

    digits = sum(ch.isdigit() for ch in text)
    upper = sum(ch.isupper() for ch in text)
    non_ascii = sum(ord(ch) > 127 for ch in text)

    put(0, len(html))
    put(1, n_text)
    put(2, n_tags)
    put(3, len(inputs))
    put(4, len(pwd_inputs))
    put(5, len(forms))
    put(6, external_form_actions)
    put(7, len(iframes))
    put(8, len(scripts))
    put(9, n_ext_scripts)
    put(10, n_inline_scripts)
    put(11, len(_safe(soup, "style")))
    put(12, len(links))
    put(13, n_ext_links)
    put(14, len(images))
    put(15, len(hidden_inputs))
    put(16, len(submits))
    put(17, meta_refresh)
    put(18, len(title))
    put(19, title_login)
    put(20, login_hits)
    put(21, pwd_like)
    put(22, brand_in_title)
    put(23, brand_in_text)
    put(24, brand_mismatch)
    put(25, n_brand_tokens)
    put(26, external_form_submit)
    put(27, social)
    put(28, n_obf)
    put(29, len(metas))
    put(30, len(_safe(soup, "base")))
    put(31, len(urls_in_text))
    put(32, n_ext_text_urls)
    put(33, digits / n_text if n_text else 0.0)
    put(34, upper / n_text if n_text else 0.0)
    put(35, non_ascii / n_text if n_text else 0.0)
    put(36, avg_maxlength)
    put(37, len(selects))
    put(38, len(textareas))
    put(39, len(email_inputs))
    # name/id/placeholder hints at card capture without inventing a new source.
    card_like = 0
    for i in inputs:
        blob = " ".join(
            str(i.get(a) or "") for a in ("name", "id", "placeholder", "aria-label")
        ).lower()
        if any(t in blob for t in ("card", "cvv", "cvc", "ccnum", "expiry", "cardnumber")):
            card_like += 1
    put(40, card_like)
    put(41, len(checkboxes))
    put(42, n_images_no_alt)
    put(43, charset_declared)
    put(44, math.log1p(n_text) / 10.0)
    return vec


def extract_html_features_batch(documents: Sequence[str | None]) -> list[list[float]]:
    return [extract_html_features(d) for d in documents]

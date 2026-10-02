"""Handcrafted URL features - an explicit, separately-ablatable branch.

Design rule from the project brief: the character model consumes **only** the
raw URL string. These handcrafted features are computed **from the URL string
alone** (no network access, no page content, nothing from the PhiUSIIL
engineered columns) and are fed into a *separate* module whose output is
concatenated explicitly before the classifier head.

This distinction matters for research integrity:

* Features derived from the **URL string** are legitimate input for a URL model.
* Features derived from the **fetched page** (PhiUSIIL's ``URLSimilarityIndex``,
  ``LineOfCode``, ``HasPasswordField``, ...) are *not available at inference
  time* for the URL-only model and are excluded from the headline config. They
  are trained only as a labelled leakage experiment in P2.

Every feature below is a deterministic function of ``url`` alone. The ordered
list of names is exported by :data:`FEATURE_NAMES` and is stored in the
checkpoint so inference cannot silently disagree with training.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence
from urllib.parse import urlsplit

#: Keyword stems commonly used in credential-harvesting URLs. Matched as
#: substrings of the *lowercased full URL* (host + path + query).
SUSPICIOUS_KEYWORDS: tuple[str, ...] = (
    "login",
    "signin",
    "sign-in",
    "log-in",
    "verify",
    "verification",
    "secure",
    "security",
    "account",
    "update",
    "confirm",
    "confirmation",
    "billing",
    "invoice",
    "payment",
    "unlock",
    "suspended",
    "suspend",
    "expired",
    "validate",
    "validation",
    "auth",
    "authorize",
    "recovery",
    "password",
    "passwd",
    "credential",
    "webscr",
    "wallet",
    "banking",
    "support",
    "helpdesk",
    "customer",
    "recover",
    "reactivate",
    "otp",
    "cvv",
    "2fa",
    "mfa",
)

#: Brand names whose presence in a URL that does NOT belong to them is a strong
#: signal. Used only as a *count* of mismatched brand tokens.
BRAND_TOKENS: tuple[str, ...] = (
    "paypal",
    "apple",
    "icloud",
    "microsoft",
    "office365",
    "outlook",
    "amazon",
    "netflix",
    "facebook",
    "instagram",
    "whatsapp",
    "linkedin",
    "twitter",
    "google",
    "gmail",
    "dropbox",
    "docusign",
    "wellsfargo",
    "chase",
    "bankofamerica",
    "citibank",
    "hsbc",
    "barclays",
    "santander",
    "coinbase",
    "binance",
    "metamask",
    "blockchain",
)

_IPV4_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_HEX_HOST_RE = re.compile(r"^[0-9a-fA-F:]+$")

#: Order is part of the checkpoint contract. Never reorder without retraining.
FEATURE_NAMES: tuple[str, ...] = (
    "url_length",
    "host_length",
    "path_length",
    "query_length",
    "n_dots",
    "n_hyphens",
    "n_underscores",
    "n_digits",
    "n_letters",
    "n_special_chars",
    "digit_ratio",
    "special_char_ratio",
    "n_subdomains",
    "is_ip_in_host",
    "is_ipv6_host",
    "is_hex_host",
    "is_https",
    "has_at_symbol",
    "n_query_params",
    "host_entropy",
    "path_entropy",
    "n_suspicious_keywords",
    "has_suspicious_keyword",
    "n_brand_tokens",
    "brand_domain_mismatch",
    "n_consecutive_digits",
    "n_encoded_chars",
    "n_dashes_in_host",
    "host_label_count",
    "max_host_label_length",
)


def shannon_entropy(text: str) -> float:
    """Shannon entropy in bits. Empty input -> 0.0."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))


def _extract_tld(host: str):
    """Cached tldextract instance using the bundled PSL (offline, deterministic)."""
    try:
        import tldextract

        extractor = getattr(_extract_tld, "_ex", None)
        if extractor is None:
            extractor = tldextract.TLDExtract(suffix_list_urls=())
            _extract_tld._ex = extractor  # type: ignore[attr-defined]
        return extractor(host)
    except Exception:  # noqa: BLE001 - feature extraction must never hard-fail
        return None


def registered_domain_of(host: str) -> str:
    """Registrable domain via the bundled PSL snapshot (offline, deterministic)."""
    if not host:
        return ""
    res = _extract_tld(host)
    reg = res.registered_domain if res is not None else ""
    return reg.lower() if reg else host.lower()


def name_label_of(host: str) -> str:
    """The second-level label, i.e. the part a domain owner actually chose.

    ``paypal-secure.tld`` -> ``paypal-secure``; ``paypal.com`` -> ``paypal``.

    Used for brand-mismatch detection: a brand name counts as "owned" only when
    it is the whole name label, not merely a substring of it. Substring matching
    would call ``paypal-secure.tld`` a legitimate PayPal domain, which is exactly
    the impersonation we want to flag.
    """
    if not host:
        return ""
    res = _extract_tld(host)
    d = (res.domain if res is not None else "") or ""
    return d.lower()


def extract_handcrafted_features(url: str) -> list[float]:
    """Compute every feature in :data:`FEATURE_NAMES` from ``url`` alone.

    Returns a list whose length and order always match ``FEATURE_NAMES``. This
    ordering is what the checkpoint records, so a feature-name drift is a hard
    failure at load time rather than a silent prediction shift.
    """
    raw = url or ""
    try:
        parts = urlsplit(raw if "://" in raw else f"//{raw}")
        host = (parts.hostname or "").lower()
        path = parts.path or ""
        query = parts.query or ""
        scheme = (parts.scheme or "").lower()
        # Reconstruct from the original so percent-encoding is counted, not
        # decoded away.
        if "@" in (parts.netloc or ""):
            n_at = (parts.netloc or "").count("@")
        else:
            n_at = raw.count("@")
    except ValueError:
        host, path, query, scheme, n_at = "", raw, "", "", raw.count("@")

    lowered = raw.lower()
    host_labels = [l for l in host.split(".") if l]

    # Subdomain depth: labels minus the registrable domain's own labels.
    reg = registered_domain_of(host)
    n_subdomains = 0
    if reg and host.endswith(reg):
        n_subdomains = max(0, len(host_labels) - len([l for l in reg.split(".") if l]))

    host_digit_dashes = host.count("-")
    max_label = max((len(l) for l in host_labels), default=0)

    digits = sum(c.isdigit() for c in raw)
    letters = sum(c.isalpha() for c in raw)
    special = sum(not c.isalnum() for c in raw)
    n = len(raw) or 1

    n_dots = raw.count(".")
    n_hyphens = raw.count("-")
    n_underscores = raw.count("_")

    # Longest run of consecutive digits (card/bin harvesting, phone numbers).
    max_digit_run, run = 0, 0
    for ch in raw:
        if ch.isdigit():
            run += 1
            max_digit_run = max(max_digit_run, run)
        else:
            run = 0
    n_consecutive_digits = max_digit_run

    n_encoded = len(re.findall(r"%[0-9a-fA-F]{2}", raw))

    kw_hits = [k for k in SUSPICIOUS_KEYWORDS if k in lowered]
    brand_hits = [b for b in BRAND_TOKENS if b in lowered]
    # Mismatch: a brand appears somewhere in the URL, but the registrable domain's
    # own name label is not that brand. `paypal-secure.tld` is an impersonation;
    # `paypal.com` is not. Compared as whole labels, never as substrings.
    name_label = name_label_of(host)
    mismatch = 1.0 if (brand_hits and name_label and not any(name_label == b for b in brand_hits)) else 0.0

    features = [
        float(len(raw)),
        float(len(host)),
        float(len(path)),
        float(len(query)),
        float(n_dots),
        float(n_hyphens),
        float(n_underscores),
        float(digits),
        float(letters),
        float(special),
        digits / n,
        special / n,
        float(n_subdomains),
        1.0 if _IPV4_RE.match(host) else 0.0,
        1.0 if ":" in host else 0.0,
        1.0 if (host and _HEX_HOST_RE.match(host) and not _IPV4_RE.match(host)) else 0.0,
        1.0 if scheme == "https" else 0.0,
        float(n_at),
        float(len([p for p in query.split("&") if p])),
        shannon_entropy(host),
        shannon_entropy(path),
        float(len(kw_hits)),
        1.0 if kw_hits else 0.0,
        float(len(brand_hits)),
        mismatch,
        float(n_consecutive_digits),
        float(n_encoded),
        float(host_digit_dashes),
        float(len(host_labels)),
        float(max_label),
    ]
    if len(features) != len(FEATURE_NAMES):
        raise AssertionError(
            f"feature vector length {len(features)} != {len(FEATURE_NAMES)} declared names"
        )
    return features


def extract_batch(urls: Sequence[str]) -> list[list[float]]:
    return [extract_handcrafted_features(u) for u in urls]


class FeatureScaler:
    """Standardise handcrafted features using training-split statistics only.

    Saved into the checkpoint. At inference the stored ``mean``/``std`` are
    reused, never refitted on the incoming URL.
    """

    def __init__(self) -> None:
        self.mean_: list[float] | None = None
        self.std_: list[float] | None = None

    def fit(self, rows: Sequence[Sequence[float]]) -> "FeatureScaler":
        if not rows:
            raise ValueError("cannot fit FeatureScaler on empty data")
        k = len(rows[0])
        mean = [0.0] * k
        for r in rows:
            for i, v in enumerate(r):
                mean[i] += v
        n = len(rows)
        mean = [m / n for m in mean]
        var = [0.0] * k
        for r in rows:
            for i, v in enumerate(r):
                var[i] += (v - mean[i]) ** 2
        std = [math.sqrt(v / n) for v in var]
        # Constant features must not blow up to 1e6.
        self.mean_ = mean
        self.std_ = [s if s > 1e-8 else 1.0 for s in std]
        return self

    def transform(self, rows: Sequence[Sequence[float]]) -> list[list[float]]:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("FeatureScaler used before fit(); call fit() on training data first")
        out = []
        for r in rows:
            out.append([(r[i] - self.mean_[i]) / self.std_[i] for i in range(len(self.mean_))])
        return out

    def fit_transform(self, rows: Sequence[Sequence[float]]) -> list[list[float]]:
        return self.fit(rows).transform(rows)

    def state_dict(self) -> dict:
        return {"feature_names": list(FEATURE_NAMES), "mean": self.mean_, "std": self.std_}

    @classmethod
    def from_state_dict(cls, state: dict) -> "FeatureScaler":
        if list(state.get("feature_names", [])) != list(FEATURE_NAMES):
            raise ValueError(
                "checkpoint feature names do not match FEATURE_NAMES; the handcrafted "
                "branch changed after training - retrain or restore the old code"
            )
        obj = cls()
        obj.mean_ = list(state["mean"])
        obj.std_ = list(state["std"])
        return obj
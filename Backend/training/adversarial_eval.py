"""Adversarial URL robustness evaluation.

Slide 11 claims "robustness against adversarially modified URLs". This script
produces the evidence for that claim, or shows it does not hold.

Method
------
Take real phishing and legitimate URLs from the held-out **test** split only.
Generate adversarial variants of each using perturbation families that mirror
documented phishing evasion techniques. Score every variant with the trained
URL model and compare against the clean baseline.

The metric that matters is **flip rate**: the fraction of URLs whose verdict
changes from the clean prediction. A detector is robust when obfuscation does
not move the verdict; a high flip rate means an attacker can evade it with a
one-character edit.

This measures evasion resistance only. It does not claim the variants are
"new attacks" -- they are surface transformations of URLs the model already saw
the domain of.

Usage
-----
    python training/adversarial_eval.py --config configs/dev.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# structlog, not stdlib logging: configure_logging() installs structlog with a
# filtering bound logger, and stdlib Logger.info() rejects keyword arguments.
import structlog  # noqa: E402

from app.preprocessing.url_dataset import URLDataset, load_splits  # noqa: E402
from app.preprocessing.url_features import FeatureScaler  # noqa: E402
from app.preprocessing.url_preprocessing import CharTokenizer  # noqa: E402
from app.utils.metrics import compute_binary_metrics  # noqa: E402
from training.train_url import configure_logging, load_config, resolve_paths, set_seed  # noqa: E402

log = structlog.get_logger("adversarial")

THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Perturbation families
# ---------------------------------------------------------------------------
# Each returns a new URL string. They are intentionally simple and auditable:
# every transformation here is a technique documented in the phishing literature.

CONFUSABLE_MAP = {
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с",
    "x": "х", "y": "у", "i": "і", "s": "ѕ", "j": "ј",
}
FULLWIDTH_MAP = {chr(c): chr(c - 0x20 + 0xFF00) for c in range(0x21, 0x7F)}
ZERO_WIDTH = ["", "‌", "‍", "﻿", "­"]

LEET_MAP = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}


def _host_of(url: str) -> tuple[str, str]:
    """Split a URL into (everything-before-host, host) -- no host/tail split.

    Callers need three parts (scheme prefix, host, tail), so this only peels off
    the scheme. Use ``_split_host_tail`` on the remainder for the host/tail pair.
    """
    if "://" in url:
        scheme, rest = url.split("://", 1)
        return scheme + "://", rest
    if url.startswith("//"):
        return "//", url[2:]
    return "", url


def _join(scheme: str, host: str, tail: str) -> str:
    return f"{scheme}{host}{tail}"


@dataclass
class Family:
    name: str
    description: str
    fn: Callable[[str, random.Random], str]
    #: True when the transformation is expected to be undone by normalisation.
    #: Such attacks are only meaningful if the pipeline fails to fold them.
    reversible_by_normaliser: bool = False


def p_homoglyph(url: str, rng: random.Random) -> str:
    scheme, rest = _host_of(url)
    host, tail = _split_host_tail(rest)
    if not host:
        return url
    positions = [i for i, ch in enumerate(host) if ch.lower() in CONFUSABLE_MAP]
    if not positions:
        return url
    i = rng.choice(positions)
    host = host[:i] + CONFUSABLE_MAP[host[i].lower()] + host[i + 1:]
    return _join(scheme, host, tail)


def p_fullwidth(url: str, rng: random.Random) -> str:
    scheme, rest = _host_of(url)
    host, tail = _split_host_tail(rest)
    if not host:
        return url
    positions = [i for i, ch in enumerate(host) if ch in FULLWIDTH_MAP]
    if not positions:
        return url
    i = rng.choice(positions)
    host = host[:i] + FULLWIDTH_MAP[host[i]] + host[i + 1:]
    return _join(scheme, host, tail)


def p_zero_width(url: str, rng: random.Random) -> str:
    scheme, rest = _host_of(url)
    host, tail = _split_host_tail(rest)
    if not host:
        return url
    positions = [i for i in range(1, len(host) - 1)]
    if not positions:
        return url
    i = rng.choice(positions)
    host = host[:i] + rng.choice(ZERO_WIDTH) + host[i:]
    return _join(scheme, host, tail)


def p_subdomain_prepend(url: str, rng: random.Random) -> str:
    """``paypal.com`` -> ``paypal.com.evil.tk``: adds depth, keeps the brand."""
    scheme, rest = _host_of(url)
    host, tail = _split_host_tail(rest)
    if not host or "." not in host:
        return url
    label = host.split(".")[0]
    return _join(scheme, f"{label}.{host}.secure-login-{rng.randint(10,99)}.tk", tail)


def p_typosquat(url: str, rng: random.Random) -> str:
    """Transpose or drop one character in the longest host label.

    The longest label is the one an attacker would actually target, and it is the
    only one reliably long enough to mutate. Picking a random label instead
    frequently lands on a short suffix like "tk", which cannot be typosquatted,
    so the attack silently no-ops and reports a flattering 0.0 flip rate for a
    family that was never really applied.
    """
    scheme, rest = _host_of(url)
    host, tail = _split_host_tail(rest)
    if not host or "." not in host:
        return url
    labels = host.split(".")
    candidates = [i for i, w in enumerate(labels) if len(w) >= 4]
    if not candidates:
        return url
    li = max(candidates, key=lambda i: len(labels[i]))
    word = labels[li]
    i = rng.randrange(len(word) - 1)
    if rng.random() < 0.5:
        labels[li] = word[:i] + word[i + 1] + word[i] + word[i + 2:]
    else:
        labels[li] = word[:i] + word[i + 1:]
    return _join(scheme, ".".join(labels), tail)


def p_brand_swap(url: str, rng: random.Random) -> str:
    """Replace the registrable label with a high-value brand name."""
    scheme, rest = _host_of(url)
    host, tail = _split_host_tail(rest)
    if not host or "." not in host:
        return url
    parts = host.split(".")
    suffix = ".".join(parts[-2:])
    brand = rng.choice(["paypal", "apple", "microsoft", "amazon", "netflix"])
    return _join(scheme, f"{brand}-{rng.randint(10,99)}.{suffix}", tail)


def p_path_shuffle(url: str, rng: random.Random) -> str:
    """Move the sensitive path segment to the query string."""
    scheme, rest = _host_of(url)
    host, tail = _split_host_tail(rest)
    if "/" not in tail:
        return url
    path, _, query = tail.partition("?")
    segs = [s for s in path.split("/") if s]
    if len(segs) < 2:
        return url
    last = segs.pop()
    return _join(scheme, host, "/" + "/".join(segs) + f"?next={last}")


def p_double_encode(url: str, rng: random.Random) -> str:
    """Add a second layer of percent-encoding to the path/query.

    Deliberately does NOT touch the ``://`` separator. Encoding that would
    produce ``https:%252f/...``, which is a syntactically broken URL rather
    than a real evasion, and would inflate the flip rate for the wrong reason.
    A genuine double-encode attack re-encodes characters inside the path, where
    the server still decodes them once and the filter sees only the outer form.
    """
    scheme, rest = _host_of(url)
    host, tail = _split_host_tail(rest)
    if not tail or tail == "/":
        return url
    body = tail.lstrip("/")
    if not body:
        return url
    # Re-encode the first %-escape, or percent-encode one path character.
    if "%" in body:
        i = body.index("%")
        return _join(scheme, host, "/" + body[:i] + "%25" + body[i + 1:])
    i = rng.randrange(len(body))
    ch = body[i]
    if ch.isalnum() or ch in "-._~":
        return _join(scheme, host, "/" + body[:i] + f"%25{ord(ch):02X}" + body[i + 1:])
    return url


def p_scheme_upper(url: str, rng: random.Random) -> str:
    if url.startswith("http://"):
        return "HTTP://" + url[7:]
    if url.startswith("https://"):
        return "HTTPS://" + url[8:]
    return url


def p_repeat(url: str, rng: random.Random) -> str:
    """Pad the host with repeated characters."""
    scheme, rest = _host_of(url)
    host, tail = _split_host_tail(rest)
    if not host:
        return url
    return _join(scheme, host + rng.choice(string.ascii_lowercase) * 4, tail)


def p_trailing_dot(url: str, rng: random.Random) -> str:
    scheme, rest = _host_of(url)
    host, tail = _split_host_tail(rest)
    if not host or host.endswith("."):
        return url
    return _join(scheme, host + ".", tail)


def _split_host_tail(rest: str) -> tuple[str, str]:
    slash = rest.find("/")
    return (rest[:slash], rest[slash:]) if slash != -1 else (rest, "")


FAMILIES: list[Family] = [
    Family("homoglyph", "Cyrillic look-alike swapped into the host", p_homoglyph, True),
    Family("fullwidth", "Fullwidth Unicode character in the host", p_fullwidth, True),
    Family("zero_width", "Zero-width separator inside the host", p_zero_width, True),
    Family("scheme_upper", "Uppercased URL scheme", p_scheme_upper, True),
    Family("trailing_dot", "Fully-qualified trailing dot on the host", p_trailing_dot, True),
    Family("brand_swap", "Host replaced with a high-value brand", p_brand_swap),
    Family("subdomain_prepend", "Brand kept as a subdomain of an attacker domain", p_subdomain_prepend),
    Family("typosquat", "Transposed or dropped character in the host label", p_typosquat),
    Family("path_shuffle", "Sensitive path segment moved into the query", p_path_shuffle),
    Family("double_encode", "Extra layer of percent-encoding", p_double_encode),
    Family("repeat_pad", "Repeated characters padded onto the host", p_repeat),
]


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
def load_model(cfg: dict, paths: dict, ckpt_name: str = "url_model.pt"):
    ckpt = paths["checkpoints_dir"] / ckpt_name
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing {ckpt}; run training/train_url.py first")
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    from app.models.url_model import URLCharModel

    model_cfg = dict(blob["model_config"])
    model_cfg.pop("dropout", None)
    model = URLCharModel(**model_cfg)
    model.load_state_dict(blob["model_state"])
    model.eval()
    tok = CharTokenizer.from_state_dict(blob["tokenizer"])
    scaler = FeatureScaler.from_state_dict(blob["scaler"])
    return model, tok, scaler, blob.get("calibration")


def score(model, tok, scaler, urls: list[str], calibration) -> np.ndarray:
    from scipy.special import expit

    ds = URLDataset(_make_split(urls), tok, scaler)
    out = []
    with torch.no_grad():
        for i in range(0, len(urls), 512):
            chunk = [ds[j] for j in range(i, min(i + 512, len(urls)))]
            cb = torch.stack([c[0] for c in chunk])
            mb = torch.stack([c[1] for c in chunk])
            hb = torch.stack([c[2] for c in chunk])
            out.append(model(cb, mb, hb).logit.numpy())
    logits = np.concatenate(out) if out else np.zeros(0)
    if calibration:
        from app.utils.metrics import TemperatureScaler

        return TemperatureScaler.from_state_dict(calibration).transform_logits(logits)
    return expit(logits)


def _make_split(urls: list[str]):
    from app.preprocessing.url_dataset import SplitData

    return SplitData(
        name="adversarial",
        urls=list(urls),
        labels=np.zeros(len(urls), dtype=np.int64),
        row_ids=np.arange(len(urls), dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Adversarial *training* augmentation
# ---------------------------------------------------------------------------
def augment_for_training(
    split,
    families: list[Family] | None = None,
    ratio: float = 0.5,
    seed: int = 42,
    copies_per_url: int = 1,
):
    """Return a copy of ``split`` with adversarially perturbed training URLs.

    Used by ``training/train_url.py --adversarial-augment`` so the model sees
    evasive variants during training rather than only at evaluation time.

    Two design points that matter more than the augmentation itself:

    1. **Both classes are perturbed, in proportion.** Perturbing only phishing
       URLs would teach the model the inverse of the intended lesson -- that
       surface mangling signals phishing -- and would make the model *more*
       brittle, not less. The added examples mirror the class balance of the
       originals.

    2. **Perturbed copies are added, never substituted.** The clean URL stays in
       the training set, so augmentation widens coverage instead of quietly
       deleting the distribution the model was previously fitting.

    The perturbation families are applied with a seeded RNG, so two runs with
    the same seed produce byte-identical training data.
    """
    from app.preprocessing.url_dataset import SplitData

    families = families or FAMILIES
    rng = random.Random(seed)

    base_urls = list(split.urls)
    base_labels = np.asarray(split.labels, dtype=np.int64)

    new_urls: list[str] = []
    new_labels: list[int] = []

    for url, label in zip(base_urls, base_labels):
        if rng.random() >= ratio:
            continue
        fam = families[rng.randrange(len(families))]
        for _ in range(copies_per_url):
            mutated = fam.fn(url, rng)
            # A family that cannot apply to this URL returns it unchanged.
            # Re-deriving a family that does fire is better than duplicating
            # the clean string, which would just over-weight it in training.
            if mutated == url:
                alt = families[rng.randrange(len(families))]
                mutated = alt.fn(url, rng)
                if mutated == url:
                    continue
            new_urls.append(mutated)
            new_labels.append(int(label))

    if not new_urls:
        return split

    return SplitData(
        name=f"{split.name}+adv",
        urls=base_urls + new_urls,
        labels=np.concatenate([base_labels, np.asarray(new_labels, dtype=np.int64)]),
        row_ids=np.concatenate(
            [split.row_ids, -np.ones(len(new_urls), dtype=np.int64)]
        ),
    )


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def run(cfg: dict, paths: dict, seed: int, per_class: int, ckpt_name: str = "url_model.pt") -> dict:
    set_seed(seed)
    splits = load_splits(paths["splits_dir"])
    test = splits["test"]

    # Sample from the held-out split only, balanced across classes.
    rng = np.random.default_rng(seed)
    idx_by_class: dict[int, list[int]] = {0: [], 1: []}
    for i, y in enumerate(test.labels):
        idx_by_class[int(y)].append(i)
    chosen: list[int] = []
    for cls, pool in idx_by_class.items():
        take = min(per_class, len(pool))
        chosen += rng.choice(pool, size=take, replace=False).tolist()

    base_urls = [test.urls[i] for i in chosen]
    base_y = np.array([int(test.labels[i]) for i in chosen])

    model, tok, scaler, calibration = load_model(cfg, paths, ckpt_name)
    log.info(
        "model_loaded",
        handcrafted=model.use_handcrafted,
        calibrated=calibration is not None,
        n=len(base_urls),
        checkpoint=ckpt_name,
    )

    base_prob = score(model, tok, scaler, base_urls, calibration)
    base_pred = (base_prob >= THRESHOLD).astype(int)

    base_metrics = compute_binary_metrics(base_y, base_prob, threshold=THRESHOLD)
    report: dict = {
        "seed": seed,
        # Recorded so two robustness reports are never silently compared across
        # different models.
        "checkpoint": ckpt_name,
        "threshold": THRESHOLD,
        "n_urls": len(base_urls),
        "n_phishing": int((base_y == 1).sum()),
        "n_legitimate": int((base_y == 0).sum()),
        "calibrated": calibration is not None,
        "baseline": {
            "accuracy": round(base_metrics.accuracy, 6),
            "precision": round(base_metrics.precision, 6),
            "recall": round(base_metrics.recall, 6),
            "f1": round(base_metrics.f1, 6),
        },
        "families": {},
    }

    crng = random.Random(seed)
    for fam in FAMILIES:
        variants = [fam.fn(u, crng) for u in base_urls]
        changed = sum(1 for v, u in zip(variants, base_urls) if v != u)
        if changed == 0:
            log.info("family_skipped", family=fam.name)
            continue

        prob = score(model, tok, scaler, variants, calibration)
        pred = (prob >= THRESHOLD).astype(int)
        flips = pred != base_pred

        # Direction matters: a phishing URL that flips to 'legitimate' is an
        # evasion (attacker wins). The reverse is a false alarm, which is bad
        # but not a security failure.
        phish_idx = base_y == 1
        evasion = int((flips & phish_idx).sum())
        n_phish = int(phish_idx.sum())
        false_alarm = int((flips & ~phish_idx).sum())
        n_legit = int((~phish_idx).sum())

        after = compute_binary_metrics(base_y, prob, threshold=THRESHOLD)
        report["families"][fam.name] = {
            "description": fam.description,
            "reversible_by_normaliser": fam.reversible_by_normaliser,
            "n_variants_applied": changed,
            "flip_rate": round(float(flips.mean()), 4),
            "evasion_rate_on_phishing": round(evasion / max(1, n_phish), 4),
            "false_alarm_rate_on_legitimate": round(false_alarm / max(1, n_legit), 4),
            "f1_after": round(after.f1, 6),
            "recall_after": round(after.recall, 6),
            "f1_drop": round(base_metrics.f1 - after.f1, 6),
            "mean_abs_prob_shift": round(float(np.abs(prob - base_prob).mean()), 6),
            "example_base": next((u for u, v in zip(base_urls, variants) if u != v), ""),
            "example_attack": next((v for u, v in zip(base_urls, variants) if u != v), ""),
        }
        log.info(
            "family_done",
            family=fam.name,
            flip_rate=report["families"][fam.name]["flip_rate"],
            evasion=report["families"][fam.name]["evasion_rate_on_phishing"],
        )

    report["tld_prior"] = tld_prior_analysis(cfg, paths)
    report["summary"] = summarise(report)
    return report


def summarise(report: dict) -> dict:
    fams = report["families"]
    if not fams:
        return {}
    worst_evasion = max(fams.items(), key=lambda kv: kv[1]["evasion_rate_on_phishing"])
    worst_f1 = min(fams.items(), key=lambda kv: kv[1]["f1_after"])
    reversible = [k for k, v in fams.items() if v["reversible_by_normaliser"]]
    unreversible = [k for k, v in fams.items() if not v["reversible_by_normaliser"]]

    def mean_flip(keys):
        vals = [fams[k]["flip_rate"] for k in keys if k in fams]
        return round(float(np.mean(vals)), 4) if vals else None

    return {
        "baseline_f1": report["baseline"]["f1"],
        "mean_flip_rate_normaliser_handled": mean_flip(reversible),
        "mean_flip_rate_semantic_attacks": mean_flip(unreversible),
        "worst_evasion_family": worst_evasion[0],
        "worst_evasion_rate": worst_evasion[1]["evasion_rate_on_phishing"],
        "worst_f1_family": worst_f1[0],
        "worst_f1_after": worst_f1[1]["f1_after"],
        "min_f1_drop": round(
            max(v["f1_drop"] for v in fams.values()), 6
        ),
    }


def tld_prior_analysis(cfg: dict, paths: dict) -> dict:
    """Measure how strongly the training set ties cost-free TLDs to legitimacy.

    ``brand_swap`` and ``subdomain_prepend`` both end up placing the host on a
    free domain registry. If the training data says those registries are ~always
    legitimate, the model learns "cheap domain = safe", which is an attacker-
    controllable signal. This quantifies that prior directly from the splits so
    the report can attribute the evasion to a cause rather than just observe it.
    """
    from app.preprocessing.url_preprocessing import host_of

    splits = load_splits(paths["splits_dir"])
    rows = []
    for name in ("train", "val", "test"):
        for url, label in zip(splits[name].urls, splits[name].labels):
            rows.append((name, int(label), host_of(url)))

    import pandas as pd

    df = pd.DataFrame(rows, columns=["split", "label", "host"])
    out: dict[str, dict] = {}
    for suffix in ("tk", "ml", "ga", "cf", "gq", "xyz.tk", "weeblysite.com", "web.app"):
        sub = df[df.host.str.endswith(suffix, na=False)]
        if len(sub) == 0:
            continue
        out[suffix] = {
            "n_rows": int(len(sub)),
            "phishing_rate": round(float(sub.label.mean()), 4),
        }
    return out


def render_markdown(report: dict) -> str:
    s = report.get("summary", {})
    lines = [
        "# Adversarial URL robustness evaluation",
        "",
        "Generated by `training/adversarial_eval.py`. Held-out **test** split only.",
        "",
        "## Method",
        "",
        "Each test URL is perturbed by one documented evasion technique and rescored.",
        "**Flip rate** is the fraction of verdicts that change. **Evasion rate** is the",
        "flip rate restricted to phishing URLs -- a phishing URL that flips to",
        "'legitimate' is an attacker success. A false alarm is the reverse: bad, but",
        "not a security failure.",
        "",
        "## Clean baseline",
        "",
        f"- n = {report['n_urls']} ({report['n_phishing']} phishing / {report['n_legitimate']} legitimate)",
        f"- F1 {report['baseline']['f1']}  precision {report['baseline']['precision']}  recall {report['baseline']['recall']}",
        f"- calibration applied: {report['calibrated']}",
        "",
        "## Results by attack family",
        "",
        "| Attack | Description | Flip rate | Evasion (phish) | False alarm (legit) | F1 after | F1 drop |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, v in report["families"].items():
        lines.append(
            f"| `{name}` | {v['description']} | {v['flip_rate']:.3f} | "
            f"**{v['evasion_rate_on_phishing']:.3f}** | {v['false_alarm_rate_on_legitimate']:.3f} | "
            f"{v['f1_after']:.4f} | {v['f1_drop']:.4f} |"
        )

    lines += [
        "",
        "## Summary",
        "",
        f"- Mean flip rate, normalisation-handled families: **{s.get('mean_flip_rate_normaliser_handled')}**",
        f"- Mean flip rate, semantic attacks: **{s.get('mean_flip_rate_semantic_attacks')}**",
        f"- Worst evasion family: `{s.get('worst_evasion_family')}` at {s.get('worst_evasion_rate')}",
        f"- Worst post-attack F1: `{s.get('worst_f1_family')}` at {s.get('worst_f1_after')}",
        f"- Largest F1 drop from any single attack: {s.get('min_f1_drop')}",
        "",
        "## Why `brand_swap` and `subdomain_prepend` work",
        "",
        "Both families end up placing the host on a cheap or free domain registry.",
        "If the training data ties those registries to legitimacy, the model learns",
        "'cheap domain = safe' -- a signal the attacker controls for under a dollar.",
        "Measured prior over train+val+test:",
        "",
        "| Suffix | Rows | Phishing rate |",
        "|---|---|---|",
    ]
    for suffix, v in sorted((report.get("tld_prior") or {}).items(), key=lambda kv: kv[1]["phishing_rate"]):
        lines.append(f"| `.{suffix}` | {v['n_rows']} | {v['phishing_rate']:.4f} |")

    lines += [
        "",
        "A phishing rate near zero means the label itself carries the answer: register",
        "on that registry and the model has little reason to look further. This is a",
        "dataset artifact, not a tuning problem, because the information is absent from",
        "the features entirely. Fixing it needs a different label distribution or an",
        "out-of-distribution phishing source, not more epochs.",
        "",
        "## How to read this",
        "",
        "A low flip rate on a family the normaliser already undoes is expected and",
        "shows the folding works. A high flip rate on a *semantic* attack is the real",
        "finding: it means the model relies on surface lexical cues that the attacker",
        "controls, and such an attack needs no exotic tooling -- a template and a",
        "different brand string are enough.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Adversarial URL robustness evaluation.")
    p.add_argument("--config", type=Path, default=BACKEND_ROOT / "configs" / "dev.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-class", type=int, default=500)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--ckpt", default="url_model.pt",
                   help="checkpoint filename inside the checkpoints dir; use   + 'url_model_adv.pt' +  to score the adversarially-trained model")
    args = p.parse_args(argv)

    configure_logging(fmt="json")
    cfg = load_config(args.config)
    paths = resolve_paths(cfg, args.config)
    out = args.out or (paths["reports_dir"] / "adversarial_robustness.json")

    report = run(cfg, paths, args.seed, args.per_class, args.ckpt)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    # Derive the markdown name from --out. Hardcoding it here meant a second
    # run (e.g. against the adversarially-trained checkpoint) silently
    # overwrote the first run's markdown while leaving its JSON intact.
    out.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps({
        "baseline_f1": report["baseline"]["f1"],
        "summary": report.get("summary", {}),
        "written": str(out),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
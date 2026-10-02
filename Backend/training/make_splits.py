"""P1 - create the train/val/test split, grouped by registered domain.

Why grouped: PhiUSIIL contains many URLs per domain. A random row-level split
leaks the domain identity across train and test, which inflates every metric
and makes the reported numbers meaningless. Splitting at the *domain* level and
moving whole domains guarantees no registered domain appears in two splits.

Splitting strategy: per-class, deterministic domain shuffling followed by
cumulative-ratio assignment (a single deterministic fold of the idea behind
StratifiedGroupKFold). Every step is seeded, so the split is reproducible from
``--seed`` alone.

Outputs (all under ``data/splits/``)
------------------------------------
``train.txt`` / ``val.txt`` / ``test.txt``
    One **row index** per line, referring to the deduplicated row order produced
    by this script (order is deterministic and documented in ``dedup.csv``).
``domain_split.csv``
    Every registered domain and the split it was assigned to. This is the
    authoritative grouping record used by the leakage tests.
``split_summary.json``
    Class balance, domain counts, overlap checks - the real numbers.
``dedup.csv``
    The deduplicated URL/label table with its stable row ids, so indices are
    reproducible without re-reading the source CSV.

Usage
-----
    python training/make_splits.py --csv path/to/PhiUSIIL.csv --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pandas as pd  # noqa: E402

from training.inspect_dataset import detect_columns

SPLITS: tuple[str, ...] = ("train", "val", "test")


# ---------------------------------------------------------------------------
@dataclass
class SplitConfig:
    seed: int = 42
    ratios: dict[str, float] | None = None
    label_column: str | None = None
    url_column: str | None = None

    def __post_init__(self) -> None:
        if self.ratios is None:
            self.ratios = {"train": 0.70, "val": 0.15, "test": 0.15}


def registered_domain_of(host: str) -> str:
    """Extract the registrable domain using the bundled PSL snapshot.

    ``suffix_list_urls=()`` disables the network fetch so the grouping is
    reproducible offline and inside tests. The extractor is built once and
    cached: constructing it per hostname costs seconds over ~175k domains.
    """
    if not host:
        return ""
    extractor = getattr(registered_domain_of, "_extractor", None)
    if extractor is None:
        import tldextract

        extractor = tldextract.TLDExtract(suffix_list_urls=())
        registered_domain_of._extractor = extractor  # type: ignore[attr-defined]
    reg = extractor(host).registered_domain
    return reg.lower() if reg else host.strip().lower()


def build_domain_index(host: str) -> str:
    """Stable, case-insensitive host -> registered domain mapping (cached)."""
    cache = getattr(build_domain_index, "_cache", None)
    if cache is None:
        cache = {}
        setattr(build_domain_index, "_cache", cache)
    key = host.lower()
    if key not in cache:
        cache[key] = registered_domain_of(key)
    return cache[key]


# ---------------------------------------------------------------------------
def load_and_deduplicate(
    csv_path: Path,
    url_column: str | None,
    label_column: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load URL + label, drop exact duplicate rows and duplicate URLs.

    The returned frame has a stable ``row_id`` column (0..n-1) which is the
    coordinate system used by the split index files.
    """
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    header = pd.read_csv(csv_path, nrows=0)
    cols = [str(c) for c in header.columns]

    url_col = url_column or _pick_url_col(csv_path, cols)
    label_col = label_column or _pick_label_col(csv_path, cols)

    df = pd.read_csv(csv_path, usecols=[url_col, label_col], dtype=str)
    raw_rows = len(df)
    df.columns = ["url", "label_raw"]
    df = df.dropna(subset=["url", "label_raw"])
    after_null_drop = len(df)

    before_exact = len(df)
    df = df.drop_duplicates(subset=["url", "label_raw"], keep="first")
    dropped_exact = before_exact - len(df)

    before_url = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first")
    dropped_dup_url = before_url - len(df)

    df = df.reset_index(drop=True)
    df["row_id"] = df.index.astype("int64")

    from urllib.parse import urlsplit

    hosts = []
    malformed = 0
    for u in df["url"]:
        try:
            h = urlsplit(u).hostname or ""
        except ValueError:
            h = ""
        if not h:
            malformed += 1
        hosts.append(h)
    df["host"] = hosts
    df["registered_domain"] = [build_domain_index(h) for h in hosts]

    stats = {
        "raw_rows": int(raw_rows),
        "rows_after_null_drop": int(after_null_drop),
        "dropped_exact_duplicate_rows": int(dropped_exact),
        "dropped_duplicate_url_rows": int(dropped_dup_url),
        "final_rows": int(len(df)),
        "rows_with_unparseable_host": int(malformed),
        "url_column": url_col,
        "label_column": label_col,
    }
    return df, stats


def _pick_url_col(csv_path: Path, cols: list[str]) -> str:
    head = pd.read_csv(csv_path, nrows=500)
    best: tuple[float, str] | None = None
    for c in cols:
        if pd.api.types.is_numeric_dtype(head[c]):
            continue
        rate = float(head[c].dropna().astype(str).str.match(r"^\s*https?://", case=False).mean())
        if rate >= 0.5 and (best is None or rate > best[0]):
            best = (rate, c)
    if best is None:
        raise RuntimeError(f"No URL column detected among {cols}")
    return best[1]


def _pick_label_col(csv_path: Path, cols: list[str]) -> str:
    """Delegate to the shared detector in ``inspect_dataset``.

    Kept as a thin wrapper so callers of this module do not have to import the
    inspection script directly. See ``detect_columns`` for why the naive
    "first low-cardinality column" heuristic is unsafe on this dataset.
    """
    _, label_col, _ = detect_columns(csv_path)
    if not label_col:
        raise RuntimeError(f"No binary label column detected among {cols}")
    return label_col


# ---------------------------------------------------------------------------
def grouped_split(
    df: pd.DataFrame,
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign every registered domain to exactly one split.

    A domain is the atomic unit: it is never split across two subsets, and it is
    never assigned twice. The assignment is a single greedy pass over domains
    that respects **both** the overall row target and the per-class row target,
    because a phishing-heavy domain must not be allowed to unbalance a split.

    Cost of placing domain ``d`` into split ``s`` is the relative overshoot of
    every target it would exceed:

        cost(s, d) = max(0, overshoot of split total)
                   + sum_c max(0, overshoot of split-class total)

    The domain goes to the cheapest split. Ties break on the fixed SPLITS order,
    so the result is deterministic for a given ``seed``.
    """
    rng = random.Random(seed)
    if "y" not in df.columns:
        df = df.copy()
        df["y"] = df["label_raw"].astype(str)

    # Per-domain, per-class row counts.
    domain_counts: dict[str, dict[str, int]] = {}
    for domain, part in df.groupby("registered_domain"):
        domain_counts[str(domain)] = {
            str(k): int(v) for k, v in part["y"].value_counts().items()
        }

    classes = sorted({c for counts in domain_counts.values() for c in counts})
    class_totals = {
        c: sum(counts.get(c, 0) for counts in domain_counts.values()) for c in classes
    }
    n_rows = int(sum(sum(v.values()) for v in domain_counts.values()))

    target_total = {s: ratios[s] * n_rows for s in SPLITS}
    target_class = {
        (s, c): ratios[s] * class_totals[c] for s in SPLITS for c in classes
    }

    cur_total = {s: 0 for s in SPLITS}
    cur_class = {(s, c): 0 for s in SPLITS for c in classes}
    n_domains = {s: 0 for s in SPLITS}

    def cost(split: str, counts: dict[str, int]) -> float:
        size = sum(counts.values())
        total_overshoot = max(
            0.0, (cur_total[split] + size - target_total[split]) / max(target_total[split], 1.0)
        )
        class_overshoot = 0.0
        for c, k in counts.items():
            class_overshoot += max(
                0.0,
                (cur_class[(split, c)] + k - target_class[(split, c)])
                / max(target_class[(split, c)], 1.0),
            )
        return total_overshoot + class_overshoot

    # Largest domains first so a few huge domains cannot dominate a split;
    # ties broken by a seeded shuffle so different seeds give different splits.
    order = sorted(domain_counts, key=lambda d: (-sum(domain_counts[d].values()), d))
    rng.shuffle(order)
    order.sort(key=lambda d: -sum(domain_counts[d].values()))

    assignment: dict[str, str] = {}
    for domain in order:
        counts = domain_counts[domain]
        size = sum(counts.values())

        # Guarantee every split receives at least one domain when the data
        # allows it, otherwise a tiny split could end up empty.
        empty = [s for s in SPLITS if n_domains[s] == 0]
        if empty and len(domain_counts) >= len(SPLITS) * 2:
            # Pick the emptiest-by-target split to keep balance reasonable.
            best = max(empty, key=lambda s: (target_total[s], -SPLITS.index(s)))
        else:
            best = min(SPLITS, key=lambda s: (cost(s, counts), SPLITS.index(s)))

        assignment[domain] = best
        cur_total[best] += size
        n_domains[best] += 1
        for c, k in counts.items():
            cur_class[(best, c)] += k

    return assignment


def apply_split(df: pd.DataFrame, domain_split: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Materialise per-split row frames from the domain assignment."""
    assigned = df["registered_domain"].map(domain_split)
    unknown = int(assigned.isna().sum())
    if unknown:
        raise RuntimeError(f"{unknown} rows were not assigned to any split")
    out = {s: df[assigned == s].reset_index(drop=True) for s in SPLITS}
    return out


def verify_no_overlap(parts: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Assert the grouped-split invariant and return the measured evidence."""
    sets = {s: set(p["registered_domain"]) for s, p in parts.items()}
    row_sets = {s: set(p["row_id"]) for s, p in parts.items()}
    overlaps: dict[str, dict[str, list[str]]] = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            shared = sorted(sets[a] & sets[b])
            if shared:
                overlaps[f"{a}|{b}"] = shared[:20]
    total_rows = sum(len(p) for p in parts.values())
    unique_rows = len(set().union(*row_sets.values()))
    return {
        "domain_overlap_counts": {f"{a}|{b}": len(sets[a] & sets[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))},
        "domain_overlap_examples": overlaps,
        "no_domain_overlap": all(len(sets[a] & sets[b]) == 0 for a, b in (("train", "val"), ("train", "test"), ("val", "test"))),
        "row_ids_total": total_rows,
        "row_ids_unique": unique_rows,
        "row_ids_disjoint": total_rows == unique_rows,
    }


def summarise(parts: dict[str, pd.DataFrame], dedup_stats: dict[str, Any]) -> dict[str, Any]:
    """Per-split class balance, domain counts and ratios - the reported numbers."""
    summary: dict[str, Any] = {"dedup": dedup_stats, "splits": {}}
    for s, part in parts.items():
        counts = part["label_raw"].value_counts().to_dict()
        n = len(part)
        summary["splits"][s] = {
            "n_rows": int(n),
            "n_registered_domains": int(part["registered_domain"].nunique()),
            "label_counts": {str(k): int(v) for k, v in counts.items()},
            "label_fractions": {str(k): round(float(v) / max(n, 1), 6) for k, v in counts.items()},
        }
    return summary


# ---------------------------------------------------------------------------
def write_outputs(
    parts: dict[str, pd.DataFrame],
    domain_split: dict[str, str],
    dedup: pd.DataFrame,
    summary: dict[str, Any],
    out_dir: Path,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for s, part in parts.items():
        p = out_dir / f"{s}.txt"
        p.write_text("\n".join(str(v) for v in part["row_id"].tolist()) + "\n", encoding="utf-8")
        written[s] = p

    dom_df = pd.DataFrame(
        [{"registered_domain": d, "split": domain_split[d]} for d in sorted(domain_split)],
        columns=["registered_domain", "split"],
    )
    p = out_dir / "domain_split.csv"
    dom_df.to_csv(p, index=False)
    written["domain_split"] = p

    p = out_dir / "dedup.csv"
    dedup[["row_id", "url", "label_raw", "host", "registered_domain"]].to_csv(p, index=False)
    written["dedup"] = p

    p = out_dir / "split_summary.json"
    p.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    written["summary"] = p
    return written


def fingerprint(summary: dict[str, Any]) -> str:
    """Short deterministic fingerprint of a split, for the report."""
    blob = json.dumps(summary["splits"], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run(
    csv_path: Path,
    out_dir: Path,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
    label_column: str | None = None,
    url_column: str | None = None,
) -> dict[str, Any]:
    """End-to-end: load, dedup, group-split, verify, persist."""
    ratios = ratios or {"train": 0.70, "val": 0.15, "test": 0.15}
    if set(ratios) != set(SPLITS):
        raise ValueError(f"ratios must define exactly {list(SPLITS)}, got {sorted(ratios)}")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios.values())}")
    if any(v <= 0 for v in ratios.values()):
        raise ValueError(f"every ratio must be positive, got {ratios}")

    dedup, dedup_stats = load_and_deduplicate(csv_path, url_column, label_column)
    domain_split = grouped_split(dedup, ratios, seed)
    parts = apply_split(dedup, domain_split)
    verification = verify_no_overlap(parts)
    summary = summarise(parts, dedup_stats)
    summary["verification"] = verification
    summary["config"] = {
        "seed": seed,
        "ratios": ratios,
        "grouping": "registered_domain (tldextract, bundled PSL snapshot)",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary["fingerprint"] = fingerprint(summary)
    written = write_outputs(parts, domain_split, dedup, summary, out_dir)
    summary["written"] = {k: str(v) for k, v in written.items()}
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Create the grouped-by-domain split.")
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--out", type=Path, default=BACKEND_ROOT / "data" / "splits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--url-column")
    p.add_argument("--label-column")
    args = p.parse_args(argv)

    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-9:
        p.error(f"ratios must sum to 1.0, got {total}")

    summary = run(
        csv_path=args.csv,
        out_dir=args.out,
        seed=args.seed,
        ratios={"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        label_column=args.label_column,
        url_column=args.url_column,
    )

    print(json.dumps(summary, indent=2, default=str))
    v = summary["verification"]
    if not v["no_domain_overlap"]:
        print("FAIL: registered domains leak across splits", file=sys.stderr)
        return 1
    print("\nOK: no registered domain appears in more than one split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

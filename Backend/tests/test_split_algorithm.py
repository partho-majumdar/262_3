"""Unit tests for the grouped-split algorithm itself.

These build a SMALL SYNTHETIC fixture in a tmp directory. They prove the
algorithm is correct; they say nothing about PhiUSIIL and their numbers must
never be quoted as dataset results.

The complementary ``test_splits.py`` reads the *real* artefacts produced from
the real dataset and skips until they exist. Keeping the two apart prevents a
synthetic fixture from ever being mistaken for evidence about the dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from training.make_splits import (
    apply_split,
    grouped_split,
    load_and_deduplicate,
    registered_domain_of,
    run,
    verify_no_overlap,
)

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Fixture: a synthetic dataset with a known, deliberately leaky structure.
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    """60 domains x 3 URLs, 50/50 labels, plus deliberate duplicate rows."""
    rows = []
    for d in range(60):
        domain = f"host{d}.example"
        for k in range(3):
            # Half the domains carry both labels, which is exactly the case a
            # row-level split would leak.
            label = d % 2 if k < 2 else (d + 1) % 2
            rows.append({"url": f"http://{domain}/p{k}", "Label": label})
    # Exact duplicate rows and duplicate URLs with conflicting labels.
    rows.append(dict(rows[0]))
    rows.append({"url": rows[1]["url"], "Label": 1})
    path = tmp_path / "synthetic.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
def test_registered_domain_uses_public_suffix_list():
    assert registered_domain_of("login.paypal.com.evil.co.uk") == "evil.co.uk"
    assert registered_domain_of("www.google.com") == "google.com"
    assert registered_domain_of("a.b.example.com") == "example.com"
    # github.io is on the PSL so its subdomains share a registrable domain.
    # This is intentional: grouping them together is the safe direction.
    assert registered_domain_of("sub.domain.github.io") == "github.io"


def test_deduplication_removes_exact_and_duplicate_url_rows(synthetic_csv):
    df, stats = load_and_deduplicate(synthetic_csv, url_column=None, label_column=None)

    assert df["url"].is_unique, "duplicate URLs survived deduplication"
    # 180 synthetic rows + 2 injected duplicates.
    assert stats["raw_rows"] == 182
    assert stats["dropped_exact_duplicate_rows"] == 1
    # The conflicting-label duplicate is dropped too (first occurrence kept).
    assert stats["dropped_duplicate_url_rows"] == 1
    assert stats["final_rows"] == 180
    assert list(df["row_id"]) == list(range(180))


def test_dedup_keeps_first_label_for_a_repeated_url(synthetic_csv):
    df, _ = load_and_deduplicate(synthetic_csv, None, None)
    first = df[df["url"] == "http://host1.example/p1"].iloc[0]
    assert first["label_raw"] == "1", "dedup must keep the first occurrence's label"


def test_grouped_split_never_shares_a_domain(synthetic_csv):
    df, _ = load_and_deduplicate(synthetic_csv, None, None)
    assignment = grouped_split(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
    parts = apply_split(df, assignment)
    report = verify_no_overlap(parts)

    assert report["no_domain_overlap"] is True
    assert report["row_ids_disjoint"] is True
    assert all(c == 0 for c in report["domain_overlap_counts"].values())


def test_domains_carrying_both_labels_stay_together(synthetic_csv):
    """The whole reason for grouping: a mixed-label domain must not be divided.

    Grouping does not (and cannot) make a mixed-label domain single-label; what
    it guarantees is that *all* of that domain's rows land in the *same* split,
    so its label mixture can never inflate a train/test comparison.
    """
    df, _ = load_and_deduplicate(synthetic_csv, None, None)
    assignment = grouped_split(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
    parts = apply_split(df, assignment)

    split_of_row: dict[int, str] = {}
    for split, part in parts.items():
        for row_id in part["row_id"]:
            split_of_row[int(row_id)] = split

    for domain, part in df.groupby("registered_domain"):
        splits_used = {split_of_row[int(i)] for i in part["row_id"]}
        assert len(splits_used) == 1, (
            f"domain {domain!r} was spread across {sorted(splits_used)}; "
            "its mixed labels would leak into the evaluation"
        )


def test_row_level_split_would_have_leaked_for_contrast(synthetic_csv):
    """Demonstrates the bug grouped splitting prevents (guards the rationale)."""
    df, _ = load_and_deduplicate(synthetic_csv, None, None)
    shuffled = df.sample(frac=1.0, random_state=0)
    n = len(shuffled)
    train_doms = set(shuffled.iloc[: int(n * 0.7)]["registered_domain"])
    test_doms = set(shuffled.iloc[int(n * 0.8) :]["registered_domain"])
    leaked = train_doms & test_doms
    assert leaked, (
        "expected a random row-level split to share domains between train and test; "
        "if this ever fails the synthetic fixture no longer exercises the leakage case"
    )


def test_split_ratios_are_close_to_target(synthetic_csv):
    df, _ = load_and_deduplicate(synthetic_csv, None, None)
    assignment = grouped_split(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
    parts = apply_split(df, assignment)
    total = len(df)
    for split, target in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        assert abs(len(parts[split]) / total - target) <= 0.05


def test_split_is_deterministic_for_a_given_seed(synthetic_csv):
    a = grouped_split(
        load_and_deduplicate(synthetic_csv, None, None)[0], {"train": 0.7, "val": 0.15, "test": 0.15}, seed=7
    )
    b = grouped_split(
        load_and_deduplicate(synthetic_csv, None, None)[0], {"train": 0.7, "val": 0.15, "test": 0.15}, seed=7
    )
    assert a == b


def test_different_seeds_produce_different_partitions(synthetic_csv):
    df, _ = load_and_deduplicate(synthetic_csv, None, None)
    a = grouped_split(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
    b = grouped_split(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=2)
    assert a != b


def test_run_writes_every_artefact_and_verifies(synthetic_csv, tmp_path):
    out = tmp_path / "splits"
    summary = run(synthetic_csv, out, seed=42)

    for name in ("train.txt", "val.txt", "test.txt", "domain_split.csv", "dedup.csv", "split_summary.json"):
        assert (out / name).is_file(), f"missing artefact {name}"

    assert summary["verification"]["no_domain_overlap"] is True
    assert summary["verification"]["row_ids_disjoint"] is True
    assert summary["config"]["seed"] == 42
    assert summary["fingerprint"]

    saved = json.loads((out / "split_summary.json").read_text())
    assert saved["fingerprint"] == summary["fingerprint"]
    for split in SPLITS:
        ids = [int(x) for x in (out / f"{split}.txt").read_text().split()]
        assert ids == sorted(ids), f"{split}.txt must be in ascending row_id order"


def test_run_raises_on_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        run(tmp_path / "nope.csv", tmp_path / "out")


def test_ratios_must_sum_to_one(synthetic_csv, tmp_path):
    with pytest.raises(Exception):
        run(synthetic_csv, tmp_path / "out", ratios={"train": 0.5, "val": 0.15, "test": 0.15})
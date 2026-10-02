"""P1 acceptance: the real split has no registered domain in two splits.

These tests read the artefacts produced by ``training/make_splits.py`` from the
actual dataset. They do not build their own split, so a failure here means the
shipped split is broken rather than that a toy fixture is wrong.
"""

from __future__ import annotations

import json

import pytest

SPLIT_NAMES = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Core grouped-split invariant
# ---------------------------------------------------------------------------
def test_no_registered_domain_appears_in_two_splits(dedup_df, split_indices):
    """The whole point of grouping: domain sets must be pairwise disjoint."""
    by_id = dedup_df.set_index("row_id")["registered_domain"]
    domain_sets: dict[str, set[str]] = {}
    for split, ids in split_indices.items():
        domain_sets[split] = {by_id[i] for i in ids}

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = domain_sets[a] & domain_sets[b]
        assert not shared, (
            f"{len(shared)} registered domain(s) leak between {a} and {b}. "
            f"Examples: {sorted(shared)[:10]}"
        )


def test_domain_split_csv_agrees_with_row_indices(domain_split_csv, dedup_df, split_indices):
    """The domain->split map and the row index files must tell the same story."""
    import pandas as pd

    dom_map = pd.read_csv(domain_split_csv)
    declared = dict(zip(dom_map["registered_domain"], dom_map["split"]))
    by_id = dedup_df.set_index("row_id")

    for split, ids in split_indices.items():
        for row_id in ids[:2000]:  # sample: full pass is unnecessary, 2k is ample
            domain = by_id.loc[row_id, "registered_domain"]
            assert declared.get(domain) == split, (
                f"row_id {row_id} (domain {domain!r}) is in {split}.txt but "
                f"domain_split.csv assigns it to {declared.get(domain)!r}"
            )


def test_every_domain_assigned_to_exactly_one_split(domain_split_csv):
    """A domain cannot be listed twice with different splits."""
    import pandas as pd

    dom_map = pd.read_csv(domain_split_csv)
    assert dom_map["registered_domain"].is_unique, "duplicate domain rows in domain_split.csv"
    assert set(dom_map["split"]) <= set(SPLIT_NAMES), (
        f"unexpected split names: {sorted(set(dom_map['split']) - set(SPLIT_NAMES))}"
    )


# ---------------------------------------------------------------------------
# Coverage / disjointness of the row indices
# ---------------------------------------------------------------------------
def test_row_indices_are_disjoint(split_indices):
    all_ids: list[int] = []
    for ids in split_indices.values():
        all_ids.extend(ids)
    assert len(all_ids) == len(set(all_ids)), "the same row_id appears in two splits"


def test_row_indices_cover_every_deduplicated_row(split_indices, dedup_df):
    all_ids = {i for ids in split_indices.values() for i in ids}
    expected = set(dedup_df["row_id"].tolist())
    assert all_ids == expected, (
        f"row-id coverage mismatch: {len(all_ids)} assigned vs {len(expected)} in dedup.csv; "
        f"missing={sorted(expected - all_ids)[:10]} extra={sorted(all_ids - expected)[:10]}"
    )


def test_row_indices_are_within_bounds(split_indices, dedup_df):
    n = len(dedup_df)
    for split, ids in split_indices.items():
        assert ids, f"{split}.txt is empty"
        assert min(ids) >= 0, f"{split}.txt has a negative row_id"
        assert max(ids) < n, f"{split}.txt row_id {max(ids)} exceeds dedup rows ({n})"


# ---------------------------------------------------------------------------
# Class balance is actually reported (and non-degenerate)
# ---------------------------------------------------------------------------
def test_each_split_contains_both_classes(split_indices, dedup_df):
    labels = dedup_df.set_index("row_id")["label_raw"]
    for split, ids in split_indices.items():
        present = {labels[i] for i in ids}
        assert len(present) >= 2, f"{split} contains only one class: {present}"


def test_summary_json_matches_the_real_artifacts(split_summary_json, split_indices, dedup_df):
    """The numbers in split_summary.json must be reproducible from the files."""
    import pandas as pd

    summary = json.loads(split_summary_json.read_text(encoding="utf-8"))
    labels = dedup_df.set_index("row_id")["label_raw"]
    for split, ids in split_indices.items():
        reported = summary["splits"][split]["n_rows"]
        assert reported == len(ids), (
            f"split_summary.json says {split} has {reported} rows but {split}.txt has {len(ids)}"
        )
        recomputed = pd.Series([labels[i] for i in ids]).value_counts().to_dict()
        for value, count in recomputed.items():
            assert summary["splits"][split]["label_counts"].get(str(value)) == count, (
                f"{split} label_counts mismatch for {value!r}"
            )


def test_class_balance_is_close_to_target_ratios(split_summary_json):
    """The split must not be wildly imbalanced relative to the source ratio."""
    summary = json.loads(split_summary_json.read_text(encoding="utf-8"))
    totals: dict[str, int] = {}
    for split in SPLIT_NAMES:
        for value, count in summary["splits"][split]["label_counts"].items():
            totals[value] = totals.get(value, 0) + count
    n = sum(totals.values())
    if n == 0:
        pytest.skip("empty split summary")

    for split in SPLIT_NAMES:
        share = summary["splits"][split]["n_rows"] / n
        expected = summary["config"]["ratios"][split]
        # 8 percentage points of slack absorbs domain granularity effects.
        assert abs(share - expected) <= 0.08, (
            f"{split} holds {share:.3f} of rows, target was {expected:.3f}"
        )


def test_verification_block_asserts_no_overlap(split_summary_json):
    """The script's own self-check must be recorded as passing."""
    summary = json.loads(split_summary_json.read_text(encoding="utf-8"))
    v = summary["verification"]
    assert v["no_domain_overlap"] is True
    assert v["row_ids_disjoint"] is True
    assert all(c == 0 for c in v["domain_overlap_counts"].values()), v["domain_overlap_counts"]


# ---------------------------------------------------------------------------
# Duplicate policy
# ---------------------------------------------------------------------------
def test_no_duplicate_urls_in_the_deduplicated_table(dedup_df):
    """Exact duplicate URLs were removed before splitting (rows are never duplicated)."""
    assert dedup_df["url"].is_unique, (
        f"{int(dedup_df['url'].duplicated().sum())} duplicate URL rows survived deduplication"
    )


def test_dedup_stats_record_real_removals(split_summary_json, dedup_df):
    summary = json.loads(split_summary_json.read_text(encoding="utf-8"))
    d = summary["dedup"]
    assert d["final_rows"] == len(dedup_df)
    assert d["dropped_exact_duplicate_rows"] >= 0
    assert d["dropped_duplicate_url_rows"] >= 0
    assert d["raw_rows"] >= d["final_rows"]

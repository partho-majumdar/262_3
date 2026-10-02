"""Tests for split loading, label mapping, and the dev-profile subsampler.

These guard the seam between the P1 artifacts on disk and the tensors the model
consumes. A mislabelled or misaligned row would silently corrupt training while
every metric still looked plausible.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.preprocessing.url_dataset import (
    PHISHING_TOKENS,
    SPLIT_NAMES,
    SplitData,
    URLDataset,
    load_splits,
    make_loader,
    subsample,
    to_label_array,
)
from app.preprocessing.url_features import FeatureScaler, extract_batch
from app.preprocessing.url_preprocessing import CharTokenizer


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("token", sorted(PHISHING_TOKENS))
def test_every_phishing_token_maps_to_one(token):
    assert to_label_array([token])[0] == 1


def test_legitimate_tokens_map_to_zero():
    assert to_label_array(["0", "legitimate", "benign", "good", "normal"]).tolist() == [0] * 5


def test_label_mapping_is_case_and_whitespace_insensitive():
    assert to_label_array([" 1 ", "PhIsHiNg", "TRUE"]).tolist() == [1, 1, 1]


def test_unknown_label_is_refused_rather_than_guessed():
    """Silently defaulting an unknown label to 0 would corrupt every metric."""
    with pytest.raises(ValueError, match="refusing to guess"):
        to_label_array(["maybe"])


def test_label_array_dtype_is_int64():
    assert to_label_array(["1", "0"]).dtype == np.int64


# ---------------------------------------------------------------------------
# Split loading
# ---------------------------------------------------------------------------
@pytest.fixture()
def fake_splits_dir(tmp_path):
    """A miniature split directory with the same on-disk layout as P1's."""
    import pandas as pd

    rows = [
        (0, "http://a.test", "1", "a.test"),
        (1, "http://b.test", "0", "b.test"),
        (2, "http://c.test", "1", "c.test"),
        (3, "http://d.test", "0", "d.test"),
        (4, "http://e.test", "1", "e.test"),
        (5, "http://f.test", "0", "f.test"),
    ]
    pd.DataFrame(
        rows, columns=["row_id", "url", "label_raw", "registered_domain"]
    ).to_csv(tmp_path / "dedup.csv", index=False)

    (tmp_path / "train.txt").write_text("0 2 4", encoding="utf-8")
    (tmp_path / "val.txt").write_text("1\n3", encoding="utf-8")
    (tmp_path / "test.txt").write_text("5", encoding="utf-8")
    return tmp_path


def test_load_splits_returns_all_three_splits(fake_splits_dir):
    splits = load_splits(fake_splits_dir)
    assert set(splits) == set(SPLIT_NAMES)
    assert [len(splits[n]) for n in SPLIT_NAMES] == [3, 2, 1]


def test_load_splits_keeps_urls_and_labels_aligned(fake_splits_dir):
    """Each URL must sit beside its own label, in the requested row order."""
    splits = load_splits(fake_splits_dir)
    train = splits["train"]
    assert train.urls == ["http://a.test", "http://c.test", "http://e.test"]
    assert train.labels.tolist() == [1, 1, 1]
    assert train.row_ids.tolist() == [0, 2, 4]


def test_load_splits_handles_newlines_and_trailing_space_in_index_files(fake_splits_dir):
    splits = load_splits(fake_splits_dir)
    assert splits["val"].row_ids.tolist() == [1, 3]


def test_load_splits_handles_a_single_row_split(fake_splits_dir):
    assert load_splits(fake_splits_dir)["test"].labels.tolist() == [0]


def test_missing_dedup_file_gives_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_splits.py"):
        load_splits(tmp_path)


def test_missing_index_file_is_reported_by_name(fake_splits_dir):
    (fake_splits_dir / "val.txt").unlink()
    with pytest.raises(FileNotFoundError, match="val.txt"):
        load_splits(fake_splits_dir)


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------
def _split(n: int, name: str = "train") -> SplitData:
    labels = np.array([i % 2 for i in range(n)], dtype=np.int64)
    return SplitData(
        name=name,
        urls=[f"http://h{i}.test" for i in range(n)],
        labels=labels,
        row_ids=np.arange(n, dtype=np.int64),
    )


def test_subsample_below_target_returns_the_original_object():
    s = _split(10)
    assert subsample(s, 100) is s


def test_subsample_none_returns_the_original_object():
    s = _split(10)
    assert subsample(s, None) is s


def test_subsample_respects_the_cap():
    assert len(subsample(_split(1000), 200)) == 200


def test_subsample_keeps_both_classes():
    out = subsample(_split(1000), 200)
    assert set(out.labels.tolist()) == {0, 1}


def test_subsample_preserves_the_class_balance_of_the_source():
    src = _split(1000)
    src.labels = np.array([1] * 700 + [0] * 300, dtype=np.int64)
    out = subsample(src, 200)
    assert abs(out.labels.mean() - src.labels.mean()) < 0.05


def test_subsample_keeps_urls_labels_and_row_ids_in_step():
    out = subsample(_split(500), 100)
    for i, rid in enumerate(out.row_ids):
        assert out.urls[i] == f"http://h{rid}.test"
        assert out.labels[i] == int(rid) % 2


def test_subsample_is_deterministic_for_a_fixed_seed():
    a = subsample(_split(1000), 200, seed=7)
    b = subsample(_split(1000), 200, seed=7)
    assert a.row_ids.tolist() == b.row_ids.tolist()


def test_different_seeds_give_different_subsamples():
    a = subsample(_split(1000), 200, seed=1)
    b = subsample(_split(1000), 200, seed=2)
    assert a.row_ids.tolist() != b.row_ids.tolist()


def test_subsample_handles_a_split_with_one_class():
    s = _split(10)
    s.labels = np.ones(10, dtype=np.int64)
    out = subsample(s, 4)
    assert len(out) > 0
    assert out.labels.tolist() == [1] * len(out)


def test_subsample_preserves_the_split_name():
    assert subsample(_split(100, name="val"), 20).name == "val"


# ---------------------------------------------------------------------------
# Dataset tensors
# ---------------------------------------------------------------------------
@pytest.fixture()
def tokenizer():
    return CharTokenizer(max_length=64, min_count=1).fit(
        ["http://a.test", "http://b.test", "http://longer.example.test/path?q=1"]
    )


def test_dataset_yields_four_aligned_tensors(tokenizer):
    ds = URLDataset(_split(4), tokenizer, scaler=None)
    ids, mask, feats, label = ds[0]
    assert ids.dtype == torch.long
    assert mask.shape == ids.shape
    assert feats.shape == (0,)
    assert label.shape == ()


def test_dataset_char_width_matches_max_length(tokenizer):
    ds = URLDataset(_split(4), tokenizer, scaler=None)
    assert ds[0][0].shape[0] == tokenizer.max_length


def test_dataset_padding_is_masked_out(tokenizer):
    """Short URLs pad with zero ids; the mask must exclude them."""
    ds = URLDataset(_split(4), tokenizer, scaler=None)
    ids, mask, _, _ = ds[0]
    n_real = int(mask.sum())
    assert ids[n_real:].eq(0).all(), "padding ids should be zero"
    assert mask[:n_real].eq(1).all()


def test_dataset_label_tensor_is_float32(tokenizer):
    ds = URLDataset(_split(4), tokenizer, scaler=None)
    assert ds[0][3].dtype == torch.float32


def test_dataset_without_scaler_uses_the_text_branch_only(tokenizer):
    ds = URLDataset(_split(4), tokenizer, scaler=None)
    assert ds.use_handcrafted is False
    assert ds[0][2].numel() == 0


def test_dataset_with_scaler_populates_the_feature_branch(tokenizer):
    urls = _split(30)
    scaler = FeatureScaler().fit(extract_batch(urls.urls))
    ds = URLDataset(urls, tokenizer, scaler)
    assert ds.use_handcrafted is True
    assert ds[0][2].shape == (len(extract_batch(urls.urls)[0]),)


def test_dataset_indexing_matches_source_order(tokenizer):
    ds = URLDataset(_split(6), tokenizer, scaler=None)
    assert ds[3][3].item() == 3 % 2


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def test_loader_yields_the_expected_batch_count(tokenizer):
    ds = URLDataset(_split(10), tokenizer, scaler=None)
    loader = make_loader(ds, batch_size=4, shuffle=False)
    assert len(loader) == 3


def test_loader_does_not_drop_the_last_partial_batch(tokenizer):
    """drop_last=True would silently discard training rows."""
    ds = URLDataset(_split(10), tokenizer, scaler=None)
    loader = make_loader(ds, batch_size=4, shuffle=False)
    assert sum(b[0].shape[0] for b in loader) == 10


def test_loader_preserves_order_when_not_shuffling(tokenizer):
    ds = URLDataset(_split(8), tokenizer, scaler=None)
    labels = torch.cat([b[3] for b in make_loader(ds, batch_size=3, shuffle=False)])
    assert labels.tolist() == [i % 2 for i in range(8)]


def test_shuffled_loader_still_covers_every_row(tokenizer):
    ds = URLDataset(_split(20), tokenizer, scaler=None)
    labels = torch.cat([b[3] for b in make_loader(ds, batch_size=6, shuffle=True)])
    assert sorted(labels.tolist()) == sorted(ds.labels.tolist())

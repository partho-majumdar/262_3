"""Dataset assembly from the saved split indices.

The split index files (``data/splits/{train,val,test}.txt``) contain row ids into
``data/splits/dedup.csv``. This module is the single place that turns those files
into tensors, so training, evaluation and the API can never disagree about which
row belongs to which split.

Only ``url`` (and optionally the label) is ever read from ``dedup.csv`` for the
URL model. The page-derived engineered columns live in the original PhiUSIIL
file and are loaded *only* by the P2 leakage experiment, which is deliberately a
separate code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from app.preprocessing.url_features import (
    FEATURE_NAMES,
    FeatureScaler,
    extract_batch,
)
from app.preprocessing.url_preprocessing import CharTokenizer

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class SplitData:
    """Materialised arrays for one split."""

    name: str
    urls: list[str]
    labels: np.ndarray  # int64 (B,), 1 = phishing
    row_ids: np.ndarray  # int64 (B,)

    def __len__(self) -> int:
        return len(self.urls)


class URLDataset(Dataset):
    """Character tensors (+ optional standardised handcrafted features) per URL."""

    def __init__(
        self,
        split: SplitData,
        tokenizer: CharTokenizer,
        scaler: FeatureScaler | None,
    ) -> None:
        self.urls = split.urls
        self.labels = torch.as_tensor(split.labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.scaler = scaler
        self.use_handcrafted = scaler is not None

        ids, mask = tokenizer.encode_batch(self.urls)
        self.char_ids = torch.tensor(ids, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.float32)

        if self.use_handcrafted:
            feats = extract_batch(self.urls)
            scaled = scaler.transform(feats)  # type: ignore[union-attr]
            self.handcrafted = torch.tensor(scaled, dtype=torch.float32)
        else:
            self.handcrafted = torch.zeros((len(self.urls), 0), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.urls)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.char_ids[idx],
            self.mask[idx],
            self.handcrafted[idx],
            self.labels[idx],
        )


def load_splits(splits_dir: Path) -> dict[str, SplitData]:
    """Read dedup.csv + the index files into per-split arrays."""
    import pandas as pd

    splits_dir = Path(splits_dir)
    dedup_path = splits_dir / "dedup.csv"
    if not dedup_path.is_file():
        raise FileNotFoundError(
            f"Missing {dedup_path}. Run: python training/make_splits.py --csv <PhiUSIIL.csv>"
        )

    df = pd.read_csv(dedup_path, dtype={"url": str, "label_raw": str})
    by_id = df.set_index("row_id")

    out: dict[str, SplitData] = {}
    for name in SPLIT_NAMES:
        idx_path = splits_dir / f"{name}.txt"
        if not idx_path.is_file():
            raise FileNotFoundError(f"Missing {idx_path}")
        ids = np.array([int(x) for x in idx_path.read_text(encoding="utf-8").split() if x.strip()], dtype=np.int64)
        rows = by_id.loc[ids]
        urls = rows["url"].astype(str).tolist()
        labels = to_label_array(rows["label_raw"].astype(str).tolist())
        out[name] = SplitData(name=name, urls=urls, labels=labels, row_ids=ids)
    return out


#: Raw label tokens meaning "phishing" in the dataset, matched by
#: ``make_splits``. Kept here so the mapping is defined in exactly one place.
PHISHING_TOKENS = {"1", "phishing", "phish", "malicious", "bad", "true", "yes", "attack"}


def to_label_array(raw: Sequence[str]) -> np.ndarray:
    """Map raw label strings to 0/1, refusing anything ambiguous."""
    out = []
    for r in raw:
        token = str(r).strip().lower()
        if token in PHISHING_TOKENS:
            out.append(1)
        elif token in {"0", "legitimate", "benign", "good", "false", "no", "normal", "safe"}:
            out.append(0)
        else:
            raise ValueError(f"uninterpretable label {r!r}; refusing to guess")
    return np.asarray(out, dtype=np.int64)


def make_loader(
    dataset: URLDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        drop_last=False,
    )


def subsample(
    split: SplitData,
    max_rows: int | None,
    seed: int = 42,
) -> SplitData:
    """Class-stratified subsample used by the dev profile.

    Stratified so a reduced run still exercises both classes in proportion; a
    naive head/truncation would silently drop a class depending on file order.
    """
    if max_rows is None or len(split) <= max_rows:
        return split
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    idx_all = np.arange(len(split))
    for cls in (0, 1):
        cls_idx = idx_all[split.labels == cls]
        if cls_idx.size == 0:
            continue
        quota = int(round(max_rows * cls_idx.size / split.labels.size))
        quota = min(quota, cls_idx.size)
        keep.extend(rng.choice(cls_idx, size=quota, replace=False).tolist())
    keep_arr = np.array(sorted(keep), dtype=np.int64)
    return SplitData(
        name=split.name,
        urls=[split.urls[i] for i in keep_arr],
        labels=split.labels[keep_arr],
        row_ids=split.row_ids[keep_arr],
    )
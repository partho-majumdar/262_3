"""Throughput probe: how fast is one training step on this CPU?

Used to size the dev/full epoch budget before committing to a long run. This is a
measurement tool, not part of the training pipeline.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np
import torch

from app.models.url_model import URLCharModel
from app.preprocessing.url_dataset import URLDataset, SplitData, load_splits
from app.preprocessing.url_features import FEATURE_NAMES, FeatureScaler, extract_batch
from app.preprocessing.url_preprocessing import CharTokenizer


def main() -> int:
    splits = load_splits(BACKEND_ROOT / "data" / "splits")
    train = splits["train"]
    n = 2000
    sub = SplitData(
        name="probe",
        urls=train.urls[:n],
        labels=train.labels[:n],
        row_ids=train.row_ids[:n],
    )

    tk = CharTokenizer(max_length=256, min_count=5)
    tk.fit(sub.urls)
    sc = FeatureScaler().fit(extract_batch(sub.urls))
    ds = URLDataset(sub, tk, sc)

    model = URLCharModel(
        vocab_size=tk.vocab_size, max_length=tk.max_length,
        embedding_dim=64, cnn_channels=48, lstm_hidden=48, embedding_out=128,
        dropout=0.35, n_handcrafted=len(FEATURE_NAMES), use_handcrafted=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = torch.nn.BCEWithLogitsLoss()

    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    model.train()
    t0 = time.perf_counter()
    steps = 0
    for char_ids, mask, hand, y in loader:
        opt.zero_grad()
        out = model(char_ids, mask, hand)
        loss = crit(out.logit, y)
        loss.backward()
        opt.step()
        steps += 1
        if steps >= 20:
            break
    elapsed = time.perf_counter() - t0
    per_step = elapsed / steps
    per_sample = per_step / 32

    full = len(splits["train"])
    print(f"vocab_size        = {tk.vocab_size}")
    print(f"max_length        = {tk.max_length}")
    print(f"steps_measured    = {steps}")
    print(f"sec_per_step      = {per_step:.4f}")
    print(f"sec_per_sample    = {per_sample:.5f}")
    print()
    for cap in (40000, 164760):
        secs = cap * per_sample
        print(
            f"epoch over {cap:>7,} rows = {secs / 60:6.1f} min"
            f"   | 5 epochs = {5 * secs / 60:7.1f} min"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
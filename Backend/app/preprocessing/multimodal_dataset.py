"""Join the acquisition manifest into tensors for the HTML / vision / fusion models.

Every modality here is *optional*. A row can arrive with a URL but no HTML, or
with HTML but no screenshot, because the page was dead, too large, of the wrong
content type, or blocked. The loader therefore emits, for each row:

* the per-modality feature vectors (zeros where unavailable), and
* a boolean **availability mask** per modality.

The mask is not a convenience. The fusion network is explicitly *mask-aware*: it
must learn to ignore a modality that is missing rather than reading the zero
vector as evidence. Conflating "absent" with "measured zero" is the single
easiest way to make a multimodal model look better than it is.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.preprocessing.html_features import (  # noqa: E402
    HTML_FEATURE_NAMES,
    N_HTML_FEATURES,
    extract_html_features,
    visible_text_of,
)

__all__ = [
    "MODALITIES",
    "MultiModalRow",
    "load_manifest",
    "build_rows",
    "MultiModalDataset",
    "collate_multimodal",
]

MODALITIES = ("url", "html", "vision")


def _soup_of(html: str):
    """Parse once; BeautifulSoup with the fast backend, falling back on failure."""
    from bs4 import BeautifulSoup

    try:
        return BeautifulSoup(html, "lxml")
    except Exception:  # pragma: no cover
        return BeautifulSoup(html, "html.parser")


@dataclass
class MultiModalRow:
    """One example, with per-modality availability."""

    row_id: int
    split: str
    url: str
    label: int
    html: str | None
    html_vec: np.ndarray
    image_path: Path | None
    url_chars: np.ndarray | None = None
    url_feats: np.ndarray | None = None
    url_mask: np.ndarray | None = None
    html_tokens: np.ndarray | None = None
    html_token_mask: np.ndarray | None = None

    @property
    def has_html(self) -> bool:
        return self.html is not None

    @property
    def has_vision(self) -> bool:
        return self.image_path is not None and self.image_path.is_file()

    def mask(self) -> dict[str, bool]:
        return {
            "url": True,
            "html": self.has_html,
            "vision": self.has_vision,
        }


def load_manifest(manifest_path: Path) -> "object":
    """Read the collector manifest."""
    import pandas as pd

    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run: python training/collect_multimodal.py"
        )
    return pd.read_csv(manifest_path, dtype={"url": str})


def _read_html(path: str, limit: int = 200_000) -> str | None:
    if not path or not str(path).strip():
        return None
    p = Path(str(path).strip())
    if not p.is_file():
        return None
    try:
        return p.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return None


def build_rows(
    manifest: "object",
    *,
    tokenizer=None,
    scaler=None,
    html_tokenizer=None,
    splits: tuple[str, ...] | None = None,
) -> list[MultiModalRow]:
    """Materialise rows with HTML features computed once, up front."""
    from app.preprocessing.url_features import extract_batch

    df = manifest
    if splits:
        df = df[df["split"].isin(splits)]
    df = df.reset_index(drop=True)

    html_docs = [
        _read_html(p, limit=200_000) if str(ok).lower() in {"true", "1"} else None
        for p, ok in zip(df.get("html_path", [""] * len(df)), df.get("html_ok", [0] * len(df)))
    ]
    html_vecs = np.array(
        [extract_html_features(d) for d in html_docs], dtype=np.float32
    ).reshape(len(df), N_HTML_FEATURES)

    shot_paths = [
        Path(str(p).strip())
        if str(ok).lower() in {"true", "1"} and str(p).strip()
        else None
        for p, ok in zip(df.get("shot_path", [""] * len(df)), df.get("shot_ok", [0] * len(df)))
    ]

    html_texts = [
        visible_text_of(_soup_of(d))[:4000] if d else "" for d in html_docs
    ]
    if html_tokenizer is not None:
        tids, tmask = html_tokenizer.encode_batch(html_texts)
        html_tokens = torch.tensor(tids, dtype=torch.long)
        html_token_mask = torch.tensor(tmask, dtype=torch.float32)
    else:
        html_tokens = html_token_mask = None

    url_chars = url_feats = None
    if tokenizer is not None:
        ids, mask = tokenizer.encode_batch([str(u) for u in df["url"]])
        url_chars = torch.tensor(ids, dtype=torch.long)
        url_mask = torch.tensor(mask, dtype=torch.float32)
    if scaler is not None:
        url_feats = torch.tensor(
            scaler.transform(extract_batch([str(u) for u in df["url"]])), dtype=torch.float32
        )
    else:
        url_mask = torch.zeros((len(df), 0), dtype=torch.float32)

    rows: list[MultiModalRow] = []
    for i in range(len(df)):
        r = df.iloc[i]
        rows.append(
            MultiModalRow(
                row_id=int(r["row_id"]),
                split=str(r["split"]),
                url=str(r["url"]),
                label=int(r["label"]),
                html=html_docs[i],
                html_vec=html_vecs[i],
                image_path=shot_paths[i],
                url_chars=None if url_chars is None else url_chars[i],
                url_feats=url_feats[i] if url_feats is not None else None,
                url_mask=None if url_mask is None else url_mask[i],
                html_tokens=None if html_tokens is None else html_tokens[i],
                html_token_mask=None if html_token_mask is None else html_token_mask[i],
            )
        )
    return rows


class MultiModalDataset(Dataset):
    """Serves the three modalities plus their availability masks."""

    def __init__(
        self,
        rows: list[MultiModalRow],
        image_size: int = 224,
        train_augment: bool = False,
    ) -> None:
        self.rows = rows
        self.image_size = image_size
        self.train_augment = train_augment
        self._tokenizer = None

    def __len__(self) -> int:
        return len(self.rows)

    def _image_tensor(self, path: Path | None):
        """Load and normalise one screenshot, or zeros plus mask=0."""
        if path is None or not Path(path).is_file():
            return torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
        from PIL import Image
        from torchvision import transforms

        if not hasattr(self, "_tf"):
            tf = []
            if self.train_augment:
                tf += [
                    transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                ]
            tf += [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
            self._tf = transforms.Compose(tf)

        try:
            with Image.open(path) as im:
                return self._tf(im.convert("RGB"))
        except Exception:
            # A corrupt or truncated PNG must not abort a batch; mask it out.
            return torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        m = r.mask()
        return {
            "label": torch.tensor(float(r.label), dtype=torch.float32),
            "url_chars": r.url_chars,
            "url_feats": r.url_feats,
            "url_mask": (
                r.url_mask
                if r.url_mask is not None
                else torch.zeros(0, dtype=torch.float32)
            ),
            "html_vec": torch.tensor(r.html_vec, dtype=torch.float32),
            "html_tokens": (
                r.html_tokens
                if r.html_tokens is not None
                else torch.zeros(0, dtype=torch.long)
            ),
            "html_token_mask": (
                r.html_token_mask
                if r.html_token_mask is not None
                else torch.zeros(0, dtype=torch.float32)
            ),
            "image": self._image_tensor(r.image_path if m["vision"] else None),
            "mask_url": torch.tensor(1.0 if m["url"] else 0.0),
            "mask_html": torch.tensor(1.0 if m["html"] else 0.0),
            "mask_vision": torch.tensor(1.0 if m["vision"] else 0.0),
        }


def collate_multimodal(batch: list[dict]) -> dict:
    """Stack a batch, substituting zeros for absent per-row tensors.

    A batch may mix rows that have a modality with rows that do not (missing HTML
    is the normal case at inference). Those rows arrive as zero-length tensors, so
    padding to the longest *present* tensor is required before stacking - simply
    stacking raises on mixed batches.
    """
    out: dict = {"label": torch.stack([b["label"] for b in batch])}
    for key in (
        "url_chars",
        "url_feats",
        "url_mask",
        "html_vec",
        "image",
        "html_tokens",
        "html_token_mask",
    ):
        vals = [b[key] for b in batch]
        present = [v for v in vals if v is not None and getattr(v, "numel", lambda: 0)() > 0]
        if not present:
            out[key] = torch.zeros((len(batch), 0), dtype=torch.float32)
            continue
        width = max(v.shape[0] for v in present)
        tail = present[0].shape[1:]
        filled = []
        for v in vals:
            if v is None or v.shape[0] == 0:
                filled.append(
                    torch.zeros((width, *tail), dtype=present[0].dtype)
                )
            else:
                filled.append(v)
        out[key] = torch.stack(filled)
    for key in ("mask_url", "mask_html", "mask_vision"):
        out[key] = torch.stack([b[key] for b in batch])
    return out
